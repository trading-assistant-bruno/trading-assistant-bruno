from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mscidata import msci

import us_backtest as base
import us_backtest_fixed6 as pit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "active_momentum_rotation"
OUT.mkdir(parents=True, exist_ok=True)

DATA_START = pd.Timestamp("2009-01-01")
TEST_START = pd.Timestamp("2011-01-03")
END_EXCLUSIVE = pd.Timestamp("2026-09-01")
INITIAL = 100_000.0
COMMISSION = 0.0008
SLIPPAGE = 0.0005
MIN_COMMISSION = 1.0
MIN_ADV20 = 20_000_000.0
MIN_PRICE = 5.0


@dataclass(frozen=True)
class Variant:
    name: str
    score_kind: str
    entry_rank: int
    exit_rank: int
    target_n: int
    stock_trend_filter: bool
    market_gate: bool


VARIANTS = [
    Variant("Mom12_Top50", "mom12", 50, 50, 50, False, False),
    Variant("Mom12_Top20_Buffer40", "mom12", 20, 40, 20, False, False),
    Variant("Composite_Top20_Buffer40", "composite", 20, 40, 20, False, False),
    Variant("Composite_Top20_Buffer40_Trend", "composite", 20, 40, 20, True, False),
    Variant("Composite_Top20_Buffer40_TrendGate", "composite", 20, 40, 20, True, True),
]


def monthly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def load_history_symbols():
    hist = pit.load_history().copy()
    hist = hist[hist.date >= DATA_START - pd.Timedelta(days=7)].reset_index(drop=True)
    symbols = set()
    for m in hist.members:
        symbols.update(m)
    symbols.add("SPY")
    return hist, sorted(symbols)


def memberships(hist, dates):
    hd = hist.date.to_numpy(dtype="datetime64[ns]")
    out = {}
    for d in dates:
        nd = pd.Timestamp(d).normalize()
        pos = hd.searchsorted(nd.to_datetime64(), side="right") - 1
        out[nd] = set(hist.iloc[int(pos)].members) if pos >= 0 else set()
    return out


def indicators(close, volume):
    ret = close.pct_change()
    spy = close.SPY
    sr = spy.pct_change()
    mom12 = close.shift(21) / close.shift(252) - 1
    mom6 = close.shift(21) / close.shift(126) - 1
    high52 = close / close.rolling(252).max()
    sma200 = close.rolling(200).mean()
    adv20 = (close * volume).rolling(20).mean()
    beta252 = ret.rolling(252).cov(sr).div(sr.rolling(252).var(), axis=0)
    spy12 = spy.shift(21) / spy.shift(252) - 1
    residual = mom12.sub(beta252.mul(spy12, axis=0))
    market_ok = (spy > spy.rolling(200).mean()) & (spy / spy.shift(252) - 1 > 0)
    return dict(mom12=mom12, mom6=mom6, high52=high52, sma200=sma200, adv20=adv20, residual=residual, market_ok=market_ok)


def section(date, members, close, ind, kind):
    cols = [t for t in members if t in close.columns and t != "SPY"]
    x = pd.DataFrame(index=cols)
    if not cols:
        return x
    for k in ["mom12", "mom6", "high52", "sma200", "adv20", "residual"]:
        x[k] = ind[k].loc[date, cols]
    x["close"] = close.loc[date, cols]
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["close", "mom12", "mom6", "high52", "adv20", "sma200"])
    if kind == "composite":
        x = x.dropna(subset=["residual"])
    if x.empty:
        return x
    if kind == "mom12":
        x["score"] = x.mom12.rank(pct=True)
    else:
        x["score"] = (
            .40 * x.mom12.rank(pct=True)
            + .20 * x.mom6.rank(pct=True)
            + .15 * x.high52.rank(pct=True)
            + .25 * x.residual.rank(pct=True)
        )
    x["rank"] = x.score.rank(ascending=False, method="first")
    x["base_eligible"] = (x.close >= MIN_PRICE) & (x.adv20 >= MIN_ADV20) & (x.mom12 > 0)
    return x.sort_values("rank")


def world_netr(start, end):
    df = msci.get_levels("990100", start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), variant="NETR").copy()
    df["DATE"] = pd.to_datetime(df.DATE, errors="coerce")
    df["LEVEL"] = pd.to_numeric(df.LEVEL, errors="coerce")
    s = df.dropna(subset=["DATE", "LEVEL"]).set_index("DATE").LEVEL.sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep="last")]


def choose_targets(v, cs, incumbents, gate_ok):
    if v.market_gate and not gate_ok:
        return []
    if cs.empty:
        return []
    elig = cs[cs.base_eligible].copy()
    if v.stock_trend_filter:
        elig = elig[elig.close > elig.sma200]
    ranks = cs["rank"].to_dict()
    survivors = [t for t in incumbents if t in elig.index and ranks.get(t, np.inf) <= v.exit_rank]
    # Keep the best surviving incumbents, then fill from entry zone.
    survivors = sorted(survivors, key=lambda t: ranks.get(t, np.inf))[:v.target_n]
    targets = list(survivors)
    for t in elig.index:
        if len(targets) >= v.target_n:
            break
        if ranks[t] <= v.entry_rank and t not in targets:
            targets.append(t)
    return targets


def commission(notional):
    return max(MIN_COMMISSION, abs(notional) * COMMISSION)


def simulate(v, open_, close, signal_dates, secs, ind):
    dates = close.index[(close.index >= TEST_START) & (close.index < END_EXCLUSIVE)]
    sigset = set(signal_dates)
    cash = INITIAL
    shares = {}
    pending = None
    trades = []
    equity = []
    turnover_notional = 0.0

    for date in dates:
        # Execute prior month-end target at next session open.
        if pending is not None:
            sig_date, targets = pending
            opens = {t: float(open_.at[date, t]) for t in set(shares) | set(targets) if t in open_.columns and pd.notna(open_.at[date, t])}
            eq_open = cash + sum(q * opens.get(t, float(close.loc[:date, t].dropna().iloc[-1])) for t, q in shares.items())
            target_w = 1.0 / len(targets) if targets else 0.0
            desired = {t: eq_open * target_w for t in targets}

            # Sell reductions/exits first.
            for t in list(shares):
                if t not in opens:
                    continue
                cur = shares[t] * opens[t]
                des = desired.get(t, 0.0)
                if cur > des + 1.0:
                    raw_qty = (cur - des) / opens[t]
                    qty = min(shares[t], raw_qty)
                    px = opens[t] * (1 - SLIPPAGE)
                    notion = qty * px
                    fee = commission(notion)
                    cash += notion - fee
                    shares[t] -= qty
                    turnover_notional += notion
                    trades.append({"date": date, "ticker": t, "side": "SELL", "notional": notion, "fee": fee})
                    if shares[t] <= 1e-10:
                        del shares[t]

            # Buy increases/new names.
            for t in targets:
                if t not in opens:
                    continue
                cur = shares.get(t, 0.0) * opens[t]
                des = desired[t]
                if des > cur + 1.0:
                    px = opens[t] * (1 + SLIPPAGE)
                    max_notional = max(0.0, cash - MIN_COMMISSION)
                    notion = min(des - cur, max_notional)
                    fee = commission(notion)
                    if notion + fee > cash:
                        notion = max(0.0, cash - fee)
                    if notion > 100:
                        qty = notion / px
                        cash -= notion + fee
                        shares[t] = shares.get(t, 0.0) + qty
                        turnover_notional += notion
                        trades.append({"date": date, "ticker": t, "side": "BUY", "notional": notion, "fee": fee})
            pending = None

        total = cash
        for t, q in shares.items():
            if t in close.columns and pd.notna(close.at[date, t]):
                total += q * float(close.at[date, t])
        equity.append((date, total, len(shares), cash))

        if date in sigset:
            cs = secs.get((v.score_kind, date), pd.DataFrame())
            gate = bool(ind["market_ok"].get(date, False))
            targets = choose_targets(v, cs, list(shares), gate)
            pending = (date, targets)

    eq = pd.DataFrame(equity, columns=["date", "equity", "positions", "cash"]).set_index("date")
    return eq, pd.DataFrame(trades), turnover_notional


def stats(series):
    s = series.dropna().astype(float)
    years = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().fillna(0)
    dd = s / s.cummax() - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    vol = r.std(ddof=0) * math.sqrt(252)
    ann = r.mean() * 252
    mdd = float(dd.min())
    return dict(cagr_pct=100*cagr, max_drawdown_pct=100*mdd, sharpe_0rf=ann/vol if vol else np.nan,
                calmar=cagr/abs(mdd) if mdd<0 else np.nan, final_value=float(s.iloc[-1]),
                total_return_pct=100*(s.iloc[-1]/s.iloc[0]-1), volatility_pct=100*vol)


def benchmark(s, start, end):
    x = s.loc[(s.index >= start) & (s.index <= end)].dropna().astype(float)
    return INITIAL * x / float(x.iloc[0])


def subperiod(name, eq):
    spans=[("2011-2017","2011-01-03","2017-12-31"),("2018-2022","2018-01-01","2022-12-31"),("2023-2026","2023-01-01","2026-08-31")]
    rows=[]
    for lab,a,b in spans:
        s=eq.loc[pd.Timestamp(a):pd.Timestamp(b)]
        if len(s)>30:
            z=stats(s); z.update(strategy=name,period=lab); rows.append(z)
    return rows


def main():
    hist, symbols = load_history_symbols()
    print(f"Downloading {len(symbols)} historical symbols")
    frames = base.download_prices(symbols, DATA_START.strftime("%Y-%m-%d"), (END_EXCLUSIVE-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    open_, high, low, close, volume = base.matrices(frames)
    close = close.loc[(close.index >= DATA_START) & (close.index < END_EXCLUSIVE)]
    open_ = open_.reindex(close.index); volume = volume.reindex(close.index)
    if "SPY" not in close.columns: raise RuntimeError("SPY unavailable")

    sigs = monthly_dates(close.index[(close.index >= TEST_START) & (close.index < END_EXCLUSIVE)])
    mem = memberships(hist, sigs)
    ind = indicators(close, volume)
    secs={}; coverage=[]
    for d in sigs:
        m=mem.get(d,set()); priced=[t for t in m if t in close.columns and pd.notna(close.at[d,t])]
        coverage.append({"date":d,"members":len(m),"priced":len(priced),"coverage_pct":100*len(priced)/max(len(m),1)})
        for k in ["mom12","composite"]:
            secs[(k,d)] = section(d,m,close,ind,k)
    cov=pd.DataFrame(coverage); cov.to_csv(OUT/"universe_coverage.csv",index=False)

    last=close.index[(close.index>=TEST_START)&(close.index<END_EXCLUSIVE)][-1]
    world=world_netr(TEST_START,last); common=min(last,world.index.max())
    print("Common end",common.date())

    rows=[]; subs=[]; annual={}; curves={}; trade_parts=[]
    for v in VARIANTS:
        eqdf,tr,turn=simulate(v,open_,close,sigs,secs,ind)
        eq=eqdf.equity.loc[:common]
        z=stats(eq)
        avg_eq=float(eq.mean())
        years=(eq.index[-1]-eq.index[0]).days/365.25
        z.update(strategy=v.name,start=str(eq.index[0].date()),end=str(eq.index[-1].date()),
                 transactions=int(len(tr)),annual_turnover_x=float(turn/max(avg_eq,1)/max(years,1e-9)))
        rows.append(z); subs.extend(subperiod(v.name,eq)); curves[v.name]=eq
        annual[v.name]=(1+eq.pct_change().fillna(0)).groupby(eq.index.year).prod()-1
        if not tr.empty:
            t=tr.copy(); t["strategy"]=v.name; trade_parts.append(t)

    for name,s in [("SPY_BuyHold",close.SPY),("MSCI_World_NETR",world)]:
        eq=benchmark(s,TEST_START,common); z=stats(eq); z.update(strategy=name,start=str(eq.index[0].date()),end=str(eq.index[-1].date()),transactions=1,annual_turnover_x=np.nan)
        rows.append(z); subs.extend(subperiod(name,eq)); curves[name]=eq
        annual[name]=(1+eq.pct_change().fillna(0)).groupby(eq.index.year).prod()-1

    res=pd.DataFrame(rows); sub=pd.DataFrame(subs); ann=pd.DataFrame(annual)*100; curve=pd.DataFrame(curves)
    tr=pd.concat(trade_parts,ignore_index=True) if trade_parts else pd.DataFrame()
    res.to_csv(OUT/"results.csv",index=False); sub.to_csv(OUT/"subperiod_results.csv",index=False)
    ann.to_csv(OUT/"annual_returns_pct.csv"); curve.to_csv(OUT/"equity_curves.csv"); tr.to_csv(OUT/"transactions.csv",index=False)
    payload={
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "method":"Monthly point-in-time S&P500 momentum rotation; signals at month-end close, execution next session open; 0.08% commission + 0.05% slippage each side.",
        "limitations":["Yahoo missing delisted/renamed constituents creates residual survivorship/data-availability bias.","Residual momentum is SPY-beta adjusted, not full multifactor residual momentum.","No tax impact is modeled."],
        "coverage":{"median_pct":float(cov.coverage_pct.median()),"min_pct":float(cov.coverage_pct.min()),"last_pct":float(cov.coverage_pct.iloc[-1])},
        "results":res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records")
    }
    (OUT/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("\nCOVERAGE\n",cov.coverage_pct.describe().to_string())
    print("\nRESULTS\n",res.to_string(index=False))
    print("\nSUBPERIODS\n",sub[["strategy","period","cagr_pct","max_drawdown_pct","sharpe_0rf"]].to_string(index=False))
    print("\nANNUAL %\n",ann.to_string())

if __name__=="__main__": main()
