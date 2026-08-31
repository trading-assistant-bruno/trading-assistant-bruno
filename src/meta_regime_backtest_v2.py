from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import active_momentum_rotation as rot
import meta_regime_backtest as meta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "meta_regime_backtest"
OUT.mkdir(parents=True, exist_ok=True)
DELIST_HAIRCUT = 0.02


def robust_simulate(v, open_, close, signal_dates, secs, ind):
    dates = close.index[(close.index >= rot.TEST_START) & (close.index < rot.END_EXCLUSIVE)]
    sigset = set(signal_dates)
    close_ffill = close.ffill()
    cash = rot.INITIAL
    shares = {}
    pending = None
    trades = []
    equity = []
    turnover_notional = 0.0

    def mark_price(date, ticker):
        if ticker not in close_ffill.columns:
            return np.nan
        x = close_ffill.at[date, ticker]
        return float(x) if pd.notna(x) else np.nan

    def sell(date, ticker, qty, raw_px, reason, extra_haircut=0.0):
        nonlocal cash, turnover_notional
        if qty <= 0 or not np.isfinite(raw_px) or raw_px <= 0:
            return False
        px = raw_px * (1 - extra_haircut) * (1 - rot.SLIPPAGE)
        notion = qty * px
        fee = rot.commission(notion)
        cash += notion - fee
        shares[ticker] = shares.get(ticker, 0.0) - qty
        turnover_notional += notion
        trades.append({"date": date, "ticker": ticker, "side": "SELL", "notional": notion, "fee": fee, "reason": reason})
        if shares.get(ticker, 0.0) <= 1e-10:
            shares.pop(ticker, None)
        return True

    for date in dates:
        if pending is not None:
            sig_date, targets = pending
            target_set = set(targets)

            # First force exits of names no longer in the target portfolio. If the
            # security no longer has an open quote, use its last known adjusted close
            # with a conservative 2% haircut rather than silently valuing it at zero.
            for t in list(shares):
                if t in target_set:
                    continue
                op = open_.at[date, t] if t in open_.columns else np.nan
                if pd.notna(op) and float(op) > 0:
                    sell(date, t, shares[t], float(op), "normal_exit")
                else:
                    lp = mark_price(date, t)
                    sell(date, t, shares[t], lp, "stale_delisted_exit", DELIST_HAIRCUT)

            # Equity at execution open; surviving names without a valid open are marked
            # at the last known close and will not receive additional purchases.
            eq_open = cash
            opens = {}
            for t, q in shares.items():
                op = open_.at[date, t] if t in open_.columns else np.nan
                if pd.notna(op) and float(op) > 0:
                    opens[t] = float(op)
                    eq_open += q * float(op)
                else:
                    lp = mark_price(date, t)
                    if np.isfinite(lp):
                        eq_open += q * lp
            for t in targets:
                if t not in opens and t in open_.columns:
                    op = open_.at[date, t]
                    if pd.notna(op) and float(op) > 0:
                        opens[t] = float(op)

            target_w = 1.0 / len(targets) if targets else 0.0
            desired = {t: eq_open * target_w for t in targets}

            # Rebalance reductions among still-live target holdings.
            for t in list(shares):
                if t not in desired or t not in opens:
                    continue
                cur = shares[t] * opens[t]
                des = desired[t]
                if cur > des + 1.0:
                    qty = min(shares[t], (cur - des) / opens[t])
                    sell(date, t, qty, opens[t], "rebalance_reduction")

            # Buy increases/new names only when a live open is available.
            for t in targets:
                if t not in opens:
                    continue
                cur = shares.get(t, 0.0) * opens[t]
                des = desired[t]
                if des > cur + 1.0:
                    px = opens[t] * (1 + rot.SLIPPAGE)
                    max_notional = max(0.0, cash - rot.MIN_COMMISSION)
                    notion = min(des - cur, max_notional)
                    fee = rot.commission(notion)
                    if notion + fee > cash:
                        notion = max(0.0, cash - fee)
                    if notion > 100:
                        qty = notion / px
                        cash -= notion + fee
                        shares[t] = shares.get(t, 0.0) + qty
                        turnover_notional += notion
                        trades.append({"date": date, "ticker": t, "side": "BUY", "notional": notion, "fee": fee, "reason": "rebalance_buy"})
            pending = None

        total = cash
        for t, q in shares.items():
            lp = mark_price(date, t)
            if np.isfinite(lp):
                total += q * lp
        equity.append((date, total, len(shares), cash))

        if date in sigset:
            cs = secs.get((v.score_kind, date), pd.DataFrame())
            gate = bool(ind["market_ok"].get(date, False))
            targets = rot.choose_targets(v, cs, list(shares), gate)
            pending = (date, targets)

    # Conservative terminal liquidation at final marked prices.
    if len(dates):
        last = dates[-1]
        for t in list(shares):
            lp = mark_price(last, t)
            sell(last, t, shares[t], lp, "terminal_exit", 0.0)
        if equity:
            equity[-1] = (last, cash, 0, cash)

    eq = pd.DataFrame(equity, columns=["date", "equity", "positions", "cash"]).set_index("date")
    return eq, pd.DataFrame(trades), turnover_notional


def main():
    hist, symbols = rot.load_history_symbols()
    # Preserve the exact stock batches used by the prior rotation test; download QQQ
    # separately so adding a regime indicator does not reshuffle Yahoo batch requests.
    print(f"Downloading {len(symbols)} point-in-time historical stock symbols")
    frames = rot.base.download_prices(symbols, rot.DATA_START.strftime("%Y-%m-%d"), (rot.END_EXCLUSIVE-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    qqq = rot.base.download_prices(["QQQ"], rot.DATA_START.strftime("%Y-%m-%d"), (rot.END_EXCLUSIVE-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    frames.update(qqq)
    open_, high, low, close, volume = rot.base.matrices(frames)
    close = close.loc[(close.index >= rot.DATA_START) & (close.index < rot.END_EXCLUSIVE)]
    open_ = open_.reindex(close.index)
    volume = volume.reindex(close.index)
    if "SPY" not in close.columns:
        raise RuntimeError("SPY unavailable")

    sigs = rot.monthly_dates(close.index[(close.index >= rot.TEST_START) & (close.index < rot.END_EXCLUSIVE)])
    mem = rot.memberships(hist, sigs)
    ind = rot.indicators(close, volume)
    secs = {("composite", d): rot.section(d, mem.get(d,set()), close, ind, "composite") for d in sigs}

    top10_df, top10_tr, top10_turn = robust_simulate(meta.TOP10, open_, close, sigs, secs, ind)
    top20_df, top20_tr, top20_turn = robust_simulate(meta.TOP20, open_, close, sigs, secs, ind)

    last = close.index[(close.index >= rot.TEST_START) & (close.index < rot.END_EXCLUSIVE)][-1]
    world_raw = rot.world_netr(rot.TEST_START,last)
    common = min(last, world_raw.index.max())
    calendar = close.index[(close.index >= rot.TEST_START) & (close.index <= common)]
    world_eq = rot.benchmark(world_raw,rot.TEST_START,common).reindex(calendar).ffill().bfill()
    spy_eq = rot.benchmark(close["SPY"],rot.TEST_START,common).reindex(calendar).ffill().bfill()
    top10_eq = top10_df.equity.reindex(calendar).ffill().bfill()
    top20_eq = top20_df.equity.reindex(calendar).ffill().bfill()
    cash_eq = pd.Series(meta.INITIAL,index=calendar,dtype=float)

    sleeves = {
        meta.SLEEVE_TOP10:top10_eq,
        meta.SLEEVE_TOP20:top20_eq,
        meta.SLEEVE_SPY:spy_eq,
        meta.SLEEVE_WORLD:world_eq,
        meta.SLEEVE_CASH:cash_eq,
    }

    feature_rows=[]; coverage=[]
    for d in sigs[sigs<=common]:
        m=mem.get(d,set())
        f=meta.regime_features(d,m,close,ind); f["signal_date"]=pd.Timestamp(d); feature_rows.append(f)
        coverage.append({"date":d,"members":len(m),"priced_members":f["priced_members"],"coverage_pct":100*f["priced_members"]/max(len(m),1)})

    meta_names=["RegimeFixed","RegimePerf6M","RegimePerf12M","PerfOnly6M"]
    results=[]; curves={meta.SLEEVE_TOP10:top10_eq,meta.SLEEVE_TOP20:top20_eq,"SPY_BuyHold":spy_eq,"MSCI_World_NETR":world_eq}
    all_decisions=[]
    for name in meta_names:
        dec=meta.build_decisions(name,sigs[sigs<=common],feature_rows,sleeves,calendar)
        mdf,switches=meta.simulate_meta(dec,sleeves,calendar)
        eq=mdf.equity
        results.append(meta.stats_with_meta(name,eq,switches,world_eq)); curves[name]=eq; all_decisions.append(dec)
    for name,eq in [(meta.SLEEVE_TOP10,top10_eq),(meta.SLEEVE_TOP20,top20_eq),("SPY_BuyHold",spy_eq),("MSCI_World_NETR",world_eq)]:
        results.append(meta.stats_with_meta(name,eq,None,world_eq))

    res=pd.DataFrame(results); decisions=pd.concat(all_decisions,ignore_index=True); features=pd.DataFrame(feature_rows); coverage_df=pd.DataFrame(coverage); curve_df=pd.DataFrame(curves)
    sub_rows=[]
    spans=[("2011-2017","2011-01-03","2017-12-31"),("2018-2022","2018-01-01","2022-12-31"),("2023-2026","2023-01-01","2026-08-31")]
    for name,eq in curves.items():
        for lab,a,b in spans:
            s=eq.loc[pd.Timestamp(a):pd.Timestamp(b)]
            if len(s)>30:
                sub_rows.append({"strategy":name,"period":lab,**rot.stats(s)})
    sub=pd.DataFrame(sub_rows)
    annual_df=pd.DataFrame({name:((1+eq.pct_change().fillna(0)).groupby(eq.index.year).prod()-1)*100 for name,eq in curves.items()})
    regime_summary=features.regime.value_counts().rename_axis("regime").reset_index(name="months"); regime_summary["pct_months"]=100*regime_summary.months/regime_summary.months.sum()
    selection_summary=decisions.groupby(["meta_strategy","chosen_sleeve"]).size().rename("months").reset_index(); selection_summary["pct_within_meta"]=selection_summary.groupby("meta_strategy").months.transform(lambda x:100*x/x.sum())

    res.to_csv(OUT/"results.csv",index=False); sub.to_csv(OUT/"subperiod_results.csv",index=False); annual_df.to_csv(OUT/"annual_returns_pct.csv"); curve_df.to_csv(OUT/"equity_curves.csv")
    decisions.to_csv(OUT/"monthly_decisions.csv",index=False); features.to_csv(OUT/"regime_features.csv",index=False); coverage_df.to_csv(OUT/"universe_coverage.csv",index=False)
    regime_summary.to_csv(OUT/"regime_summary.csv",index=False); selection_summary.to_csv(OUT/"selection_summary.csv",index=False)
    pd.concat([top10_tr.assign(strategy=meta.SLEEVE_TOP10),top20_tr.assign(strategy=meta.SLEEVE_TOP20)],ignore_index=True).to_csv(OUT/"active_transactions.csv",index=False)

    payload={
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "method":"Robust v2 monthly walk-forward regime/meta-strategy test. Active sleeve stale/delisted holdings remain marked at last known close and are force-liquidated with a 2% haircut at rebalance. Meta decisions use only information available at month-end and become effective after an implementation delay. 10bp meta switch cost plus sleeve costs.",
        "active_turnover":{"top10_notional":float(top10_turn),"top20_notional":float(top20_turn)},
        "limitations":["Yahoo missing historical delisted/renamed constituents leaves residual data-availability bias.","2% stale/delisting haircut is an approximation, not actual merger/bankruptcy proceeds.","No tax impact is modeled.","Residual momentum is SPY-beta adjusted, not a full multifactor residual.","Round regime thresholds were fixed ex ante; no optimization."],
        "coverage":{"median_pct":float(coverage_df.coverage_pct.median()),"min_pct":float(coverage_df.coverage_pct.min()),"last_pct":float(coverage_df.coverage_pct.iloc[-1])},
        "results":res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records"),
    }
    (OUT/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")

    print("COMMON END",common.date())
    print("\nREGIME SUMMARY\n",regime_summary.to_string(index=False))
    print("\nSELECTION SUMMARY\n",selection_summary.to_string(index=False))
    print("\nRESULTS\n",res.to_string(index=False))
    print("\nSUBPERIODS\n",sub[["strategy","period","cagr_pct","max_drawdown_pct","sharpe_0rf"]].to_string(index=False))
    print("\nLAST 12 DECISIONS\n",decisions.sort_values("signal_date").tail(12).to_string(index=False))

if __name__=="__main__":
    main()
