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
OUT = ROOT / "data" / "active_momentum"
OUT.mkdir(parents=True, exist_ok=True)

DATA_START = pd.Timestamp("2009-01-01")
TEST_START = pd.Timestamp("2011-01-03")
END_EXCLUSIVE = pd.Timestamp("2026-09-01")
INITIAL_CAPITAL = 100_000.0
COMMISSION_PCT = 0.0008
MIN_COMMISSION = 1.0
SLIPPAGE_PCT = 0.0005
RISK_PER_TRADE = 0.005
ATR_MULT = 3.0
MIN_ADV20 = 20_000_000.0
MIN_PRICE = 5.0


@dataclass(frozen=True)
class Variant:
    name: str
    score_kind: str
    entry_rank: int
    exit_rank: int
    max_positions: int
    market_gate: bool


VARIANTS = [
    Variant("SimpleMom_Top10_Buffer20", "simple", 10, 20, 10, True),
    Variant("ResidualMom_Top10_Buffer20", "residual", 10, 20, 10, True),
    Variant("ResidualMom_Top20_Buffer40", "residual", 20, 40, 20, True),
    Variant("ResidualMom_Top10_NoMarketGate", "residual", 10, 20, 10, False),
]


def weekly_signal_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W-FRI")).max().values)


def history_and_symbols():
    hist = pit.load_history().copy()
    hist = hist[hist["date"] >= DATA_START - pd.Timedelta(days=7)].reset_index(drop=True)
    symbols: set[str] = set()
    for members in hist["members"]:
        symbols.update(members)
    symbols.add("SPY")
    return hist, sorted(symbols)


def membership_lookup(hist: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[pd.Timestamp, set[str]]:
    hd = hist["date"].to_numpy(dtype="datetime64[ns]")
    out = {}
    for d in dates:
        nd = pd.Timestamp(d).normalize()
        pos = hd.searchsorted(nd.to_datetime64(), side="right") - 1
        out[nd] = set(hist.iloc[int(pos)]["members"]) if pos >= 0 else set()
    return out


def compute_indicators(open_, high, low, close, volume):
    ret = close.pct_change()
    spy = close["SPY"]
    spy_ret = spy.pct_change()

    mom12_1 = close.shift(21) / close.shift(252) - 1.0
    mom6_1 = close.shift(21) / close.shift(126) - 1.0
    high52 = close / close.rolling(252).max()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()
    adv20 = (close * volume).rolling(20).mean()

    prev_close = close.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy(),
        ]),
        index=close.index,
        columns=close.columns,
    )
    atr14 = tr.rolling(14).mean()

    beta126 = ret.rolling(126).cov(spy_ret).div(spy_ret.rolling(126).var(), axis=0)
    beta252 = ret.rolling(252).cov(spy_ret).div(spy_ret.rolling(252).var(), axis=0)
    spy6_1 = spy.shift(21) / spy.shift(126) - 1.0
    spy12_1 = spy.shift(21) / spy.shift(252) - 1.0
    resid6 = mom6_1.sub(beta126.mul(spy6_1, axis=0))
    resid12 = mom12_1.sub(beta252.mul(spy12_1, axis=0))
    residual = 0.5 * resid6 + 0.5 * resid12

    spy_sma200 = spy.rolling(200).mean()
    spy_mom12 = spy / spy.shift(252) - 1.0
    market_ok = (spy > spy_sma200) & (spy_mom12 > 0)

    return {
        "mom12_1": mom12_1,
        "mom6_1": mom6_1,
        "high52": high52,
        "sma100": sma100,
        "sma200": sma200,
        "adv20": adv20,
        "atr14": atr14,
        "residual": residual,
        "market_ok": market_ok,
    }


def cross_section(date, members, close, ind, kind: str):
    cols = [t for t in members if t in close.columns and t != "SPY"]
    if not cols:
        return pd.DataFrame()
    x = pd.DataFrame(index=cols)
    x["close"] = close.loc[date, cols]
    x["mom12"] = ind["mom12_1"].loc[date, cols]
    x["mom6"] = ind["mom6_1"].loc[date, cols]
    x["high52"] = ind["high52"].loc[date, cols]
    x["residual"] = ind["residual"].loc[date, cols]
    x["sma100"] = ind["sma100"].loc[date, cols]
    x["sma200"] = ind["sma200"].loc[date, cols]
    x["adv20"] = ind["adv20"].loc[date, cols]
    x["atr14"] = ind["atr14"].loc[date, cols]
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["close", "mom12", "mom6", "high52", "sma200", "adv20", "atr14"])
    if kind == "residual":
        x = x.dropna(subset=["residual"])
    if x.empty:
        return x

    # Cross-sectional percentile scoring: fixed ex-ante weights, no optimization.
    p12 = x["mom12"].rank(pct=True)
    p6 = x["mom6"].rank(pct=True)
    ph = x["high52"].rank(pct=True)
    if kind == "simple":
        x["score"] = 0.50 * p12 + 0.30 * p6 + 0.20 * ph
    else:
        pr = x["residual"].rank(pct=True)
        x["score"] = 0.35 * p12 + 0.20 * p6 + 0.20 * ph + 0.25 * pr

    x["rank"] = x["score"].rank(ascending=False, method="first")
    x["entry_eligible"] = (
        (x["close"] >= MIN_PRICE)
        & (x["adv20"] >= MIN_ADV20)
        & (x["close"] > x["sma200"])
        & (x["high52"] >= 0.70)
        & (x["mom12"] > 0)
        & (x["mom6"] > 0)
    )
    return x.sort_values("rank")


def buy_cost(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_PCT)


def benchmark_equity(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    s = series.loc[(series.index >= start) & (series.index <= end)].dropna().astype(float)
    return INITIAL_CAPITAL * s / float(s.iloc[0])


def get_world_netr(start, end):
    df = msci.get_levels("990100", start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), variant="NETR").copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["LEVEL"] = pd.to_numeric(df["LEVEL"], errors="coerce")
    s = df.dropna(subset=["DATE", "LEVEL"]).set_index("DATE")["LEVEL"].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep="last")]


def simulate(variant: Variant, open_, high, low, close, ind, signal_dates, memberships, cross_sections):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity = []
    pending_signal = None
    exit_next = set()
    signal_set = set(signal_dates)
    dates = close.index[(close.index >= TEST_START) & (close.index < END_EXCLUSIVE)]

    def px(date, ticker, field):
        mat = {"open": open_, "high": high, "low": low, "close": close}[field]
        v = mat.at[date, ticker] if ticker in mat.columns else np.nan
        return float(v) if pd.notna(v) else np.nan

    def sell(date, ticker, raw_price, reason):
        nonlocal cash
        p = positions.get(ticker)
        if p is None or not math.isfinite(raw_price) or raw_price <= 0:
            return
        exec_price = raw_price * (1.0 - SLIPPAGE_PCT)
        notional = p["shares"] * exec_price
        fee = buy_cost(notional)
        cash += notional - fee
        pnl = (exec_price - p["entry_price"]) * p["shares"] - p["entry_fee"] - fee
        trades.append({
            "ticker": ticker,
            "entry_date": p["entry_date"],
            "exit_date": date,
            "entry_price": p["entry_price"],
            "exit_price": exec_price,
            "shares": p["shares"],
            "pnl": pnl,
            "return_pct": 100.0 * pnl / max(p["entry_notional"] + p["entry_fee"], 1e-9),
            "reason": reason,
            "holding_days": (date - p["entry_date"]).days,
        })
        del positions[ticker]

    for i, date in enumerate(dates):
        # 1) Overnight stop / scheduled exits at today's open.
        for t in list(positions):
            op = px(date, t, "open")
            if not math.isfinite(op):
                continue
            stop = positions[t]["stop"]
            if op <= stop:
                sell(date, t, op, "gap_stop")
        for t in list(exit_next):
            if t in positions:
                op = px(date, t, "open")
                if math.isfinite(op):
                    sell(date, t, op, "trend_exit")
            exit_next.discard(t)

        # 2) Execute previous weekly signal at today's open.
        if pending_signal is not None:
            sig_date, cs, gate_ok = pending_signal
            if (not variant.market_gate) or gate_ok:
                ranks = cs["rank"].to_dict() if not cs.empty else {}
                # Keep incumbents inside buffer; exit anything outside it or below SMA100 at signal close.
                for t in list(positions):
                    r = ranks.get(t, np.inf)
                    s100 = cs.at[t, "sma100"] if t in cs.index else np.nan
                    c = cs.at[t, "close"] if t in cs.index else np.nan
                    if r > variant.exit_rank or not math.isfinite(float(c)) or not math.isfinite(float(s100)) or c <= s100:
                        op = px(date, t, "open")
                        if math.isfinite(op):
                            sell(date, t, op, "rank_or_trend")

                candidates = cs[(cs["rank"] <= variant.entry_rank) & cs["entry_eligible"]].index.tolist() if not cs.empty else []
                # Portfolio equity at open before new buys.
                eq_open = cash
                for t, p in positions.items():
                    op = px(date, t, "open")
                    if math.isfinite(op):
                        eq_open += p["shares"] * op
                position_cap = eq_open / variant.max_positions
                for t in candidates:
                    if len(positions) >= variant.max_positions or t in positions:
                        continue
                    op = px(date, t, "open")
                    atr = float(cs.at[t, "atr14"]) if t in cs.index else np.nan
                    if not math.isfinite(op) or not math.isfinite(atr) or op <= 0 or atr <= 0:
                        continue
                    buy_price = op * (1.0 + SLIPPAGE_PCT)
                    stop_dist = ATR_MULT * atr
                    risk_budget = eq_open * RISK_PER_TRADE
                    shares_risk = risk_budget / stop_dist
                    shares_cap = position_cap / buy_price
                    shares_cash = max(0.0, (cash - MIN_COMMISSION) / buy_price)
                    shares = min(shares_risk, shares_cap, shares_cash)
                    if shares * buy_price < 200:
                        continue
                    notional = shares * buy_price
                    fee = buy_cost(notional)
                    if notional + fee > cash:
                        shares = max(0.0, (cash - fee) / buy_price)
                        notional = shares * buy_price
                    if shares <= 0 or notional + fee > cash:
                        continue
                    cash -= notional + fee
                    positions[t] = {
                        "shares": shares,
                        "entry_price": buy_price,
                        "entry_date": date,
                        "entry_fee": fee,
                        "entry_notional": notional,
                        "stop": buy_price - stop_dist,
                        "peak_close": buy_price,
                    }
            else:
                for t in list(positions):
                    op = px(date, t, "open")
                    if math.isfinite(op):
                        sell(date, t, op, "market_gate_off")
            pending_signal = None

        # 3) Intraday protective stops, including same-day new positions (conservative).
        for t in list(positions):
            lo = px(date, t, "low")
            op = px(date, t, "open")
            stop = positions[t]["stop"]
            if math.isfinite(lo) and lo <= stop:
                raw = op if math.isfinite(op) and op <= stop else stop
                sell(date, t, raw, "protective_stop")

        # 4) Mark close, update trailing stops and daily SMA100 exit flag.
        total = cash
        for t, p in list(positions.items()):
            c = px(date, t, "close")
            if not math.isfinite(c):
                continue
            total += p["shares"] * c
            p["peak_close"] = max(p["peak_close"], c)
            atr = ind["atr14"].at[date, t] if t in ind["atr14"].columns else np.nan
            if pd.notna(atr):
                p["stop"] = max(p["stop"], p["peak_close"] - ATR_MULT * float(atr))
            s100 = ind["sma100"].at[date, t] if t in ind["sma100"].columns else np.nan
            if pd.notna(s100) and c <= float(s100):
                exit_next.add(t)
        equity.append((date, total, len(positions), cash))

        # 5) Generate weekly signal using today's completed close; execute next session open.
        if date in signal_set:
            cs = cross_sections.get((variant.score_kind, date), pd.DataFrame())
            gate_ok = bool(ind["market_ok"].get(date, False))
            pending_signal = (date, cs, gate_ok)

    # Liquidate for an apples-to-apples end value.
    if len(dates):
        last = dates[-1]
        for t in list(positions):
            c = px(last, t, "close")
            if math.isfinite(c):
                sell(last, t, c, "final_liquidation")
        if equity:
            equity[-1] = (last, cash, 0, cash)

    eq = pd.DataFrame(equity, columns=["date", "equity", "positions", "cash"]).set_index("date")
    return eq, pd.DataFrame(trades)


def perf_stats(eq: pd.Series):
    e = eq.dropna().astype(float)
    years = (e.index[-1] - e.index[0]).days / 365.25
    r = e.pct_change().fillna(0.0)
    dd = e / e.cummax() - 1.0
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
    vol = r.std(ddof=0) * math.sqrt(252)
    ann = r.mean() * 252
    mdd = float(dd.min())
    return {
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * mdd,
        "sharpe_0rf": ann / vol if vol > 0 else np.nan,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "final_value": float(e.iloc[-1]),
        "total_return_pct": 100 * (e.iloc[-1] / e.iloc[0] - 1),
        "volatility_pct": 100 * vol,
    }


def trade_stats(trades: pd.DataFrame):
    if trades.empty:
        return {"trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan, "avg_trade_pct": np.nan, "median_holding_days": np.nan}
    wins = trades[trades.pnl > 0].pnl.sum()
    losses = -trades[trades.pnl < 0].pnl.sum()
    return {
        "trades": int(len(trades)),
        "win_rate_pct": 100 * float((trades.pnl > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "avg_trade_pct": float(trades.return_pct.mean()),
        "median_holding_days": float(trades.holding_days.median()),
    }


def subperiod_rows(name, eq):
    spans = [
        ("2011-2017", pd.Timestamp("2011-01-03"), pd.Timestamp("2017-12-31")),
        ("2018-2022", pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-31")),
        ("2023-2026", pd.Timestamp("2023-01-01"), pd.Timestamp("2026-08-31")),
    ]
    out = []
    for label, a, b in spans:
        s = eq.loc[(eq.index >= a) & (eq.index <= b)]
        if len(s) > 30:
            st = perf_stats(s)
            st.update({"strategy": name, "period": label})
            out.append(st)
    return out


def main():
    hist, symbols = history_and_symbols()
    print(f"Downloading {len(symbols)} point-in-time historical symbols + SPY")
    frames = base.download_prices(symbols, DATA_START.strftime("%Y-%m-%d"), (END_EXCLUSIVE - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    open_, high, low, close, volume = base.matrices(frames)
    if "SPY" not in close.columns:
        raise RuntimeError("SPY unavailable")
    close = close.loc[(close.index >= DATA_START) & (close.index < END_EXCLUSIVE)]
    open_ = open_.reindex(close.index); high = high.reindex(close.index); low = low.reindex(close.index); volume = volume.reindex(close.index)

    signals = weekly_signal_dates(close.index[(close.index >= TEST_START) & (close.index < END_EXCLUSIVE)])
    memberships = membership_lookup(hist, signals)
    ind = compute_indicators(open_, high, low, close, volume)

    cross_sections = {}
    coverage = []
    for date in signals:
        members = memberships.get(date, set())
        priced = [t for t in members if t in close.columns and pd.notna(close.at[date, t])]
        coverage.append({"date": date, "members": len(members), "priced": len(priced), "coverage_pct": 100 * len(priced) / max(len(members), 1)})
        for kind in ["simple", "residual"]:
            cross_sections[(kind, date)] = cross_section(date, members, close, ind, kind)
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(OUT / "universe_coverage.csv", index=False)

    last_common = close.index[(close.index >= TEST_START) & (close.index < END_EXCLUSIVE)][-1]
    world = get_world_netr(TEST_START, last_common)
    common_end = min(last_common, world.index.max())
    if common_end < last_common:
        print("Trimming strategy/benchmarks to MSCI last date", common_end.date())

    results = []
    subperiods = []
    annual = {}
    all_equity = {}
    all_trades = []

    for v in VARIANTS:
        eqdf, trades = simulate(v, open_, high, low, close, ind, signals, memberships, cross_sections)
        eq = eqdf.equity.loc[:common_end]
        st = perf_stats(eq)
        ts = trade_stats(trades[trades.exit_date <= common_end] if not trades.empty else trades)
        st.update(ts)
        st.update({"strategy": v.name, "start": str(eq.index[0].date()), "end": str(eq.index[-1].date())})
        results.append(st)
        subperiods.extend(subperiod_rows(v.name, eq))
        annual[v.name] = (1 + eq.pct_change().fillna(0)).groupby(eq.index.year).prod() - 1
        all_equity[v.name] = eq
        if not trades.empty:
            t = trades.copy(); t["strategy"] = v.name; all_trades.append(t)

    spy_eq = benchmark_equity(close["SPY"], TEST_START, common_end)
    world_eq = benchmark_equity(world, TEST_START, common_end)
    for name, eq in [("SPY_BuyHold", spy_eq), ("MSCI_World_NETR", world_eq)]:
        st = perf_stats(eq)
        st.update({"strategy": name, "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "trades": 1, "win_rate_pct": np.nan, "profit_factor": np.nan, "avg_trade_pct": np.nan, "median_holding_days": np.nan})
        results.append(st)
        subperiods.extend(subperiod_rows(name, eq))
        annual[name] = (1 + eq.pct_change().fillna(0)).groupby(eq.index.year).prod() - 1
        all_equity[name] = eq

    res = pd.DataFrame(results)
    sub = pd.DataFrame(subperiods)
    ann = pd.DataFrame(annual) * 100
    curves = pd.DataFrame(all_equity)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    res.to_csv(OUT / "results.csv", index=False)
    sub.to_csv(OUT / "subperiod_results.csv", index=False)
    ann.to_csv(OUT / "annual_returns_pct.csv")
    curves.to_csv(OUT / "equity_curves.csv")
    trades_df.to_csv(OUT / "trades.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "universe": "Point-in-time S&P 500 snapshots from chinobing/historical_sp500_constituents",
            "signal_frequency": "weekly; signal at completed weekly close, execution next trading-day open",
            "simple_score": "50% 12-1m momentum + 30% 6-1m momentum + 20% proximity to 52-week high, cross-sectional percentile ranks",
            "residual_score": "35% 12-1m + 20% 6-1m + 20% 52-week-high proximity + 25% beta-adjusted residual momentum",
            "entry_filters": "price>$5, ADV20>$20m, close>SMA200, >=70% of 52-week high, positive 12-1m and 6-1m momentum",
            "risk": "3xATR14 initial/trailing stop; 0.5% equity risk budget per new trade; equal notional cap based on max positions",
            "market_gate": "SPY>SMA200 and positive 12-month SPY momentum (except explicit no-gate variant)",
            "costs": "0.08% commission each side with $1 minimum + 0.05% adverse slippage each side",
            "benchmark": "MSCI World Net Total Return USD (MSCI code 990100) and SPY adjusted",
        },
        "important_limitations": [
            "Yahoo historical ticker availability is incomplete for renamed/delisted securities. Missing historical constituents are excluded, leaving residual survivorship/data-availability bias.",
            "Residual momentum is approximated with rolling market beta to SPY, not a full Fama-French residual regression.",
            "No fundamental quality/earnings data are used in this first price-only test.",
            "Backtest evidence is not a forecast and parameters were not optimized in-sample here.",
        ],
        "coverage": {
            "median_pct": float(coverage_df.coverage_pct.median()),
            "min_pct": float(coverage_df.coverage_pct.min()),
            "last_pct": float(coverage_df.coverage_pct.iloc[-1]),
        },
        "results": res.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nUNIVERSE COVERAGE\n", coverage_df.coverage_pct.describe().to_string())
    print("\nRESULTS\n", res.to_string(index=False))
    print("\nSUBPERIODS\n", sub[["strategy", "period", "cagr_pct", "max_drawdown_pct", "sharpe_0rf"]].to_string(index=False))
    print("\nANNUAL RETURNS %\n", ann.to_string())


if __name__ == "__main__":
    main()
