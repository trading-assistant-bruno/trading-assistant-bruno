from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "world_top5"
OUT.mkdir(parents=True, exist_ok=True)

DATA_START = "2017-01-01"
START = pd.Timestamp("2018-01-01")
END_EXCLUSIVE = "2026-08-14"
MONTHLY_CONTRIBUTION = 100.0
INITIAL_CAPITAL = 10_000.0
TURNOVER_COST_ONE_WAY = 0.001  # 10 bps stress assumption, not an ETF/broker quote

# Broad set of US mega-caps that could plausibly occupy the very top of MSCI World
# during the test window. MSCI World top ranks were US-dominated in this period.
# This is NOT an official historical MSCI constituent file: the limitation is reported
# explicitly in the output.
CANDIDATES = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "TSLA",
    "JPM", "JNJ", "V", "WMT", "XOM", "UNH", "PG", "MA", "AVGO", "HD",
    "LLY", "ORCL", "COST", "NFLX", "ADBE", "CRM", "BAC", "KO", "PEP",
    "MRK", "ABBV", "CVX"
]

BENCHMARKS = {
    "MSCI_World_proxy_URTH": "URTH",
    "MSCI_World_Momentum_proxy_IWMO": "IWMO.L",
}


def normalize_download(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if data.empty:
        return out
    if len(tickers) == 1:
        x = data.copy()
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        out[tickers[0]] = x
        return out
    for ticker in tickers:
        try:
            x = data[ticker].copy().dropna(how="all")
            if not x.empty:
                out[ticker] = x
        except Exception:
            continue
    return out


def download_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        start=DATA_START,
        end=END_EXCLUSIVE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=30,
    )
    frames = normalize_download(data, tickers)
    clean: dict[str, pd.DataFrame] = {}
    for ticker, x in frames.items():
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        required = ["Open", "Close"]
        if not all(c in x.columns for c in required):
            continue
        if "Adj Close" not in x.columns:
            x["Adj Close"] = x["Close"]
        x.index = pd.to_datetime(x.index).tz_localize(None)
        clean[ticker] = x.sort_index()
    return clean


def get_shares_series(ticker: str) -> pd.Series:
    try:
        s = yf.Ticker(ticker).get_shares_full(start="2016-12-01", end=END_EXCLUSIVE)
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        s = pd.Series(s).astype(float)
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        s.index = idx.normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s
    except Exception as exc:
        print(f"shares unavailable {ticker}: {exc}")
        return pd.Series(dtype=float)


def last_value_on_or_before(series: pd.Series, date: pd.Timestamp) -> float:
    if series.empty:
        return float("nan")
    x = series.loc[:date]
    if x.empty:
        return float("nan")
    v = float(x.iloc[-1])
    return v if math.isfinite(v) and v > 0 else float("nan")


def close_on_or_before(df: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    x = df.loc[:date, col].dropna()
    if x.empty:
        return float("nan")
    return float(x.iloc[-1])


def trading_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    # Use URTH calendar for developed-equity trading days.
    idx = frames["URTH"].index
    return idx[(idx >= START) & (idx < pd.Timestamp(END_EXCLUSIVE))]


def first_trading_day_each_month(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
    s = pd.Series(calendar, index=calendar)
    return set(s.groupby(calendar.to_period("M")).first().tolist())


def quarter_rebalance_days(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    first = first_trading_day_each_month(calendar)
    return [d for d in sorted(first) if d.month in (1, 4, 7, 10)]


def previous_trading_day(calendar_all: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    earlier = calendar_all[calendar_all < date]
    return pd.Timestamp(earlier[-1]) if len(earlier) else None


def top5_history(
    prices: dict[str, pd.DataFrame],
    shares: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    full_cal = prices["URTH"].index
    history: dict[pd.Timestamp, dict[str, float]] = {}
    rows = []

    for reb_date in quarter_rebalance_days(calendar):
        rank_date = previous_trading_day(full_cal, reb_date)
        if rank_date is None:
            continue
        caps: dict[str, float] = {}
        missing = []
        for ticker in CANDIDATES:
            if ticker not in prices:
                missing.append(ticker)
                continue
            px = close_on_or_before(prices[ticker], rank_date, "Close")
            sh = last_value_on_or_before(shares.get(ticker, pd.Series(dtype=float)), rank_date)
            if math.isfinite(px) and px > 0 and math.isfinite(sh) and sh > 0:
                caps[ticker] = px * sh
            else:
                missing.append(ticker)

        ranked = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) < 5:
            raise RuntimeError(f"Only {len(ranked)} market caps available at {rank_date}; missing={missing}")
        selected = ranked[:5]
        total = sum(v for _, v in selected)
        weights = {t: v / total for t, v in selected}
        history[reb_date] = weights

        row = {
            "rebalance_date": reb_date.date().isoformat(),
            "ranking_date": rank_date.date().isoformat(),
            "available_market_caps": len(ranked),
            "missing_market_caps": ",".join(missing),
        }
        for i, (t, cap) in enumerate(selected, 1):
            row[f"rank_{i}_ticker"] = t
            row[f"rank_{i}_market_cap"] = cap
            row[f"rank_{i}_cap_weight"] = cap / total
        rows.append(row)

    return history, pd.DataFrame(rows)


def adjusted_price_matrix(prices: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for t in CANDIDATES:
        if t in prices:
            cols[t] = prices[t]["Adj Close"].reindex(calendar).ffill()
    return pd.DataFrame(cols, index=calendar)


def simulate_top5(
    adj: pd.DataFrame,
    target_history: dict[pd.Timestamp, dict[str, float]],
    equal_weight: bool,
    dca: bool,
) -> tuple[pd.Series, list[tuple[pd.Timestamp, float]], float, float]:
    calendar = adj.index
    month_first = first_trading_day_each_month(calendar)
    rebal_dates = set(target_history.keys())
    holdings: dict[str, float] = {}
    cash = 0.0 if dca else INITIAL_CAPITAL
    current_weights: dict[str, float] = {}
    equity_curve = {}
    cashflows: list[tuple[pd.Timestamp, float]] = []
    total_cost = 0.0

    def portfolio_value(dt: pd.Timestamp) -> float:
        value = cash
        for t, units in holdings.items():
            px = float(adj.at[dt, t]) if t in adj.columns else float("nan")
            if math.isfinite(px):
                value += units * px
        return value

    for dt in calendar:
        # Monthly contribution. A DCA means cash arrives over time; this is not a staged
        # deployment of money that was available from day one.
        if dca and dt in month_first:
            cash += MONTHLY_CONTRIBUTION
            cashflows.append((dt, -MONTHLY_CONTRIBUTION))

        # Quarterly point-in-time rank and full rebalance.
        if dt in rebal_dates:
            capw = target_history[dt]
            names = list(capw)
            current_weights = ({t: 1.0 / len(names) for t in names} if equal_weight else dict(capw))

            before = portfolio_value(dt)
            current_values = {t: holdings.get(t, 0.0) * float(adj.at[dt, t]) for t in set(holdings) | set(current_weights) if t in adj.columns and pd.notna(adj.at[dt, t])}
            desired_values = {t: before * w for t, w in current_weights.items()}
            turnover_notional = 0.5 * sum(abs(desired_values.get(t, 0.0) - current_values.get(t, 0.0)) for t in set(current_values) | set(desired_values))
            cost = turnover_notional * TURNOVER_COST_ONE_WAY
            total_cost += cost
            after_cost = max(0.0, before - cost)
            holdings = {}
            cash = 0.0
            for t, w in current_weights.items():
                px = float(adj.at[dt, t])
                if math.isfinite(px) and px > 0:
                    holdings[t] = after_cost * w / px
        elif dca and dt in month_first and current_weights:
            # Invest only the new monthly contribution between quarterly rebalances.
            investable = min(cash, MONTHLY_CONTRIBUTION)
            cost = investable * TURNOVER_COST_ONE_WAY
            total_cost += cost
            amount = max(0.0, investable - cost)
            cash -= investable
            for t, w in current_weights.items():
                px = float(adj.at[dt, t])
                if math.isfinite(px) and px > 0:
                    holdings[t] = holdings.get(t, 0.0) + amount * w / px

        equity_curve[dt] = portfolio_value(dt)

    equity = pd.Series(equity_curve).astype(float)
    final_value = float(equity.iloc[-1])
    if dca:
        cashflows.append((equity.index[-1], final_value))
    return equity, cashflows, final_value, total_cost


def xnpv(rate: float, cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    t0 = cashflows[0][0]
    return sum(v / ((1 + rate) ** (((d - t0).days) / 365.25)) for d, v in cashflows)


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    lo, hi = -0.9999, 10.0
    flo, fhi = xnpv(lo, cashflows), xnpv(hi, cashflows)
    while flo * fhi > 0 and hi < 1e6:
        hi *= 2
        fhi = xnpv(hi, cashflows)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(300):
        mid = (lo + hi) / 2
        fm = xnpv(mid, cashflows)
        if abs(fm) < 1e-9:
            return mid
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return (lo + hi) / 2


def performance_stats(equity: pd.Series) -> dict:
    e = equity.dropna()
    if len(e) < 2 or e.iloc[0] <= 0:
        return {"cagr_pct": np.nan, "max_drawdown_pct": np.nan, "sharpe": np.nan, "final_value": float(e.iloc[-1]) if len(e) else np.nan}
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1 / 365.25)
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
    dd = e / e.cummax() - 1
    ret = e.pct_change().fillna(0.0)
    vol = ret.std(ddof=0) * math.sqrt(252)
    ann = ret.mean() * 252
    return {
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * float(dd.min()),
        "sharpe": float(ann / vol) if vol > 0 else np.nan,
        "final_value": float(e.iloc[-1]),
    }


def benchmark_stats(df: pd.DataFrame, name: str) -> tuple[dict, dict]:
    x = df.loc[df.index >= START].copy()
    adj = x["Adj Close"].dropna().astype(float)
    normalized = adj / adj.iloc[0] * INITIAL_CAPITAL
    lump = performance_stats(normalized)

    periods = adj.index.to_period("M")
    dates = [adj.index[periods == p][0] for p in periods.unique()]
    units = 0.0
    cashflows = []
    for d in dates:
        px = float(adj.at[d])
        units += MONTHLY_CONTRIBUTION / px
        cashflows.append((d, -MONTHLY_CONTRIBUTION))
    final = units * float(adj.iloc[-1])
    cashflows.append((adj.index[-1], final))
    dca = {
        "dca_xirr_pct": 100 * xirr(cashflows),
        "dca_contributed": MONTHLY_CONTRIBUTION * len(dates),
        "dca_final_value": final,
        "dca_months": len(dates),
    }
    return lump, dca


def main() -> None:
    tickers = list(dict.fromkeys(CANDIDATES + list(BENCHMARKS.values())))
    prices = download_prices(tickers)
    if "URTH" not in prices:
        raise RuntimeError("URTH data unavailable")
    if "IWMO.L" not in prices:
        print("WARNING: IWMO.L unavailable; momentum benchmark will be omitted")

    print("Downloading historical shares outstanding...")
    shares = {t: get_shares_series(t) for t in CANDIDATES}
    coverage_rows = []
    for t in CANDIDATES:
        s = shares[t]
        coverage_rows.append({
            "ticker": t,
            "share_observations": int(len(s)),
            "first_share_date": s.index.min().date().isoformat() if len(s) else None,
            "last_share_date": s.index.max().date().isoformat() if len(s) else None,
        })
    pd.DataFrame(coverage_rows).to_csv(OUT / "share_data_coverage.csv", index=False)

    cal = trading_calendar(prices)
    target_history, rank_df = top5_history(prices, shares, cal)
    rank_df.to_csv(OUT / "top5_rebalance_history.csv", index=False)
    print("\n=== TOP 5 HISTORY ===")
    print(rank_df[[c for c in rank_df.columns if c.startswith("rebalance") or c.startswith("rank_") and (c.endswith("ticker") or c.endswith("cap_weight"))]].to_string(index=False))

    adj = adjusted_price_matrix(prices, cal)
    rows = []
    curves = {}

    for label, equal in [("Top5_equal_weight", True), ("Top5_cap_weight", False)]:
        lump_eq, _, _, lump_cost = simulate_top5(adj, target_history, equal, dca=False)
        dca_eq, dca_cf, dca_final, dca_cost = simulate_top5(adj, target_history, equal, dca=True)
        lump = performance_stats(lump_eq)
        rows.append({
            "strategy": label,
            **lump,
            "dca_xirr_pct": 100 * xirr(dca_cf),
            "dca_contributed": -sum(v for _, v in dca_cf if v < 0),
            "dca_final_value": dca_final,
            "estimated_lump_turnover_cost": lump_cost,
            "estimated_dca_turnover_cost": dca_cost,
        })
        curves[label] = lump_eq

    for name, ticker in BENCHMARKS.items():
        if ticker not in prices:
            continue
        lump, dca = benchmark_stats(prices[ticker], name)
        rows.append({
            "strategy": name,
            **lump,
            **dca,
            "estimated_lump_turnover_cost": 0.0,
            "estimated_dca_turnover_cost": 0.0,
        })

    results = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)
    results.to_csv(OUT / "world_top5_comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "top5_lump_equity_curves.csv")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_period": [START.date().isoformat(), str(cal[-1].date())],
        "method": {
            "selection": "Top 5 by full historical market capitalization at the prior trading day before each quarterly rebalance, from the declared mega-cap candidate pool.",
            "equal_weight": "20% each at quarterly rebalance; monthly DCA cash is allocated to current holdings between rebalances.",
            "cap_weight": "Proportional to full market cap within the five selected names at quarterly rebalance.",
            "transaction_cost_assumption": TURNOVER_COST_ONE_WAY,
            "benchmarks": BENCHMARKS,
        },
        "important_limitations": [
            "This is not an official point-in-time reconstruction of MSCI World. Historical constituent-level free-float-adjusted market-cap data are proprietary and are not present in this repository.",
            "Ranking uses full market capitalization from Yahoo historical prices times historical shares outstanding, whereas MSCI uses free-float-adjusted market capitalization.",
            "The candidate universe is a broad US mega-cap pool chosen to cover plausible top-rank MSCI World names in 2018-2026; omission of a true historical top-five constituent would bias results.",
            "BRK-B share-count representation may not perfectly reproduce Berkshire Hathaway aggregate market capitalization across share classes.",
            "URTH is a liquid ETF proxy for MSCI World; IWMO.L is an ETF proxy for MSCI World Momentum. ETF tracking, fees, listing currency and taxes can differ from index returns.",
            "Backtest results are historical and are not forecasts."
        ],
        "results": results.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "top5_rebalances": rank_df.to_dict(orient="records"),
    }
    (OUT / "world_top5_comparison.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== WORLD TOP-5 COMPARISON ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
