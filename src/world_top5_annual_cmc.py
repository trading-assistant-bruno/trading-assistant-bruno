from __future__ import annotations

import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "world_top5_annual"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-01-01")
END_EXCLUSIVE = "2026-08-14"
INITIAL_CAPITAL = 10_000.0
MONTHLY_CONTRIBUTION = 100.0
TURNOVER_COST_ONE_WAY = 0.001  # 10 bps stress assumption

# Plausible mega-cap candidates for the developed-market global top 5 during 2017-2025.
# XOM, JNJ and V are included as sanity-check challengers, not because they are expected winners.
CANDIDATES = {
    "AAPL": "apple",
    "MSFT": "microsoft",
    "AMZN": "amazon",
    "GOOGL": "alphabet",
    "META": "meta-platforms",
    "NVDA": "nvidia",
    "BRK-B": "berkshire-hathaway",
    "TSLA": "tesla",
    "XOM": "exxon-mobil",
    "JNJ": "johnson-and-johnson",
    "V": "visa",
}

BENCHMARKS = {
    "MSCI_World_proxy_URTH": "URTH",
    "MSCI_World_Momentum_proxy_IWMO": "IWMO.L",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; trading-assistant-bruno/1.0)"}


def parse_market_cap(value) -> float:
    s = str(value).replace("$", "").replace(",", "").strip()
    m = re.search(r"([0-9.]+)\s*([TtBbMm])", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return float("nan")
    n = float(m.group(1))
    unit = m.group(2).upper()
    return n * {"T": 1e12, "B": 1e9, "M": 1e6}[unit]


def historical_caps_from_cmc(ticker: str, slug: str) -> dict[int, float]:
    url = f"https://companiesmarketcap.com/{slug}/marketcap/"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("year" in c for c in cols) and any("marketcap" in c.replace(" ", "") or "market cap" in c for c in cols):
            target = t.copy()
            break
    if target is None:
        raise RuntimeError(f"No annual market-cap table found for {ticker} at {url}")
    year_col = next(c for c in target.columns if "year" in str(c).lower())
    cap_col = next(c for c in target.columns if "market" in str(c).lower())
    out = {}
    for _, row in target.iterrows():
        try:
            y = int(row[year_col])
        except Exception:
            continue
        cap = parse_market_cap(row[cap_col])
        if math.isfinite(cap) and cap > 0:
            out[y] = cap
    if 2017 not in out:
        raise RuntimeError(f"Historical cap series for {ticker} does not reach 2017")
    return out


def build_annual_rankings() -> pd.DataFrame:
    all_caps = {}
    for ticker, slug in CANDIDATES.items():
        print(f"Fetching historical market caps: {ticker}")
        all_caps[ticker] = historical_caps_from_cmc(ticker, slug)

    rows = []
    # Prior year end determines holdings for the next calendar year.
    for source_year in range(2017, 2026):
        caps = {t: series.get(source_year, np.nan) for t, series in all_caps.items()}
        caps = {t: float(v) for t, v in caps.items() if math.isfinite(v) and v > 0}
        ranked = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) < 5:
            raise RuntimeError(f"Insufficient cap observations for {source_year}")
        top5 = ranked[:5]
        fifth = top5[-1][1]
        sixth = ranked[5][1] if len(ranked) > 5 else np.nan
        total = sum(v for _, v in top5)
        row = {
            "source_year_end": source_year,
            "holding_year": source_year + 1,
            "fifth_market_cap": fifth,
            "sixth_market_cap": sixth,
            "fifth_to_sixth_ratio": fifth / sixth if math.isfinite(sixth) and sixth > 0 else np.nan,
        }
        for i, (t, cap) in enumerate(top5, 1):
            row[f"rank_{i}_ticker"] = t
            row[f"rank_{i}_market_cap"] = cap
            row[f"rank_{i}_weight"] = cap / total
        rows.append(row)
    return pd.DataFrame(rows)


def download_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        start="2017-12-01",
        end=END_EXCLUSIVE,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=30,
    )
    out = {}
    for t in tickers:
        try:
            x = data[t].copy().dropna(how="all") if len(tickers) > 1 else data.copy()
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)
            if "Adj Close" not in x.columns:
                x["Adj Close"] = x["Close"]
            x.index = pd.to_datetime(x.index).tz_localize(None)
            out[t] = x.sort_index()
        except Exception:
            pass
    return out


def calendar_from_urth(prices: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = prices["URTH"].index
    return idx[(idx >= START) & (idx < pd.Timestamp(END_EXCLUSIVE))]


def first_trading_days(calendar: pd.DatetimeIndex) -> dict[pd.Period, pd.Timestamp]:
    return {p: calendar[calendar.to_period("M") == p][0] for p in calendar.to_period("M").unique()}


def target_by_year(rankings: pd.DataFrame, equal_weight: bool) -> dict[int, dict[str, float]]:
    out = {}
    for _, r in rankings.iterrows():
        year = int(r["holding_year"])
        names = [r[f"rank_{i}_ticker"] for i in range(1, 6)]
        if equal_weight:
            out[year] = {t: 0.2 for t in names}
        else:
            out[year] = {r[f"rank_{i}_ticker"]: float(r[f"rank_{i}_weight"]) for i in range(1, 6)}
    return out


def adj_matrix(prices: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: prices[t]["Adj Close"].reindex(calendar).ffill() for t in CANDIDATES if t in prices}, index=calendar)


def simulate_top5(adj: pd.DataFrame, targets: dict[int, dict[str, float]], dca: bool):
    calendar = adj.index
    month_first = set(first_trading_days(calendar).values())
    holdings: dict[str, float] = {}
    cash = 0.0 if dca else INITIAL_CAPITAL
    current_weights: dict[str, float] = {}
    equity = {}
    cashflows = []
    total_cost = 0.0
    current_year = None

    def value(dt):
        total = cash
        for t, units in holdings.items():
            px = float(adj.at[dt, t])
            total += units * px
        return total

    for dt in calendar:
        if dca and dt in month_first:
            cash += MONTHLY_CONTRIBUTION
            cashflows.append((dt, -MONTHLY_CONTRIBUTION))

        if dt.year != current_year:
            current_year = dt.year
            current_weights = targets.get(current_year, current_weights)
            if current_weights:
                before = value(dt)
                current_values = {t: holdings.get(t, 0.0) * float(adj.at[dt, t]) for t in set(holdings) | set(current_weights) if t in adj.columns}
                desired = {t: before * w for t, w in current_weights.items()}
                turnover = 0.5 * sum(abs(desired.get(t, 0.0) - current_values.get(t, 0.0)) for t in set(desired) | set(current_values))
                cost = turnover * TURNOVER_COST_ONE_WAY
                total_cost += cost
                invest = max(0.0, before - cost)
                holdings = {t: invest * w / float(adj.at[dt, t]) for t, w in current_weights.items()}
                cash = 0.0
        elif dca and dt in month_first and current_weights:
            investable = min(cash, MONTHLY_CONTRIBUTION)
            cost = investable * TURNOVER_COST_ONE_WAY
            total_cost += cost
            amount = max(0.0, investable - cost)
            cash -= investable
            for t, w in current_weights.items():
                holdings[t] = holdings.get(t, 0.0) + amount * w / float(adj.at[dt, t])

        equity[dt] = value(dt)

    series = pd.Series(equity).astype(float)
    if dca:
        cashflows.append((series.index[-1], float(series.iloc[-1])))
    return series, cashflows, total_cost


def xnpv(rate: float, cashflows):
    t0 = cashflows[0][0]
    return sum(v / ((1 + rate) ** (((d - t0).days) / 365.25)) for d, v in cashflows)


def xirr(cashflows):
    lo, hi = -0.9999, 10.0
    flo, fhi = xnpv(lo, cashflows), xnpv(hi, cashflows)
    while flo * fhi > 0 and hi < 1e6:
        hi *= 2
        fhi = xnpv(hi, cashflows)
    if flo * fhi > 0:
        return np.nan
    for _ in range(300):
        mid = (lo + hi) / 2
        fm = xnpv(mid, cashflows)
        if abs(fm) < 1e-9:
            return mid
        if flo * fm <= 0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return (lo + hi) / 2


def perf(series: pd.Series):
    e = series.dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
    dd = e / e.cummax() - 1
    r = e.pct_change().fillna(0.0)
    vol = r.std(ddof=0) * math.sqrt(252)
    ann = r.mean() * 252
    return {
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * float(dd.min()),
        "sharpe": float(ann / vol) if vol > 0 else np.nan,
        "final_value_10000": float(e.iloc[-1]),
    }


def benchmark(prices: pd.DataFrame):
    adj = prices.loc[prices.index >= START, "Adj Close"].dropna().astype(float)
    lump = adj / adj.iloc[0] * INITIAL_CAPITAL
    p = perf(lump)
    periods = adj.index.to_period("M")
    dates = [adj.index[periods == m][0] for m in periods.unique()]
    units = 0.0
    cf = []
    for d in dates:
        units += MONTHLY_CONTRIBUTION / float(adj.at[d])
        cf.append((d, -MONTHLY_CONTRIBUTION))
    final = units * float(adj.iloc[-1])
    cf.append((adj.index[-1], final))
    p.update({
        "dca_xirr_pct": 100 * xirr(cf),
        "dca_contributed": MONTHLY_CONTRIBUTION * len(dates),
        "dca_final_value": final,
        "estimated_turnover_cost": 0.0,
    })
    return p


def main():
    rankings = build_annual_rankings()
    rankings.to_csv(OUT / "annual_top5_rankings.csv", index=False)
    print("\n=== ANNUAL TOP 5 (prior year end -> holding year) ===")
    print(rankings[["holding_year"] + [f"rank_{i}_ticker" for i in range(1, 6)]].to_string(index=False))

    tickers = list(CANDIDATES) + list(BENCHMARKS.values())
    prices = download_prices(tickers)
    if "URTH" not in prices:
        raise RuntimeError("URTH unavailable")
    calendar = calendar_from_urth(prices)
    adj = adj_matrix(prices, calendar)

    rows = []
    curves = {}
    for label, equal in [("Top5_equal_weight_annual", True), ("Top5_cap_weight_annual", False)]:
        targets = target_by_year(rankings, equal)
        lump, _, lump_cost = simulate_top5(adj, targets, dca=False)
        _, cf, dca_cost = simulate_top5(adj, targets, dca=True)
        stats = perf(lump)
        stats.update({
            "strategy": label,
            "dca_xirr_pct": 100 * xirr(cf),
            "dca_contributed": -sum(v for _, v in cf if v < 0),
            "dca_final_value": cf[-1][1],
            "estimated_turnover_cost": lump_cost + dca_cost,
        })
        rows.append(stats)
        curves[label] = lump

    for name, ticker in BENCHMARKS.items():
        if ticker not in prices:
            continue
        s = benchmark(prices[ticker])
        s["strategy"] = name
        rows.append(s)

    results = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)
    results.to_csv(OUT / "annual_top5_comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "annual_top5_equity_curves.csv")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": [str(calendar[0].date()), str(calendar[-1].date())],
        "selection_data_source": "CompaniesMarketCap annual end-of-year market-cap tables",
        "return_data_source": "Yahoo Finance adjusted daily prices",
        "method": "At the first trading day of each year, hold the five largest candidate companies based only on the previous calendar year-end market cap. Rebalance annually. DCA adds 100 units monthly.",
        "limitations": [
            "This is a transparent proxy, not an official point-in-time MSCI World reconstruction. MSCI uses free-float-adjusted market capitalization while CompaniesMarketCap reports full market capitalization.",
            "Candidate set is limited to declared mega-cap names plus sanity-check challengers. A missing historical developed-market company that exceeded the fifth-ranked cap would bias selection.",
            "Annual rebalancing can miss intra-year changes in the true index top five.",
            "URTH and IWMO.L are ETF proxies; tracking, fund fees, currency/listing effects and taxes can differ from index returns.",
            "Historical backtests are not forecasts."
        ],
        "results": results.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "rankings": rankings.to_dict(orient="records"),
    }
    (OUT / "annual_top5_comparison.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n=== RESULTS ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
