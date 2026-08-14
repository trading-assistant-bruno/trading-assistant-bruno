from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import world_top5_annual_cmc as annual

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "world_top5_blend"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
MONTHLY_CONTRIBUTION = 100.0
TURNOVER_COST_ONE_WAY = 0.001  # 10 bps stress assumption
PRIMARY_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]  # Top-5 sleeve weights
GRID_WEIGHTS = [i / 10 for i in range(11)]


def first_trading_days(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
    periods = calendar.to_period("M")
    return {calendar[periods == p][0] for p in periods.unique()}


def build_prices_and_targets():
    rankings = annual.build_annual_rankings()
    top5_targets = annual.target_by_year(rankings, equal_weight=False)

    tickers = list(annual.CANDIDATES) + ["URTH"]
    prices = annual.download_prices(tickers)
    if "URTH" not in prices:
        raise RuntimeError("URTH unavailable")

    calendar = annual.calendar_from_urth(prices)
    adj = annual.adj_matrix(prices, calendar)
    world = prices["URTH"]["Adj Close"].reindex(calendar).ffill().astype(float)
    return rankings, top5_targets, adj, world


def desired_weights_for_year(year: int, top5_weight: float, targets: dict[int, dict[str, float]]) -> dict[str, float]:
    top = targets.get(year)
    if not top:
        raise RuntimeError(f"No Top-5 target for {year}")
    w = {"URTH": 1.0 - top5_weight}
    for ticker, inner_weight in top.items():
        w[ticker] = top5_weight * inner_weight
    return {k: v for k, v in w.items() if v > 1e-12}


def price_at(dt: pd.Timestamp, ticker: str, adj: pd.DataFrame, world: pd.Series) -> float:
    if ticker == "URTH":
        return float(world.at[dt])
    return float(adj.at[dt, ticker])


def simulate(top5_weight: float, adj: pd.DataFrame, world: pd.Series, targets: dict[int, dict[str, float]], dca: bool):
    calendar = world.index
    month_first = first_trading_days(calendar)
    holdings: dict[str, float] = {}
    cash = 0.0 if dca else INITIAL_CAPITAL
    equity: dict[pd.Timestamp, float] = {}
    cashflows: list[tuple[pd.Timestamp, float]] = []
    total_cost = 0.0
    current_year = None
    current_weights: dict[str, float] = {}

    def value(dt: pd.Timestamp) -> float:
        total = cash
        for ticker, units in holdings.items():
            total += units * price_at(dt, ticker, adj, world)
        return total

    for dt in calendar:
        # Cash actually arrives monthly for DCA; no assumption that it was available at t0.
        if dca and dt in month_first:
            cash += MONTHLY_CONTRIBUTION
            cashflows.append((dt, -MONTHLY_CONTRIBUTION))

        # Rebalance the whole portfolio once per year to the World/Top-5 target and
        # simultaneously refresh the Top-5 constituents from prior-year market caps.
        if dt.year != current_year:
            current_year = dt.year
            current_weights = desired_weights_for_year(current_year, top5_weight, targets)
            before = value(dt)
            current_values = {
                t: holdings.get(t, 0.0) * price_at(dt, t, adj, world)
                for t in set(holdings) | set(current_weights)
            }
            desired_values = {t: before * w for t, w in current_weights.items()}
            turnover_notional = 0.5 * sum(
                abs(desired_values.get(t, 0.0) - current_values.get(t, 0.0))
                for t in set(current_values) | set(desired_values)
            )
            cost = turnover_notional * TURNOVER_COST_ONE_WAY
            total_cost += cost
            investable = max(0.0, before - cost)
            holdings = {
                t: investable * w / price_at(dt, t, adj, world)
                for t, w in current_weights.items()
            }
            cash = 0.0

        elif dca and dt in month_first:
            # Allocate only the new contribution according to the strategic target.
            investable = min(cash, MONTHLY_CONTRIBUTION)
            cost = investable * TURNOVER_COST_ONE_WAY
            total_cost += cost
            amount = max(0.0, investable - cost)
            cash -= investable
            for t, w in current_weights.items():
                px = price_at(dt, t, adj, world)
                holdings[t] = holdings.get(t, 0.0) + amount * w / px

        equity[dt] = value(dt)

    e = pd.Series(equity).astype(float)
    if dca:
        cashflows.append((e.index[-1], float(e.iloc[-1])))
    return e, cashflows, total_cost


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
        if abs(fm) < 1e-10:
            return mid
        if flo * fm <= 0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return (lo + hi) / 2


def stats(equity: pd.Series) -> dict:
    e = equity.dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
    dd = e / e.cummax() - 1
    r = e.pct_change().fillna(0.0)
    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=0) * math.sqrt(252)
    downside = r[r < 0].std(ddof=0) * math.sqrt(252)
    max_dd = float(dd.min())
    annual_returns = (1.0 + r).groupby(r.index.year).prod() - 1.0
    return {
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * max_dd,
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "sortino": float(ann_ret / downside) if downside > 0 else np.nan,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "final_value_10000": float(e.iloc[-1]),
        "worst_calendar_year_pct": 100 * float(annual_returns.min()),
        "best_calendar_year_pct": 100 * float(annual_returns.max()),
    }


def main():
    rankings, targets, adj, world = build_prices_and_targets()
    rankings.to_csv(OUT / "annual_top5_rankings.csv", index=False)

    weights = sorted(set(PRIMARY_WEIGHTS + GRID_WEIGHTS))
    rows = []
    curves = {}
    annual_returns_out = {}

    for top5_w in weights:
        label = f"World_{int(round((1-top5_w)*100))}_Top5_{int(round(top5_w*100))}"
        lump, _, lump_cost = simulate(top5_w, adj, world, targets, dca=False)
        dca_equity, cf, dca_cost = simulate(top5_w, adj, world, targets, dca=True)
        s = stats(lump)
        s.update({
            "strategy": label,
            "world_weight_pct": 100 * (1 - top5_w),
            "top5_weight_pct": 100 * top5_w,
            "dca_xirr_pct": 100 * xirr(cf),
            "dca_contributed": -sum(v for _, v in cf if v < 0),
            "dca_final_value": float(dca_equity.iloc[-1]),
            "estimated_total_cost_lump": lump_cost,
            "estimated_total_cost_dca": dca_cost,
        })
        rows.append(s)
        curves[label] = lump
        ret = lump.pct_change().fillna(0.0)
        annual_returns_out[label] = (1.0 + ret).groupby(ret.index.year).prod() - 1.0

    results = pd.DataFrame(rows).sort_values("top5_weight_pct")
    results["is_primary_mix"] = results["top5_weight_pct"].round().isin([0, 25, 50, 75, 100])
    results.to_csv(OUT / "blend_results.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "blend_equity_curves.csv")
    (pd.DataFrame(annual_returns_out) * 100.0).to_csv(OUT / "blend_annual_returns_pct.csv")

    best_calmar = results.loc[results["calmar"].idxmax()].to_dict()
    best_sharpe = results.loc[results["sharpe"].idxmax()].to_dict()
    best_dca = results.loc[results["dca_xirr_pct"].idxmax()].to_dict()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": [str(world.index[0].date()), str(world.index[-1].date())],
        "method": "Annual strategic rebalance between URTH (World proxy) and a point-in-time annual Top-5 cap-weight sleeve. Top-5 constituents are chosen only from prior calendar year-end market caps. DCA contributes 100 units monthly according to the strategic sleeve weights.",
        "cost_assumption": "10 bps one-way turnover stress cost applied to annual rebalances and DCA purchases.",
        "primary_mixes": ["100/0", "75/25", "50/50", "25/75", "0/100"],
        "grid": "Top-5 weight 0% to 100% by 10% increments, plus 25% and 75%.",
        "best_calmar": best_calmar,
        "best_sharpe": best_sharpe,
        "best_dca_xirr": best_dca,
        "limitations": [
            "URTH is a proxy for MSCI World, not the index itself.",
            "Top-5 selection is a transparent annual proxy using CompaniesMarketCap full market caps, not official MSCI free-float-adjusted point-in-time constituent data.",
            "Annual selection may miss intra-year changes in the true MSCI World Top 5.",
            "The 2018-2026 period was unusually favorable to US mega-cap technology; concentration risk is substantial.",
            "Taxes, FX conversion, bid-ask spreads and security-specific broker commissions are not modeled beyond the generic 10 bps turnover stress cost.",
            "Historical performance is not a forecast."
        ],
        "results": results.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
    }
    (OUT / "blend_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== WORLD / TOP-5 BLEND RESULTS ===")
    cols = ["strategy", "cagr_pct", "max_drawdown_pct", "sharpe", "sortino", "calmar", "final_value_10000", "dca_xirr_pct", "dca_final_value"]
    print(results[cols].to_string(index=False))
    print("\nBest Calmar:", best_calmar["strategy"], best_calmar["calmar"])
    print("Best Sharpe:", best_sharpe["strategy"], best_sharpe["sharpe"])
    print("Best DCA XIRR:", best_dca["strategy"], best_dca["dca_xirr_pct"])


if __name__ == "__main__":
    main()
