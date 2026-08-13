from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dca_comparison"
OUT.mkdir(parents=True, exist_ok=True)

START = "2018-01-01"
END = "2026-08-13"  # yfinance end is exclusive
MONTHLY = 100.0

ASSETS = {
    "BTC": "BTC-USD",
    "MSCI_World_proxy_URTH": "URTH",
}


def download(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, start=START, end=END, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "Close"]].dropna().copy()


def cagr_and_dd(df: pd.DataFrame) -> tuple[float, float, float]:
    px = df["Close"].astype(float)
    years = (px.index[-1] - px.index[0]).days / 365.25
    cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1
    dd = px / px.cummax() - 1
    return 100 * cagr, 100 * float(dd.min()), float(px.iloc[-1] / px.iloc[0])


def monthly_investment_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    periods = df.index.to_period("M")
    dates = []
    for p in periods.unique():
        mask = periods == p
        dates.append(df.index[mask][0])
    return dates


def xnpv(rate: float, cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    t0 = cashflows[0][0]
    return sum(v / ((1 + rate) ** (((d - t0).days) / 365.25)) for d, v in cashflows)


def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = xnpv(lo, cashflows), xnpv(hi, cashflows)
    while f_lo * f_hi > 0 and hi < 1e6:
        hi *= 2
        f_hi = xnpv(hi, cashflows)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(300):
        mid = (lo + hi) / 2
        f_mid = xnpv(mid, cashflows)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2


def dca_stats(df: pd.DataFrame) -> dict:
    dates = monthly_investment_dates(df)
    units = 0.0
    total = 0.0
    cashflows: list[tuple[pd.Timestamp, float]] = []
    for d in dates:
        px = float(df.at[d, "Open"])
        units += MONTHLY / px
        total += MONTHLY
        cashflows.append((d, -MONTHLY))
    final_date = df.index[-1]
    final_px = float(df.iloc[-1]["Close"])
    final_value = units * final_px
    cashflows.append((final_date, final_value))
    irr = xirr(cashflows)
    return {
        "monthly_contribution": MONTHLY,
        "months": len(dates),
        "total_contributed": total,
        "final_value": final_value,
        "wealth_multiple_on_contributions": final_value / total,
        "xirr_pct": 100 * irr,
    }


def main() -> None:
    rows = []
    details = {}
    for name, ticker in ASSETS.items():
        df = download(ticker)
        cagr, max_dd, lump_multiple = cagr_and_dd(df)
        dca = dca_stats(df)
        rows.append({
            "asset": name,
            "ticker": ticker,
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "lump_sum_cagr_pct": cagr,
            "lump_sum_max_drawdown_pct": max_dd,
            "lump_sum_wealth_multiple": lump_multiple,
            "dca_xirr_pct": dca["xirr_pct"],
            "dca_total_contributed": dca["total_contributed"],
            "dca_final_value": dca["final_value"],
            "dca_wealth_multiple_on_contributions": dca["wealth_multiple_on_contributions"],
            "dca_months": dca["months"],
        })
        details[name] = dca

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "dca_benchmark_comparison.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_requested": [START, END],
        "note": "URTH is used only as a liquid historical proxy for the MSCI World Index; results exclude taxes, ETF TER, FX conversion costs and trading commissions. BTC DCA likewise excludes venue fees to isolate asset-path performance.",
        "results": rows,
    }
    (OUT / "dca_benchmark_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
