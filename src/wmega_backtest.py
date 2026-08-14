from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from mscidata import msci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "wmega_backtest"
OUT.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2026-08-14"
WORLD_CODE = "990100"
MEGA_CODE = "761936"
WMEGA_TER = 0.0012
MONTHLY_CONTRIBUTION = 100.0
INITIAL = 10_000.0


def levels(code: str) -> pd.Series:
    df = msci.get_levels(code, START, END, variant="NETR")
    if df is None or len(df) == 0:
        raise RuntimeError(f"MSCI levels unavailable for {code}")
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["LEVEL"] = pd.to_numeric(df["LEVEL"], errors="coerce")
    s = df.dropna(subset=["DATE", "LEVEL"]).set_index("DATE")["LEVEL"].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep="last")].astype(float)


def xnpv(rate: float, cf: list[tuple[pd.Timestamp, float]]) -> float:
    t0 = cf[0][0]
    return sum(v / ((1 + rate) ** (((d - t0).days) / 365.25)) for d, v in cf)


def xirr(cf: list[tuple[pd.Timestamp, float]]) -> float:
    lo, hi = -0.9999, 10.0
    flo, fhi = xnpv(lo, cf), xnpv(hi, cf)
    while flo * fhi > 0 and hi < 1e6:
        hi *= 2
        fhi = xnpv(hi, cf)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(250):
        mid = (lo + hi) / 2
        fm = xnpv(mid, cf)
        if flo * fm <= 0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return (lo + hi) / 2


def stats(equity: pd.Series) -> dict:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    r = equity.pct_change().fillna(0.0)
    dd = equity / equity.cummax() - 1
    vol = r.std(ddof=0) * math.sqrt(252)
    ann = r.mean() * 252
    annual = (1 + r).groupby(r.index.year).prod() - 1
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * float(dd.min()),
        "sharpe": ann / vol if vol else float("nan"),
        "final_value_10000": float(INITIAL * equity.iloc[-1] / equity.iloc[0]),
        "worst_year_pct": 100 * float(annual.min()),
        "best_year_pct": 100 * float(annual.max()),
    }


def dca_xirr(index: pd.Series) -> tuple[float, float, float, int]:
    periods = index.index.to_period("M")
    first_days = [index.index[periods == p][0] for p in periods.unique()]
    units = 0.0
    cf: list[tuple[pd.Timestamp, float]] = []
    for d in first_days:
        units += MONTHLY_CONTRIBUTION / float(index.loc[d])
        cf.append((d, -MONTHLY_CONTRIBUTION))
    final = units * float(index.iloc[-1])
    cf.append((index.index[-1], final))
    return 100 * xirr(cf), MONTHLY_CONTRIBUTION * len(first_days), final, len(first_days)


def fee_drag(index: pd.Series, annual_fee: float) -> pd.Series:
    r = index.pct_change().fillna(0.0)
    daily_fee = (1 + annual_fee) ** (1 / 252) - 1
    net_r = r - daily_fee
    out = (1 + net_r).cumprod()
    out.iloc[0] = 1.0
    return out


def normalized(index: pd.Series) -> pd.Series:
    return index / float(index.iloc[0])


def main() -> None:
    world_raw = levels(WORLD_CODE)
    mega_raw = levels(MEGA_CODE)
    common = world_raw.index.intersection(mega_raw.index).sort_values()
    if len(common) < 2500:
        raise RuntimeError(f"Common history too short: {len(common)} sessions; mega starts {mega_raw.index.min()}")

    world = normalized(world_raw.reindex(common))
    mega_index = normalized(mega_raw.reindex(common))
    wmega_proxy = normalized(fee_drag(mega_index, WMEGA_TER))

    # 50/50 blend, annual rebalance on the first common trading day of each year.
    blend = pd.Series(index=common, dtype=float)
    capital = 1.0
    units_world = 0.5 / world.iloc[0]
    units_mega = 0.5 / wmega_proxy.iloc[0]
    current_year = common[0].year
    for d in common:
        if d.year != current_year:
            capital = units_world * world.loc[d] + units_mega * wmega_proxy.loc[d]
            units_world = 0.5 * capital / world.loc[d]
            units_mega = 0.5 * capital / wmega_proxy.loc[d]
            current_year = d.year
        blend.loc[d] = units_world * world.loc[d] + units_mega * wmega_proxy.loc[d]
    blend = normalized(blend)

    series = {
        "MSCI_World_NETR": world,
        "MSCI_World_MegaCap_index_NETR": mega_index,
        "WMEGA_proxy_after_0.12pct_TER": wmega_proxy,
        "World50_WMEGA50_annual_rebalance": blend,
    }

    rows = []
    annual = {}
    for name, s in series.items():
        row = {"strategy": name, **stats(s)}
        dx, contrib, final, months = dca_xirr(s)
        row.update({
            "dca_xirr_pct": dx,
            "dca_contributed": contrib,
            "dca_final_value": final,
            "dca_months": months,
        })
        rows.append(row)
        annual[name] = ((1 + s.pct_change().fillna(0)).groupby(s.index.year).prod() - 1) * 100

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(series).to_csv(OUT / "equity.csv")
    pd.DataFrame(annual).to_csv(OUT / "annual_returns_pct.csv")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_period": [START, END],
        "actual_common_period": [str(common[0].date()), str(common[-1].date())],
        "mega_index_code": MEGA_CODE,
        "world_index_code": WORLD_CODE,
        "wmega_ter_assumption": WMEGA_TER,
        "notes": [
            "WMEGA itself launched in 2025; pre-launch history is the MSCI underlying index, not live ETF returns.",
            "WMEGA proxy subtracts 0.12% annual TER from the underlying index return path.",
            "No taxes, spreads, tracking difference or investor FX effects.",
            "The 50/50 World-WMEGA blend is rebalanced annually.",
            "Historical backtests are not forecasts.",
        ],
        "results": results.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Mega raw history:", mega_raw.index.min(), "->", mega_raw.index.max(), "rows", len(mega_raw))
    print("World raw history:", world_raw.index.min(), "->", world_raw.index.max(), "rows", len(world_raw))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
