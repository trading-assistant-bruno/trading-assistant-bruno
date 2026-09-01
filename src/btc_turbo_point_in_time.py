from __future__ import annotations

import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "btc_turbo_point_in_time"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-01-07")
TODAY = pd.Timestamp.utcnow().tz_localize(None).normalize()
END = TODAY - pd.Timedelta(days=(TODAY.weekday() - 6) % 7)
BTC = "BTC"
CASH = "__CASH__"
COST_ONE_WAY = 0.0023  # 0.18% commission + 0.05% slippage, per traded notional
UNIVERSE_TOP_N = 30
PRICE_DEPTH = 100

STABLE = {
    "USDT","USDC","BUSD","DAI","TUSD","USDP","PAX","GUSD","USDD","FDUSD","USDE",
    "FRAX","PYUSD","UST","USTC","EURT","EURC","SUSD","LUSD","USDS","USD1","USDE"
}
WRAPPED = {"WBTC","WETH","STETH","WSTETH","RETH","CBETH","WEETH","WBNB"}
EXCLUDE = STABLE | WRAPPED
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

VARIANTS = [
    {"name": "BTC_HOLD", "n_alts": 0, "alt_weight": 0.0, "rel4": 0.0, "riskoff_cash": 0.0},
    {"name": "PIT_Top2_Turbo30_rel5", "n_alts": 2, "alt_weight": 0.30, "rel4": 0.05, "riskoff_cash": 0.0},
    {"name": "PIT_Top2_Turbo50_rel5", "n_alts": 2, "alt_weight": 0.50, "rel4": 0.05, "riskoff_cash": 0.0},
    {"name": "PIT_Top2_Turbo75_rel5", "n_alts": 2, "alt_weight": 0.75, "rel4": 0.05, "riskoff_cash": 0.0},
    {"name": "PIT_Top1_Turbo50_rel5", "n_alts": 1, "alt_weight": 0.50, "rel4": 0.05, "riskoff_cash": 0.0},
    {"name": "PIT_Top2_Turbo50_rel10", "n_alts": 2, "alt_weight": 0.50, "rel4": 0.10, "riskoff_cash": 0.0},
    {"name": "PIT_Top2_Turbo50_rel5_RiskOff50", "n_alts": 2, "alt_weight": 0.50, "rel4": 0.05, "riskoff_cash": 0.50},
    {"name": "PIT_Top2_Turbo75_rel5_RiskOff50", "n_alts": 2, "alt_weight": 0.75, "rel4": 0.05, "riskoff_cash": 0.50},
]


def sundays(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    d = start + pd.Timedelta(days=(6 - start.weekday()) % 7)
    return list(pd.date_range(d, end, freq="7D"))


def parse_number(value) -> float:
    if pd.isna(value):
        return float("nan")
    s = str(value).replace("$", "").replace(",", "").replace("<", "").strip()
    s = s.replace("—", "").replace("-", "-")
    try:
        return float(s)
    except Exception:
        import re
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else float("nan")


def fetch_snapshot(dt: pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame | None, str | None]:
    url = f"https://coinmarketcap.com/historical/{dt:%Y%m%d}/"
    last_error = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=35)
            if r.status_code in (429, 403):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            tables = pd.read_html(io.StringIO(r.text))
            best = None
            for t in tables:
                cols = [str(c).strip() for c in t.columns]
                if "Rank" in cols and "Symbol" in cols and "Price" in cols:
                    best = t.copy()
                    break
            if best is None:
                raise RuntimeError("ranking table not found")
            best.columns = [str(c).strip() for c in best.columns]
            keep = [c for c in ["Rank", "Name", "Symbol", "Market Cap", "Price", "Volume (24h)", "volume (24h)"] if c in best.columns]
            best = best[keep].copy()
            best["Rank"] = pd.to_numeric(best["Rank"], errors="coerce")
            best["Symbol"] = best["Symbol"].astype(str).str.upper().str.strip()
            best["Price"] = best["Price"].map(parse_number)
            best = best.dropna(subset=["Rank", "Symbol", "Price"])
            best = best[(best["Rank"] >= 1) & (best["Rank"] <= PRICE_DEPTH)]
            best = best.drop_duplicates("Symbol", keep="first").sort_values("Rank")
            best["snapshot_date"] = dt
            return dt, best, None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.7 * (attempt + 1))
    return dt, None, last_error


def collect_snapshots() -> pd.DataFrame:
    dates = sundays(START, END)
    rows: list[pd.DataFrame] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_snapshot, dt): dt for dt in dates}
        for fut in as_completed(futures):
            dt, table, err = fut.result()
            if table is None:
                failures.append({"date": str(dt.date()), "error": err})
                print(f"FAIL {dt.date()}: {err}")
            else:
                rows.append(table)
                print(f"OK {dt.date()} rows={len(table)}")
    if not rows:
        raise RuntimeError("No CMC historical snapshots collected")
    snap = pd.concat(rows, ignore_index=True).sort_values(["snapshot_date", "Rank"])
    snap.to_csv(OUT / "snapshot_universe.csv", index=False)
    pd.DataFrame(failures).sort_values("date").to_csv(OUT / "snapshot_failures.csv", index=False)
    return snap


def build_panels(snap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = snap.pivot_table(index="snapshot_date", columns="Symbol", values="Price", aggfunc="first").sort_index()
    ranks = snap.pivot_table(index="snapshot_date", columns="Symbol", values="Rank", aggfunc="first").sort_index()
    return prices, ranks


def momentum_exact(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    past = prices.copy()
    past.index = past.index + pd.Timedelta(days=days)
    past = past.reindex(prices.index)
    return prices / past - 1.0


def target_weights(dt: pd.Timestamp, prices: pd.DataFrame, ranks: pd.DataFrame, mom4: pd.DataFrame, mom8: pd.DataFrame, cfg: dict) -> tuple[dict[str, float], dict]:
    if cfg["name"] == "BTC_HOLD":
        return {BTC: 1.0}, {"btc_mom4": np.nan, "btc_mom8": np.nan, "selected": ""}

    btc4 = mom4.at[dt, BTC] if BTC in mom4.columns else np.nan
    btc8 = mom8.at[dt, BTC] if BTC in mom8.columns else np.nan
    info = {"btc_mom4": btc4, "btc_mom8": btc8, "selected": ""}

    # Strong risk-off only when both horizons are negative. Otherwise BTC remains the default asset.
    if pd.notna(btc4) and pd.notna(btc8) and btc4 < 0 and btc8 < 0 and cfg["riskoff_cash"] > 0:
        c = float(cfg["riskoff_cash"])
        return {BTC: 1.0 - c, CASH: c}, info

    # Turbo activates only in a positive BTC regime.
    if pd.isna(btc4) or pd.isna(btc8) or btc4 <= 0 or btc8 <= 0:
        return {BTC: 1.0}, info

    candidates = []
    if dt not in ranks.index:
        return {BTC: 1.0}, info
    for sym, rank in ranks.loc[dt].dropna().items():
        if sym == BTC or sym in EXCLUDE or rank > UNIVERSE_TOP_N:
            continue
        if sym not in prices.columns:
            continue
        p = prices.at[dt, sym]
        m4 = mom4.at[dt, sym]
        m8 = mom8.at[dt, sym]
        if any(pd.isna(x) for x in (p, m4, m8)):
            continue
        if m4 <= 0 or m8 <= 0 or (m4 - btc4) < cfg["rel4"]:
            continue
        score = 0.60 * m4 + 0.40 * m8
        candidates.append((score, -float(rank), sym, m4, m8))

    candidates.sort(reverse=True)
    chosen = candidates[: int(cfg["n_alts"])]
    if not chosen:
        return {BTC: 1.0}, info

    aw = float(cfg["alt_weight"])
    w = {BTC: 1.0 - aw}
    each = aw / len(chosen)
    for _, _, sym, _, _ in chosen:
        w[sym] = each
    info["selected"] = ";".join(x[2] for x in chosen)
    return w, info


def asset_return(prices: pd.DataFrame, dt: pd.Timestamp, nxt: pd.Timestamp, sym: str) -> float:
    if sym == CASH:
        return 0.0
    if sym not in prices.columns:
        return -1.0
    p0 = prices.at[dt, sym] if dt in prices.index else np.nan
    p1 = prices.at[nxt, sym] if nxt in prices.index else np.nan
    if pd.isna(p0) or p0 <= 0:
        return -1.0
    # If a held top-30 coin disappears even from the next top-100 snapshot, assume total loss.
    # This is intentionally conservative and avoids silently forward-filling dead coins.
    if pd.isna(p1) or p1 <= 0:
        return -1.0
    return float(p1 / p0 - 1.0)


def simulate(prices: pd.DataFrame, ranks: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    mom4 = momentum_exact(prices, 28)
    mom8 = momentum_exact(prices, 56)
    dates = prices.index[(prices.index >= START) & (prices.index <= END)]
    valid_start = None
    for dt in dates:
        if BTC in mom8.columns and pd.notna(mom8.at[dt, BTC]):
            valid_start = dt
            break
    if valid_start is None:
        raise RuntimeError("No 8-week BTC history")
    dates = dates[dates >= valid_start]

    equity = 1.0
    eq = {dates[0]: equity}
    pre_weights = {CASH: 1.0}
    logs = []
    holdings = []

    for i in range(len(dates) - 1):
        dt, nxt = dates[i], dates[i + 1]
        target, info = target_weights(dt, prices, ranks, mom4, mom8, cfg)

        risky = set(k for k in pre_weights if k != CASH) | set(k for k in target if k != CASH)
        turnover = sum(abs(target.get(k, 0.0) - pre_weights.get(k, 0.0)) for k in risky)
        cost_frac = turnover * COST_ONE_WAY
        equity *= max(0.0, 1.0 - cost_frac)

        r_by = {sym: asset_return(prices, dt, nxt, sym) for sym in target}
        period_ret = sum(w * r_by[sym] for sym, w in target.items())
        equity *= max(0.0, 1.0 + period_ret)
        eq[nxt] = equity

        gross_end = {sym: w * (1.0 + r_by[sym]) for sym, w in target.items()}
        denom = sum(gross_end.values())
        if denom > 0:
            pre_weights = {sym: val / denom for sym, val in gross_end.items() if val > 0}
        else:
            pre_weights = {CASH: 1.0}

        logs.append({
            "strategy": cfg["name"], "date": dt, "next_date": nxt, "turnover": turnover,
            "cost_pct_equity": 100 * cost_frac, "period_return_pct": 100 * period_ret,
            "btc_mom4_pct": 100 * info["btc_mom4"] if pd.notna(info["btc_mom4"]) else np.nan,
            "btc_mom8_pct": 100 * info["btc_mom8"] if pd.notna(info["btc_mom8"]) else np.nan,
            "selected": info["selected"], "target_btc_pct": 100 * target.get(BTC, 0.0),
            "target_cash_pct": 100 * target.get(CASH, 0.0),
            "target_alt_pct": 100 * (1.0 - target.get(BTC, 0.0) - target.get(CASH, 0.0)),
        })
        for sym, w in target.items():
            holdings.append({"strategy": cfg["name"], "date": dt, "symbol": sym, "weight": w, "next_return": r_by[sym]})

    return pd.Series(eq, name=cfg["name"]).sort_index(), pd.DataFrame(logs), pd.DataFrame(holdings)


def metrics(name: str, eq: pd.Series, logs: pd.DataFrame) -> tuple[dict, pd.Series]:
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0
    dd = eq / eq.cummax() - 1.0
    vol = r.std(ddof=0) * math.sqrt(52)
    ann_mean = r.mean() * 52
    sharpe = ann_mean / vol if vol > 0 else np.nan
    annual = eq.resample("YE").last().pct_change()
    # include first partial year from first equity value
    first_year = eq.index[0].year
    first_end = eq[eq.index.year == first_year].iloc[-1]
    annual.loc[pd.Timestamp(f"{first_year}-12-31")] = first_end / eq.iloc[0] - 1.0
    annual = annual.sort_index()
    roll52 = eq / eq.shift(52) - 1.0
    row = {
        "strategy": name,
        "cagr_pct": 100 * cagr,
        "max_drawdown_pct": 100 * dd.min(),
        "sharpe_0rf": sharpe,
        "volatility_pct": 100 * vol,
        "final_multiple": float(eq.iloc[-1]),
        "best_calendar_year_pct": 100 * annual.max(),
        "worst_calendar_year_pct": 100 * annual.min(),
        "calendar_years_ge_50": int((annual >= 0.50).sum()),
        "calendar_years_ge_100": int((annual >= 1.00).sum()),
        "best_52week_return_pct": 100 * roll52.max(),
        "rolling_52week_windows_ge_100": int((roll52 >= 1.00).sum()),
        "annual_turnover_x": float(logs["turnover"].sum() / years) if len(logs) else 0.0,
        "total_modeled_cost_pct_initial_equity": float(logs["cost_pct_equity"].sum()) if len(logs) else 0.0,
        "weeks_with_alt_exposure_pct": 100 * float((logs["target_alt_pct"] > 0).mean()) if len(logs) else 0.0,
    }
    annual.index = annual.index.year
    return row, annual * 100


def main() -> None:
    snap = collect_snapshots()
    prices, ranks = build_panels(snap)
    if BTC not in prices.columns:
        raise RuntimeError("BTC missing from CMC snapshots")

    results = []
    annuals = {}
    curves = []
    all_logs = []
    all_holdings = []
    for cfg in VARIANTS:
        eq, logs, holdings = simulate(prices, ranks, cfg)
        row, annual = metrics(cfg["name"], eq, logs)
        results.append(row)
        annuals[cfg["name"]] = annual
        curves.append(eq)
        all_logs.append(logs)
        all_holdings.append(holdings)

    res = pd.DataFrame(results).sort_values("cagr_pct", ascending=False)
    annual_df = pd.DataFrame(annuals).sort_index()
    equity_df = pd.concat(curves, axis=1)
    logs_df = pd.concat(all_logs, ignore_index=True)
    holdings_df = pd.concat(all_holdings, ignore_index=True)

    res.to_csv(OUT / "results.csv", index=False)
    annual_df.to_csv(OUT / "annual_returns_pct.csv")
    equity_df.to_csv(OUT / "equity_curves.csv")
    logs_df.to_csv(OUT / "selection_log.csv", index=False)
    holdings_df.to_csv(OUT / "holdings_log.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "CoinMarketCap historical weekly snapshots",
        "start": str(equity_df.index.min().date()),
        "end": str(equity_df.index.max().date()),
        "universe_rule": f"Top {UNIVERSE_TOP_N} by market cap at each historical snapshot; prices retained to rank {PRICE_DEPTH}",
        "missing_held_coin_next_snapshot": "-100% return (conservative; no forward fill)",
        "cost_one_way_pct": 100 * COST_ONE_WAY,
        "results": res.replace({np.nan: None}).to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== TRUE POINT-IN-TIME BTC TURBO ===")
    print(res.to_string(index=False))
    print("\n=== ANNUAL RETURNS (%) ===")
    print(annual_df.to_string())
    print(f"\nSnapshots requested={len(sundays(START, END))}, collected={snap['snapshot_date'].nunique()}")


if __name__ == "__main__":
    main()
