from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_backtest as base
import crypto_backtest_yahoo as yahoo
import crypto_risk_backtest as risk

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "crypto_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_cfg(items: list[dict], name: str) -> dict:
    for item in items:
        if item.get("name") == name:
            return item
    raise KeyError(name)


def mask_exclusions(hybrid: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    out = hybrid.copy()
    for symbol in excluded:
        if symbol in out.columns:
            out[symbol] = 0.0
    return out


def liquidity_mask(prices: dict[str, pd.DataFrame], close: pd.DataFrame, top_n: int, window: int = 30) -> pd.DataFrame:
    qv = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for symbol, df in prices.items():
        if "quote_asset_volume" in df.columns:
            qv.loc[df.index.intersection(qv.index), symbol] = df.loc[df.index.intersection(qv.index), "quote_asset_volume"]
    liq = qv.rolling(window, min_periods=max(7, window // 3)).median()
    mask = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for dt in liq.index:
        row = liq.loc[dt].dropna().sort_values(ascending=False)
        chosen = row.head(top_n).index
        mask.loc[dt, chosen] = 1.0
    return mask


def run_scenario(
    scenario: str,
    hybrid_scenario: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    close_ffill: pd.DataFrame,
    atrs: dict[str, pd.Series],
    regime: pd.Series,
    ranking_momentum: pd.DataFrame,
    sma100: pd.DataFrame,
    stop_cfg: dict,
    exit_cfg: dict,
) -> tuple[dict, pd.DataFrame]:
    result, equity, trades = risk.run_variant(
        prices,
        close,
        close_ffill,
        atrs,
        hybrid_scenario,
        regime,
        ranking_momentum,
        sma100,
        stop_cfg,
        exit_cfg,
    )
    row = result.__dict__.copy()
    row["scenario"] = scenario
    row["variant"] = f"{scenario}__{result.variant}"

    if not trades.empty:
        trades = trades.copy()
        trades.insert(0, "scenario", scenario)
        trades.insert(1, "risk_variant", f"{stop_cfg['name']}__{exit_cfg['name']}")
    return row, trades


def main() -> None:
    prices = yahoo.load_prices_yahoo()
    benchmark = base.CONFIG["universe"]["benchmark"]
    if benchmark not in prices:
        raise RuntimeError("BTC benchmark unavailable")

    close, atrs, hybrid, regime, ranking_momentum, sma100 = risk.build_indicators(prices)
    close_ffill = close.ffill()

    stop_variants = [
        pick_cfg(risk.RISK_CONFIG["risk"]["stops"], "atr_2_0"),
        pick_cfg(risk.RISK_CONFIG["risk"]["stops"], "atr_2_5"),
    ]
    exit_cfg = pick_cfg(risk.RISK_CONFIG["exits"], "sma100")

    symbols = [s for s in base.CONFIG["universe"]["symbols"] if s in close.columns]

    scenario_signals: dict[str, pd.DataFrame] = {"full_universe": hybrid.copy()}

    # Leave-one-out: the excluded asset cannot be traded, but BTC can still serve as market-regime proxy.
    for symbol in symbols:
        scenario_signals[f"exclude_{symbol}"] = mask_exclusions(hybrid, {symbol})

    # Explicit concentration stress tests.
    scenario_signals["exclude_BTC_SOL_DOGE"] = mask_exclusions(hybrid, {"BTCUSDT", "SOLUSDT", "DOGEUSDT"})
    scenario_signals["exclude_BTC_ETH_SOL"] = mask_exclusions(hybrid, {"BTCUSDT", "ETHUSDT", "SOLUSDT"})
    scenario_signals["alts_only_no_BTC_ETH"] = mask_exclusions(hybrid, {"BTCUSDT", "ETHUSDT"})

    # Dynamic liquidity subsets inside the fixed currently-supported universe.
    for n in [4, 6, 8]:
        scenario_signals[f"top{n}_trailing_liquidity"] = hybrid * liquidity_mask(prices, close, top_n=n, window=30)

    rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []

    for stop_cfg in stop_variants:
        for scenario, signal in scenario_signals.items():
            print(f"Running {scenario} with {stop_cfg['name']} + sma100")
            row, trades = run_scenario(
                scenario,
                signal,
                prices,
                close,
                close_ffill,
                atrs,
                regime,
                ranking_momentum,
                sma100,
                stop_cfg,
                exit_cfg,
            )
            rows.append(row)
            if not trades.empty:
                all_trades.append(trades)

    results = pd.DataFrame(rows)
    results = results.sort_values(["stop_rule", "calmar", "cagr_pct"], ascending=[True, False, False])
    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Contribution concentration for the baseline best candidate only.
    contributions = pd.DataFrame()
    if not trades_out.empty:
        baseline = trades_out[
            (trades_out["scenario"] == "full_universe")
            & (trades_out["risk_variant"] == "atr_2_0__sma100")
        ].copy()
        if not baseline.empty:
            contributions = (
                baseline.groupby("symbol")
                .agg(
                    trades=("pnl", "size"),
                    pnl=("pnl", "sum"),
                    avg_r=("r_multiple", "mean"),
                    win_rate_pct=("pnl", lambda s: 100.0 * float((s > 0).mean())),
                )
                .sort_values("pnl", ascending=False)
            )
            total_abs = float(contributions["pnl"].abs().sum())
            total_net = float(contributions["pnl"].sum())
            contributions["share_abs_pnl_pct"] = 100.0 * contributions["pnl"].abs() / total_abs if total_abs else np.nan
            contributions["share_net_pnl_pct"] = 100.0 * contributions["pnl"] / total_net if total_net else np.nan

    # Summary statistics across leave-one-out scenarios.
    loo = results[results["scenario"].str.startswith("exclude_") & ~results["scenario"].isin(["exclude_BTC_SOL_DOGE", "exclude_BTC_ETH_SOL"])].copy()
    summary_rows = []
    for stop_rule, group in loo.groupby("stop_rule"):
        summary_rows.append(
            {
                "stop_rule": stop_rule,
                "n_leave_one_out": int(len(group)),
                "cagr_min_pct": float(group["cagr_pct"].min()),
                "cagr_median_pct": float(group["cagr_pct"].median()),
                "cagr_max_pct": float(group["cagr_pct"].max()),
                "max_dd_worst_pct": float(group["max_drawdown_pct"].min()),
                "max_dd_median_pct": float(group["max_drawdown_pct"].median()),
                "calmar_min": float(group["calmar"].min()),
                "calmar_median": float(group["calmar"].median()),
            }
        )
    loo_summary = pd.DataFrame(summary_rows)

    results.to_csv(OUT_DIR / "robustness_results.csv", index=False)
    loo_summary.to_csv(OUT_DIR / "leave_one_out_summary.csv", index=False)
    trades_out.to_csv(OUT_DIR / "robustness_trades.csv", index=False)
    contributions.to_csv(OUT_DIR / "baseline_symbol_contributions.csv")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance daily OHLC via yfinance",
        "important_limitation": (
            "This is not a true all-crypto point-in-time universe. It stress-tests the fixed current 11-asset practical universe "
            "with leave-one-out and trailing-liquidity masks. Historical delisted/failed coins outside this set remain unmodeled."
        ),
        "results": results.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "leave_one_out_summary": loo_summary.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
        "baseline_symbol_contributions": contributions.reset_index().replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records"),
    }
    (OUT_DIR / "robustness_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n=== CRYPTO ROBUSTNESS RESULTS ===")
    print(results.to_string(index=False))
    print("\n=== LEAVE-ONE-OUT SUMMARY ===")
    print(loo_summary.to_string(index=False))
    print("\n=== BASELINE SYMBOL CONTRIBUTIONS ===")
    print(contributions.to_string())
    print("\nIMPORTANT LIMITATION: fixed current 11-asset universe; not a full historical all-crypto point-in-time universe.")


if __name__ == "__main__":
    main()
