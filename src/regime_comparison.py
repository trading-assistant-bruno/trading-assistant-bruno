from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import crypto_backtest as base
import crypto_backtest_yahoo as yahoo
import crypto_risk_backtest as risk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "regime_comparison"
OUT.mkdir(parents=True, exist_ok=True)
START = "2018-01-01"
END = "2026-08-14"  # yfinance end is exclusive
MONTHLY = 100.0


def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close"]].dropna().copy()


def regime_from_close(close: pd.Series, sma_days: int = 200) -> pd.Series:
    sma = close.rolling(sma_days).mean()
    regime = pd.Series(index=close.index, dtype="object")
    regime.loc[close >= sma] = "BULL"
    regime.loc[close < sma] = "BEAR"
    return regime


def conditional_return_summary(returns: pd.Series, regime: pd.Series, label: str) -> list[dict]:
    rows = []
    aligned = pd.concat([returns.rename("r"), regime.rename("regime")], axis=1).dropna()
    for state in ["BULL", "BEAR"]:
        x = aligned.loc[aligned["regime"] == state, "r"]
        if x.empty:
            continue
        compound = (1 + x).prod() - 1
        ann = (1 + x.mean()) ** 365 - 1
        vol = x.std(ddof=0) * np.sqrt(365)
        sharpe = (x.mean() * 365 / vol) if vol > 0 else np.nan
        rows.append({
            "asset_or_strategy": label,
            "regime": state,
            "days": int(len(x)),
            "share_days_pct": float(100 * len(x) / len(aligned)),
            "compounded_contribution_pct": float(100 * compound),
            "annualized_mean_return_pct": float(100 * ann),
            "annualized_vol_pct": float(100 * vol),
            "sharpe_like": float(sharpe),
            "positive_day_pct": float(100 * (x > 0).mean()),
        })
    return rows


def dca_cohorts(df: pd.DataFrame, regime: pd.Series, asset: str) -> list[dict]:
    periods = df.index.to_period("M")
    first_dates = []
    for p in periods.unique():
        idx = df.index[periods == p]
        if len(idx):
            first_dates.append(idx[0])
    final_px = float(df["Close"].iloc[-1])
    final_date = df.index[-1]
    rows = []
    for state in ["BULL", "BEAR"]:
        contrib_dates = [d for d in first_dates if d in regime.index and regime.loc[d] == state]
        if not contrib_dates:
            continue
        units = sum(MONTHLY / float(df.at[d, "Open"]) for d in contrib_dates)
        contributed = MONTHLY * len(contrib_dates)
        final_value = units * final_px
        # Contribution-cohort annualized return: each monthly lot is annualized to the final date,
        # then summarized by median; this avoids pretending a DCA has a single CAGR.
        lot_ann = []
        for d in contrib_dates:
            years = max((final_date - d).days / 365.25, 1 / 365.25)
            multiple = final_px / float(df.at[d, "Open"])
            lot_ann.append(multiple ** (1 / years) - 1)
        rows.append({
            "asset": asset,
            "purchase_regime": state,
            "months": len(contrib_dates),
            "contributed": contributed,
            "final_value": final_value,
            "wealth_multiple": final_value / contributed,
            "median_lot_annualized_return_pct": 100 * float(np.median(lot_ann)),
            "mean_lot_annualized_return_pct": 100 * float(np.mean(lot_ann)),
        })
    return rows


def episode_table(benchmark: pd.DataFrame, trading_equity: pd.Series) -> pd.DataFrame:
    close = benchmark["Close"].astype(float)
    regime = regime_from_close(close)
    # Ignore pre-SMA200 warm-up.
    valid = regime.dropna()
    if valid.empty:
        return pd.DataFrame()
    starts = valid.ne(valid.shift(1)).cumsum()
    rows = []
    for _, group in valid.groupby(starts):
        state = group.iloc[0]
        start = group.index[0]
        end = group.index[-1]
        if len(group) < 14:  # suppress tiny whipsaw episodes from headline table
            continue
        b = close.loc[start:end]
        eq = trading_equity.reindex(b.index).ffill().dropna()
        if len(b) < 2 or len(eq) < 2:
            continue
        rows.append({
            "regime": state,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "days": len(group),
            "btc_return_pct": 100 * (b.iloc[-1] / b.iloc[0] - 1),
            "crypto_trading_return_pct": 100 * (eq.iloc[-1] / eq.iloc[0] - 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    btc = download("BTC-USD")
    world = download("URTH")
    btc_regime = regime_from_close(btc["Close"])
    world_regime = regime_from_close(world["Close"])

    rows = []
    rows += conditional_return_summary(btc["Close"].pct_change(), btc_regime.shift(1), "BTC_buy_hold")
    rows += conditional_return_summary(world["Close"].pct_change(), world_regime.shift(1), "MSCI_World_proxy_URTH_buy_hold")

    # Re-run the previously selected risk-normalized crypto variant (2 ATR + SMA100).
    prices = yahoo.load_prices_yahoo()
    close, atrs, hybrid, strategy_regime, ranking_momentum, sma100 = risk.build_indicators(prices)
    close_ffill = close.ffill()
    stop_cfg = next(x for x in risk.RISK_CONFIG["risk"]["stops"] if x["name"] == "atr_2_0")
    exit_cfg = next(x for x in risk.RISK_CONFIG["exits"] if x["name"] == "sma100")
    result, equity, trades = risk.run_variant(
        prices, close, close_ffill, atrs, hybrid, strategy_regime,
        ranking_momentum, sma100, stop_cfg, exit_cfg
    )
    trading_returns = equity.pct_change().fillna(0.0)
    btc_regime_for_equity = btc_regime.reindex(equity.index).ffill().shift(1)
    rows += conditional_return_summary(trading_returns, btc_regime_for_equity, "Crypto_trading_D_2ATR_SMA100")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "regime_conditional_returns.csv", index=False)

    dca = pd.DataFrame(dca_cohorts(btc, btc_regime, "BTC") + dca_cohorts(world, world_regime, "MSCI_World_proxy_URTH"))
    dca.to_csv(OUT / "dca_purchase_regime_cohorts.csv", index=False)

    episodes = episode_table(btc, equity)
    episodes.to_csv(OUT / "btc_regime_episodes.csv", index=False)

    # Current state using last complete downloaded bar.
    current = {
        "btc_last_date": btc.index[-1].date().isoformat(),
        "btc_close": float(btc["Close"].iloc[-1]),
        "btc_sma200": float(btc["Close"].rolling(200).mean().iloc[-1]),
        "btc_macro_regime_sma200": str(btc_regime.iloc[-1]),
        "btc_momentum_28d_pct": float(100 * (btc["Close"].iloc[-1] / btc["Close"].iloc[-29] - 1)) if len(btc) >= 29 else None,
        "strategy_new_long_gate": "ON" if (btc["Close"].iloc[-1] / btc["Close"].iloc[-29] - 1) > 0 else "OFF",
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": "Macro bull/bear is classified mechanically from prior close versus SMA200. Trading strategy itself still uses its original 28-day BTC momentum gate; SMA200 is only an independent analysis label.",
        "current": current,
        "conditional_returns": summary.replace({np.nan: None}).to_dict(orient="records"),
        "dca_purchase_cohorts": dca.replace({np.nan: None}).to_dict(orient="records"),
        "episodes": episodes.replace({np.nan: None}).to_dict(orient="records"),
        "crypto_variant": result.__dict__,
    }
    (OUT / "regime_comparison.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=== CURRENT ===")
    print(json.dumps(current, indent=2))
    print("\n=== CONDITIONAL RETURNS ===")
    print(summary.to_string(index=False))
    print("\n=== DCA PURCHASE COHORTS ===")
    print(dca.to_string(index=False))
    print("\n=== BTC REGIME EPISODES >=14 DAYS ===")
    print(episodes.to_string(index=False))


if __name__ == "__main__":
    main()
