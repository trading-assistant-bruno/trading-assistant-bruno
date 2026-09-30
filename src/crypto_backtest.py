from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "crypto_config.yml").read_text(encoding="utf-8"))
DATA_DIR = ROOT / "data" / "crypto"
DOCS_DIR = ROOT / "docs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class Metrics:
    strategy: str
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    total_return_pct: float
    exposure_pct: float
    trades: int


def fetch_binance_daily(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000) if end_date else None
    rows: list[list] = []

    while True:
        params = {"symbol": symbol, "interval": "1d", "limit": 1000, "startTime": start_ms}
        if end_ms is not None:
            params["endTime"] = end_ms
        response = requests.get(BINANCE_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + 86_400_000
        if next_start <= start_ms or len(batch) < 1000:
            break
        start_ms = next_start
        if end_ms is not None and start_ms > end_ms:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("date")[["open", "high", "low", "close", "volume", "quote_asset_volume"]].sort_index()


def load_prices() -> dict[str, pd.DataFrame]:
    cfg = CONFIG["backtest"]
    symbols = CONFIG["universe"]["symbols"]
    output: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        print(f"Downloading {symbol}")
        try:
            df = fetch_binance_daily(symbol, cfg["start_date"], cfg.get("end_date"))
        except Exception as exc:
            print(f"Failed {symbol}: {exc}")
            continue
        if len(df) >= int(CONFIG["universe"].get("min_history_days", 180)):
            output[symbol] = df
            df.to_csv(DATA_DIR / f"{symbol}.csv")
    return output


def align_close(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat({s: d["close"] for s, d in prices.items()}, axis=1).sort_index()


def signal_momentum(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return (close.pct_change(lookback) > 0).astype(float)


def signal_sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return (close > close.rolling(window).mean()).astype(float)


def signal_breakout(prices: dict[str, pd.DataFrame], entry_days: int, exit_days: int) -> pd.DataFrame:
    signals = {}
    for symbol, df in prices.items():
        entry = df["close"] > df["high"].shift(1).rolling(entry_days).max()
        exit_ = df["close"] < df["low"].shift(1).rolling(exit_days).min()
        state = pd.Series(0.0, index=df.index)
        current = 0.0
        for idx in df.index:
            if bool(entry.loc[idx]):
                current = 1.0
            elif bool(exit_.loc[idx]):
                current = 0.0
            state.loc[idx] = current
        signals[symbol] = state
    return pd.DataFrame(signals).sort_index()


def build_signals(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = align_close(prices)
    s = CONFIG["strategies"]
    benchmark = CONFIG["universe"]["benchmark"]

    mom = signal_momentum(close, int(s["momentum"]["lookback_days"]))
    sma = signal_sma(close, int(s["sma"]["window_days"]))
    brk = signal_breakout(prices, int(s["breakout"]["entry_days"]), int(s["breakout"]["exit_days"]))

    market_mom = close[benchmark].pct_change(int(s["regime_momentum"]["market_momentum_days"])) > 0
    regime = pd.DataFrame(np.repeat(market_mom.values[:, None], close.shape[1], axis=1), index=close.index, columns=close.columns).astype(float)

    hybrid = (
        regime
        * signal_sma(close, int(s["hybrid"]["sma_days"]))
        * signal_momentum(close, int(s["hybrid"]["momentum_days"]))
        * signal_breakout(prices, int(s["hybrid"]["breakout_days"]), int(s["breakout"]["exit_days"]))
    )
    regime_momentum = regime * signal_momentum(close, int(s["regime_momentum"]["momentum_days"]))

    return {
        "A_momentum": mom,
        "B_sma": sma,
        "C_breakout": brk,
        "D_hybrid": hybrid,
        "E_regime_momentum": regime_momentum,
    }


def cap_positions(signal: pd.DataFrame, close: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    momentum = close.pct_change(28)
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for dt in signal.index:
        eligible = signal.loc[dt]
        candidates = eligible[eligible > 0].index.tolist()
        if not candidates:
            continue
        ranked = momentum.loc[dt, candidates].dropna().sort_values(ascending=False).index.tolist()
        chosen = ranked[:max_positions]
        if chosen:
            out.loc[dt, chosen] = 1.0 / len(chosen)
    return out


def portfolio_returns(signal: pd.DataFrame, close: pd.DataFrame, one_way_cost_pct: float, max_positions: int) -> tuple[pd.Series, int, float]:
    common = close.index.intersection(signal.index)
    close = close.loc[common]
    signal = signal.reindex(common).reindex(columns=close.columns).fillna(0.0)
    weights = cap_positions(signal.shift(1).fillna(0.0), close, max_positions)
    asset_ret = close.pct_change().fillna(0.0)
    gross = (weights * asset_ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * (one_way_cost_pct / 100.0)
    net = gross - cost
    trades = int((turnover > 0).sum())
    exposure = float((weights.abs().sum(axis=1) > 0).mean())
    return net, trades, exposure


def calc_metrics(name: str, returns: pd.Series, trades: int, exposure: float) -> Metrics:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    if len(equity) < 2:
        return Metrics(name, *(float("nan"),) * 7, trades)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    ann_vol = returns.std(ddof=0) * math.sqrt(365)
    sharpe = (returns.mean() * 365 / ann_vol) if ann_vol > 0 else float("nan")
    downside = returns[returns < 0].std(ddof=0) * math.sqrt(365)
    sortino = (returns.mean() * 365 / downside) if downside > 0 else float("nan")
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    return Metrics(
        strategy=name,
        cagr_pct=100 * cagr,
        max_drawdown_pct=100 * max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        total_return_pct=100 * total_return,
        exposure_pct=100 * exposure,
        trades=trades,
    )


def benchmark_returns(close: pd.DataFrame, symbol: str, one_way_cost_pct: float) -> tuple[pd.Series, int, float]:
    r = close[symbol].pct_change().fillna(0.0)
    first_valid = close[symbol].first_valid_index()
    if first_valid is not None and first_valid in r.index:
        r.loc[first_valid] -= one_way_cost_pct / 100.0
    return r, 1, 1.0


def robustness_summary(close: pd.DataFrame) -> list[dict]:
    bench = CONFIG["universe"]["benchmark"]
    market = close[bench]
    cost = float(CONFIG["backtest"]["transaction_cost_pct_one_way"])
    max_positions = int(CONFIG["backtest"]["max_positions"])
    rows: list[dict] = []

    for lb in CONFIG["robustness"]["momentum_lookbacks"]:
        signal = signal_momentum(close, int(lb))
        ret, trades, exp = portfolio_returns(signal, close, cost, max_positions)
        m = calc_metrics(f"momentum_{lb}", ret, trades, exp)
        rows.append(m.__dict__)

    for window in CONFIG["robustness"]["sma_windows"]:
        signal = signal_sma(close, int(window))
        ret, trades, exp = portfolio_returns(signal, close, cost, max_positions)
        m = calc_metrics(f"sma_{window}", ret, trades, exp)
        rows.append(m.__dict__)

    for lb in CONFIG["robustness"]["momentum_lookbacks"]:
        regime = (market.pct_change(int(lb)) > 0).astype(float)
        sig = signal_momentum(close, int(lb)).mul(regime, axis=0)
        ret, trades, exp = portfolio_returns(sig, close, cost, max_positions)
        m = calc_metrics(f"regime_momentum_{lb}", ret, trades, exp)
        rows.append(m.__dict__)

    return rows


def main() -> None:
    prices = load_prices()
    benchmark = CONFIG["universe"]["benchmark"]
    if benchmark not in prices:
        raise RuntimeError("BTCUSDT benchmark unavailable")

    close = align_close(prices)
    signals = build_signals(prices)
    cost = float(CONFIG["backtest"]["transaction_cost_pct_one_way"])
    max_positions = int(CONFIG["backtest"]["max_positions"])

    metrics: list[Metrics] = []
    benchmark_ret, benchmark_trades, benchmark_exp = benchmark_returns(close, benchmark, cost)
    metrics.append(calc_metrics("BTC_buy_hold", benchmark_ret, benchmark_trades, benchmark_exp))

    returns_by_strategy: dict[str, pd.Series] = {"BTC_buy_hold": benchmark_ret}
    for name, sig in signals.items():
        ret, trades, exposure = portfolio_returns(sig, close, cost, max_positions)
        metrics.append(calc_metrics(name, ret, trades, exposure))
        returns_by_strategy[name] = ret

    result_df = pd.DataFrame([m.__dict__ for m in metrics]).sort_values("calmar", ascending=False)
    annual = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in returns_by_strategy.items()}) * 100
    robust_df = pd.DataFrame(robustness_summary(close)).sort_values("calmar", ascending=False)

    result_df.to_csv(DATA_DIR / "backtest_results.csv", index=False)
    annual.to_csv(DATA_DIR / "backtest_annual_returns.csv")
    robust_df.to_csv(DATA_DIR / "backtest_robustness.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols_loaded": sorted(prices.keys()),
        "metrics": result_df.replace({np.nan: None}).to_dict(orient="records"),
        "annual_returns_pct": annual.replace({np.nan: None}).to_dict(),
        "robustness": robust_df.replace({np.nan: None}).to_dict(orient="records"),
    }
    (DATA_DIR / "backtest_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== CRYPTO BACKTEST RESULTS ===")
    print(result_df.to_string(index=False))
    print("\n=== ANNUAL RETURNS (%) ===")
    print(annual.to_string())
    print("\n=== ROBUSTNESS ===")
    print(robust_df.to_string(index=False))


if __name__ == "__main__":
    main()
