from __future__ import annotations

import pandas as pd
import yfinance as yf

import crypto_backtest as base

YAHOO_SYMBOLS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "ADAUSDT": "ADA-USD",
    "LINKUSDT": "LINK-USD",
    "AVAXUSDT": "AVAX-USD",
    "DOGEUSDT": "DOGE-USD",
    "LTCUSDT": "LTC-USD",
    "BCHUSDT": "BCH-USD",
    "SUIUSDT": "SUI20947-USD",
}


def fetch_yahoo_daily(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    yahoo_symbol = YAHOO_SYMBOLS[symbol]
    ticker = yf.Ticker(yahoo_symbol)
    kwargs = {
        "start": start_date,
        "interval": "1d",
        "auto_adjust": False,
        "actions": False,
    }
    if end_date:
        # yfinance treats end as exclusive; add one day to include requested end date.
        kwargs["end"] = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = ticker.history(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if not all(c in df.columns for c in keep):
        return pd.DataFrame()

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    df.index.name = "date"

    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quote_asset_volume"] = df["close"] * df["volume"]
    return df.dropna(subset=["close"]).sort_index()


def load_prices_yahoo() -> dict[str, pd.DataFrame]:
    cfg = base.CONFIG["backtest"]
    symbols = base.CONFIG["universe"]["symbols"]
    min_history = int(base.CONFIG["universe"].get("min_history_days", 180))
    output: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        yahoo_symbol = YAHOO_SYMBOLS.get(symbol)
        if yahoo_symbol is None:
            print(f"No Yahoo mapping for {symbol}; skipping")
            continue

        print(f"Downloading {symbol} via Yahoo Finance ({yahoo_symbol})")
        try:
            df = fetch_yahoo_daily(symbol, cfg["start_date"], cfg.get("end_date"))
        except Exception as exc:
            print(f"Failed {symbol}: {exc}")
            continue

        if len(df) >= min_history:
            output[symbol] = df
            df.to_csv(base.DATA_DIR / f"{symbol}.csv")
            print(f"Loaded {symbol}: {len(df)} daily bars")
        else:
            print(f"Skipping {symbol}: only {len(df)} bars (< {min_history})")

    return output


def cap_positions_lagged(signal: pd.DataFrame, close: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """Cap positions without using today's close to rank today's return.

    portfolio_returns already shifts the eligibility signal by one day. Ranking must
    therefore also be based only on information known at the previous close.
    """
    lagged_momentum = close.pct_change(28).shift(1)
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for dt in signal.index:
        eligible = signal.loc[dt]
        candidates = eligible[eligible > 0].index.tolist()
        if not candidates:
            continue

        ranked = (
            lagged_momentum.loc[dt, candidates]
            .dropna()
            .sort_values(ascending=False)
            .index.tolist()
        )
        chosen = ranked[:max_positions]
        if chosen:
            out.loc[dt, chosen] = 1.0 / len(chosen)

    return out


if __name__ == "__main__":
    # GitHub-hosted runners can receive HTTP 451 from Binance depending on region.
    # Keep the strategy engine unchanged and swap only the historical data source.
    base.load_prices = load_prices_yahoo

    # Critical anti-look-ahead fix: the original prototype ranked eligible coins
    # using the current day's close while also applying that day's return.
    # This replacement uses only prior-close information for ranking.
    base.cap_positions = cap_positions_lagged

    base.main()
