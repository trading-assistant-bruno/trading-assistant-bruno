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


if __name__ == "__main__":
    # GitHub-hosted runners can receive HTTP 451 from Binance depending on region.
    # Keep the strategy engine unchanged and swap only the historical data source.
    base.load_prices = load_prices_yahoo
    base.main()
