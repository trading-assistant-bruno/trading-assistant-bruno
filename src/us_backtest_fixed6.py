from __future__ import annotations

import pandas as pd

import us_backtest as base

HISTORY_URL = "https://raw.githubusercontent.com/chinobing/historical_sp500_constituents/main/sp_500_historical_components.csv"
_HISTORY: pd.DataFrame | None = None


def load_history() -> pd.DataFrame:
    global _HISTORY
    if _HISTORY is not None:
        return _HISTORY
    df = pd.read_csv(HISTORY_URL)
    if not {"date", "tickers"}.issubset(df.columns):
        raise RuntimeError(f"Unexpected point-in-time CSV columns: {df.columns.tolist()}")
    df = df[["date", "tickers"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    df["members"] = df["tickers"].astype(str).apply(
        lambda s: {base.norm_symbol(x) for x in s.split(",") if base.norm_symbol(x)}
    )
    if len(df) < 1000:
        raise RuntimeError(f"Historical S&P snapshot file unexpectedly short: {len(df)} rows")
    _HISTORY = df.reset_index(drop=True)
    print(
        f"Loaded point-in-time S&P snapshots: {len(_HISTORY)} rows, "
        f"{_HISTORY.iloc[0]['date'].date()} -> {_HISTORY.iloc[-1]['date'].date()}"
    )
    return _HISTORY


def get_sp500_point_in_time_data_snapshot():
    hist = load_history()
    current = set(hist.iloc[-1]["members"])
    all_symbols: set[str] = set()
    for members in hist["members"]:
        all_symbols.update(members)
    # The second return value is deliberately the history dataframe; the
    # membership function below is monkey-patched to consume it directly.
    print(f"Point-in-time universe: current={len(current)}, unique historical={len(all_symbols)}")
    return current, hist, all_symbols


def membership_by_date_snapshot(dates: pd.DatetimeIndex, current_set, history: pd.DataFrame):
    hist = history.sort_values("date").reset_index(drop=True)
    hist_dates = hist["date"].to_numpy(dtype="datetime64[ns]")
    out = {}
    for d in dates:
        nd = pd.Timestamp(d).normalize()
        pos = hist_dates.searchsorted(nd.to_datetime64(), side="right") - 1
        if pos >= 0:
            out[nd] = set(hist.iloc[int(pos)]["members"])
        else:
            out[nd] = set()
    return out


def main():
    base.get_sp500_point_in_time_data = get_sp500_point_in_time_data_snapshot
    base.membership_by_date = membership_by_date_snapshot
    base.main()


if __name__ == "__main__":
    main()
