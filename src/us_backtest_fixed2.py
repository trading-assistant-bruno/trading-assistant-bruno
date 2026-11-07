from __future__ import annotations

import io
import pandas as pd
import requests

import us_backtest as base


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [" ".join(str(v) for v in c if str(v) != "nan").strip() for c in x.columns]
    else:
        x.columns = [str(c) for c in x.columns]
    return x


def get_sp500_point_in_time_data_fixed2() -> tuple[set[str], list[dict], set[str]]:
    r = requests.get(base.WIKI, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if len(tables) < 2:
        raise RuntimeError(f"Wikipedia returned only {len(tables)} tables")

    # On the canonical page the first table is the constituents list and
    # the second table is the selected index changes history.
    current = _flatten_columns(tables[0])
    changes = _flatten_columns(tables[1])
    print("Current columns:", current.columns.tolist())
    print("Changes columns:", changes.columns.tolist())

    symbol_col = next((c for c in current.columns if "symbol" in c.lower()), None)
    if not symbol_col:
        raise RuntimeError(f"No Symbol column in current table: {current.columns.tolist()}")
    current_set = {base.norm_symbol(v) for v in current[symbol_col].tolist()}
    current_set.discard(None)

    date_col = next((c for c in changes.columns if "date" in c.lower()), None)
    added_col = next((c for c in changes.columns if "added" in c.lower() and "ticker" in c.lower()), None)
    removed_col = next((c for c in changes.columns if "removed" in c.lower() and "ticker" in c.lower()), None)

    # Fallback to positional columns matching the published table:
    # Effective Date | Added Ticker | Added Security | Removed Ticker | Removed Security | Reason
    if date_col is None and len(changes.columns) >= 1:
        date_col = changes.columns[0]
    if added_col is None and len(changes.columns) >= 2:
        added_col = changes.columns[1]
    if removed_col is None and len(changes.columns) >= 4:
        removed_col = changes.columns[3]

    events: list[dict] = []
    all_symbols = set(current_set)
    for _, row in changes.iterrows():
        date = pd.to_datetime(row.get(date_col), errors="coerce")
        if pd.isna(date):
            continue
        added = base.norm_symbol(row.get(added_col))
        removed = base.norm_symbol(row.get(removed_col))
        events.append({"date": pd.Timestamp(date).normalize(), "added": added, "removed": removed})
        if added:
            all_symbols.add(added)
        if removed:
            all_symbols.add(removed)

    events.sort(key=lambda x: x["date"], reverse=True)
    if len(current_set) < 490 or len(events) < 50:
        raise RuntimeError(f"Bad S&P parse: current={len(current_set)}, events={len(events)}, columns={changes.columns.tolist()}")

    print(f"Parsed S&P history: {len(current_set)} current constituents, {len(events)} change rows, {len(all_symbols)} unique symbols")
    return current_set, events, all_symbols


if __name__ == "__main__":
    base.get_sp500_point_in_time_data = get_sp500_point_in_time_data_fixed2
    base.main()
