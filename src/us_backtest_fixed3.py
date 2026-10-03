from __future__ import annotations

import io
import pandas as pd
import requests

import us_backtest as base


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [" ".join(str(v) for v in c if str(v) != "nan").strip() for c in x.columns]
    else:
        x.columns = [str(c) for c in x.columns]
    return x


def get_sp500_point_in_time_data_fixed3() -> tuple[set[str], list[dict], set[str]]:
    r = requests.get(base.WIKI, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    raw_tables = pd.read_html(io.StringIO(r.text))
    tables = [flatten(t) for t in raw_tables]

    current = next((t for t in tables if len(t) > 490 and any("symbol" in c.lower() for c in t.columns)), None)
    if current is None:
        raise RuntimeError("Unable to identify current S&P constituents table")

    changes = None
    best_valid_dates = 0
    for t in tables:
        if t.shape[1] < 5 or len(t) < 20:
            continue
        first = t.iloc[:, 0]
        parsed = pd.to_datetime(first, errors="coerce")
        valid = int(parsed.notna().sum())
        cols = " | ".join(t.columns).lower()
        # Prefer an explicit Added/Removed table; otherwise require many date rows.
        if valid > best_valid_dates and (valid >= 20 or ("added" in cols and "removed" in cols)):
            changes = t
            best_valid_dates = valid

    if changes is None:
        debug = [{"shape": t.shape, "cols": t.columns.tolist()} for t in tables]
        raise RuntimeError(f"Unable to identify changes table. Tables={debug}")

    print("Selected changes table shape:", changes.shape)
    print("Selected changes columns:", changes.columns.tolist())

    symbol_col = next(c for c in current.columns if "symbol" in c.lower())
    current_set = {base.norm_symbol(v) for v in current[symbol_col].tolist()}
    current_set.discard(None)

    cols = changes.columns.tolist()
    date_col = next((c for c in cols if "date" in c.lower()), cols[0])
    added_col = next((c for c in cols if "added" in c.lower() and "ticker" in c.lower()), cols[1] if len(cols) > 1 else None)
    removed_col = next((c for c in cols if "removed" in c.lower() and "ticker" in c.lower()), cols[3] if len(cols) > 3 else None)
    if added_col is None or removed_col is None:
        raise RuntimeError(f"Changes table has insufficient columns: {cols}")

    events = []
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
        raise RuntimeError(f"Bad S&P parse: current={len(current_set)}, events={len(events)}, shape={changes.shape}, cols={cols}")

    print(f"Parsed S&P history: {len(current_set)} current, {len(events)} changes, {len(all_symbols)} unique symbols")
    return current_set, events, all_symbols


if __name__ == "__main__":
    base.get_sp500_point_in_time_data = get_sp500_point_in_time_data_fixed3
    base.main()
