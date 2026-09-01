from __future__ import annotations

import io
import pandas as pd
import requests

import us_backtest as base


def get_sp500_point_in_time_data_fixed() -> tuple[set[str], list[dict], set[str]]:
    r = requests.get(base.WIKI, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))

    current = None
    changes = None
    for table in tables:
        flat_cols = [" ".join(str(x) for x in c if str(x) != "nan").strip() if isinstance(c, tuple) else str(c) for c in table.columns]
        joined = " | ".join(flat_cols).lower()
        if current is None and "symbol" in joined and "security" in joined and len(table) > 400:
            current = table.copy()
        if changes is None and "date" in joined and "added" in joined and "removed" in joined:
            changes = table.copy()

    if current is None or changes is None:
        raise RuntimeError("Unable to parse S&P 500 constituent/change tables")

    symbol_col = next(c for c in current.columns if "symbol" in str(c).lower())
    current_set = {base.norm_symbol(v) for v in current[symbol_col].tolist()}
    current_set.discard(None)

    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = [" ".join(str(x) for x in c if str(x) != "nan").strip() for c in changes.columns]
    else:
        changes.columns = [str(c) for c in changes.columns]

    date_col = next((c for c in changes.columns if "date" in c.lower()), None)
    added_col = next((c for c in changes.columns if "added" in c.lower() and "ticker" in c.lower()), None)
    removed_col = next((c for c in changes.columns if "removed" in c.lower() and "ticker" in c.lower()), None)
    if not date_col or not added_col or not removed_col:
        raise RuntimeError(f"Unexpected S&P changes columns: {changes.columns.tolist()}")

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
    if len(events) < 50:
        raise RuntimeError(f"Too few S&P history events parsed: {len(events)}")

    print(f"Parsed S&P history: {len(current_set)} current constituents, {len(events)} change rows, {len(all_symbols)} unique symbols")
    return current_set, events, all_symbols


if __name__ == "__main__":
    base.get_sp500_point_in_time_data = get_sp500_point_in_time_data_fixed
    base.main()
