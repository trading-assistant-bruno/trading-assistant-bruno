from __future__ import annotations

import io
import pandas as pd
import requests

import us_backtest as base

REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/List_of_S%26P_500_companies"


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [" ".join(str(v) for v in c if str(v) != "nan").strip() for c in x.columns]
    else:
        x.columns = [str(c) for c in x.columns]
    return x


def get_sp500_point_in_time_data_fixed5() -> tuple[set[str], list[dict], set[str]]:
    r = requests.get(REST_HTML, headers={"User-Agent": "trading-assistant-bruno/1.0"}, timeout=30)
    r.raise_for_status()
    html = r.text
    if not html or "S&P 500 component stocks" not in html:
        raise RuntimeError(f"Wikipedia REST HTML invalid; status={r.status_code}, bytes={len(html)}")

    tables = [flatten(t) for t in pd.read_html(io.StringIO(html))]
    print("Rendered tables:", [(t.shape, t.columns.tolist()) for t in tables])

    current = next((t for t in tables if len(t) > 490 and any("symbol" in c.lower() for c in t.columns)), None)
    if current is None:
        raise RuntimeError("Unable to identify current S&P constituents table")

    changes = None
    for t in tables:
        cols_text = " | ".join(t.columns).lower()
        if len(t) >= 50 and t.shape[1] >= 5 and "added" in cols_text and "removed" in cols_text:
            changes = t
            break

    if changes is None:
        best_valid = 0
        for t in tables:
            if len(t) < 50 or t.shape[1] < 5:
                continue
            parsed = pd.to_datetime(t.iloc[:, 0], errors="coerce")
            valid = int(parsed.notna().sum())
            if valid > best_valid:
                best_valid = valid
                changes = t
        if best_valid < 50:
            changes = None

    if changes is None:
        raise RuntimeError("Rendered HTML did not expose a usable S&P changes table")

    print("Selected changes table:", changes.shape, changes.columns.tolist())
    symbol_col = next(c for c in current.columns if "symbol" in c.lower())
    current_set = {base.norm_symbol(v) for v in current[symbol_col].tolist()}
    current_set.discard(None)

    cols = changes.columns.tolist()
    date_col = next((c for c in cols if "date" in c.lower()), cols[0])
    added_col = next((c for c in cols if "added" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())), None)
    removed_col = next((c for c in cols if "removed" in c.lower() and ("ticker" in c.lower() or "symbol" in c.lower())), None)
    if added_col is None and len(cols) >= 2:
        added_col = cols[1]
    if removed_col is None and len(cols) >= 4:
        removed_col = cols[3]

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
        raise RuntimeError(f"Bad S&P REST parse: current={len(current_set)}, events={len(events)}")

    print(f"Parsed S&P history: {len(current_set)} current, {len(events)} changes, {len(all_symbols)} unique symbols")
    return current_set, events, all_symbols


if __name__ == "__main__":
    base.get_sp500_point_in_time_data = get_sp500_point_in_time_data_fixed5
    base.main()
