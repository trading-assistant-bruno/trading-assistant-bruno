from __future__ import annotations

import io
import pandas as pd
import requests

import us_backtest as base

MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [" ".join(str(v) for v in c if str(v) != "nan").strip() for c in x.columns]
    else:
        x.columns = [str(c) for c in x.columns]
    return x


def get_sp500_point_in_time_data_fixed4() -> tuple[set[str], list[dict], set[str]]:
    params = {
        "action": "parse",
        "page": "List of S&P 500 companies",
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }
    r = requests.get(MEDIAWIKI_API, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    html = payload.get("parse", {}).get("text")
    if isinstance(html, dict):
        html = html.get("*")
    if not html:
        raise RuntimeError(f"MediaWiki parse API returned no rendered HTML: {payload}")

    tables = [flatten(t) for t in pd.read_html(io.StringIO(html))]
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
        best = (0, None)
        for t in tables:
            if t.shape[1] < 5 or len(t) < 50:
                continue
            parsed = pd.to_datetime(t.iloc[:, 0], errors="coerce")
            valid = int(parsed.notna().sum())
            if valid > best[0]:
                best = (valid, t)
        if best[0] >= 50:
            changes = best[1]

    if changes is None:
        debug = [{"shape": t.shape, "cols": t.columns.tolist()} for t in tables]
        raise RuntimeError(f"Unable to identify rendered changes table. Tables={debug}")

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
        raise RuntimeError(f"Bad rendered S&P parse: current={len(current_set)}, events={len(events)}")

    print(f"Parsed S&P history: {len(current_set)} current, {len(events)} changes, {len(all_symbols)} unique symbols")
    return current_set, events, all_symbols


if __name__ == "__main__":
    base.get_sp500_point_in_time_data = get_sp500_point_in_time_data_fixed4
    base.main()
