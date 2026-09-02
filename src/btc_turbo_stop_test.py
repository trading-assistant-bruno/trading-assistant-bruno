from __future__ import annotations

import io
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "btc_turbo_stop_test"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-01-07")
END = pd.Timestamp("2026-08-02")  # last complete monthly point-in-time interval available
BTC = "BTC"
COST = 0.0023  # 23 bps per side, deliberately conservative
UNIVERSE_N = 20
TURBO_WEIGHT = 0.30
REL_THRESHOLD = 0.05
FIXED_ATR_MULT = 2.5
TRAIL_ATR_MULT = 3.0
ATR_N = 14

EXCLUDE = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "PAX", "GUSD", "USDD",
    "FDUSD", "USDE", "FRAX", "PYUSD", "UST", "USTC", "EURT", "EURC", "SUSD",
    "LUSD", "USDS", "WBTC", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "WEETH"
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/128 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
# Historical ticker -> Yahoo ticker overrides where the current Yahoo symbol differs.
YF_OVERRIDES = {
    "MIOTA": "IOTA-USD",
    "IOTA": "IOTA-USD",
    "BCH": "BCH-USD",
    "XRB": "XNO-USD",
    "NANO": "XNO-USD",
}


def first_sundays():
    out = []
    cur = pd.Timestamp(START.year, START.month, 1)
    while cur <= END:
        d = cur + pd.Timedelta(days=(6 - cur.weekday()) % 7)
        if START <= d <= END:
            out.append(d)
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return out


def num(x):
    s = str(x).replace("$", "").replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan


def fetch_cmc(dt):
    url = f"https://coinmarketcap.com/historical/{dt:%Y%m%d}/"
    err = ""
    for a in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            table = None
            for t in pd.read_html(io.StringIO(r.text)):
                cols = [str(c).strip() for c in t.columns]
                if "Rank" in cols and "Symbol" in cols and "Price" in cols:
                    table = t.copy()
                    break
            if table is None:
                raise RuntimeError("CMC ranking table missing")
            table.columns = [str(c).strip() for c in table.columns]
            z = table[["Rank", "Name", "Symbol", "Price"]].copy()
            z["Rank"] = pd.to_numeric(z["Rank"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            z["Symbol"] = z["Symbol"].astype(str).str.upper().str.strip()
            z["Price"] = z["Price"].map(num)
            z = z.dropna(subset=["Rank", "Symbol", "Price"])
            z = z[(z.Rank >= 1) & (z.Rank <= UNIVERSE_N)].drop_duplicates("Symbol").sort_values("Rank")
            z["date"] = dt
            if len(z) < 10:
                raise RuntimeError(f"only {len(z)} rows")
            return z, None
        except Exception as e:
            err = str(e)
            time.sleep(0.8 * (a + 1))
    return None, err


def collect_snapshots():
    rows, fails = [], []
    for d in first_sundays():
        z, e = fetch_cmc(d)
        if z is None:
            fails.append({"date": d, "error": e})
            print("FAIL CMC", d.date(), e)
        else:
            rows.append(z)
            print("OK CMC", d.date(), len(z))
        time.sleep(0.08)
    if not rows:
        raise RuntimeError("no CMC snapshots")
    s = pd.concat(rows, ignore_index=True).sort_values(["date", "Rank"])
    s.to_csv(OUT / "snapshots.csv", index=False)
    pd.DataFrame(fails, columns=["date", "error"]).to_csv(OUT / "snapshot_failures.csv", index=False)
    return s


def build_panels(s):
    p = s.pivot_table(index="date", columns="Symbol", values="Price", aggfunc="first").sort_index()
    r = s.pivot_table(index="date", columns="Symbol", values="Rank", aggfunc="first").sort_index()
    return p, r


def select(i, p, ranks, top_n):
    d = p.index[i]
    if i < 2:
        return []
    if BTC not in p.columns:
        return []
    b0 = p.at[d, BTC]
    b1 = p.at[p.index[i - 1], BTC]
    b2 = p.at[p.index[i - 2], BTC]
    if any(pd.isna(x) or x <= 0 for x in (b0, b1, b2)):
        return []
    bm1 = b0 / b1 - 1
    bm2 = b0 / b2 - 1
    if bm1 <= 0 or bm2 <= 0:
        return []
    cand = []
    for sym, rank in ranks.loc[d].dropna().items():
        if sym == BTC or sym in EXCLUDE or rank > UNIVERSE_N or sym not in p.columns:
            continue
        a0, a1, a2 = p.at[d, sym], p.at[p.index[i - 1], sym], p.at[p.index[i - 2], sym]
        if any(pd.isna(x) or x <= 0 for x in (a0, a1, a2)):
            continue
        m1 = a0 / a1 - 1
        m2 = a0 / a2 - 1
        if m1 <= 0 or m2 <= 0 or (m1 - bm1) < REL_THRESHOLD:
            continue
        score = 0.60 * m1 + 0.40 * m2
        cand.append((score, -rank, sym, m1, m2, bm1))
    cand.sort(reverse=True)
    return cand[:top_n]


def yf_symbol(sym):
    return YF_OVERRIDES.get(sym, f"{sym}-USD")


def download_daily(sym):
    ticker = yf_symbol(sym)
    start = (START - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end = (END + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, actions=False,
                         progress=False, threads=False, timeout=30)
        if df is None or df.empty:
            return None, f"empty {ticker}"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        need = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in need):
            return None, f"missing OHLC {ticker}: {list(df.columns)}"
        z = df[need].copy().dropna(how="any")
        idx = pd.to_datetime(z.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        z.index = idx.normalize()
        z = z[~z.index.duplicated(keep="last")].sort_index()
        if len(z) < 30:
            return None, f"too few bars {ticker}: {len(z)}"
        return z, None
    except Exception as e:
        return None, str(e)


def atr(df, n=ATR_N):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def first_bar_on_or_after(df, d):
    x = df.index[df.index >= pd.Timestamp(d)]
    return x[0] if len(x) else None


def sleeve_path(sym, start_signal, end_signal, daily, mode):
    """Return sleeve gross return and whether it ends as BTC.

    Signal is known at the Sunday CMC snapshot. Execution occurs at the first
    daily open on/after Monday. Exit/rebalance occurs at the first daily open
    on/after the Monday following the next signal date.
    """
    if sym not in daily or daily[sym] is None or BTC not in daily or daily[BTC] is None:
        return None
    a = daily[sym]
    b = daily[BTC]
    entry_day = first_bar_on_or_after(a, start_signal + pd.Timedelta(days=1))
    end_day = first_bar_on_or_after(a, end_signal + pd.Timedelta(days=1))
    if entry_day is None or end_day is None or end_day <= entry_day:
        return None
    entry = float(a.at[entry_day, "Open"])
    end_open = float(a.at[end_day, "Open"])
    if entry <= 0 or end_open <= 0:
        return None
    if sym == BTC or mode == "nostop":
        return {"ret": end_open / entry - 1, "ended_asset": sym, "stopped": False,
                "stop_day": None, "stop_price": None, "entry_day": entry_day, "end_day": end_day}

    aa = a.copy()
    aa["ATR"] = atr(aa)
    pre = aa.loc[:start_signal, "ATR"].dropna()
    if pre.empty:
        return None
    atr0 = float(pre.iloc[-1])
    if not np.isfinite(atr0) or atr0 <= 0:
        return None

    if mode == "fixed":
        stop = entry - FIXED_ATR_MULT * atr0
    elif mode == "trailing":
        stop = entry - TRAIL_ATR_MULT * atr0
    else:
        raise ValueError(mode)

    highest = entry
    bars = aa[(aa.index >= entry_day) & (aa.index < end_day)]
    stop_day = None
    stop_px = None
    for day, row in bars.iterrows():
        # Today's stop uses only information available before today's low is observed.
        if day > entry_day and mode == "trailing":
            prev_slice = aa[(aa.index >= entry_day) & (aa.index < day)]
            if not prev_slice.empty:
                highest = max(highest, float(prev_slice["High"].max()))
                atr_prev = aa.loc[: day - pd.Timedelta(days=1), "ATR"].dropna()
                if not atr_prev.empty:
                    stop = max(stop, highest - TRAIL_ATR_MULT * float(atr_prev.iloc[-1]))
        if float(row["Low"]) <= stop:
            op = float(row["Open"])
            stop_px = min(op, stop) if op < stop else stop
            stop_day = day
            break

    if stop_day is None:
        return {"ret": end_open / entry - 1, "ended_asset": sym, "stopped": False,
                "stop_day": None, "stop_price": None, "entry_day": entry_day, "end_day": end_day}

    # After an alt stop, proceeds rotate back to BTC at the next BTC daily open.
    # Two extra one-way costs are charged: sell alt + buy BTC.
    btc_re = first_bar_on_or_after(b, stop_day + pd.Timedelta(days=1))
    btc_end = first_bar_on_or_after(b, end_signal + pd.Timedelta(days=1))
    alt_factor = stop_px / entry
    switch_factor = (1 - COST) ** 2
    if btc_re is None or btc_end is None or btc_end <= btc_re:
        factor = alt_factor * switch_factor
    else:
        br = float(b.at[btc_re, "Open"])
        be = float(b.at[btc_end, "Open"])
        factor = alt_factor * switch_factor * (be / br if br > 0 else 1.0)
    return {"ret": factor - 1, "ended_asset": BTC, "stopped": True,
            "stop_day": stop_day, "stop_price": stop_px, "entry_day": entry_day, "end_day": end_day}


def simulate(p, ranks, daily, top_n, mode):
    eq = 1.0
    curves = {p.index[0]: 1.0}
    prev_w = {BTC: 1.0}
    logs = []
    missing = []

    for i in range(len(p.index) - 1):
        d, nxt = p.index[i], p.index[i + 1]
        picks = select(i, p, ranks, top_n)
        chosen = [x[2] for x in picks]
        target = {BTC: 1.0 - TURBO_WEIGHT if chosen else 1.0}
        if chosen:
            each = TURBO_WEIGHT / len(chosen)
            for s in chosen:
                target[s] = each

        risky = set(prev_w) | set(target)
        turnover = sum(abs(target.get(s, 0.0) - prev_w.get(s, 0.0)) for s in risky)
        eq *= max(0.0, 1 - turnover * COST)

        end_amounts = {}
        stopped_syms = []
        period_gross = 0.0
        valid = True
        for s, w in target.items():
            path = sleeve_path(s, d, nxt, daily, mode if s != BTC else "nostop")
            if path is None:
                valid = False
                missing.append({"date": d, "next": nxt, "symbol": s, "mode": mode})
                break
            val = w * (1 + path["ret"])
            period_gross += val
            end_asset = path["ended_asset"]
            end_amounts[end_asset] = end_amounts.get(end_asset, 0.0) + val
            if path["stopped"]:
                stopped_syms.append(s)
        if not valid:
            # To avoid optimistic imputation, fall back to the original CMC snapshot-to-snapshot
            # return for the affected month; this month cannot benefit from a stop.
            period_gross = 0.0
            end_amounts = {}
            for s, w in target.items():
                if s in p.columns and pd.notna(p.at[d, s]) and pd.notna(p.at[nxt, s]) and p.at[d, s] > 0:
                    rr = p.at[nxt, s] / p.at[d, s] - 1
                else:
                    rr = -1.0
                val = w * (1 + rr)
                period_gross += val
                end_amounts[s] = end_amounts.get(s, 0.0) + val
            stopped_syms = []

        eq *= max(0.0, period_gross)
        curves[nxt] = eq
        den = sum(end_amounts.values())
        prev_w = {s: v / den for s, v in end_amounts.items() if v > 0} if den > 0 else {BTC: 1.0}
        logs.append({
            "strategy": f"Top{top_n}_T30_{mode}", "date": d, "next": nxt,
            "selected": ";".join(chosen), "stopped": ";".join(stopped_syms),
            "turnover": turnover, "equity": eq, "daily_path_complete": valid,
        })

    return pd.Series(curves, name=f"Top{top_n}_T30_{mode}"), pd.DataFrame(logs), pd.DataFrame(missing)


def metrics(name, eq, log):
    x = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = eq / eq.cummax() - 1
    vol = x.std(ddof=0) * math.sqrt(12)
    sharpe = x.mean() * 12 / vol if vol > 0 else np.nan
    ann = {y: g.iloc[-1] / g.iloc[0] - 1 for y, g in eq.groupby(eq.index.year)}
    ann = pd.Series(ann, dtype=float)
    return {
        "strategy": name,
        "cagr_pct": 100 * cagr,
        "max_monthly_drawdown_pct": 100 * dd.min(),
        "sharpe_0rf": sharpe,
        "volatility_pct": 100 * vol,
        "final_multiple": float(eq.iloc[-1]),
        "best_year_pct": 100 * ann.max(),
        "worst_year_pct": 100 * ann.min(),
        "best_month_pct": 100 * x.max(),
        "worst_month_pct": 100 * x.min(),
        "stops_count": int(log["stopped"].astype(bool).sum()),
        "months_with_complete_daily_path_pct": 100 * log["daily_path_complete"].mean(),
        "annual_turnover_x": log["turnover"].sum() / yrs,
    }, ann * 100


def main():
    snapshots = collect_snapshots()
    p, ranks = build_panels(snapshots)
    if BTC not in p.columns:
        raise RuntimeError("BTC missing from snapshots")

    # Determine every symbol that can actually be selected in either Top1 or Top2 configuration.
    needed = {BTC}
    selection_rows = []
    for i in range(len(p.index)):
        for n in (1, 2):
            picks = select(i, p, ranks, n)
            syms = [x[2] for x in picks]
            needed.update(syms)
            selection_rows.append({"date": p.index[i], "top_n": n, "selected": ";".join(syms)})
    pd.DataFrame(selection_rows).to_csv(OUT / "signals.csv", index=False)
    print("Daily symbols needed:", sorted(needed))

    daily = {}
    coverage = []
    for s in sorted(needed):
        z, err = download_daily(s)
        daily[s] = z
        coverage.append({"symbol": s, "yahoo_ticker": yf_symbol(s), "bars": 0 if z is None else len(z), "error": err or ""})
        print("DAILY", s, "OK" if z is not None else "FAIL", 0 if z is None else len(z), err or "")
        time.sleep(0.15)
    pd.DataFrame(coverage).to_csv(OUT / "daily_coverage.csv", index=False)

    curves, logs, misses, rows, annuals = [], [], [], [], {}
    for top_n in (1, 2):
        for mode in ("nostop", "fixed", "trailing"):
            eq, lg, ms = simulate(p, ranks, daily, top_n, mode)
            m, a = metrics(eq.name, eq, lg)
            rows.append(m)
            annuals[eq.name] = a
            curves.append(eq)
            logs.append(lg)
            if not ms.empty:
                misses.append(ms)

    # BTC benchmark in the same daily-open execution engine.
    btc_eq = 1.0
    btc_curve = {p.index[0]: 1.0}
    for i in range(len(p.index) - 1):
        path = sleeve_path(BTC, p.index[i], p.index[i + 1], daily, "nostop")
        if path is None:
            rr = p.at[p.index[i + 1], BTC] / p.at[p.index[i], BTC] - 1
        else:
            rr = path["ret"]
        btc_eq *= 1 + rr
        btc_curve[p.index[i + 1]] = btc_eq
    btc_eqs = pd.Series(btc_curve, name="BTC_HOLD_daily_engine")
    dummy = pd.DataFrame({"stopped": [""] * (len(p.index) - 1), "daily_path_complete": [True] * (len(p.index) - 1), "turnover": [0.0] * (len(p.index) - 1)})
    bm, ba = metrics(btc_eqs.name, btc_eqs, dummy)
    rows.append(bm); annuals[btc_eqs.name] = ba; curves.append(btc_eqs)

    results = pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)
    equity = pd.concat(curves, axis=1)
    annual = pd.DataFrame(annuals)
    all_logs = pd.concat(logs, ignore_index=True)
    all_missing = pd.concat(misses, ignore_index=True) if misses else pd.DataFrame(columns=["date", "next", "symbol", "mode"])

    results.to_csv(OUT / "results.csv", index=False)
    equity.to_csv(OUT / "equity.csv")
    annual.to_csv(OUT / "annual.csv")
    all_logs.to_csv(OUT / "trade_log.csv", index=False)
    all_missing.to_csv(OUT / "missing_daily_paths.csv", index=False)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": "CoinMarketCap historical first-Sunday snapshots, point-in-time Top20",
        "pnl_source": "Yahoo Finance daily OHLC for selected assets; CMC monthly fallback only when daily path unavailable",
        "turbo_weight": TURBO_WEIGHT,
        "relative_momentum_threshold": REL_THRESHOLD,
        "signal": "BTC 1m>0 & 2m>0; alt 1m>0 & 2m>0 & alt 1m - BTC 1m >=5pp; score .6*1m+.4*2m",
        "fixed_stop": f"{FIXED_ATR_MULT} x ATR{ATR_N} from entry",
        "trailing_stop": f"highest prior high - {TRAIL_ATR_MULT} x prior ATR{ATR_N}, never loosened",
        "stop_execution": "if daily low breaches stop, fill at stop unless daily open is lower; after alt stop rotate to BTC at next daily open",
        "cost_per_side": COST,
        "no_leverage": True,
        "notes": [
            "Stops are tested only on the alt Turbo sleeve; BTC core is never stopped.",
            "Monthly selection rules are unchanged; no parameter optimization grid was run.",
            "Max drawdown is measured from monthly equity points, so intramonth drawdown can be worse.",
            "Yahoo symbol/history gaps are reported and affected months fall back to the original CMC monthly return without stop benefit.",
            "This is a research backtest, not evidence of guaranteed future outperformance."
        ],
        "results": results.replace({np.nan: None}).to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(meta, indent=2))
    print("\nRESULTS\n", results.to_string(index=False))
    print("\nDAILY COVERAGE\n", pd.DataFrame(coverage).to_string(index=False))
    print("\nMISSING DAILY PATHS", len(all_missing))


if __name__ == "__main__":
    main()
