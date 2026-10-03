from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "us_backtest_config.yml").read_text(encoding="utf-8"))
DATA_DIR = ROOT / "data" / "us_backtest"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BENCHMARKS = ["SPY", "QQQ", "^IXIC", "^VIX"]


def norm_symbol(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper().replace(".", "-")
    return s or None


def get_sp500_point_in_time_data() -> tuple[set[str], list[dict], set[str]]:
    r = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))

    current = None
    changes = None
    for table in tables:
        flat_cols = [" ".join(c).strip() if isinstance(c, tuple) else str(c) for c in table.columns]
        joined = " | ".join(flat_cols).lower()
        if current is None and "symbol" in joined and "security" in joined and len(table) > 400:
            current = table.copy()
        if changes is None and "date" in joined and "added" in joined and "removed" in joined:
            changes = table.copy()

    if current is None:
        raise RuntimeError("Unable to parse current S&P 500 constituents")

    # Current table has a simple Symbol column on Wikipedia.
    symbol_col = next(c for c in current.columns if "symbol" in str(c).lower())
    current_set = {norm_symbol(v) for v in current[symbol_col].tolist()}
    current_set.discard(None)

    events: list[dict] = []
    all_symbols = set(current_set)

    if changes is not None:
        # Flatten MultiIndex headings such as ('Added', 'Ticker').
        if isinstance(changes.columns, pd.MultiIndex):
            changes.columns = [" ".join(str(x) for x in c if str(x) != "nan").strip() for c in changes.columns]
        else:
            changes.columns = [str(c) for c in changes.columns]

        date_col = next((c for c in changes.columns if c.lower().startswith("date")), None)
        added_col = next((c for c in changes.columns if "added" in c.lower() and "ticker" in c.lower()), None)
        removed_col = next((c for c in changes.columns if "removed" in c.lower() and "ticker" in c.lower()), None)

        if date_col and (added_col or removed_col):
            for _, row in changes.iterrows():
                date = pd.to_datetime(row.get(date_col), errors="coerce")
                if pd.isna(date):
                    continue
                added = norm_symbol(row.get(added_col)) if added_col else None
                removed = norm_symbol(row.get(removed_col)) if removed_col else None
                events.append({"date": pd.Timestamp(date).normalize(), "added": added, "removed": removed})
                if added:
                    all_symbols.add(added)
                if removed:
                    all_symbols.add(removed)

    events.sort(key=lambda x: x["date"], reverse=True)
    return current_set, events, all_symbols


def membership_by_date(dates: pd.DatetimeIndex, current_set: set[str], events: list[dict]) -> dict[pd.Timestamp, set[str]]:
    members = set(current_set)
    out: dict[pd.Timestamp, set[str]] = {}
    i = 0
    ev = events
    for date in sorted([pd.Timestamp(d).normalize() for d in dates], reverse=True):
        while i < len(ev) and ev[i]["date"] > date:
            added, removed = ev[i]["added"], ev[i]["removed"]
            # Reverse the index change to reconstruct the prior membership.
            if added:
                members.discard(added)
            if removed:
                members.add(removed)
            i += 1
        out[date] = set(members)
    return out


def extract_batch(data: pd.DataFrame, batch: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    if data is None or data.empty:
        return out
    if len(batch) == 1:
        frame = data.dropna(how="all")
        if not frame.empty:
            out[batch[0]] = frame
        return out
    for ticker in batch:
        try:
            frame = data[ticker].dropna(how="all")
            if not frame.empty:
                out[ticker] = frame
        except Exception:
            pass
    return out


def download_prices(symbols: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    output = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        print(f"Downloading batch {i//100 + 1}: {len(batch)} symbols")
        kwargs = dict(
            tickers=batch,
            start=start,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=30,
        )
        if end:
            kwargs["end"] = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            data = yf.download(**kwargs)
            output.update(extract_batch(data, batch))
        except Exception as exc:
            print(f"Batch failed: {exc}")
    return output


def normalized_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        for level in range(x.columns.nlevels):
            vals = [str(v).title() for v in x.columns.get_level_values(level)]
            if "Close" in vals and "High" in vals:
                x.columns = vals
                break
    else:
        x.columns = [str(c).title() for c in x.columns]
    x = x.loc[:, ~x.columns.duplicated()]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in x.columns for c in needed):
        return pd.DataFrame()
    x = x[needed].copy()
    x.index = pd.to_datetime(x.index).tz_localize(None) if getattr(pd.to_datetime(x.index), "tz", None) else pd.to_datetime(x.index)
    return x.sort_index()


def matrices(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = {s: normalized_ohlcv(d) for s, d in frames.items()}
    clean = {s: d for s, d in clean.items() if not d.empty}
    close = pd.concat({s: d["Close"] for s, d in clean.items()}, axis=1).sort_index()
    open_ = pd.concat({s: d["Open"] for s, d in clean.items()}, axis=1).reindex(close.index)
    high = pd.concat({s: d["High"] for s, d in clean.items()}, axis=1).reindex(close.index)
    low = pd.concat({s: d["Low"] for s, d in clean.items()}, axis=1).reindex(close.index)
    volume = pd.concat({s: d["Volume"] for s, d in clean.items()}, axis=1).reindex(close.index)
    return open_, high, low, close, volume


def piecewise_quality(depth, contraction, dryup, steps, tightness):
    q = pd.DataFrame(0.0, index=depth.index, columns=depth.columns)
    q += pd.DataFrame(np.select([depth <= 10, depth <= 15, depth <= 20, depth <= 25], [25, 22, 16, 8], default=0), index=q.index, columns=q.columns)
    q += pd.DataFrame(np.select([contraction <= .60, contraction <= .80, contraction <= 1.00], [25, 20, 10], default=0), index=q.index, columns=q.columns)
    q += pd.DataFrame(np.select([dryup <= .70, dryup <= .85, dryup <= 1.00], [20, 15, 8], default=0), index=q.index, columns=q.columns)
    q += pd.DataFrame(np.select([steps >= 2, steps >= 1], [20, 10], default=0), index=q.index, columns=q.columns)
    q += pd.DataFrame(np.select([tightness <= 3, tightness <= 5, tightness <= 7], [10, 7, 4], default=0), index=q.index, columns=q.columns)
    return q.clip(upper=100)


def indicators(open_, high, low, close, volume):
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    high52 = high.rolling(252).max()
    low52 = low.rolling(252).min()
    adv20 = (close * volume).rolling(20).mean()
    perf3 = close / close.shift(63) - 1
    perf6 = close / close.shift(126) - 1
    perf12 = close / close.shift(252) - 1
    momscore = perf3 * .40 + perf6 * .30 + perf12 * .30

    base_high = high.rolling(50).max()
    base_low = low.rolling(50).min()
    depth = (base_high - base_low) / base_high * 100
    tight = (high.rolling(5).max() - low.rolling(5).min()) / close * 100
    day_range = high - low
    recent_range = day_range.rolling(10).mean() / close
    previous_range = day_range.shift(10).rolling(20).mean() / close
    contraction = recent_range / previous_range
    dryup = volume.shift(1).rolling(10).mean() / volume.shift(11).rolling(30).mean()

    block = (high.rolling(10).max() - low.rolling(10).min()) / close.rolling(10).mean() * 100
    old, mid, recent = block.shift(20), block.shift(10), block
    steps = (mid < old).astype(int) + (recent < mid).astype(int)
    base_quality = piecewise_quality(depth, contraction, dryup, steps, tight)

    pivot = high.shift(5).rolling(55).max()
    swing_low = low.rolling(10).min() * .995
    vol_ratio = volume / volume.shift(1).rolling(20).mean()

    return dict(sma50=sma50, sma150=sma150, sma200=sma200, high52=high52, low52=low52,
                adv20=adv20, perf3=perf3, perf6=perf6, perf12=perf12, momscore=momscore,
                depth=depth, tight=tight, contraction=contraction, dryup=dryup, steps=steps,
                base_quality=base_quality, pivot=pivot, swing_low=swing_low, vol_ratio=vol_ratio)


def market_regime(close_bench: pd.DataFrame) -> pd.Series:
    slope_days = int(CONFIG["strategy"]["sma50_slope_days"])
    positives = pd.Series(0, index=close_bench.index, dtype=float)
    for t in ["QQQ", "SPY", "^IXIC"]:
        c = close_bench[t]
        s50 = c.rolling(50).mean()
        s200 = c.rolling(200).mean()
        positives += (c > s50).astype(int)
        positives += (c > s200).astype(int)
        positives += (s50 > s50.shift(slope_days)).astype(int)
    positives += (close_bench["^VIX"] < float(CONFIG["market"]["vix_green_max"])).astype(int)
    out = pd.Series("ROUGE", index=positives.index)
    out[positives >= 6] = "ORANGE"
    out[positives >= 9] = "VERT"
    return out


def candidates_for_day(date, members, close, volume, ind, spy_perf6):
    s = CONFIG["strategy"]
    u = CONFIG["universe"]
    members = [m for m in members if m in close.columns and pd.notna(close.at[date, m])]
    if not members:
        return pd.DataFrame()

    ms = ind["momscore"].loc[date, members].dropna()
    if ms.empty:
        return pd.DataFrame()
    pct = ms.rank(pct=True, method="average")
    rs_rank = (1 + pct * 98).round().clip(1, 99)

    c = close.loc[date, rs_rank.index]
    sma50 = ind["sma50"].loc[date, rs_rank.index]
    sma150 = ind["sma150"].loc[date, rs_rank.index]
    sma200 = ind["sma200"].loc[date, rs_rank.index]
    sma200_old = ind["sma200"].shift(int(s["sma200_slope_days"])).loc[date, rs_rank.index]
    h52 = ind["high52"].loc[date, rs_rank.index]
    l52 = ind["low52"].loc[date, rs_rank.index]
    adv = ind["adv20"].loc[date, rs_rank.index]
    p6 = ind["perf6"].loc[date, rs_rank.index]
    p12 = ind["perf12"].loc[date, rs_rank.index]
    bq = ind["base_quality"].loc[date, rs_rank.index]
    depth = ind["depth"].loc[date, rs_rank.index]
    pivot = ind["pivot"].loc[date, rs_rank.index]
    swing = ind["swing_low"].loc[date, rs_rank.index]
    vr = ind["vol_ratio"].loc[date, rs_rank.index]

    distance_high = (h52 - c) / h52 * 100
    distance_low = (c / l52 - 1) * 100
    rs6 = ((1 + p6) / (1 + spy_perf6.loc[date]) - 1) * 100

    entry = pivot * (1 + float(s["entry_buffer_pct"]) / 100)
    buymax = pivot * (1 + float(s["max_distance_above_pivot_pct"]) / 100)
    below_pct = (pivot - c) / pivot * 100
    valid_status = ((c >= entry) & (c <= buymax)) | ((c <= pivot) & (below_pct >= 0) & (below_pct <= float(s["watchlist_below_pivot_pct"])))

    tech_stop_pct = (entry - swing) / entry * 100
    stop = swing.copy()
    min_stop = float(s["min_stop_distance_pct"])
    stop[tech_stop_pct < min_stop] = entry[tech_stop_pct < min_stop] * (1 - min_stop / 100)
    stop_pct = (entry - stop) / entry * 100

    mask = (
        (rs_rank >= int(s["min_rs_rank"])) &
        (c >= float(u["min_price"])) &
        (adv >= float(u["min_avg_dollar_volume"])) &
        (c > sma50) & (sma50 > sma150) & (sma150 > sma200) &
        (sma200 > sma200_old) &
        (distance_high <= float(s["max_distance_from_52w_high_pct"])) &
        (distance_low >= float(s["min_distance_above_52w_low_pct"])) &
        (p6 * 100 >= float(s["min_perf_6m_pct"])) &
        (rs6 >= float(s["min_rs_6m_vs_spy_pct"])) &
        (depth <= float(s["max_base_depth_pct"])) &
        (bq >= float(s["min_base_quality_score"])) &
        valid_status &
        (tech_stop_pct > 0) & (tech_stop_pct <= float(s["max_stop_distance_pct"]))
    )

    names = mask[mask].index
    if len(names) == 0:
        return pd.DataFrame()

    perf6_pct = p6.loc[names] * 100
    score = rs_rank.loc[names] * .35 + bq.loc[names] * .35
    score += np.minimum(15, np.maximum(0, perf6_pct / 6))
    score += np.maximum(0, 10 - distance_high.loc[names] * .4)
    score += np.where(vr.loc[names] >= 1.5, 5, np.where(vr.loc[names] >= 1.2, 3, 0))

    return pd.DataFrame({
        "score": score.clip(upper=100),
        "entry": entry.loc[names],
        "limit": entry.loc[names] * (1 + float(CONFIG["backtest"]["stop_limit_max_slippage_pct"]) / 100),
        "stop": stop.loc[names],
        "stop_pct": stop_pct.loc[names],
    }).sort_values("score", ascending=False)


@dataclass
class Position:
    ticker: str
    shares: int
    entry: float
    stop: float
    risk_per_share: float
    entry_date: str
    entry_fee: float
    exit_on_open: bool = False


def fee(value: float) -> float:
    pct = float(CONFIG["backtest"]["commission_pct_one_way"]) / 100
    return max(float(CONFIG["backtest"]["min_commission_usd"]), value * pct)


def metrics(equity: pd.Series, trades: pd.DataFrame, name: str) -> dict:
    r = equity.pct_change().fillna(0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1/365.25)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = equity / equity.cummax() - 1
    maxdd = dd.min()
    vol = r.std(ddof=0) * math.sqrt(252)
    sharpe = r.mean() * 252 / vol if vol > 0 else np.nan
    downside = r[r < 0].std(ddof=0) * math.sqrt(252)
    sortino = r.mean() * 252 / downside if downside > 0 else np.nan
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    wins = trades["pnl"] > 0 if not trades.empty else pd.Series(dtype=bool)
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum() if not trades.empty else 0
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum() if not trades.empty else 0
    return {
        "strategy": name,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": maxdd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "total_return_pct": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "trades": int(len(trades)),
        "win_rate_pct": float(wins.mean() * 100) if len(wins) else np.nan,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "final_equity": float(equity.iloc[-1]),
    }


def simulate(mode, dates, membership, open_, high, low, close, ind, regime):
    capital = float(CONFIG["capital_usd"])
    cash = capital
    risk_pct = float(CONFIG["risk_per_trade_pct"]) / 100
    positions: dict[str, Position] = {}
    pending: list[dict] = []
    equity_points = []
    trades = []
    spy_perf6 = close["SPY"].pct_change(126)

    for idx, date in enumerate(dates):
        if date not in open_.index:
            continue

        # 1) Exit-on-open signals from prior close (SMA50 mode).
        for ticker in list(positions):
            p = positions[ticker]
            if p.exit_on_open and pd.notna(open_.at[date, ticker]):
                px = float(open_.at[date, ticker])
                f = fee(px * p.shares)
                cash += px * p.shares - f
                pnl = (px - p.entry) * p.shares - p.entry_fee - f
                trades.append({"ticker": ticker, "entry_date": p.entry_date, "exit_date": str(date.date()), "entry": p.entry, "exit": px, "pnl": pnl, "reason": "SMA50"})
                del positions[ticker]

        # 2) Process pending stop-limit entries from previous close.
        if pending:
            for order in pending:
                if order["ticker"] in positions:
                    continue
                ticker = order["ticker"]
                if ticker not in open_.columns or pd.isna(open_.at[date, ticker]) or pd.isna(high.at[date, ticker]):
                    continue
                o = float(open_.at[date, ticker]); h = float(high.at[date, ticker])
                if o > order["limit"] or h < order["entry"]:
                    continue
                fill = max(o, order["entry"])
                if fill > order["limit"]:
                    continue
                equity_now = cash + sum(p.shares * float(close.at[date, t]) for t, p in positions.items() if pd.notna(close.at[date, t]))
                risk_budget = equity_now * risk_pct
                rps = fill - order["stop"]
                if rps <= 0:
                    continue
                shares = math.floor(risk_budget / rps)
                if shares < 1:
                    continue
                max_affordable = math.floor(max(0, cash - 1) / fill)
                shares = min(shares, max_affordable)
                if shares < 1:
                    continue
                f = fee(fill * shares)
                while shares > 0 and fill * shares + f > cash:
                    shares -= 1
                    f = fee(fill * shares) if shares else 0
                if shares < 1:
                    continue
                cash -= fill * shares + f
                positions[ticker] = Position(ticker, shares, fill, order["stop"], rps, str(date.date()), f)

        pending = []

        # 3) Intraday stops / targets. If stop and target both hit, assume stop first (conservative).
        for ticker in list(positions):
            p = positions[ticker]
            if pd.isna(low.at[date, ticker]) or pd.isna(high.at[date, ticker]):
                continue
            lo = float(low.at[date, ticker]); hi = float(high.at[date, ticker])
            exit_px = None; reason = None
            if lo <= p.stop:
                exit_px = p.stop; reason = "STOP"
            elif mode == "target_2r" and hi >= p.entry + 2 * p.risk_per_share:
                exit_px = p.entry + 2 * p.risk_per_share; reason = "2R"
            elif mode == "target_3r" and hi >= p.entry + 3 * p.risk_per_share:
                exit_px = p.entry + 3 * p.risk_per_share; reason = "3R"
            if exit_px is not None:
                f = fee(exit_px * p.shares)
                cash += exit_px * p.shares - f
                pnl = (exit_px - p.entry) * p.shares - p.entry_fee - f
                trades.append({"ticker": ticker, "entry_date": p.entry_date, "exit_date": str(date.date()), "entry": p.entry, "exit": exit_px, "pnl": pnl, "reason": reason})
                del positions[ticker]

        # 4) Mark-to-market equity at close.
        eq = cash
        for ticker, p in positions.items():
            px = close.at[date, ticker]
            if pd.notna(px):
                eq += p.shares * float(px)
        equity_points.append((date, eq))

        # 5) End-of-day trend exit flag.
        if mode == "sma50":
            for ticker, p in positions.items():
                if pd.notna(close.at[date, ticker]) and pd.notna(ind["sma50"].at[date, ticker]):
                    p.exit_on_open = bool(close.at[date, ticker] < ind["sma50"].at[date, ticker])

        # 6) Generate next-day orders from close, respecting market regime and total-position cap.
        color = regime.get(date, "ROUGE")
        allowed = int(CONFIG["max_positions_green"] if color == "VERT" else CONFIG["max_positions_orange"] if color == "ORANGE" else 0)
        slots = max(0, allowed - len(positions))
        if slots <= 0 or idx == len(dates) - 1:
            continue
        members = membership.get(pd.Timestamp(date).normalize(), set())
        cand = candidates_for_day(date, members, close, volume_global, ind, spy_perf6)
        if cand.empty:
            continue
        for ticker, row in cand.iterrows():
            if ticker in positions:
                continue
            pending.append({"ticker": ticker, "entry": float(row["entry"]), "limit": float(row["limit"]), "stop": float(row["stop"]), "score": float(row["score"])})
            if len(pending) >= slots:
                break

    equity = pd.Series(dict(equity_points)).sort_index()
    trades_df = pd.DataFrame(trades)
    return equity, trades_df


def benchmark_spy(dates, close):
    s = close["SPY"].reindex(dates).dropna()
    eq = float(CONFIG["capital_usd"]) * s / s.iloc[0]
    return eq


def main():
    global volume_global
    current, events, all_symbols = get_sp500_point_in_time_data()
    cfg = CONFIG["backtest"]
    symbols = sorted(all_symbols | set(BENCHMARKS))
    frames = download_prices(symbols, cfg["warmup_start_date"], cfg.get("end_date"))
    open_, high, low, close, volume = matrices(frames)
    volume_global = volume

    missing_bench = [b for b in BENCHMARKS if b not in close.columns]
    if missing_bench:
        raise RuntimeError(f"Missing benchmarks: {missing_bench}")

    start = pd.Timestamp(cfg["start_date"])
    dates = close.index[close.index >= start]
    membership = membership_by_date(dates, current, events)
    ind = indicators(open_, high, low, close, volume)
    regime = market_regime(close[BENCHMARKS]).reindex(dates).fillna("ROUGE")

    results = []
    annual = {}
    trade_summaries = []
    for mode in cfg["exit_modes"]:
        equity, trades = simulate(mode, dates, membership, open_, high, low, close, ind, regime)
        m = metrics(equity, trades, f"US_{mode}")
        results.append(m)
        annual[m["strategy"]] = (1 + equity.pct_change().fillna(0)).groupby(equity.index.year).prod() - 1
        if not trades.empty:
            trades.to_csv(DATA_DIR / f"trades_{mode}.csv", index=False)
        trade_summaries.append({"mode": mode, "trades": len(trades)})

    spy_eq = benchmark_spy(dates, close)
    results.append(metrics(spy_eq, pd.DataFrame(), "SPY_buy_hold"))
    annual["SPY_buy_hold"] = (1 + spy_eq.pct_change().fillna(0)).groupby(spy_eq.index.year).prod() - 1

    result_df = pd.DataFrame(results).sort_values("calmar", ascending=False)
    annual_df = pd.DataFrame(annual) * 100
    coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_sp500_count": len(current),
        "historical_symbols_requested": len(all_symbols),
        "symbols_with_price_data": len([s for s in all_symbols if s in close.columns]),
        "events_parsed": len(events),
        "note": "Point-in-time S&P 500 proxy reconstructed from Wikipedia changes. This is not the full 5,000-stock scanner universe and may miss/rename some delisted tickers.",
    }

    result_df.to_csv(DATA_DIR / "backtest_results.csv", index=False)
    annual_df.to_csv(DATA_DIR / "backtest_annual_returns.csv")
    (DATA_DIR / "backtest_metadata.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (DATA_DIR / "backtest_results.json").write_text(json.dumps({"metrics": result_df.replace({np.nan: None}).to_dict(orient="records"), "coverage": coverage}, indent=2), encoding="utf-8")

    print("=== US BACKTEST RESULTS ===")
    print(result_df.to_string(index=False))
    print("\n=== ANNUAL RETURNS (%) ===")
    print(annual_df.to_string())
    print("\n=== COVERAGE ===")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
