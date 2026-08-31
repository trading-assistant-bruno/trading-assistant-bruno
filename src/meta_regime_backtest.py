from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import active_momentum_rotation as rot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "meta_regime_backtest"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL = 100_000.0
META_SWITCH_COST = 0.0010  # 10 bps whenever sleeve/exposure changes

TOP10 = rot.Variant("Composite_Top10_Buffer20", "composite", 10, 20, 10, False, False)
TOP20 = rot.Variant("Composite_Top20_Buffer40", "composite", 20, 40, 20, False, False)

SLEEVE_TOP10 = "Active_Top10"
SLEEVE_TOP20 = "Active_Top20"
SLEEVE_SPY = "SPY"
SLEEVE_WORLD = "WORLD"
SLEEVE_CASH = "CASH"


def regime_features(date, members, close, ind):
    spy = float(close.at[date, "SPY"])
    spy_sma200 = float(ind["sma200"].at[date, "SPY"])
    spy_mom12 = float(spy / close["SPY"].shift(252).at[date] - 1) if pd.notna(close["SPY"].shift(252).at[date]) else np.nan
    spy_mom3 = float(spy / close["SPY"].shift(63).at[date] - 1) if pd.notna(close["SPY"].shift(63).at[date]) else np.nan
    vol20 = float(close["SPY"].pct_change().rolling(20).std(ddof=0).at[date] * math.sqrt(252))

    qqq_ok = True
    qqq_above200 = np.nan
    qqq_mom12 = np.nan
    if "QQQ" in close.columns and pd.notna(close.at[date, "QQQ"]):
        q = float(close.at[date, "QQQ"])
        qs = float(close["QQQ"].rolling(200).mean().at[date]) if pd.notna(close["QQQ"].rolling(200).mean().at[date]) else np.nan
        qold = close["QQQ"].shift(252).at[date]
        qqq_above200 = bool(math.isfinite(qs) and q > qs)
        qqq_mom12 = float(q / qold - 1) if pd.notna(qold) and qold > 0 else np.nan
        qqq_ok = bool(qqq_above200 and (not math.isfinite(qqq_mom12) or qqq_mom12 > 0))

    priced = [t for t in members if t in close.columns and pd.notna(close.at[date, t]) and t in ind["sma200"].columns and pd.notna(ind["sma200"].at[date, t])]
    breadth = float(np.mean([close.at[date, t] > ind["sma200"].at[date, t] for t in priced])) if priced else np.nan

    spy_above = bool(spy > spy_sma200)
    if spy_above and spy_mom12 > 0 and qqq_ok and breadth >= 0.55:
        regime = "BULL_BROAD"
    elif spy_above and spy_mom12 > 0:
        regime = "BULL_NARROW"
    elif (not spy_above) and spy_mom3 > 0 and breadth >= 0.40:
        regime = "RECOVERY"
    elif (not spy_above) and breadth < 0.40 and spy_mom12 <= 0:
        regime = "BEAR"
    else:
        regime = "NEUTRAL"

    return {
        "regime": regime,
        "breadth_above_sma200": breadth,
        "spy_above_sma200": spy_above,
        "spy_mom12": spy_mom12,
        "spy_mom3": spy_mom3,
        "qqq_above_sma200": qqq_above200,
        "qqq_mom12": qqq_mom12,
        "spy_vol20_ann": vol20,
        "priced_members": len(priced),
    }


def regime_exposure(regime):
    return {
        "BULL_BROAD": 1.00,
        "BULL_NARROW": 0.75,
        "RECOVERY": 0.50,
        "NEUTRAL": 0.50,
        "BEAR": 0.00,
    }[regime]


def fixed_choice(regime):
    return {
        "BULL_BROAD": SLEEVE_TOP20,
        "BULL_NARROW": SLEEVE_TOP10,
        "RECOVERY": SLEEVE_TOP10,
        "NEUTRAL": SLEEVE_WORLD,
        "BEAR": SLEEVE_CASH,
    }[regime]


def allowed_choices(regime):
    return {
        "BULL_BROAD": [SLEEVE_TOP10, SLEEVE_TOP20, SLEEVE_SPY],
        "BULL_NARROW": [SLEEVE_TOP10, SLEEVE_TOP20, SLEEVE_SPY],
        "RECOVERY": [SLEEVE_TOP10, SLEEVE_SPY, SLEEVE_WORLD],
        "NEUTRAL": [SLEEVE_TOP20, SLEEVE_WORLD, SLEEVE_CASH],
        "BEAR": [SLEEVE_CASH],
    }[regime]


def trailing_score(eq, date, trading_days):
    s = eq.loc[:date].dropna()
    if len(s) < trading_days + 5:
        return np.nan
    s = s.iloc[-(trading_days + 1):]
    r = s.pct_change().dropna()
    if len(r) < int(trading_days * 0.8):
        return np.nan
    cum = float(s.iloc[-1] / s.iloc[0] - 1)
    vol = float(r.std(ddof=0) * math.sqrt(252))
    sharpe = float(r.mean() * 252 / vol) if vol > 1e-12 else (5.0 if cum >= 0 else -5.0)
    # Avoid selecting a recently losing sleeve purely because of very low vol.
    return sharpe if cum > 0 else sharpe - 1.0


def choose_by_score(regime, sleeves, date, window):
    choices = allowed_choices(regime)
    if choices == [SLEEVE_CASH]:
        return SLEEVE_CASH, {SLEEVE_CASH: 0.0}
    scores = {}
    for name in choices:
        if name == SLEEVE_CASH:
            scores[name] = 0.0
        else:
            scores[name] = trailing_score(sleeves[name], date, window)
    valid = {k: v for k, v in scores.items() if pd.notna(v) and math.isfinite(v)}
    if not valid:
        return fixed_choice(regime), scores
    chosen = max(valid, key=valid.get)
    return chosen, scores


def choose_perf_only(sleeves, date, window):
    choices = [SLEEVE_TOP10, SLEEVE_TOP20, SLEEVE_SPY, SLEEVE_WORLD, SLEEVE_CASH]
    scores = {}
    for name in choices:
        scores[name] = 0.0 if name == SLEEVE_CASH else trailing_score(sleeves[name], date, window)
    valid = {k: v for k, v in scores.items() if pd.notna(v) and math.isfinite(v)}
    if not valid:
        return SLEEVE_WORLD, scores
    return max(valid, key=valid.get), scores


def build_decisions(name, sigs, feature_rows, sleeves, calendar):
    rows = []
    feature_map = {r["signal_date"]: r for r in feature_rows}
    for sig in sigs:
        sig = pd.Timestamp(sig)
        if sig not in feature_map:
            continue
        f = feature_map[sig]
        regime = f["regime"]
        if name == "RegimeFixed":
            choice = fixed_choice(regime)
            exposure = regime_exposure(regime)
            scores = {}
        elif name == "RegimePerf6M":
            choice, scores = choose_by_score(regime, sleeves, sig, 126)
            exposure = regime_exposure(regime)
        elif name == "RegimePerf12M":
            choice, scores = choose_by_score(regime, sleeves, sig, 252)
            exposure = regime_exposure(regime)
        elif name == "PerfOnly6M":
            choice, scores = choose_perf_only(sleeves, sig, 126)
            exposure = 0.0 if choice == SLEEVE_CASH else 1.0
        else:
            raise ValueError(name)

        future = calendar[calendar > sig]
        # Conservative: signal at month-end close, one full next session for implementation,
        # then the selected return stream starts on the following session.
        effective = future[1] if len(future) > 1 else (future[0] if len(future) else pd.NaT)
        row = dict(f)
        row.update({
            "meta_strategy": name,
            "chosen_sleeve": choice,
            "exposure": exposure,
            "effective_date": effective,
        })
        for k, v in scores.items():
            row[f"score_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_meta(decisions, sleeves, calendar):
    dec = decisions.dropna(subset=["effective_date"]).copy()
    dec["effective_date"] = pd.to_datetime(dec["effective_date"])
    by_date = {pd.Timestamp(r.effective_date): r for r in dec.itertuples(index=False)}
    daily_returns = {k: v.reindex(calendar).ffill().pct_change().fillna(0.0) for k, v in sleeves.items() if k != SLEEVE_CASH}

    value = INITIAL
    sleeve = SLEEVE_CASH
    exposure = 0.0
    switches = 0
    out = []
    for d in calendar:
        if d in by_date:
            r = by_date[d]
            new_sleeve = r.chosen_sleeve
            new_exposure = float(r.exposure)
            if new_sleeve != sleeve or abs(new_exposure - exposure) > 1e-12:
                # Charge only on the capital whose strategy/exposure specification changes.
                changed_fraction = 1.0 if new_sleeve != sleeve else abs(new_exposure - exposure)
                value *= (1.0 - META_SWITCH_COST * changed_fraction)
                switches += 1
            sleeve = new_sleeve
            exposure = new_exposure
        if sleeve != SLEEVE_CASH and exposure > 0:
            rr = float(daily_returns[sleeve].get(d, 0.0))
            value *= (1.0 + exposure * rr)
        out.append((d, value, sleeve, exposure))
    df = pd.DataFrame(out, columns=["date", "equity", "sleeve", "exposure"]).set_index("date")
    return df, switches


def rolling_relative_win(meta, world, days=756):
    a = meta.reindex(world.index).ffill()
    w = world.reindex(a.index).ffill()
    ar = a / a.shift(days) - 1
    wr = w / w.shift(days) - 1
    x = (ar > wr).dropna()
    return float(100 * x.mean()) if len(x) else np.nan


def stats_with_meta(name, eq, switches=None, world=None):
    z = rot.stats(eq)
    z.update(strategy=name, start=str(eq.index[0].date()), end=str(eq.index[-1].date()))
    if switches is not None:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        z["switches"] = int(switches)
        z["switches_per_year"] = float(switches / years)
    else:
        z["switches"] = np.nan
        z["switches_per_year"] = np.nan
    z["rolling_3y_win_vs_world_pct"] = rolling_relative_win(eq, world) if world is not None else np.nan
    return z


def main():
    hist, symbols = rot.load_history_symbols()
    symbols = sorted(set(symbols) | {"QQQ"})
    print(f"Downloading {len(symbols)} point-in-time historical symbols + QQQ")
    frames = rot.base.download_prices(symbols, rot.DATA_START.strftime("%Y-%m-%d"), (rot.END_EXCLUSIVE - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    open_, high, low, close, volume = rot.base.matrices(frames)
    close = close.loc[(close.index >= rot.DATA_START) & (close.index < rot.END_EXCLUSIVE)]
    open_ = open_.reindex(close.index)
    volume = volume.reindex(close.index)
    if "SPY" not in close.columns:
        raise RuntimeError("SPY unavailable")

    sigs = rot.monthly_dates(close.index[(close.index >= rot.TEST_START) & (close.index < rot.END_EXCLUSIVE)])
    mem = rot.memberships(hist, sigs)
    ind = rot.indicators(close, volume)
    secs = {( "composite", d): rot.section(d, mem.get(d, set()), close, ind, "composite") for d in sigs}

    top10_df, top10_tr, top10_turn = rot.simulate(TOP10, open_, close, sigs, secs, ind)
    top20_df, top20_tr, top20_turn = rot.simulate(TOP20, open_, close, sigs, secs, ind)

    last = close.index[(close.index >= rot.TEST_START) & (close.index < rot.END_EXCLUSIVE)][-1]
    world_raw = rot.world_netr(rot.TEST_START, last)
    common = min(last, world_raw.index.max())
    calendar = close.index[(close.index >= rot.TEST_START) & (close.index <= common)]

    world_eq_raw = rot.benchmark(world_raw, rot.TEST_START, common)
    world_eq = world_eq_raw.reindex(calendar).ffill().bfill()
    spy_eq = rot.benchmark(close["SPY"], rot.TEST_START, common).reindex(calendar).ffill().bfill()
    top10_eq = top10_df.equity.reindex(calendar).ffill().bfill()
    top20_eq = top20_df.equity.reindex(calendar).ffill().bfill()
    cash_eq = pd.Series(INITIAL, index=calendar, dtype=float)

    sleeves = {
        SLEEVE_TOP10: top10_eq,
        SLEEVE_TOP20: top20_eq,
        SLEEVE_SPY: spy_eq,
        SLEEVE_WORLD: world_eq,
        SLEEVE_CASH: cash_eq,
    }

    feature_rows = []
    coverage = []
    for d in sigs[sigs <= common]:
        m = mem.get(d, set())
        f = regime_features(d, m, close, ind)
        f["signal_date"] = pd.Timestamp(d)
        feature_rows.append(f)
        coverage.append({"date": d, "members": len(m), "priced_members": f["priced_members"], "coverage_pct": 100*f["priced_members"]/max(len(m),1)})

    meta_names = ["RegimeFixed", "RegimePerf6M", "RegimePerf12M", "PerfOnly6M"]
    results = []
    curves = {SLEEVE_TOP10: top10_eq, SLEEVE_TOP20: top20_eq, "SPY_BuyHold": spy_eq, "MSCI_World_NETR": world_eq}
    all_decisions = []
    meta_dfs = {}

    for name in meta_names:
        dec = build_decisions(name, sigs[sigs <= common], feature_rows, sleeves, calendar)
        mdf, switches = simulate_meta(dec, sleeves, calendar)
        eq = mdf.equity
        results.append(stats_with_meta(name, eq, switches, world_eq))
        curves[name] = eq
        meta_dfs[name] = mdf
        all_decisions.append(dec)

    for name, eq in [(SLEEVE_TOP10, top10_eq), (SLEEVE_TOP20, top20_eq), ("SPY_BuyHold", spy_eq), ("MSCI_World_NETR", world_eq)]:
        results.append(stats_with_meta(name, eq, None, world_eq))

    res = pd.DataFrame(results)
    decisions = pd.concat(all_decisions, ignore_index=True)
    features = pd.DataFrame(feature_rows)
    coverage_df = pd.DataFrame(coverage)
    curve_df = pd.DataFrame(curves)

    sub_rows = []
    spans = [("2011-2017","2011-01-03","2017-12-31"),("2018-2022","2018-01-01","2022-12-31"),("2023-2026","2023-01-01","2026-08-31")]
    for name, eq in curves.items():
        for lab, a, b in spans:
            s = eq.loc[pd.Timestamp(a):pd.Timestamp(b)]
            if len(s) > 30:
                z = rot.stats(s)
                sub_rows.append({"strategy": name, "period": lab, **z})
    sub = pd.DataFrame(sub_rows)

    annual = {}
    for name, eq in curves.items():
        annual[name] = ((1 + eq.pct_change().fillna(0)).groupby(eq.index.year).prod() - 1) * 100
    annual_df = pd.DataFrame(annual)

    regime_summary = features["regime"].value_counts().rename_axis("regime").reset_index(name="months")
    regime_summary["pct_months"] = 100 * regime_summary.months / regime_summary.months.sum()

    selection_summary = decisions.groupby(["meta_strategy","chosen_sleeve"]).size().rename("months").reset_index()
    selection_summary["pct_within_meta"] = selection_summary.groupby("meta_strategy").months.transform(lambda x: 100*x/x.sum())

    res.to_csv(OUT/"results.csv", index=False)
    sub.to_csv(OUT/"subperiod_results.csv", index=False)
    annual_df.to_csv(OUT/"annual_returns_pct.csv")
    curve_df.to_csv(OUT/"equity_curves.csv")
    decisions.to_csv(OUT/"monthly_decisions.csv", index=False)
    features.to_csv(OUT/"regime_features.csv", index=False)
    coverage_df.to_csv(OUT/"universe_coverage.csv", index=False)
    regime_summary.to_csv(OUT/"regime_summary.csv", index=False)
    selection_summary.to_csv(OUT/"selection_summary.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Monthly walk-forward regime/meta-strategy test. Regime uses only month-end SPY/QQQ trend, S&P500 point-in-time breadth and SPY momentum. RegimePerf variants select the best allowed shadow sleeve by trailing 6m or 12m Sharpe-like score, then apply it only after a one-session implementation delay. 10bp meta switch cost is charged in addition to sleeve trading costs.",
        "fixed_regime_rules": {
            "BULL_BROAD": "SPY>200d, 12m momentum>0, QQQ trend positive, breadth>=55%; Top20 composite; 100% exposure",
            "BULL_NARROW": "SPY>200d and 12m momentum>0 but broad-bull test fails; Top10 composite; 75% exposure",
            "RECOVERY": "SPY<200d, 3m momentum>0, breadth>=40%; Top10; 50% exposure",
            "NEUTRAL": "all other cases; World; 50% exposure",
            "BEAR": "SPY<200d, breadth<40%, 12m momentum<=0; cash",
        },
        "limitations": [
            "Yahoo lacks many delisted/renamed S&P500 constituents; residual survivorship/data-availability bias remains.",
            "Active sleeves are shadow portfolios. Switching into a sleeve assumes the current shadow holdings can be established at the decision transition; an extra 10bp switch cost is charged but exact tax lots are not modeled.",
            "No taxes are modeled. This is important for a taxable CTO and can materially reduce active-strategy advantage.",
            "Residual momentum is a SPY-beta-adjusted approximation, not a full multifactor residual model.",
            "Regime thresholds are fixed round values chosen ex ante for this test; no parameter optimization was performed.",
        ],
        "coverage": {"median_pct": float(coverage_df.coverage_pct.median()), "min_pct": float(coverage_df.coverage_pct.min()), "last_pct": float(coverage_df.coverage_pct.iloc[-1])},
        "results": res.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records"),
    }
    (OUT/"results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("COMMON END", common.date())
    print("\nREGIME SUMMARY\n", regime_summary.to_string(index=False))
    print("\nSELECTION SUMMARY\n", selection_summary.to_string(index=False))
    print("\nRESULTS\n", res.to_string(index=False))
    print("\nSUBPERIODS\n", sub[["strategy","period","cagr_pct","max_drawdown_pct","sharpe_0rf"]].to_string(index=False))
    print("\nLAST 12 DECISIONS\n", decisions.sort_values("signal_date").tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
