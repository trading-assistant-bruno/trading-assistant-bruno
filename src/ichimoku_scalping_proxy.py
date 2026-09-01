from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ichimoku_scalping_proxy"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.005
ATR_N = 14
STOP_ATR = 1.5
MAX_HOLD_BARS = 24

ASSETS = {
    "DAX": "^GDAXI",
    "DOW": "^DJI",
    "FTSE100": "^FTSE",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
}

@dataclass(frozen=True)
class Variant:
    name: str
    rr: float
    mtf_filter: bool
    pivot_filter: bool

VARIANTS = [
    Variant("Ichi_R1", 1.0, True, True),
    Variant("Ichi_R1_5", 1.5, True, True),
    Variant("Ichi_R2", 2.0, True, True),
    Variant("Ichi_NoMTF_R1_5", 1.5, False, True),
    Variant("Ichi_NoPivot_R1_5", 1.5, True, False),
]


def download_60m(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="730d", interval="60m", auto_adjust=True, actions=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close"]].dropna()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def midpoint(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
    return (h.rolling(n).max() + l.rolling(n).min()) / 2.0


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["tenkan"] = midpoint(x.High, x.Low, 9)
    x["kijun"] = midpoint(x.High, x.Low, 26)
    a_raw = (x.tenkan + x.kijun) / 2.0
    b_raw = midpoint(x.High, x.Low, 52)
    # Values visible at time t are those projected 26 bars earlier.
    x["span_a"] = a_raw.shift(26)
    x["span_b"] = b_raw.shift(26)
    x["cloud_top"] = x[["span_a", "span_b"]].max(axis=1)
    x["cloud_bot"] = x[["span_a", "span_b"]].min(axis=1)
    prev_close = x.Close.shift(1)
    tr = pd.concat([(x.High-x.Low), (x.High-prev_close).abs(), (x.Low-prev_close).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(ATR_N).mean()
    # Chikou confirmation expressed without future data: current close vs close 26 bars ago.
    x["chikou_long"] = x.Close > x.Close.shift(26)
    x["chikou_short"] = x.Close < x.Close.shift(26)
    return x


def add_daily_pivots(x: pd.DataFrame) -> pd.DataFrame:
    d = x[["High","Low","Close"]].resample("1D").agg({"High":"max","Low":"min","Close":"last"}).dropna()
    p = (d.High + d.Low + d.Close) / 3.0
    piv = pd.DataFrame(index=d.index)
    piv["pivot"] = p.shift(1)
    piv["r1"] = (2*p - d.Low).shift(1)
    piv["s1"] = (2*p - d.High).shift(1)
    key = x.index.floor("1D")
    x = x.copy()
    x["pivot"] = pd.Series(key.map(piv.pivot), index=x.index).values
    x["r1"] = pd.Series(key.map(piv.r1), index=x.index).values
    x["s1"] = pd.Series(key.map(piv.s1), index=x.index).values
    return x


def add_mtf_filter(x: pd.DataFrame) -> pd.DataFrame:
    # 4-hour Ichimoku trend, only completed 4h bars. Shift one 4h bar before forward fill.
    h4 = x[["Open","High","Low","Close"]].resample("4h", label="right", closed="right").agg({
        "Open":"first","High":"max","Low":"min","Close":"last"
    }).dropna()
    h4 = ichimoku(h4)
    bull = (h4.Close > h4.cloud_top) & (h4.tenkan > h4.kijun)
    bear = (h4.Close < h4.cloud_bot) & (h4.tenkan < h4.kijun)
    bull = bull.shift(1).reindex(x.index, method="ffill").fillna(False)
    bear = bear.shift(1).reindex(x.index, method="ffill").fillna(False)
    x = x.copy()
    x["mtf_bull"] = bull.astype(bool)
    x["mtf_bear"] = bear.astype(bool)
    return x


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = ichimoku(df)
    x = add_daily_pivots(x)
    x = add_mtf_filter(x)
    x["long_core"] = (x.Close > x.cloud_top) & (x.tenkan > x.kijun) & x.chikou_long
    x["short_core"] = (x.Close < x.cloud_bot) & (x.tenkan < x.kijun) & x.chikou_short
    return x


def adverse_prices(side: int, entry: float, exit_: float, roundtrip_bps: float) -> tuple[float,float]:
    half = roundtrip_bps / 20000.0
    if side == 1:
        return entry*(1+half), exit_*(1-half)
    return entry*(1-half), exit_*(1+half)


def simulate(x: pd.DataFrame, v: Variant, roundtrip_bps: float = 5.0):
    equity = INITIAL_EQUITY
    peak = equity
    maxdd = 0.0
    trades = []
    eq_rows = []
    i = 80
    n = len(x)
    while i < n-1:
        row = x.iloc[i]
        if not np.isfinite(row.atr) or row.atr <= 0:
            eq_rows.append((x.index[i], equity)); i += 1; continue
        long_sig = bool(row.long_core)
        short_sig = bool(row.short_core)
        if v.mtf_filter:
            long_sig = long_sig and bool(row.mtf_bull)
            short_sig = short_sig and bool(row.mtf_bear)
        if v.pivot_filter:
            long_sig = long_sig and np.isfinite(row.pivot) and row.Close > row.pivot
            short_sig = short_sig and np.isfinite(row.pivot) and row.Close < row.pivot
        if long_sig == short_sig:
            eq_rows.append((x.index[i], equity)); i += 1; continue

        side = 1 if long_sig else -1
        nxt = x.iloc[i+1]
        raw_entry = float(nxt.Open)
        stop_dist = STOP_ATR * float(row.atr)
        if not np.isfinite(raw_entry) or stop_dist <= 0:
            i += 1; continue
        stop = raw_entry - side*stop_dist
        target = raw_entry + side*v.rr*stop_dist
        risk_cash = equity * RISK_PER_TRADE
        units = risk_cash / stop_dist

        exit_raw = None; reason = None; exit_i = None
        last_j = min(n-1, i+1+MAX_HOLD_BARS)
        for j in range(i+1, last_j+1):
            b = x.iloc[j]
            if side == 1:
                hit_stop = b.Low <= stop
                hit_target = b.High >= target
            else:
                hit_stop = b.High >= stop
                hit_target = b.Low <= target
            # Conservative when both touched in same hourly bar: stop first.
            if hit_stop:
                exit_raw = stop; reason = "STOP"; exit_i = j; break
            if hit_target:
                exit_raw = target; reason = "TARGET"; exit_i = j; break
            # Ichimoku invalidation at bar close.
            if side == 1 and b.Close < b.kijun:
                exit_raw = float(b.Close); reason = "KIJUN"; exit_i = j; break
            if side == -1 and b.Close > b.kijun:
                exit_raw = float(b.Close); reason = "KIJUN"; exit_i = j; break
        if exit_i is None:
            exit_i = last_j; exit_raw = float(x.iloc[exit_i].Close); reason = "TIME"

        entry, exit_px = adverse_prices(side, raw_entry, exit_raw, roundtrip_bps)
        pnl = units * side * (exit_px-entry)
        eq_before = equity
        equity += pnl
        peak = max(peak, equity)
        maxdd = min(maxdd, equity/peak-1)
        r_mult = pnl/risk_cash if risk_cash else np.nan
        trades.append({
            "entry_time": x.index[i+1], "exit_time": x.index[exit_i], "side": "LONG" if side==1 else "SHORT",
            "entry": entry, "exit": exit_px, "stop_raw": stop, "target_raw": target,
            "pnl": pnl, "r_multiple": r_mult, "reason": reason, "equity_before": eq_before, "equity_after": equity,
        })
        for k in range(i, exit_i+1):
            eq_rows.append((x.index[k], equity if k==exit_i else eq_before))
        i = max(i+1, exit_i+1)

    eq = pd.Series(dict(eq_rows)).sort_index()
    tr = pd.DataFrame(trades)
    if eq.empty:
        return {}, tr, eq
    years = max((eq.index[-1]-eq.index[0]).total_seconds()/(365.25*86400), 1e-9)
    cagr = (equity/INITIAL_EQUITY)**(1/years)-1 if equity > 0 else -1
    wins = tr[tr.pnl>0] if not tr.empty else tr
    losses = tr[tr.pnl<0] if not tr.empty else tr
    gp = wins.pnl.sum() if len(wins) else 0.0
    gl = -losses.pnl.sum() if len(losses) else 0.0
    stats = {
        "cagr_pct": 100*cagr,
        "max_drawdown_pct": 100*maxdd,
        "final_equity": equity,
        "trades": int(len(tr)),
        "win_rate_pct": 100*float((tr.pnl>0).mean()) if len(tr) else np.nan,
        "profit_factor": gp/gl if gl>0 else np.nan,
        "avg_r": float(tr.r_multiple.mean()) if len(tr) else np.nan,
        "median_r": float(tr.r_multiple.median()) if len(tr) else np.nan,
        "target_pct": 100*float((tr.reason=="TARGET").mean()) if len(tr) else np.nan,
        "stop_pct": 100*float((tr.reason=="STOP").mean()) if len(tr) else np.nan,
    }
    return stats, tr, eq


def main():
    all_results=[]; all_trades=[]; curves={}; coverage=[]
    prepared={}
    for name,ticker in ASSETS.items():
        print("Downloading", name, ticker)
        raw=download_60m(ticker)
        if len(raw)<500:
            print("SKIP insufficient", name, len(raw)); continue
        x=prepare(raw); prepared[name]=x
        coverage.append({"asset":name,"ticker":ticker,"bars":len(x),"start":str(x.index.min()),"end":str(x.index.max())})
        for v in VARIANTS:
            for cost in [2.0,5.0,10.0]:
                st,tr,eq=simulate(x,v,cost)
                if not st: continue
                st.update(asset=name,ticker=ticker,variant=v.name,roundtrip_bps=cost,start=str(eq.index.min()),end=str(eq.index.max()))
                all_results.append(st)
                if cost==5.0:
                    curves[f"{name}_{v.name}"]=eq
                    if not tr.empty:
                        z=tr.copy(); z["asset"]=name; z["variant"]=v.name; z["roundtrip_bps"]=cost; all_trades.append(z)

    res=pd.DataFrame(all_results)
    cov=pd.DataFrame(coverage)
    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    curve=pd.DataFrame(curves).sort_index() if curves else pd.DataFrame()
    res.to_csv(OUT/"results.csv",index=False)
    cov.to_csv(OUT/"coverage.csv",index=False)
    trades.to_csv(OUT/"trades_5bps.csv",index=False)
    curve.to_csv(OUT/"equity_curves_5bps.csv")

    # Aggregate cross-asset summary at base 5 bps cost.
    base=res[res.roundtrip_bps==5.0].copy()
    agg=[]
    for v,g in base.groupby("variant"):
        agg.append({
            "variant":v,
            "assets":len(g),
            "median_cagr_pct":g.cagr_pct.median(),
            "mean_cagr_pct":g.cagr_pct.mean(),
            "median_maxdd_pct":g.max_drawdown_pct.median(),
            "total_trades":int(g.trades.sum()),
            "mean_win_rate_pct":g.win_rate_pct.mean(),
            "median_profit_factor":g.profit_factor.median(),
            "mean_avg_r":g.avg_r.mean(),
            "profitable_assets":int((g.cagr_pct>0).sum()),
        })
    agg=pd.DataFrame(agg).sort_values("median_cagr_pct",ascending=False)
    agg.to_csv(OUT/"aggregate_5bps.csv",index=False)

    payload={
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "description":"Reproducible public-information proxy inspired by an Ichimoku scalping course outline. It is NOT the instructor's exact proprietary method.",
        "timeframe":"Yahoo Finance 60-minute bars, up to 730 days",
        "rules":{
            "core":"Long: close above visible Kumo, Tenkan>Kijun, current close>close 26 bars ago. Short inverse.",
            "mtf":"Optional completed 4-hour Ichimoku trend confirmation.",
            "pivot":"Optional previous-day classic pivot confirmation.",
            "stop":"1.5 ATR(14)",
            "targets":"1R, 1.5R or 2R depending on variant",
            "risk_per_trade_pct":100*RISK_PER_TRADE,
            "max_holding_hours":MAX_HOLD_BARS,
            "same_bar_stop_target":"stop assumed first (conservative)",
            "cost_sensitivity_roundtrip_bps":[2,5,10],
        },
        "limitations":[
            "This is hourly, not true 5-minute scalping, because free long-history intraday data are limited.",
            "Course public page does not disclose exact entry/exit parameters; no claim that this replicates Antoine Legay's proprietary rules.",
            "Yahoo intraday data quality and trading-session conventions may differ by instrument.",
            "No broker-specific financing, contract size, margin, taxes or overnight financing modeled.",
        ],
        "aggregate_5bps":agg.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records"),
    }
    (OUT/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("\nCOVERAGE\n",cov.to_string(index=False))
    print("\nAGGREGATE 5 BPS\n",agg.to_string(index=False))
    print("\nRESULTS 5 BPS\n",base[["asset","variant","cagr_pct","max_drawdown_pct","trades","win_rate_pct","profit_factor","avg_r"]].to_string(index=False))
    print("\nCOST SENSITIVITY MEDIAN CAGR\n",res.groupby(["variant","roundtrip_bps"]).cagr_pct.median().unstack().to_string())

if __name__=="__main__":
    main()
