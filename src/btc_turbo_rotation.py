from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_backtest as base
import crypto_backtest_yahoo as yahoo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "btc_turbo_rotation"
OUT.mkdir(parents=True, exist_ok=True)

COST_ONE_WAY = 0.0023  # 0.18% commission + 0.05% slippage
BTC = "BTCUSDT"

VARIANTS = [
    {"name":"BTC_HOLD"},
    {"name":"Turbo1_50_rel0", "alts":1, "alt_weight":0.50, "rel28":0.00, "riskoff":0.00},
    {"name":"Turbo2_50_rel0", "alts":2, "alt_weight":0.50, "rel28":0.00, "riskoff":0.00},
    {"name":"Turbo2_75_rel0", "alts":2, "alt_weight":0.75, "rel28":0.00, "riskoff":0.00},
    {"name":"Turbo2_75_rel5", "alts":2, "alt_weight":0.75, "rel28":0.05, "riskoff":0.00},
    {"name":"Turbo2_75_rel10", "alts":2, "alt_weight":0.75, "rel28":0.10, "riskoff":0.00},
    {"name":"Turbo2_75_rel5_RiskOff50Cash", "alts":2, "alt_weight":0.75, "rel28":0.05, "riskoff":0.50},
    {"name":"Turbo2_75_rel5_RiskOff100Cash", "alts":2, "alt_weight":0.75, "rel28":0.05, "riskoff":1.00},
]


def metrics(name: str, r: pd.Series, w: pd.DataFrame, turnover: pd.Series) -> dict:
    r = r.fillna(0.0)
    eq = (1+r).cumprod()
    years = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = eq.iloc[-1]**(1/years)-1
    dd = eq/eq.cummax()-1
    vol = r.std(ddof=0)*math.sqrt(365)
    ann = r.mean()*365
    sharpe = ann/vol if vol>0 else np.nan
    annual = (1+r).groupby(r.index.year).prod()-1
    return {
        "strategy":name,
        "cagr_pct":100*cagr,
        "max_drawdown_pct":100*dd.min(),
        "sharpe_0rf":sharpe,
        "volatility_pct":100*vol,
        "final_multiple":eq.iloc[-1],
        "best_year_pct":100*annual.max(),
        "worst_year_pct":100*annual.min(),
        "years_ge_40":int((annual>=0.40).sum()),
        "years_ge_50":int((annual>=0.50).sum()),
        "years_ge_100":int((annual>=1.00).sum()),
        "avg_btc_weight_pct":100*w.get(BTC,pd.Series(0,index=w.index)).mean(),
        "avg_alt_weight_pct":100*(w.drop(columns=[BTC],errors="ignore").sum(axis=1)).mean(),
        "annual_turnover_x":float(turnover.sum()/years),
    }


def target_weights(close: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = list(close.columns)
    w = pd.DataFrame(np.nan, index=close.index, columns=cols)
    sma100 = close.rolling(100).mean()
    mom28 = close.pct_change(28)
    mom56 = close.pct_change(56)
    score = 0.60*mom28 + 0.40*mom56

    signal_dates = close.index[close.index.dayofweek == 6]
    for dt in signal_dates:
        row = pd.Series(0.0,index=cols)
        if cfg["name"] == "BTC_HOLD":
            row[BTC]=1.0
            w.loc[dt]=row
            continue

        btc_px = close.at[dt,BTC]
        btc_sma = sma100.at[dt,BTC]
        btc28 = mom28.at[dt,BTC]
        btc56 = mom56.at[dt,BTC]
        bull = pd.notna(btc_sma) and btc_px > btc_sma and btc28 > 0 and btc56 > 0
        bear = pd.notna(btc_sma) and btc_px < btc_sma and btc28 < 0

        if bear and cfg.get("riskoff",0)>0:
            row[BTC] = 1.0-cfg["riskoff"]
            w.loc[dt]=row
            continue

        candidates=[]
        if bull:
            for s in cols:
                if s==BTC: continue
                vals=[close.at[dt,s],sma100.at[dt,s],mom28.at[dt,s],mom56.at[dt,s],score.at[dt,s]]
                if any(pd.isna(x) for x in vals): continue
                if close.at[dt,s] <= sma100.at[dt,s]: continue
                if mom28.at[dt,s] <= 0 or mom56.at[dt,s] <= 0: continue
                if mom28.at[dt,s] - btc28 < cfg["rel28"]: continue
                candidates.append((s,float(score.at[dt,s])))
        candidates.sort(key=lambda x:x[1],reverse=True)
        chosen=[s for s,_ in candidates[:cfg["alts"]]]

        if chosen:
            aw=cfg["alt_weight"]
            row[BTC]=1.0-aw
            each=aw/len(chosen)
            for s in chosen: row[s]=each
        else:
            row[BTC]=1.0
        w.loc[dt]=row

    w=w.ffill().fillna(0.0)
    if len(signal_dates):
        w.loc[:signal_dates[0],BTC]=1.0
    return w


def run_variant(close: pd.DataFrame, cfg: dict):
    target=target_weights(close,cfg)
    live=target.shift(1).fillna(0.0)
    zero=live.sum(axis=1)==0
    live.loc[zero,BTC]=1.0
    asset_ret=close.pct_change().fillna(0.0)
    gross=(live*asset_ret).sum(axis=1)
    turnover=live.diff().abs().sum(axis=1).fillna(live.abs().sum(axis=1))
    net=gross-turnover*COST_ONE_WAY
    return net,live,turnover


def subperiod(name,r):
    periods=[("2018-2020","2018-01-01","2020-12-31"),("2021-2023","2021-01-01","2023-12-31"),("2024-2026","2024-01-01","2026-12-31")]
    out=[]
    for label,a,b in periods:
        x=r.loc[a:b]
        if len(x)<20: continue
        eq=(1+x).cumprod(); yrs=(eq.index[-1]-eq.index[0]).days/365.25
        cagr=eq.iloc[-1]**(1/yrs)-1; dd=eq/eq.cummax()-1
        out.append({"strategy":name,"period":label,"cagr_pct":100*cagr,"max_drawdown_pct":100*dd.min()})
    return out


def main():
    prices=yahoo.load_prices_yahoo()
    close=base.align_close(prices).sort_index()
    close=close.loc[close.index>=pd.Timestamp("2018-01-01")]
    if BTC not in close: raise RuntimeError("BTC unavailable")

    rows=[]; annual={}; curves={}; subs=[]; weights_out=[]
    for cfg in VARIANTS:
        r,w,t=run_variant(close,cfg)
        rows.append(metrics(cfg["name"],r,w,t))
        annual[cfg["name"]]=(1+r).groupby(r.index.year).prod()-1
        curves[cfg["name"]]=(1+r).cumprod()
        subs.extend(subperiod(cfg["name"],r))
        tmp=w.copy(); tmp["strategy"]=cfg["name"]; tmp["date"]=tmp.index; weights_out.append(tmp.reset_index(drop=True))

    res=pd.DataFrame(rows).sort_values("cagr_pct",ascending=False)
    ann=pd.DataFrame(annual)*100
    eq=pd.DataFrame(curves)
    sp=pd.DataFrame(subs)
    ww=pd.concat(weights_out,ignore_index=True)
    res.to_csv(OUT/"results.csv",index=False); ann.to_csv(OUT/"annual_returns_pct.csv"); eq.to_csv(OUT/"equity_curves.csv"); sp.to_csv(OUT/"subperiod_results.csv",index=False); ww.to_csv(OUT/"weights.csv",index=False)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"cost_one_way":COST_ONE_WAY,"variants":VARIANTS,"results":res.replace({np.nan:None}).to_dict(orient="records")}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("RESULTS"); print(res.to_string(index=False)); print("\nANNUAL"); print(ann.to_string()); print("\nSUBPERIODS"); print(sp.to_string(index=False))

if __name__=="__main__": main()
