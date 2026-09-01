from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import active_momentum_rotation as rot
import meta_regime_backtest_v2 as robust
import momentum_burst_backtest as mb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"momentum_concentrated_rotation"
OUT.mkdir(parents=True,exist_ok=True)

VARIANTS=[
    rot.Variant("MomTop3_NoBuffer","composite",3,3,3,False,False),
    rot.Variant("MomTop3_Buffer6","composite",3,6,3,False,False),
    rot.Variant("MomTop5_NoBuffer","composite",5,5,5,False,False),
    rot.Variant("MomTop5_Buffer10","composite",5,10,5,False,False),
    rot.Variant("MomTop3_Buffer6_Trend","composite",3,6,3,True,False),
    rot.Variant("MomTop5_Buffer10_Trend","composite",5,10,5,True,False),
    rot.Variant("MomTop3_Buffer6_Gate","composite",3,6,3,False,True),
    rot.Variant("MomTop5_Buffer10_Gate","composite",5,10,5,False,True),
]


def main():
    hist,symbols=rot.load_history_symbols()
    frames=rot.base.download_prices(symbols,rot.DATA_START.strftime("%Y-%m-%d"),(rot.END_EXCLUSIVE-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    open_,high,low,close,volume=rot.base.matrices(frames)
    close=close.loc[(close.index>=rot.DATA_START)&(close.index<rot.END_EXCLUSIVE)]
    open_=open_.reindex(close.index); volume=volume.reindex(close.index)
    sigs=rot.monthly_dates(close.index[(close.index>=rot.TEST_START)&(close.index<rot.END_EXCLUSIVE)])
    mem=rot.memberships(hist,sigs); ind=rot.indicators(close,volume)
    secs={("composite",d):rot.section(d,mem.get(d,set()),close,ind,"composite") for d in sigs}
    last=close.index[(close.index>=rot.TEST_START)&(close.index<rot.END_EXCLUSIVE)][-1]
    world_raw=rot.world_netr(rot.TEST_START,last); common=min(last,world_raw.index.max())
    calendar=close.index[(close.index>=rot.TEST_START)&(close.index<=common)]
    world=rot.benchmark(world_raw,rot.TEST_START,common).reindex(calendar).ffill().bfill(); spy=rot.benchmark(close.SPY,rot.TEST_START,common).reindex(calendar).ffill().bfill()

    rows=[]; curves={"MSCI_World_NETR":world,"SPY_BuyHold":spy}; annual={}; blends=[]; txs=[]; subs=[]
    for v in VARIANTS:
        eqdf,tr,turn=robust.robust_simulate(v,open_,close,sigs,secs,ind)
        e=eqdf.equity.reindex(calendar).ffill().bfill(); z=rot.stats(e); ann=((1+e.pct_change().fillna(0)).groupby(e.index.year).prod()-1)*100
        years=(e.index[-1]-e.index[0]).days/365.25
        z.update(strategy=v.name,start=str(e.index[0].date()),end=str(e.index[-1].date()),transactions=int(len(tr)),annual_turnover_x=float(turn/max(float(e.mean()),1)/max(years,1e-9)),
                 best_year_pct=float(ann.max()),worst_year_pct=float(ann.min()),years_ge_40=int((ann>=40).sum()),years_ge_50=int((ann>=50).sum()))
        rows.append(z); curves[v.name]=e; annual[v.name]=ann
        if not tr.empty: txs.append(tr.assign(strategy=v.name))
        for lab,a,b in [("2011-2017","2011-01-03","2017-12-31"),("2018-2022","2018-01-01","2022-12-31"),("2023-2026","2023-01-01","2026-08-31")]:
            s=e.loc[pd.Timestamp(a):pd.Timestamp(b)]
            if len(s)>30:
                q=rot.stats(s); q.update(strategy=v.name,period=lab); subs.append(q)
        for w in [0.05,0.10]:
            b=mb.blend_annual(world,e,w); q=rot.stats(b); q.update(strategy=f"World{int((1-w)*100)}_Mom{int(w*100)}__{v.name}",satellite_weight_pct=100*w,satellite=v.name); blends.append(q)

    for name,e in [("MSCI_World_NETR",world),("SPY_BuyHold",spy)]:
        z=rot.stats(e); ann=((1+e.pct_change().fillna(0)).groupby(e.index.year).prod()-1)*100
        z.update(strategy=name,start=str(e.index[0].date()),end=str(e.index[-1].date()),best_year_pct=float(ann.max()),worst_year_pct=float(ann.min()),years_ge_40=int((ann>=40).sum()),years_ge_50=int((ann>=50).sum())); rows.append(z); annual[name]=ann
    res=pd.DataFrame(rows).sort_values("cagr_pct",ascending=False); ann_df=pd.DataFrame(annual); curve=pd.DataFrame(curves); blends_df=pd.DataFrame(blends).sort_values("cagr_pct",ascending=False); sub=pd.DataFrame(subs); tx=pd.concat(txs,ignore_index=True) if txs else pd.DataFrame()
    res.to_csv(OUT/"results.csv",index=False); ann_df.to_csv(OUT/"annual_returns_pct.csv"); curve.to_csv(OUT/"equity_curves.csv"); blends_df.to_csv(OUT/"world_satellite_blends.csv",index=False); sub.to_csv(OUT/"subperiod_results.csv",index=False); tx.to_csv(OUT/"transactions.csv",index=False)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"method":"Monthly concentrated cross-sectional momentum, point-in-time S&P500. Composite score 12-1m/6-1m/52w-high/residual momentum, Top3 or Top5, next-open execution, 0.08% commission + 0.05% one-way slippage, no leverage.","limitations":["Yahoo residual missing-delisted bias.","No taxes.","Choosing best variant after seeing test is data mining."],"results":res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records")}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    print("COMMON END",common.date()); print("\nRESULTS\n",res[["strategy","cagr_pct","max_drawdown_pct","sharpe_0rf","final_value","best_year_pct","worst_year_pct","years_ge_40","years_ge_50","transactions","annual_turnover_x"]].to_string(index=False)); print("\nTOP BLENDS\n",blends_df[["strategy","cagr_pct","max_drawdown_pct","sharpe_0rf","final_value"]].head(16).to_string(index=False)); print("\nSUBPERIODS\n",sub[["strategy","period","cagr_pct","max_drawdown_pct","sharpe_0rf"]].to_string(index=False))

if __name__=="__main__": main()
