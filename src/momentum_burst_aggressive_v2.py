from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import active_momentum_rotation as rot
import meta_regime_backtest as meta
import meta_regime_backtest_v2 as robust
import momentum_burst_backtest as mb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "momentum_burst_aggressive_v2"
OUT.mkdir(parents=True, exist_ok=True)
INITIAL = 100_000.0
STOP_ATR = 2.5
DELIST_HAIRCUT = 0.02


@dataclass(frozen=True)
class Variant:
    name: str
    max_positions: int
    pool_rank: int
    risk_pct: float
    gate_kind: str
    breakout_n: int
    trail_atr: float
    pyramid: bool
    max_position_pct: float


VARIANTS = [
    Variant("Agg3_R150_Select_B20_T5_Pyr",3,20,0.015,"selective",20,5.0,True,0.50),
    Variant("Agg3_R200_Select_B20_T5_Pyr",3,20,0.020,"selective",20,5.0,True,0.50),
    Variant("Agg3_R300_Select_B20_T5_Pyr",3,20,0.030,"selective",20,5.0,True,0.50),
    Variant("Agg3_R200_Strong_B20_T5_Pyr",3,20,0.020,"strong",20,5.0,True,0.50),
    Variant("Agg5_R200_Select_B20_T5_Pyr",5,25,0.020,"selective",20,5.0,True,0.35),
    Variant("Agg3_R200_Select_B20_T4_Pyr",3,20,0.020,"selective",20,4.0,True,0.50),
    Variant("Agg3_R200_Select_B55_T5_Pyr",3,20,0.020,"selective",55,5.0,True,0.50),
    Variant("Agg3_R200_Select_B20_T5_NoPyr",3,20,0.020,"selective",20,5.0,False,0.50),
]


def simulate(v,open_,high,low,close,close_ffill,ind,atr20,sma50,breakout_map,weeks,mem,sections,gates):
    dates=close.index[(close.index>=rot.TEST_START)&(close.index<rot.END_EXCLUSIVE)]
    weekset=set(weeks)
    cash=INITIAL; positions={}; current_gate=False; current_cs=pd.DataFrame(); current_members=set()
    pending_entries=[]; pending_adds={}; pending_universe_exits=set()
    eq_rows=[]; fills=[]; closed=[]

    def mark(d,t):
        if t not in close_ffill.columns: return np.nan
        z=close_ffill.at[d,t]
        return float(z) if pd.notna(z) else np.nan

    def eq(d):
        total=cash; invested=0.0
        for t,p in positions.items():
            px=mark(d,t)
            if np.isfinite(px):
                val=p["shares"]*px; invested+=val; total+=val
        return total,invested

    def exit_pos(d,t,reason,raw=None,haircut=0.0):
        nonlocal cash
        if t not in positions: return
        p=positions[t]
        if raw is None or not np.isfinite(raw) or raw<=0:
            raw=mark(d,t); haircut=max(haircut,DELIST_HAIRCUT)
        cash,notion,fee,px=mb.sell_fill(cash,float(raw),p["shares"],haircut)
        if notion<=0: return
        proceeds=notion-fee; pnl=proceeds-p["buy_cash"]
        closed.append({"variant":v.name,"ticker":t,"entry_date":p["entry_date"],"exit_date":d,"reason":reason,
                       "pnl":pnl,"r_multiple":pnl/p["initial_risk_cash"] if p["initial_risk_cash"]>0 else np.nan,
                       "days_held":(d-p["entry_date"]).days,"adds":p["adds"],"initial_risk_cash":p["initial_risk_cash"],
                       "buy_cash":p["buy_cash"],"sell_proceeds":proceeds})
        fills.append({"variant":v.name,"date":d,"ticker":t,"side":"SELL","reason":reason,"notional":notion,"fee":fee,"price":px})
        positions.pop(t,None); pending_adds.pop(t,None)

    for d in dates:
        # Only universe removals force a non-price exit. Market regime never forces out a live winner.
        for t in list(pending_universe_exits):
            if t in positions:
                op=open_.at[d,t] if t in open_.columns else np.nan
                exit_pos(d,t,"UNIVERSE_EXIT",float(op) if pd.notna(op) else None)
        pending_universe_exits=set()

        # Pyramids from prior close; only if the entry gate is still favorable.
        total_open,_=eq(d)
        for t,stage in list(pending_adds.items()):
            if not current_gate or t not in positions: continue
            p=positions[t]; op=open_.at[d,t] if t in open_.columns else np.nan
            if pd.isna(op) or float(op)<=0: continue
            cap=total_open*v.max_position_pct; cur=p["shares"]*float(op); room=max(0.0,cap-cur)
            qty0=p["initial_units"]*0.50
            cash,qty,notion,fee=mb.buy_fill(cash,float(op),qty0,room)
            if qty>0:
                p["shares"]+=qty; p["buy_cash"]+=notion+fee; p["adds"]+=1
                if stage==1:
                    p["add1_done"]=True; p["stop"]=max(p["stop"],p["entry_price"])
                else:
                    p["add2_done"]=True; p["stop"]=max(p["stop"],p["entry_price"]+p["stop_dist"])
                fills.append({"variant":v.name,"date":d,"ticker":t,"side":"BUY_ADD","reason":f"ADD{stage}","notional":notion,"fee":fee,"price":float(op)*(1+rot.SLIPPAGE)})
        pending_adds={}

        # New entries.
        total_open,_=eq(d)
        for ent in pending_entries:
            t=ent["ticker"]
            if not current_gate or t in positions or len(positions)>=v.max_positions: continue
            op=open_.at[d,t] if t in open_.columns else np.nan
            if pd.isna(op) or float(op)<=0: continue
            dist=STOP_ATR*ent["atr"]
            if not np.isfinite(dist) or dist<=0: continue
            risk=total_open*v.risk_pct; wanted=risk/dist; maxcash=total_open*v.max_position_pct
            cash,qty,notion,fee=mb.buy_fill(cash,float(op),wanted,maxcash)
            if qty<=0: continue
            fill=float(op)*(1+rot.SLIPPAGE); actual=qty*dist
            positions[t]={"shares":qty,"initial_units":qty,"entry_date":d,"entry_price":fill,"stop_dist":dist,
                          "stop":fill-dist,"highest_close":fill,"initial_risk_cash":actual,"buy_cash":notion+fee,
                          "adds":0,"add1_done":False,"add2_done":False}
            fills.append({"variant":v.name,"date":d,"ticker":t,"side":"BUY","reason":"BREAKOUT","notional":notion,"fee":fee,"price":fill})
        pending_entries=[]

        # Price stop.
        for t in list(positions):
            p=positions[t]
            if t not in low.columns or pd.isna(low.at[d,t]): continue
            lo=float(low.at[d,t]); op=open_.at[d,t] if t in open_.columns else np.nan
            if lo<=p["stop"]:
                raw=float(op) if pd.notna(op) and float(op)<p["stop"] else p["stop"]
                exit_pos(d,t,"TRAIL_STOP",raw)

        total,invested=eq(d); exposure=invested/total if total>0 else 0.0
        eq_rows.append((d,total,cash,len(positions),exposure,current_gate))

        # Let winners run with a wide chandelier trail.
        for t,p in list(positions.items()):
            if t not in close.columns or pd.isna(close.at[d,t]): continue
            cl=float(close.at[d,t]); p["highest_close"]=max(p["highest_close"],cl)
            a=atr20.at[d,t] if t in atr20.columns else np.nan
            if pd.notna(a) and float(a)>0:
                p["stop"]=max(p["stop"],p["highest_close"]-v.trail_atr*float(a))

        # Weekly gate/ranking update. Gate controls entries only.
        if d in weekset:
            current_cs=sections.get(d,pd.DataFrame()); current_members=mem.get(d,set())
            current_gate=bool(gates.get(d,{}).get(v.gate_kind,False)) if v.gate_kind!="none" else True
            for t in positions:
                # Exit only if the security truly leaves the point-in-time universe or has no recent quotation.
                recent=close.loc[:d,t].tail(5) if t in close.columns else pd.Series(dtype=float)
                stale=(len(recent)==0 or recent.notna().sum()==0)
                if t not in current_members or stale:
                    pending_universe_exits.add(t)

        # Add to winners only, never average down.
        if v.pyramid and current_gate:
            for t,p in positions.items():
                if t not in close.columns or pd.isna(close.at[d,t]): continue
                cl=float(close.at[d,t])
                if (not p["add1_done"]) and cl>=p["entry_price"]+p["stop_dist"]:
                    pending_adds[t]=1
                elif p["add1_done"] and (not p["add2_done"]) and cl>=p["entry_price"]+2*p["stop_dist"]:
                    pending_adds[t]=2

        # New leader breakouts. Ranking can refresh weekly, breakout is checked daily.
        if current_gate and not current_cs.empty:
            slots=max(0,v.max_positions-len(positions))
            if slots>0:
                elig=current_cs[current_cs.base_eligible].copy(); elig=elig[elig["rank"]<=v.pool_rank]
                cand=[]
                for t,row in elig.iterrows():
                    if t in positions: continue
                    if any(t not in x.columns for x in [close,sma50,ind["sma200"],ind["high52"],ind["mom6"],atr20]): continue
                    vals=[close.at[d,t],sma50.at[d,t],ind["sma200"].at[d,t],ind["high52"].at[d,t],ind["mom6"].at[d,t],atr20.at[d,t]]
                    if any(pd.isna(z) for z in vals): continue
                    cl,s50,s200,h52,m6,a=map(float,vals)
                    if not (cl>s50>s200 and h52>=0.90 and m6>0 and a>0): continue
                    br=breakout_map[v.breakout_n].at[d,t]
                    if bool(br): cand.append((float(row["rank"]),t,a))
                cand.sort(); pending_entries=[{"ticker":t,"atr":a,"rank":r} for r,t,a in cand[:slots]]

    if len(dates):
        last=dates[-1]
        for t in list(positions): exit_pos(last,t,"TERMINAL",mark(last,t),0.0)
        if eq_rows: eq_rows[-1]=(last,cash,cash,0,0.0,current_gate)
    eqdf=pd.DataFrame(eq_rows,columns=["date","equity","cash","positions","exposure","gate_on"]).set_index("date")
    return eqdf,pd.DataFrame(closed),pd.DataFrame(fills)


def main():
    hist,symbols=rot.load_history_symbols(); symbols=sorted(set(symbols)|{"QQQ"})
    print(f"Downloading {len(symbols)} point-in-time symbols + QQQ")
    frames=rot.base.download_prices(symbols,rot.DATA_START.strftime("%Y-%m-%d"),(rot.END_EXCLUSIVE-pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    open_,high,low,close,volume=rot.base.matrices(frames)
    close=close.loc[(close.index>=rot.DATA_START)&(close.index<rot.END_EXCLUSIVE)]
    open_=open_.reindex(close.index); high=high.reindex(close.index); low=low.reindex(close.index); volume=volume.reindex(close.index)
    close_ffill=close.ffill(); ind=rot.indicators(close,volume)
    tr=mb.true_range(high,low,close); atr20=tr.rolling(20).mean(); sma50=close.rolling(50).mean()
    qqq_sma200=close.QQQ.rolling(200).mean(); spy_mom63=close.SPY/close.SPY.shift(63)-1; qqq_mom63=close.QQQ/close.QQQ.shift(63)-1
    spy_vol20=close.SPY.pct_change().rolling(20).std(ddof=0)*math.sqrt(252)
    breakout_map={n:close>close.shift(1).rolling(n).max() for n in sorted({v.breakout_n for v in VARIANTS})}
    test_index=close.index[(close.index>=rot.TEST_START)&(close.index<rot.END_EXCLUSIVE)]
    weeks=mb.weekly_dates(test_index); mem=rot.memberships(hist,weeks); sections={}; gates={}; grows=[]; cov=[]
    for d in weeks:
        m=mem.get(d,set()); sections[d]=rot.section(d,m,close,ind,"composite")
        gf=mb.gate_features(d,m,close,ind,qqq_sma200,spy_mom63,qqq_mom63,spy_vol20); gates[d]=gf; grows.append(gf)
        cov.append({"date":d,"members":len(m),"priced_members":gf["priced_members"],"coverage_pct":100*gf["priced_members"]/max(len(m),1)})

    last=test_index[-1]; world_raw=rot.world_netr(rot.TEST_START,last); common=min(last,world_raw.index.max()); calendar=test_index[test_index<=common]
    world=rot.benchmark(world_raw,rot.TEST_START,common).reindex(calendar).ffill().bfill(); spy=rot.benchmark(close.SPY,rot.TEST_START,common).reindex(calendar).ffill().bfill()
    msigs=rot.monthly_dates(test_index); mmem=rot.memberships(hist,msigs); msecs={("composite",d):rot.section(d,mmem.get(d,set()),close,ind,"composite") for d in msigs}
    top20df,_,_=robust.robust_simulate(meta.TOP20,open_,close,msigs,msecs,ind); top20=top20df.equity.reindex(calendar).ffill().bfill()

    rows=[]; curves={"MSCI_World_NETR":world,"SPY_BuyHold":spy,"Active_Top20":top20}; annual={}; trades_all=[]; fills_all=[]; blends=[]; subs=[]
    for v in VARIANTS:
        eqdf,trades,fills=simulate(v,open_,high,low,close,close_ffill,ind,atr20,sma50,breakout_map,weeks,mem,sections,gates)
        e=eqdf.equity.loc[:common]; z=rot.stats(e); ts=mb.trade_stats(trades); ann=((1+e.pct_change().fillna(0)).groupby(e.index.year).prod()-1)*100
        z.update(strategy=v.name,start=str(e.index[0].date()),end=str(e.index[-1].date()),avg_exposure_pct=100*float(eqdf.loc[:common,"exposure"].mean()),
                 gate_on_pct=100*float(eqdf.loc[:common,"gate_on"].mean()),best_year_pct=float(ann.max()),worst_year_pct=float(ann.min()),
                 years_ge_40=int((ann>=40).sum()),years_ge_50=int((ann>=50).sum()),**ts)
        rows.append(z); curves[v.name]=e; annual[v.name]=ann
        if not trades.empty: trades_all.append(trades)
        if not fills.empty: fills_all.append(fills)
        for lab,a,b in [("2011-2017","2011-01-03","2017-12-31"),("2018-2022","2018-01-01","2022-12-31"),("2023-2026","2023-01-01","2026-08-31")]:
            s=e.loc[pd.Timestamp(a):pd.Timestamp(b)]
            if len(s)>30:
                q=rot.stats(s); q.update(strategy=v.name,period=lab); subs.append(q)
        for w in [0.05,0.10]:
            b=mb.blend_annual(world,e,w); q=rot.stats(b); q.update(strategy=f"World{int((1-w)*100)}_Agg{int(w*100)}__{v.name}",burst_weight_pct=100*w,burst_variant=v.name); blends.append(q)

    for name,e in [("MSCI_World_NETR",world),("SPY_BuyHold",spy),("Active_Top20",top20)]:
        z=rot.stats(e); ann=((1+e.pct_change().fillna(0)).groupby(e.index.year).prod()-1)*100
        z.update(strategy=name,start=str(e.index[0].date()),end=str(e.index[-1].date()),best_year_pct=float(ann.max()),worst_year_pct=float(ann.min()),years_ge_40=int((ann>=40).sum()),years_ge_50=int((ann>=50).sum())); rows.append(z); annual[name]=ann

    res=pd.DataFrame(rows).sort_values("cagr_pct",ascending=False); ann_df=pd.DataFrame(annual); curve=pd.DataFrame(curves); blends_df=pd.DataFrame(blends).sort_values("cagr_pct",ascending=False)
    trdf=pd.concat(trades_all,ignore_index=True) if trades_all else pd.DataFrame(); fdf=pd.concat(fills_all,ignore_index=True) if fills_all else pd.DataFrame(); gate_df=pd.DataFrame(grows); cov_df=pd.DataFrame(cov); sub=pd.DataFrame(subs)
    res.to_csv(OUT/"results.csv",index=False); ann_df.to_csv(OUT/"annual_returns_pct.csv"); curve.to_csv(OUT/"equity_curves.csv"); blends_df.to_csv(OUT/"world_aggressive_blends.csv",index=False)
    trdf.to_csv(OUT/"closed_trades.csv",index=False); fdf.to_csv(OUT/"fills.csv",index=False); gate_df.to_csv(OUT/"weekly_gate_features.csv",index=False); cov_df.to_csv(OUT/"universe_coverage.csv",index=False); sub.to_csv(OUT/"subperiod_results.csv",index=False)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"method":"Aggressive Momentum Burst v2: 3-5 concentrated leaders, 1.5-3% pocket risk/trade, entry gate only, no regime/rank forced exit, wide 4-5 ATR trail, optional pyramiding, no leverage.",
             "limitations":["Yahoo missing delisted/renamed names leaves residual bias.","No taxes.","Daily OHLC cannot resolve exact intraday path.","Choosing the best tested variant after inspection creates data-mining risk."],
             "results":res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records"),"top_blends":blends_df.head(10).replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict("records")}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    cols=["strategy","cagr_pct","max_drawdown_pct","sharpe_0rf","final_value","volatility_pct","avg_exposure_pct","best_year_pct","worst_year_pct","years_ge_40","years_ge_50","trades","win_rate_pct","profit_factor","avg_r","avg_win_r","avg_loss_r","max_trade_r","trades_ge_5r","trades_ge_10r"]
    print("COMMON END",common.date()); print("\nRESULTS\n",res.reindex(columns=cols).to_string(index=False)); print("\nTOP BLENDS\n",blends_df[["strategy","cagr_pct","max_drawdown_pct","sharpe_0rf","final_value"]].head(16).to_string(index=False)); print("\nSUBPERIODS\n",sub[["strategy","period","cagr_pct","max_drawdown_pct","sharpe_0rf"]].to_string(index=False)); print("\nGATE FREQ\n",gate_df[["broad","strong","selective"]].mean().mul(100).to_string())

if __name__=="__main__": main()
