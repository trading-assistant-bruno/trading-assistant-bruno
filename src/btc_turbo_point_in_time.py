from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "btc_turbo_point_in_time"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-01-01")
END = pd.Timestamp.utcnow().tz_localize(None).normalize()
BTC = "BTC"
COST_ONE_WAY = 0.0023  # 0.18% commission + 0.05% slippage
TOP_N = 30
STABLE = {"USDT","USDC","BUSD","DAI","TUSD","USDP","PAX","GUSD","USDD","FDUSD","USDE","FRAX","PYUSD","UST","USTC","EURT","EURC","SUSD","LUSD","USDS"}
EXCLUDE = STABLE | {"WBTC","WETH","STETH","WSTETH","RETH","CBETH"}

YF_OVERRIDES = {
    "BTC":"BTC-USD","ETH":"ETH-USD","XRP":"XRP-USD","BCH":"BCH-USD","LTC":"LTC-USD",
    "EOS":"EOS-USD","BNB":"BNB-USD","BSV":"BSV-USD","XMR":"XMR-USD","XLM":"XLM-USD",
    "TRX":"TRX-USD","ADA":"ADA-USD","XTZ":"XTZ-USD","ATOM":"ATOM-USD","LINK":"LINK-USD",
    "NEO":"NEO-USD","DOGE":"DOGE-USD","ETC":"ETC-USD","MIOTA":"IOTA-USD","IOTA":"IOTA-USD",
    "DASH":"DASH-USD","CRO":"CRO-USD","MKR":"MKR-USD","ONT":"ONT-USD","VET":"VET-USD",
    "XEM":"XEM-USD","BAT":"BAT-USD","ZEC":"ZEC-USD","FTT":"FTT-USD","DCR":"DCR-USD",
    "SNX":"SNX-USD","QTUM":"QTUM-USD","ALGO":"ALGO-USD","ZRX":"ZRX-USD","HOT":"HOT-USD",
    "OKB":"OKB-USD","BTG":"BTG-USD","WAVES":"WAVES-USD","OMG":"OMG-USD","NANO":"XNO-USD",
    "THETA":"THETA-USD","LUNA":"LUNC-USD","LUNC":"LUNC-USD","KCS":"KCS-USD","DGB":"DGB-USD",
    "NEXO":"NEXO-USD","BTT":"BTT-USD","ENJ":"ENJ-USD","ZEN":"ZEN-USD","IOST":"IOST-USD",
    "ICX":"ICX-USD","FIL":"FIL-USD","UNI":"UNI7083-USD","AAVE":"AAVE-USD","SOL":"SOL-USD",
    "AVAX":"AVAX-USD","DOT":"DOT-USD","MATIC":"MATIC-USD","POL":"POL-USD","SHIB":"SHIB-USD",
    "ICP":"ICP-USD","APT":"APT21794-USD","ARB":"ARB11841-USD","OP":"OP-USD","SUI":"SUI20947-USD",
    "TON":"TON11419-USD","INJ":"INJ-USD","TIA":"TIA22861-USD","PEPE":"PEPE24478-USD",
    "HBAR":"HBAR-USD","LEO":"LEO-USD","TAO":"TAO22974-USD","WIF":"WIF-USD","RENDER":"RENDER-USD",
    "RNDR":"RNDR-USD","SEI":"SEI-USD","IMX":"IMX10603-USD","KAS":"KAS-USD","GRT":"GRT6719-USD"
}
HEADERS = {"User-Agent":"Mozilla/5.0 (compatible; research-backtest/1.0)"}


def first_sundays(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    out=[]; cur=pd.Timestamp(start.year,start.month,1)
    while cur<=end:
        d=cur+pd.Timedelta(days=(6-cur.weekday())%7)
        if start<=d<=end: out.append(d)
        cur=(cur+pd.offsets.MonthBegin(1)).normalize()
    return out


def parse_snapshot(dt: pd.Timestamp) -> pd.DataFrame:
    url=f"https://coinmarketcap.com/historical/{dt:%Y%m%d}/"
    r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status()
    best=None
    for t in pd.read_html(r.text):
        cols=[str(c).strip() for c in t.columns]
        if "Rank" in cols and "Symbol" in cols:
            best=t.copy(); break
    if best is None: raise RuntimeError(f"No ranking table found {url}")
    best.columns=[str(c).strip() for c in best.columns]
    best=best[["Rank","Name","Symbol"]].copy(); best["Rank"]=pd.to_numeric(best["Rank"],errors="coerce")
    best["Symbol"]=best["Symbol"].astype(str).str.upper().str.strip(); best=best.dropna(subset=["Rank"])
    best=best[best["Rank"]<=TOP_N]; best=best[~best["Symbol"].isin(EXCLUDE)]; best["snapshot_date"]=dt
    return best.sort_values("Rank")


def collect_snapshots() -> pd.DataFrame:
    rows=[]; failures=[]
    for dt in first_sundays(START,END):
        try:
            t=parse_snapshot(dt); rows.append(t); print(f"snapshot {dt.date()} rows={len(t)}")
        except Exception as e:
            print(f"snapshot failed {dt.date()}: {e}"); failures.append({"date":str(dt.date()),"error":str(e)})
        time.sleep(0.05)
    if not rows: raise RuntimeError("No CMC snapshots collected")
    snap=pd.concat(rows,ignore_index=True); snap.to_csv(OUT/"snapshot_universe.csv",index=False)
    pd.DataFrame(failures).to_csv(OUT/"snapshot_failures.csv",index=False)
    return snap


def yf_symbol(sym: str) -> str:
    return YF_OVERRIDES.get(sym, f"{sym}-USD")


def fetch_prices(symbols: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    close={}; status=[]
    for sym in symbols:
        ticker=yf_symbol(sym)
        try:
            df=yf.Ticker(ticker).history(start=(START-pd.Timedelta(days=140)).strftime("%Y-%m-%d"),interval="1d",auto_adjust=False,actions=False)
            if df is None or df.empty or "Close" not in df:
                status.append({"symbol":sym,"ticker":ticker,"bars":0,"first":None,"last":None}); continue
            idx=pd.to_datetime(df.index)
            if getattr(idx,"tz",None) is not None: idx=idx.tz_convert(None)
            s=pd.Series(pd.to_numeric(df["Close"],errors="coerce").values,index=idx,name=sym).dropna(); close[sym]=s
            status.append({"symbol":sym,"ticker":ticker,"bars":len(s),"first":str(s.index.min().date()),"last":str(s.index.max().date())})
            print(f"price {sym} {ticker}: {len(s)} bars")
        except Exception as e:
            status.append({"symbol":sym,"ticker":ticker,"bars":0,"first":None,"last":None,"error":str(e)}); print(f"price failed {sym}: {e}")
        time.sleep(0.03)
    if BTC not in close: raise RuntimeError("BTC price unavailable")
    st=pd.DataFrame(status); st.to_csv(OUT/"price_coverage_by_symbol.csv",index=False)
    return pd.concat(close,axis=1).sort_index(),st


def latest_universes(snap: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[pd.Timestamp,list[str]]:
    by={pd.Timestamp(k):g.sort_values("Rank")["Symbol"].tolist() for k,g in snap.groupby("snapshot_date")}; keys=sorted(by); out={}; j=-1
    for dt in dates:
        while j+1<len(keys) and keys[j+1]<=dt: j+=1
        out[dt]=by[keys[j]] if j>=0 else []
    return out


def calc_metrics(name: str,r: pd.Series,turnover: pd.Series,coverage: pd.Series) -> dict:
    r=r.fillna(0.0); eq=(1+r).cumprod(); years=(eq.index[-1]-eq.index[0]).days/365.25; cagr=eq.iloc[-1]**(1/years)-1
    dd=eq/eq.cummax()-1; vol=r.std(ddof=0)*math.sqrt(365); ann=r.mean()*365; annual=(1+r).groupby(r.index.year).prod()-1
    return {"strategy":name,"cagr_pct":100*cagr,"max_drawdown_pct":100*dd.min(),"sharpe_0rf":ann/vol if vol>0 else np.nan,
            "volatility_pct":100*vol,"final_multiple":eq.iloc[-1],"best_year_pct":100*annual.max(),"worst_year_pct":100*annual.min(),
            "years_ge_40":int((annual>=0.40).sum()),"years_ge_50":int((annual>=0.50).sum()),"years_ge_100":int((annual>=1.0).sum()),
            "annual_turnover_x":float(turnover.sum()/years),"mean_universe_price_coverage_pct":100*float(coverage.mean())}


def run(close: pd.DataFrame,snap: pd.DataFrame):
    close=close.loc[close.index>=START].copy(); rets=close.pct_change(fill_method=None).fillna(0.0)
    mom28=close.pct_change(28,fill_method=None); mom56=close.pct_change(56,fill_method=None); sma100=close.rolling(100).mean(); universes=latest_universes(snap,close.index)
    periods=pd.Series(close.index.to_period("W-SUN"),index=close.index); weekly=periods.ne(periods.shift(1))
    variants=[("BTC_HOLD",0,0.0,0.0),("PIT_Top1_50_rel5",1,0.50,0.05),("PIT_Top2_50_rel5",2,0.50,0.05),("PIT_Top2_75_rel5",2,0.75,0.05),("PIT_Top2_75_rel10",2,0.75,0.10)]
    results=[]; annuals={}; curves={}; covs=[]
    for name,nalts,alt_weight,rel_thresh in variants:
        w=pd.DataFrame(0.0,index=close.index,columns=close.columns); current={BTC:1.0}; coverage=pd.Series(index=close.index,dtype=float)
        for i,dt in enumerate(close.index):
            universe=[s for s in universes[dt] if s!=BTC]; avail=[s for s in universe if s in close.columns and pd.notna(close.at[dt,s])]; coverage.at[dt]=len(avail)/len(universe) if universe else np.nan
            if name=="BTC_HOLD": current={BTC:1.0}
            elif weekly.iloc[i] and i>0:
                sigdt=close.index[i-1]; ub=[s for s in universes[sigdt] if s!=BTC and s in close.columns]; candidates=[]; btc28=mom28.at[sigdt,BTC]
                for s in ub:
                    vals=(close.at[sigdt,s],sma100.at[sigdt,s],mom28.at[sigdt,s],mom56.at[sigdt,s])
                    if pd.isna(btc28) or not all(pd.notna(v) for v in vals): continue
                    if vals[0]<=vals[1] or vals[2]<=0 or vals[3]<=0 or vals[2]-btc28<rel_thresh: continue
                    candidates.append((0.6*vals[2]+0.4*vals[3],s))
                chosen=[s for _,s in sorted(candidates,reverse=True)[:nalts]]
                current={BTC:1.0} if not chosen else {BTC:1-alt_weight,**{s:alt_weight/len(chosen) for s in chosen}}
            for s,val in current.items():
                if s in w.columns: w.at[dt,s]=val
        turnover=w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum()); gross=(w.shift(1).fillna(0.0)*rets).sum(axis=1); net=gross-turnover*COST_ONE_WAY
        if name=="BTC_HOLD":
            net=rets[BTC].copy(); net.iloc[0]-=COST_ONE_WAY; turnover=pd.Series(0.0,index=close.index); turnover.iloc[0]=1.0
        results.append(calc_metrics(name,net,turnover,coverage)); curves[name]=(1+net).cumprod(); annuals[name]=(1+net).groupby(net.index.year).prod()-1
        covs.append(pd.DataFrame({"date":coverage.index,"strategy":name,"coverage":coverage.values}))
    return pd.DataFrame(results).sort_values("cagr_pct",ascending=False),pd.DataFrame(annuals)*100,pd.DataFrame(curves),pd.concat(covs,ignore_index=True)


def main():
    snap=collect_snapshots(); syms=sorted(set(snap["Symbol"]).union({BTC})); close,_=fetch_prices(syms); res,annual,equity,cov=run(close,snap)
    res.to_csv(OUT/"results.csv",index=False); annual.to_csv(OUT/"annual_returns_pct.csv"); equity.to_csv(OUT/"equity_curves.csv"); cov.to_csv(OUT/"daily_coverage.csv",index=False)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"start":str(START.date()),"end":str(close.index.max().date()),"top_n_snapshot":TOP_N,"stablecoins_excluded":sorted(STABLE),"results":res.replace({np.nan:None}).to_dict("records")}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("RESULTS\n",res.to_string(index=False)); print("\nANNUAL\n",annual.to_string())

if __name__=="__main__": main()
