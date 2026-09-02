from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'seed'/'snapshots.csv'
OUT=ROOT/'data'/'btc_turbo_catastrophe_stop'; OUT.mkdir(parents=True,exist_ok=True)
COST=.0023; T=.30; REL=.05
THRESHOLDS=[None,-.30,-.35,-.40,-.45,-.50]
EX={'USDT','USDC','BUSD','DAI','TUSD','USDP','PAX','GUSD','USDD','FDUSD','USDE','FRAX','PYUSD','UST','USTC','EURT','EURC','SUSD','LUSD','USDS','WBTC','WETH','STETH','WSTETH','RETH','CBETH','WEETH'}
OV={'MIOTA':'IOTA-USD','IOTA':'IOTA-USD','BCH':'BCH-USD','XRB':'XNO-USD','NANO':'XNO-USD'}

def yf_sym(s): return OV.get(s,f'{s}-USD')
def dl(s,start,end):
 d=yf.download(yf_sym(s),start=start,end=end,auto_adjust=False,actions=False,progress=False,threads=False,timeout=30)
 if d is None or d.empty:return None
 if isinstance(d.columns,pd.MultiIndex):d.columns=[x[0] for x in d.columns]
 if not all(x in d for x in ['Open','High','Low','Close']):return None
 d=d[['Open','High','Low','Close']].dropna(); idx=pd.to_datetime(d.index)
 if getattr(idx,'tz',None) is not None:idx=idx.tz_convert(None)
 d.index=idx.normalize();return d[~d.index.duplicated(keep='last')].sort_index()
def bar(d,x):
 a=d.index[d.index>=pd.Timestamp(x)];return a[0] if len(a) else None

def select(i,p,r):
 if i<2:return []
 dt=p.index[i]; b=[p.at[p.index[j],'BTC'] for j in [i,i-1,i-2]]
 if any(pd.isna(x) or x<=0 for x in b):return []
 bm1=b[0]/b[1]-1;bm2=b[0]/b[2]-1
 if bm1<=0 or bm2<=0:return []
 c=[]
 for s,rank in r.loc[dt].dropna().items():
  if s=='BTC' or s in EX or rank>20:continue
  a=[p.at[p.index[j],s] if s in p.columns else np.nan for j in [i,i-1,i-2]]
  if any(pd.isna(x) or x<=0 for x in a):continue
  m1=a[0]/a[1]-1;m2=a[0]/a[2]-1
  if m1>0 and m2>0 and m1-bm1>=REL:c.append((.6*m1+.4*m2,-rank,s))
 c.sort(reverse=True);return [x[2] for x in c[:1]]

def sleeve(s,d0,d1,daily,thr):
 a=daily.get(s);b=daily.get('BTC')
 if a is None or b is None:return None
 e=bar(a,d0+pd.Timedelta(days=1));x=bar(a,d1+pd.Timedelta(days=1))
 if e is None or x is None or x<=e:return None
 ep=float(a.at[e,'Open']);xp=float(a.at[x,'Open'])
 if s=='BTC' or thr is None:return xp/ep-1,False
 stop=ep*(1+thr); hit=None;px=None
 for day,row in a[(a.index>=e)&(a.index<x)].iterrows():
  if float(row.Low)<=stop:
   op=float(row.Open);px=op if op<stop else stop;hit=day;break
 if hit is None:return xp/ep-1,False
 re=bar(b,hit+pd.Timedelta(days=1));be=bar(b,d1+pd.Timedelta(days=1));f=px/ep*(1-COST)**2
 if re is not None and be is not None and be>re:
  br=float(b.at[re,'Open']);bp=float(b.at[be,'Open']);f*=bp/br if br>0 else 1
 return f-1,True

def metrics(curve,stops):
 s=pd.Series(curve).sort_index();yrs=(s.index[-1]-s.index[0]).days/365.2425;c=s.iloc[-1]**(1/yrs)-1
 rr=s.pct_change().dropna();dd=(s/s.cummax()-1).min();vol=rr.std(ddof=1)*np.sqrt(12);sh=rr.mean()/rr.std(ddof=1)*np.sqrt(12) if rr.std(ddof=1)>0 else np.nan
 ann=s.resample('YE').last().pct_change();ann.iloc[0]=s.resample('YE').last().iloc[0]/s.iloc[0]-1
 return dict(cagr_pct=100*c,max_monthly_drawdown_pct=100*dd,sharpe_0rf=sh,volatility_pct=100*vol,final_multiple=s.iloc[-1],best_year_pct=100*ann.max(),worst_year_pct=100*ann.min(),best_month_pct=100*rr.max(),worst_month_pct=100*rr.min(),stops_count=stops)

def main():
 s=pd.read_csv(SEED);s['date']=pd.to_datetime(s.date);s=s.sort_values(['date','Rank']);p=s.pivot_table(index='date',columns='Symbol',values='Price',aggfunc='first').sort_index();r=s.pivot_table(index='date',columns='Symbol',values='Rank',aggfunc='first').sort_index()
 chosen={i:select(i,p,r) for i in range(len(p)-1)};syms={'BTC'}|{x for v in chosen.values() for x in v}
 daily={}; start=(p.index[0]-pd.Timedelta(days=40)).strftime('%Y-%m-%d');end=(p.index[-1]+pd.Timedelta(days=10)).strftime('%Y-%m-%d')
 for x in sorted(syms):daily[x]=dl(x,start,end);print(x,0 if daily[x] is None else len(daily[x]))
 rows=[];curves={};logs=[]
 for thr in THRESHOLDS:
  name='NoStop' if thr is None else f'CatStop_{int(abs(thr)*100)}pct';eq=1.;curve={p.index[0]:1.};prev={'BTC':1.};stops=0
  for i in range(len(p)-1):
   d,n=p.index[i],p.index[i+1];pick=chosen[i];target={'BTC':.7 if pick else 1.}
   if pick:target[pick[0]]=.3
   turn=sum(abs(target.get(k,0)-prev.get(k,0)) for k in set(target)|set(prev));eq*=max(0,1-turn*COST)
   vals={};gross=0;valid=True;hit=False
   for a,w in target.items():
    z=sleeve(a,d,n,daily,thr)
    if z is None:valid=False;break
    ret,h=z;hit|=h;v=w*(1+ret);gross+=v;vals['BTC' if h else a]=vals.get('BTC' if h else a,0)+v
   if not valid:
    gross=0;vals={}
    for a,w in target.items():
     ret=p.at[n,a]/p.at[d,a]-1 if a in p.columns and pd.notna(p.at[n,a]) and pd.notna(p.at[d,a]) and p.at[d,a]>0 else -1
     v=w*(1+ret);gross+=v;vals[a]=v
    hit=False
   if hit:stops+=1
   eq*=max(0,gross);curve[n]=eq;den=sum(vals.values());prev={a:v/den for a,v in vals.items() if v>0} if den>0 else {'BTC':1.}
   logs.append({'strategy':name,'date':d,'next':n,'selected':';'.join(pick),'catastrophe_stop_hit':hit,'equity':eq})
  m=metrics(curve,stops);m['strategy']=name;rows.append(m);curves[name]=pd.Series(curve)
 res=pd.DataFrame(rows).sort_values('cagr_pct',ascending=False);res.to_csv(OUT/'results.csv',index=False);pd.concat(curves,axis=1).to_csv(OUT/'equity.csv');pd.DataFrame(logs).to_csv(OUT/'trade_log.csv',index=False)
 (OUT/'methodology.json').write_text(json.dumps({'turbo_weight':T,'thresholds':THRESHOLDS,'cost_per_side':COST,'rule':'Top1 point-in-time unchanged; catastrophe stop is fixed percent below entry, checked on daily lows; after stop rotate to BTC next daily open; no re-entry until next monthly signal.'},indent=2))
 print(res.to_string(index=False))
if __name__=='__main__':main()
