from __future__ import annotations
import io, json, math, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'btc_turbo_pit_v2'; OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2018-01-07')
TODAY=pd.Timestamp.now('UTC').tz_localize(None).normalize()
END=TODAY-pd.Timedelta(days=(TODAY.weekday()-6)%7)
BTC='BTC'; CASH='__CASH__'; COST=0.0023
UNIVERSE_N=15; PRICE_DEPTH=20
EXCLUDE={'USDT','USDC','BUSD','DAI','TUSD','USDP','PAX','GUSD','USDD','FDUSD','USDE','FRAX','PYUSD','UST','USTC','EURT','EURC','SUSD','LUSD','USDS','WBTC','WETH','STETH','WSTETH','RETH','CBETH','WEETH'}
HEADERS={'User-Agent':'Mozilla/5.0 AppleWebKit/537.36 Chrome/128 Safari/537.36','Accept-Language':'en-US,en;q=0.9'}
VARIANTS=[
 ('BTC_HOLD',0,0,0,0),
 ('PIT_Top2_Turbo30_rel5',2,.30,.05,0),
 ('PIT_Top2_Turbo50_rel5',2,.50,.05,0),
 ('PIT_Top2_Turbo75_rel5',2,.75,.05,0),
 ('PIT_Top1_Turbo50_rel5',1,.50,.05,0),
 ('PIT_Top2_Turbo50_rel10',2,.50,.10,0),
 ('PIT_Top2_Turbo50_rel5_RiskOff50',2,.50,.05,.50),
 ('PIT_Top2_Turbo75_rel5_RiskOff50',2,.75,.05,.50),
]

def dates(): return list(pd.date_range(START,END,freq='7D'))
def num(x):
 s=str(x).replace('$','').replace(',','').strip(); m=re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',s)
 return float(m.group()) if m else np.nan

def fetch(dt):
 url=f'https://coinmarketcap.com/historical/{dt:%Y%m%d}/'; err=''
 for a in range(4):
  try:
   r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status()
   table=None
   for t in pd.read_html(io.StringIO(r.text)):
    cc=[str(c).strip() for c in t.columns]
    if 'Rank' in cc and 'Symbol' in cc and 'Price' in cc: table=t.copy(); break
   if table is None: raise RuntimeError('ranking table missing')
   table.columns=[str(c).strip() for c in table.columns]
   z=table[['Rank','Name','Symbol','Price']].copy()
   z['Rank']=pd.to_numeric(z['Rank'].astype(str).str.extract(r'(\d+)')[0],errors='coerce')
   z['Symbol']=z['Symbol'].astype(str).str.upper().str.strip()
   z['Price']=z['Price'].map(num)
   z=z.dropna(subset=['Rank','Symbol','Price']); z=z[(z.Rank>=1)&(z.Rank<=PRICE_DEPTH)]
   z=z.drop_duplicates('Symbol').sort_values('Rank'); z['date']=dt
   if len(z)<10: raise RuntimeError(f'only {len(z)} usable rows')
   return z,None
  except Exception as e: err=str(e); time.sleep(.5*(a+1))
 return None,err

def collect():
 rows=[]; fail=[]
 with ThreadPoolExecutor(max_workers=5) as ex:
  fs={ex.submit(fetch,d):d for d in dates()}
  for f in as_completed(fs):
   d=fs[f]; z,e=f.result()
   if z is None: fail.append({'date':d,'error':e}); print('FAIL',d.date(),e)
   else: rows.append(z); print('OK',d.date(),len(z))
 if not rows: raise RuntimeError('no snapshots')
 s=pd.concat(rows,ignore_index=True).sort_values(['date','Rank']); s.to_csv(OUT/'snapshots.csv',index=False)
 pd.DataFrame(fail,columns=['date','error']).to_csv(OUT/'failures.csv',index=False)
 return s

def panels(s):
 p=s.pivot_table(index='date',columns='Symbol',values='Price',aggfunc='first').sort_index()
 r=s.pivot_table(index='date',columns='Symbol',values='Rank',aggfunc='first').sort_index()
 return p,r

def mom(p,days):
 q=p.copy(); q.index=q.index+pd.Timedelta(days=days); return p/q.reindex(p.index)-1

def target(dt,p,r,m4,m8,cfg):
 name,n,aw,rel,riskoff=cfg
 if name=='BTC_HOLD': return {BTC:1.0},''
 b4=m4.at[dt,BTC] if BTC in m4 else np.nan; b8=m8.at[dt,BTC] if BTC in m8 else np.nan
 if pd.isna(b4) or pd.isna(b8): return {BTC:1.0},''
 if b4<0 and b8<0 and riskoff>0: return {BTC:1-riskoff,CASH:riskoff},''
 if b4<=0 or b8<=0: return {BTC:1.0},''
 cand=[]
 for s,rank in r.loc[dt].dropna().items():
  if s==BTC or s in EXCLUDE or rank>UNIVERSE_N: continue
  a=m4.at[dt,s] if s in m4 else np.nan; b=m8.at[dt,s] if s in m8 else np.nan
  if pd.isna(a) or pd.isna(b) or a<=0 or b<=0 or a-b4<rel: continue
  cand.append((.6*a+.4*b,-rank,s))
 cand.sort(reverse=True); chosen=[x[2] for x in cand[:n]]
 if not chosen: return {BTC:1.0},''
 w={BTC:1-aw}; each=aw/len(chosen)
 for s in chosen:w[s]=each
 return w,';'.join(chosen)

def ret(p,d,nxt,s):
 if s==CASH:return 0.0
 if s not in p:return -1.0
 a=p.at[d,s]; b=p.at[nxt,s]
 if pd.isna(a) or a<=0 or pd.isna(b) or b<=0:return -1.0
 return float(b/a-1)

def sim(p,r,cfg):
 m4,m8=mom(p,28),mom(p,56); ds=p.index
 ds=ds[(ds>=START)&(ds<=END)]; ds=ds[[pd.notna(m8.at[d,BTC]) for d in ds]]
 eq=1.0; curve={ds[0]:1.0}; pre={CASH:1.0}; logs=[]
 for i in range(len(ds)-1):
  d,nxt=ds[i],ds[i+1]; w,sel=target(d,p,r,m4,m8,cfg)
  risky=(set(pre)|set(w))-{CASH}; turnover=sum(abs(w.get(s,0)-pre.get(s,0)) for s in risky)
  eq*=max(0,1-turnover*COST)
  rr={s:ret(p,d,nxt,s) for s in w}; pr=sum(w[s]*rr[s] for s in w); eq*=max(0,1+pr); curve[nxt]=eq
  endw={s:w[s]*(1+rr[s]) for s in w}; den=sum(endw.values()); pre={s:v/den for s,v in endw.items() if v>0} if den>0 else {CASH:1.0}
  b4=m4.at[d,BTC]; b8=m8.at[d,BTC]
  logs.append({'strategy':cfg[0],'date':d,'next':nxt,'selected':sel,'btc_mom4':b4,'btc_mom8':b8,'btc_weight':w.get(BTC,0),'alt_weight':1-w.get(BTC,0)-w.get(CASH,0),'cash_weight':w.get(CASH,0),'turnover':turnover,'period_return':pr})
 return pd.Series(curve,name=cfg[0]),pd.DataFrame(logs)

def metric(name,eq,lg):
 x=eq.pct_change().dropna(); yrs=(eq.index[-1]-eq.index[0]).days/365.25; cagr=eq.iloc[-1]**(1/yrs)-1; dd=eq/eq.cummax()-1
 vol=x.std(ddof=0)*math.sqrt(52); sh=x.mean()*52/vol if vol>0 else np.nan
 ann={}
 for y,g in eq.groupby(eq.index.year): ann[y]=g.iloc[-1]/g.iloc[0]-1
 ann=pd.Series(ann); roll=eq/eq.shift(52)-1
 return {'strategy':name,'cagr_pct':100*cagr,'max_drawdown_pct':100*dd.min(),'sharpe_0rf':sh,'volatility_pct':100*vol,'final_multiple':eq.iloc[-1],'best_year_pct':100*ann.max(),'worst_year_pct':100*ann.min(),'years_ge_50':int((ann>=.5).sum()),'years_ge_100':int((ann>=1).sum()),'best_52w_pct':100*roll.max(),'rolling52_windows_ge_100':int((roll>=1).sum()),'annual_turnover_x':lg.turnover.sum()/yrs,'weeks_alt_pct':100*(lg.alt_weight>0).mean()},ann*100

def main():
 s=collect(); p,r=panels(s)
 if BTC not in p: raise RuntimeError('BTC missing')
 rows=[]; anns={}; curves=[]; logs=[]
 for cfg in VARIANTS:
  eq,lg=sim(p,r,cfg); m,a=metric(cfg[0],eq,lg); rows.append(m); anns[cfg[0]]=a; curves.append(eq); logs.append(lg)
 res=pd.DataFrame(rows).sort_values('cagr_pct',ascending=False); annual=pd.DataFrame(anns); equity=pd.concat(curves,axis=1); log=pd.concat(logs,ignore_index=True)
 res.to_csv(OUT/'results.csv',index=False); annual.to_csv(OUT/'annual.csv'); equity.to_csv(OUT/'equity.csv'); log.to_csv(OUT/'selection_log.csv',index=False)
 (OUT/'results.json').write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'source':'CoinMarketCap weekly historical snapshots','universe_top_n':UNIVERSE_N,'price_depth':PRICE_DEPTH,'missing_next_price':'-100% return','cost_one_way_pct':100*COST,'snapshots_requested':len(dates()),'snapshots_collected':int(s.date.nunique()),'results':res.replace({np.nan:None}).to_dict('records')},indent=2))
 print('\nRESULTS\n'+res.to_string(index=False)); print('\nANNUAL\n'+annual.to_string()); print('snapshots',len(dates()),s.date.nunique())
if __name__=='__main__':main()
