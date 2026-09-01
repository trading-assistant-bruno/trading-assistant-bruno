from __future__ import annotations
import io, json, math, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'btc_turbo_pit_monthly'; OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2018-01-07'); TODAY=pd.Timestamp.now('UTC').tz_localize(None).normalize()
BTC='BTC'; CASH='__CASH__'; COST=0.0023; UNIVERSE_N=15; PRICE_DEPTH=20
EXCLUDE={'USDT','USDC','BUSD','DAI','TUSD','USDP','PAX','GUSD','USDD','FDUSD','USDE','FRAX','PYUSD','UST','USTC','EURT','EURC','SUSD','LUSD','USDS','WBTC','WETH','STETH','WSTETH','RETH','CBETH','WEETH'}
HEADERS={'User-Agent':'Mozilla/5.0 AppleWebKit/537.36 Chrome/128 Safari/537.36','Accept-Language':'en-US,en;q=0.9'}
VARIANTS=[('BTC_HOLD',0,0,0,0),('PIT_M_Top2_T30_rel5',2,.30,.05,0),('PIT_M_Top2_T50_rel5',2,.50,.05,0),('PIT_M_Top2_T75_rel5',2,.75,.05,0),('PIT_M_Top1_T50_rel5',1,.50,.05,0),('PIT_M_Top2_T50_rel10',2,.50,.10,0),('PIT_M_Top2_T50_rel5_RO50',2,.50,.05,.50),('PIT_M_Top2_T75_rel5_RO50',2,.75,.05,.50)]

def first_sundays():
 out=[]; cur=pd.Timestamp(START.year,START.month,1)
 while cur<=TODAY:
  d=cur+pd.Timedelta(days=(6-cur.weekday())%7)
  if d>=START and d<=TODAY: out.append(d)
  cur=(cur+pd.offsets.MonthBegin(1)).normalize()
 return out

def num(x):
 s=str(x).replace('$','').replace(',','').strip(); m=re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',s); return float(m.group()) if m else np.nan

def fetch(dt):
 url=f'https://coinmarketcap.com/historical/{dt:%Y%m%d}/'; err=''
 for a in range(3):
  try:
   r=requests.get(url,headers=HEADERS,timeout=25); r.raise_for_status(); table=None
   for t in pd.read_html(io.StringIO(r.text)):
    cols=[str(c).strip() for c in t.columns]
    if 'Rank' in cols and 'Symbol' in cols and 'Price' in cols: table=t.copy(); break
   if table is None: raise RuntimeError('ranking table missing')
   table.columns=[str(c).strip() for c in table.columns]; z=table[['Rank','Name','Symbol','Price']].copy()
   z['Rank']=pd.to_numeric(z['Rank'].astype(str).str.extract(r'(\d+)')[0],errors='coerce'); z['Symbol']=z['Symbol'].astype(str).str.upper().str.strip(); z['Price']=z['Price'].map(num)
   z=z.dropna(subset=['Rank','Symbol','Price']); z=z[(z.Rank>=1)&(z.Rank<=PRICE_DEPTH)].drop_duplicates('Symbol').sort_values('Rank'); z['date']=dt
   if len(z)<10: raise RuntimeError(f'only {len(z)} rows')
   return z,None
  except Exception as e: err=str(e); time.sleep(.6*(a+1))
 return None,err

def collect():
 rows=[]; fail=[]; ds=first_sundays()
 with ThreadPoolExecutor(max_workers=3) as ex:
  fs={ex.submit(fetch,d):d for d in ds}
  for f in as_completed(fs):
   d=fs[f]; z,e=f.result()
   if z is None: fail.append({'date':d,'error':e}); print('FAIL',d.date(),e)
   else: rows.append(z); print('OK',d.date(),len(z))
 if not rows: raise RuntimeError('no snapshots')
 s=pd.concat(rows,ignore_index=True).sort_values(['date','Rank']); s.to_csv(OUT/'snapshots.csv',index=False); pd.DataFrame(fail,columns=['date','error']).to_csv(OUT/'failures.csv',index=False); return s,len(ds)

def panels(s): return s.pivot_table(index='date',columns='Symbol',values='Price',aggfunc='first').sort_index(),s.pivot_table(index='date',columns='Symbol',values='Rank',aggfunc='first').sort_index()

def target(i,p,r,cfg):
 name,n,aw,rel,ro=cfg; d=p.index[i]
 if name=='BTC_HOLD':return {BTC:1.0},''
 if i<2:return {BTC:1.0},''
 b1=p.at[d,BTC]/p.at[p.index[i-1],BTC]-1 if pd.notna(p.at[p.index[i-1],BTC]) else np.nan
 b2=p.at[d,BTC]/p.at[p.index[i-2],BTC]-1 if pd.notna(p.at[p.index[i-2],BTC]) else np.nan
 if pd.isna(b1) or pd.isna(b2):return {BTC:1.0},''
 if b1<0 and b2<0 and ro>0:return {BTC:1-ro,CASH:ro},''
 if b1<=0 or b2<=0:return {BTC:1.0},''
 cand=[]
 for s,rank in r.loc[d].dropna().items():
  if s==BTC or s in EXCLUDE or rank>UNIVERSE_N or s not in p.columns:continue
  a0=p.at[d,s]; a1=p.at[p.index[i-1],s]; a2=p.at[p.index[i-2],s]
  if pd.isna(a0) or pd.isna(a1) or pd.isna(a2) or a1<=0 or a2<=0:continue
  m1=a0/a1-1; m2=a0/a2-1
  if m1<=0 or m2<=0 or m1-b1<rel:continue
  cand.append((.6*m1+.4*m2,-rank,s))
 cand.sort(reverse=True); chosen=[x[2] for x in cand[:n]]
 if not chosen:return {BTC:1.0},''
 w={BTC:1-aw}; each=aw/len(chosen)
 for s in chosen:w[s]=each
 return w,';'.join(chosen)

def aret(p,i,s):
 if s==CASH:return 0.0
 a=p.at[p.index[i],s] if s in p.columns else np.nan; b=p.at[p.index[i+1],s] if s in p.columns else np.nan
 if pd.isna(a) or a<=0 or pd.isna(b) or b<=0:return -1.0
 return float(b/a-1)

def sim(p,r,cfg):
 ds=p.index; eq=1.; curve={ds[0]:1.}; pre={CASH:1.}; logs=[]
 for i in range(len(ds)-1):
  d,nxt=ds[i],ds[i+1]; w,sel=target(i,p,r,cfg); risky=(set(pre)|set(w))-{CASH}; turn=sum(abs(w.get(s,0)-pre.get(s,0)) for s in risky); eq*=max(0,1-turn*COST)
  rr={s:aret(p,i,s) for s in w}; pr=sum(w[s]*rr[s] for s in w); eq*=max(0,1+pr); curve[nxt]=eq
  endw={s:w[s]*(1+rr[s]) for s in w}; den=sum(endw.values()); pre={s:v/den for s,v in endw.items() if v>0} if den>0 else {CASH:1.}
  logs.append({'strategy':cfg[0],'date':d,'next':nxt,'selected':sel,'btc_weight':w.get(BTC,0),'alt_weight':1-w.get(BTC,0)-w.get(CASH,0),'cash_weight':w.get(CASH,0),'turnover':turn,'period_return':pr})
 return pd.Series(curve,name=cfg[0]),pd.DataFrame(logs)

def metrics(name,eq,lg):
 x=eq.pct_change().dropna(); yrs=(eq.index[-1]-eq.index[0]).days/365.25; cagr=eq.iloc[-1]**(1/yrs)-1; dd=eq/eq.cummax()-1; vol=x.std(ddof=0)*math.sqrt(12); sh=x.mean()*12/vol if vol>0 else np.nan
 ann={y:g.iloc[-1]/g.iloc[0]-1 for y,g in eq.groupby(eq.index.year)}; ann=pd.Series(ann); roll=eq/eq.shift(12)-1
 return {'strategy':name,'cagr_pct':100*cagr,'max_drawdown_pct':100*dd.min(),'sharpe_0rf':sh,'volatility_pct':100*vol,'final_multiple':eq.iloc[-1],'best_year_pct':100*ann.max(),'worst_year_pct':100*ann.min(),'years_ge_50':int((ann>=.5).sum()),'years_ge_100':int((ann>=1).sum()),'best_12m_pct':100*roll.max(),'rolling12_windows_ge_100':int((roll>=1).sum()),'annual_turnover_x':lg.turnover.sum()/yrs,'months_alt_pct':100*(lg.alt_weight>0).mean()},ann*100

def main():
 s,requested=collect(); p,r=panels(s)
 if BTC not in p:raise RuntimeError('BTC missing')
 rows=[]; anns={}; curves=[]; logs=[]
 for cfg in VARIANTS:
  eq,lg=sim(p,r,cfg); m,a=metrics(cfg[0],eq,lg); rows.append(m); anns[cfg[0]]=a; curves.append(eq); logs.append(lg)
 res=pd.DataFrame(rows).sort_values('cagr_pct',ascending=False); annual=pd.DataFrame(anns); equity=pd.concat(curves,axis=1); log=pd.concat(logs,ignore_index=True)
 res.to_csv(OUT/'results.csv',index=False); annual.to_csv(OUT/'annual.csv'); equity.to_csv(OUT/'equity.csv'); log.to_csv(OUT/'selection_log.csv',index=False)
 (OUT/'results.json').write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'source':'CoinMarketCap historical monthly snapshots','universe_top_n':UNIVERSE_N,'price_depth':PRICE_DEPTH,'missing_next_price':'-100%','cost_one_way_pct':100*COST,'snapshots_requested':requested,'snapshots_collected':int(s.date.nunique()),'results':res.replace({np.nan:None}).to_dict('records')},indent=2))
 print('\nRESULTS\n'+res.to_string(index=False)); print('\nANNUAL\n'+annual.to_string()); print('snapshots',requested,s.date.nunique())
if __name__=='__main__':main()
