from __future__ import annotations

import json, math, re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from mscidata import msci

import world_top5_long_backtest as bt
import world_top5_long_backtest_wiki as top5src

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'topn_long'
OUT.mkdir(parents=True,exist_ok=True)
END='2026-08-14'
MONTHLY=100.0
COST=0.001

INDEXES={
    'MSCI_World':'990100',
    'MSCI_World_30':'750288',
    'MSCI_World_40':'750290',
    'MSCI_World_50':'750292',
    'MSCI_World_MegaCap_18Capped':'761936',
}

DEVELOPED={'USA','UK','France','Germany','Switzerland','Japan','Canada','Australia','Netherlands','Denmark','Sweden','Norway','Finland','Belgium','Austria','Ireland','Spain','Italy','Portugal','Israel','New Zealand','Singapore','Hong Kong'}
DATE_BY_SOURCE_YEAR={2023:'2023-12-29',2024:'2024-12-31',2025:'2025-12-31'}
HEAD={'User-Agent':'Mozilla/5.0'}


def index_series(code):
    df=msci.get_levels(code,'2000-12-29',END,variant='NETR')
    if df is None or len(df)==0: raise RuntimeError(f'No MSCI data for {code}')
    df=df.copy(); df['DATE']=pd.to_datetime(df['DATE'],errors='coerce'); df['LEVEL']=pd.to_numeric(df['LEVEL'],errors='coerce')
    s=df.dropna(subset=['DATE','LEVEL']).set_index('DATE')['LEVEL'].sort_index()
    s.index=pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep='last')].astype(float)


def stats(e):
    e=e.dropna(); years=(e.index[-1]-e.index[0]).days/365.25
    cagr=(e.iloc[-1]/e.iloc[0])**(1/years)-1
    r=e.pct_change().fillna(0); dd=e/e.cummax()-1; vol=r.std(ddof=0)*math.sqrt(252); ann=r.mean()*252; mdd=float(dd.min())
    annual=(1+r).groupby(r.index.year).prod()-1
    return {'cagr_pct':100*cagr,'max_drawdown_pct':100*mdd,'sharpe':ann/vol if vol else np.nan,'calmar':cagr/abs(mdd) if mdd<0 else np.nan,'final_value_10000':10000*float(e.iloc[-1]/e.iloc[0]),'worst_year_pct':100*float(annual.min()),'best_year_pct':100*float(annual.max())}


def xnpv(rate,cf):
    t0=cf[0][0]; return sum(v/((1+rate)**(((d-t0).days)/365.25)) for d,v in cf)

def xirr(cf):
    lo,hi=-.9999,10.; flo,fhi=xnpv(lo,cf),xnpv(hi,cf)
    while flo*fhi>0 and hi<1e6: hi*=2; fhi=xnpv(hi,cf)
    if flo*fhi>0:return np.nan
    for _ in range(250):
        mid=(lo+hi)/2; fm=xnpv(mid,cf)
        if flo*fm<=0:hi=mid
        else:lo=mid;flo=fm
    return (lo+hi)/2


def dca_single(s):
    s=s.dropna(); p=s.index.to_period('M'); months=[s[p==m].index[0] for m in p.unique()]
    units=0.; contributed=0.; cf=[]
    for d in months:
        units += MONTHLY/float(s.at[d]); contributed+=MONTHLY; cf.append((d,-MONTHLY))
    final=units*float(s.iloc[-1]); cf.append((s.index[-1],final))
    return 100*xirr(cf),contributed,final,len(months)


def normalize(s,cal):
    x=s.reindex(cal).ffill().dropna()
    return x/x.iloc[0]


def parse_cap(txt):
    m=re.search(r'\$\s*([0-9,.]+)\s*([TBM])',txt)
    if not m:return np.nan
    v=float(m.group(1).replace(',','')); mult={'T':1e12,'B':1e9,'M':1e6}[m.group(2)]
    return v*mult


def fetch_developed_ranking(source_year,n=20):
    ds=DATE_BY_SOURCE_YEAR[source_year]
    url=f'https://companiesmarketcap.com/time-machine/{ds}/'
    r=requests.get(url,headers=HEAD,timeout=30,allow_redirects=True); r.raise_for_status()
    if ds not in r.url:
        raise RuntimeError(f'Time-machine redirected for {ds}: {r.url}')
    soup=BeautifulSoup(r.text,'html.parser'); rows=[]
    for tr in soup.select('table tbody tr'):
        tds=tr.find_all('td')
        if len(tds)<7: continue
        ranktxt=tds[1].get_text(' ',strip=True) if len(tds)>1 else ''
        nametxt=tds[2].get_text(' ',strip=True) if len(tds)>2 else ''
        captxt=tds[3].get_text(' ',strip=True) if len(tds)>3 else ''
        country=tds[-1].get_text(' ',strip=True).replace('🇺🇸 ','').replace('🇬🇧 ','').strip()
        # flag emoji varies, so retain only trailing alphabetic country label after emoji.
        country=re.sub(r'^[^A-Za-z]+','',country).strip()
        if country not in DEVELOPED: continue
        try: rank=int(re.sub(r'\D','',ranktxt))
        except: continue
        cap=parse_cap(captxt)
        # ticker is the final whitespace-delimited token in the name cell.
        ticker=nametxt.split()[-1] if nametxt else ''
        if not ticker or not math.isfinite(cap): continue
        rows.append({'source_year':source_year,'holding_year':source_year+1,'global_rank':rank,'name_ticker':nametxt,'ticker':ticker,'market_cap':cap,'country':country})
        if len(rows)>=n: break
    if len(rows)<n: raise RuntimeError(f'Only {len(rows)} developed names for {source_year}')
    total=sum(x['market_cap'] for x in rows[:n])
    for i,x in enumerate(rows[:n],1): x['rank_within_developed']=i; x['weight']=x['market_cap']/total
    return rows[:n]


def download_adj(t):
    x=yf.download(t,start='2023-12-01',end=END,auto_adjust=False,progress=False,threads=False,timeout=30)
    if x is None or x.empty:return None
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    col='Adj Close' if 'Adj Close' in x.columns else 'Close'
    s=x[col].astype(float).dropna(); s.index=pd.to_datetime(s.index).tz_localize(None); return s


def recent_topn_curves(world):
    ranking20=[]
    for y in [2023,2024,2025]: ranking20 += fetch_developed_ranking(y,20)
    rdf=pd.DataFrame(ranking20); rdf.to_csv(OUT/'recent_rankings_top20.csv',index=False)
    tickers=sorted(set(rdf.ticker)); px={}
    for t in tickers:
        s=download_adj(t)
        if s is None or s.empty: raise RuntimeError(f'Yahoo missing {t}')
        px[t]=s
    cal=world.index[(world.index>=pd.Timestamp('2024-01-02')) & (world.index<pd.Timestamp(END))]
    mat=pd.DataFrame({t:px[t].reindex(cal).ffill() for t in tickers},index=cal)
    curves={}; dca={}
    for n in [5,10,20]:
        holdings={}; cash=10000.; eq={}; cur=None; target={}
        units_d={}; cash_d=0.; eq_d={}; cf=[]; pmon=cal.to_period('M'); monthdays={cal[pmon==m][0] for m in pmon.unique()}
        for d in cal:
            if d.year!=cur:
                cur=d.year; g=rdf[rdf.holding_year==cur].sort_values('rank_within_developed').head(n).copy()
                if len(g)!=n: raise RuntimeError(f'No Top{n} for {cur}')
                total=g.market_cap.sum(); target={r.ticker:float(r.market_cap/total) for _,r in g.iterrows()}
                val=cash+sum(u*float(mat.at[d,t]) for t,u in holdings.items())
                current={t:holdings.get(t,0)*float(mat.at[d,t]) for t in set(holdings)|set(target)}
                desired={t:val*w for t,w in target.items()}; turnover=.5*sum(abs(desired.get(t,0)-current.get(t,0)) for t in set(current)|set(desired)); val-=turnover*COST
                holdings={t:val*w/float(mat.at[d,t]) for t,w in target.items()}; cash=0.
                # DCA portfolio also rebalanced annually before January contribution.
                vd=cash_d+sum(u*float(mat.at[d,t]) for t,u in units_d.items())
                if vd>0:
                    currentd={t:units_d.get(t,0)*float(mat.at[d,t]) for t in set(units_d)|set(target)}; desiredd={t:vd*w for t,w in target.items()}; tod=.5*sum(abs(desiredd.get(t,0)-currentd.get(t,0)) for t in set(currentd)|set(desiredd)); vd-=tod*COST
                    units_d={t:vd*w/float(mat.at[d,t]) for t,w in target.items()}; cash_d=0.
            if d in monthdays:
                cash_d += MONTHLY; cf.append((d,-MONTHLY)); amt=MONTHLY*(1-COST); cash_d-=MONTHLY
                for t,w in target.items(): units_d[t]=units_d.get(t,0)+amt*w/float(mat.at[d,t])
            eq[d]=sum(u*float(mat.at[d,t]) for t,u in holdings.items())
            eq_d[d]=cash_d+sum(u*float(mat.at[d,t]) for t,u in units_d.items())
        e=pd.Series(eq,dtype=float); de=pd.Series(eq_d,dtype=float); cf.append((de.index[-1],float(de.iloc[-1])))
        curves[f'Exact_recent_Top{n}']=e; dca[f'Exact_recent_Top{n}']=(100*xirr(cf),-sum(v for _,v in cf if v<0),float(de.iloc[-1]),len(monthdays))
    return curves,dca


def main():
    # Long-horizon official MSCI concentration ladder.
    raw={k:index_series(v) for k,v in INDEXES.items()}
    start=max(s.index.min() for s in raw.values()); end=min(s.index.max() for s in raw.values())
    cal=raw['MSCI_World'].index[(raw['MSCI_World'].index>=start)&(raw['MSCI_World'].index<=end)]
    curves={k:10000*normalize(s,cal) for k,s in raw.items()}

    # Re-run the exact same historical Top-5 reconstruction on the common official-index window.
    ranks=top5src.frozen_rankings(); c5,mat,w=top5src.build_prices(ranks); tg=bt.targets(ranks)
    e5,_,_=bt.simulate(1.0,mat,w,tg,False); d5,cf5,_=bt.simulate(1.0,mat,w,tg,True)
    e5=e5.reindex(cal).ffill().dropna(); curves['Historical_Top5_reconstruction']=10000*e5/e5.iloc[0]

    rows=[]
    for name,e in curves.items():
        st=stats(e)
        if name=='Historical_Top5_reconstruction':
            # Recompute DCA on the common start rather than reuse the earlier 2000 cash flows.
            # Use the normalized Top-5 equity curve as a synthetic investable index for like-for-like DCA timing.
            dx,dc,df,dm=dca_single(e)
        else: dx,dc,df,dm=dca_single(e)
        st.update({'strategy':name,'start':str(e.index[0].date()),'end':str(e.index[-1].date()),'dca_xirr_pct':dx,'dca_contributed':dc,'dca_final_value':df,'dca_months':dm})
        rows.append(st)

    # Exact point-in-time Top 5/10/20 recent subtest from CompaniesMarketCap.
    recent,drecent=recent_topn_curves(raw['MSCI_World'])
    for name,e in recent.items():
        st=stats(e); dx,dc,df,dm=drecent[name]; st.update({'strategy':name,'start':str(e.index[0].date()),'end':str(e.index[-1].date()),'dca_xirr_pct':dx,'dca_contributed':dc,'dca_final_value':df,'dca_months':dm}); rows.append(st)

    res=pd.DataFrame(rows); res.to_csv(OUT/'results.csv',index=False)
    pd.DataFrame({k:v for k,v in curves.items()}).to_csv(OUT/'long_equity.csv')
    pd.DataFrame(recent).to_csv(OUT/'recent_equity.csv')
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'long_test':'Official MSCI World/World30/World40/World50/MegaCap NETR plus prior Top5 reconstruction on common 2001-2026 window','recent_exact_test':'CompaniesMarketCap point-in-time developed-market Top5/10/20, annual rebalance, 2024-2026','limitations':['No reliable public point-in-time ranks 11-20 for the full 2001-2026 period, so no fabricated long Top20 series.','Historical Top5 is the previously documented reconstruction and includes its known NTT Docomo 2001 proxy.','Recent CompaniesMarketCap test is short and should not be treated as long-run evidence.','10 bps one-way turnover/contribution implementation cost used for recent direct portfolios; official index series are NETR index levels before ETF TER.','Historical results are not forecasts.'],'results':res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict('records')}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(res.to_string(index=False))
    print('\nRECENT RANKINGS\n',pd.read_csv(OUT/'recent_rankings_top20.csv').to_string(index=False))

if __name__=='__main__': main()
