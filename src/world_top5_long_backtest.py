from __future__ import annotations

import json, math, re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'world_top5_long'
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2000-01-03')
END_EXCLUSIVE = '2026-08-14'
INITIAL_CAPITAL = 10_000.0
MONTHLY_CONTRIBUTION = 100.0
TURNOVER_COST = 0.001
WORLD = '^990100-USD-NETR'
FT_MIRROR = 'https://www.olps.co.za/kiwix/content/wikipedia_en_all_maxi_2024-01/A/List_of_public_corporations_by_market_capitalization'

# Only developed-market companies that plausibly appear in the FT global top ten.
# Foreign names use USD ADRs when practical; NTT Docomo uses Tokyo local shares + USDJPY conversion.
ALIASES = {
    'Microsoft': ('MSFT','USD'), 'General Electric': ('GE','USD'), 'GE': ('GE','USD'),
    'Cisco Systems': ('CSCO','USD'), 'Cisco': ('CSCO','USD'),
    'Walmart': ('WMT','USD'), 'Wal-Mart': ('WMT','USD'), 'Wal-Mart Stores': ('WMT','USD'),
    'ExxonMobil': ('XOM','USD'), 'Exxon Mobil': ('XOM','USD'),
    'Intel': ('INTC','USD'), 'Intel Corporation': ('INTC','USD'),
    'Pfizer': ('PFE','USD'), 'Citigroup': ('C','USD'), 'BP': ('BP','USD'),
    'Johnson & Johnson': ('JNJ','USD'), 'Bank of America': ('BAC','USD'),
    'American International Group': ('AIG','USD'), 'AIG': ('AIG','USD'),
    'Toyota': ('TM','USD'), 'Toyota Motor': ('TM','USD'),
    'AT&T': ('T','USD'), 'IBM': ('IBM','USD'), 'Chevron': ('CVX','USD'), 'Chevron Corporation': ('CVX','USD'),
    'BHP Billiton': ('BHP','USD'), 'BHP': ('BHP','USD'),
    'Apple': ('AAPL','USD'), 'Google': ('GOOGL','USD'), 'Alphabet': ('GOOGL','USD'),
    'Amazon': ('AMZN','USD'), 'Facebook': ('META','USD'), 'Meta Platforms': ('META','USD'),
    'Berkshire Hathaway': ('BRK-B','USD'), 'NVIDIA': ('NVDA','USD'), 'Nvidia': ('NVDA','USD'),
    'Tesla': ('TSLA','USD'), 'Visa': ('V','USD'), 'JPMorgan Chase': ('JPM','USD'),
    'Procter & Gamble': ('PG','USD'), 'Oracle': ('ORCL','USD'), 'Home Depot': ('HD','USD'),
    'Coca-Cola': ('KO','USD'), 'The Coca-Cola Company': ('KO','USD'), 'Merck': ('MRK','USD'),
    'HSBC': ('HSBC','USD'), 'Vodafone': ('VOD','USD'), 'Nokia': ('NOK','USD'),
    'NTT Docomo': ('9437.T','JPY'), 'NTT DoCoMo': ('9437.T','JPY'),
    'Nippon Telegraph & Telephone': ('9432.T','JPY'),
    'Deutsche Telekom': ('DTEGY','USD'), 'Nestlé': ('NSRGY','USD'), 'Nestle': ('NSRGY','USD'),
    'Novartis': ('NVS','USD'), 'Roche': ('RHHBY','USD'), 'SAP': ('SAP','USD'), 'ASML': ('ASML','USD'),
    'LVMH': ('LVMUY','USD'), 'Eli Lilly': ('LLY','USD'), 'Broadcom': ('AVGO','USD'),
    'Royal Dutch Shell': ('SHEL','USD'), 'Shell': ('SHEL','USD'),
}

# Validated 2017-2025 year-end rankings from the prior experiment.
FROZEN = ROOT / 'data' / 'world_top5_reference_rankings.csv'


def num(text: str) -> float:
    vals = re.findall(r'\d[\d,.]*', text.replace('\xa0',' '))
    if not vals: return float('nan')
    return float(vals[-1].replace(',',''))


def identify(text: str):
    low = text.lower()
    for alias in sorted(ALIASES, key=len, reverse=True):
        if alias.lower() in low:
            ticker, ccy = ALIASES[alias]
            return alias, ticker, ccy
    return None


def parse_ft_year(year: int):
    html = requests.get(FT_MIRROR, timeout=30, headers={'User-Agent':'Mozilla/5.0'}).text
    soup = BeautifulSoup(html, 'html.parser')
    heading = None
    for h in soup.find_all(['h2','h3','h4']):
        if h.get_text(' ', strip=True) == str(year):
            heading = h; break
    if heading is None: raise RuntimeError(f'FT heading missing {year}')
    table = heading.find_next('table')
    rows = table.find_all('tr')
    header = [x.get_text(' ',strip=True) for x in rows[0].find_all(['th','td'])]
    quarterly = any('Fourth quarter' in x for x in header)
    ranked=[]
    for tr in rows[1:]:
        cells=[x.get_text(' ',strip=True) for x in tr.find_all(['th','td'])]
        if not cells: continue
        try: rank=int(re.sub(r'\D','',cells[0]))
        except: continue
        target = cells[-1] if quarterly else (cells[1]+' '+cells[-1])
        hit=identify(target)
        if not hit: continue  # emerging-market or unmapped company
        alias,ticker,ccy=hit
        cap=num(target if quarterly else cells[-1])
        if math.isfinite(cap) and cap>0:
            ranked.append((rank,alias,ticker,ccy,cap))
    ranked=sorted(ranked,key=lambda x:x[0])[:5]
    if len(ranked)<5: raise RuntimeError(f'Only {len(ranked)} developed mapped names for {year}: {ranked}')
    total=sum(x[4] for x in ranked)
    return [{'source_year':year,'holding_year':year+1,'rank':i+1,'name':x[1],'ticker':x[2],'currency':x[3],'market_cap':x[4],'weight':x[4]/total} for i,x in enumerate(ranked)]


def build_rankings():
    out=[]
    # 1999 through 2016 from FT/Wikipedia archive; prior-year list sets following-year holdings.
    for y in range(1999,2017):
        out.extend(parse_ft_year(y))
    frozen=pd.read_csv(FROZEN)
    for _,r in frozen.iterrows():
        y=int(r.source_year_end)
        if y < 2017: continue
        for i in range(1,6):
            t=str(r[f'rank_{i}_ticker'])
            out.append({'source_year':y,'holding_year':int(r.holding_year),'rank':i,'name':t,'ticker':t,'currency':'USD','market_cap':float(r[f'rank_{i}_market_cap']),'weight':float(r[f'rank_{i}_weight'])})
    df=pd.DataFrame(out).sort_values(['holding_year','rank'])
    return df


def download_one(ticker):
    x=yf.download(ticker,start='1999-12-01',end=END_EXCLUSIVE,auto_adjust=False,progress=False,threads=False,timeout=30)
    if x.empty: return None
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    if 'Adj Close' not in x: x['Adj Close']=x['Close']
    x.index=pd.to_datetime(x.index).tz_localize(None)
    return x['Adj Close'].astype(float).dropna()


def build_prices(rankings):
    tickers=sorted(set(rankings.ticker)|{WORLD})
    px={}
    for t in tickers:
        print('download',t)
        s=download_one(t)
        if s is not None: px[t]=s
    if WORLD not in px: raise RuntimeError('MSCI World NETR unavailable')
    cal=px[WORLD].index[(px[WORLD].index>=START)&(px[WORLD].index<pd.Timestamp(END_EXCLUSIVE))]
    if cal.min()>START+pd.Timedelta(days=10): raise RuntimeError(f'World history starts too late: {cal.min()}')
    # FX for JPY local stocks. JPY=X is USDJPY; convert JPY price to USD by division.
    if any(rankings.currency=='JPY'):
        fx=download_one('JPY=X')
        if fx is None: raise RuntimeError('USDJPY unavailable')
        px['JPY=X']=fx
    mat=pd.DataFrame(index=cal)
    for _,r in rankings[['ticker','currency']].drop_duplicates().iterrows():
        t=r.ticker
        if t not in px:
            raise RuntimeError(f'Missing price history for selected ticker {t}')
        s=px[t].reindex(cal).ffill()
        if r.currency=='JPY':
            fx=px['JPY=X'].reindex(cal).ffill()
            s=s/fx
        mat[t]=s
    world=px[WORLD].reindex(cal).ffill()
    return cal,mat,world


def targets(rankings):
    out={}
    for y,g in rankings.groupby('holding_year'):
        out[int(y)]={r.ticker:float(r.weight) for _,r in g.iterrows()}
    return out


def first_month_days(cal):
    p=cal.to_period('M'); return {cal[p==m][0] for m in p.unique()}


def simulate(topw, mat, world, tg, dca=False):
    cal=world.index; month=first_month_days(cal); holdings={}; cash=0.0 if dca else INITIAL_CAPITAL; eq={}; cf=[]; cur=None; weights={}; cost=0.0
    def p(dt,t): return float(world.at[dt]) if t=='WORLD' else float(mat.at[dt,t])
    def value(dt): return cash+sum(u*p(dt,t) for t,u in holdings.items())
    for dt in cal:
        if dca and dt in month:
            cash+=MONTHLY_CONTRIBUTION; cf.append((dt,-MONTHLY_CONTRIBUTION))
        if dt.year!=cur:
            cur=dt.year
            top=tg.get(cur)
            if not top: raise RuntimeError(f'No ranking for holding year {cur}')
            weights={'WORLD':1-topw}
            for t,w in top.items(): weights[t]=weights.get(t,0)+topw*w
            weights={t:w for t,w in weights.items() if w>1e-12}
            before=value(dt)
            cv={t:holdings.get(t,0)*p(dt,t) for t in set(holdings)|set(weights)}
            dv={t:before*w for t,w in weights.items()}
            turnover=.5*sum(abs(dv.get(t,0)-cv.get(t,0)) for t in set(cv)|set(dv))
            c=turnover*TURNOVER_COST; cost+=c; invest=max(0,before-c)
            holdings={t:invest*w/p(dt,t) for t,w in weights.items()}; cash=0.0
        elif dca and dt in month:
            invest=min(cash,MONTHLY_CONTRIBUTION); c=invest*TURNOVER_COST; cost+=c; amt=invest-c; cash-=invest
            for t,w in weights.items(): holdings[t]=holdings.get(t,0)+amt*w/p(dt,t)
        eq[dt]=value(dt)
    e=pd.Series(eq,dtype=float)
    if dca: cf.append((e.index[-1],float(e.iloc[-1])))
    return e,cf,cost


def xnpv(rate,cf):
    t0=cf[0][0]; return sum(v/((1+rate)**(((d-t0).days)/365.25)) for d,v in cf)

def xirr(cf):
    lo,hi=-.9999,10.; flo,fhi=xnpv(lo,cf),xnpv(hi,cf)
    while flo*fhi>0 and hi<1e6: hi*=2; fhi=xnpv(hi,cf)
    if flo*fhi>0: return np.nan
    for _ in range(250):
        mid=(lo+hi)/2; fm=xnpv(mid,cf)
        if flo*fm<=0: hi=mid
        else: lo=mid; flo=fm
    return (lo+hi)/2


def stats(e):
    years=(e.index[-1]-e.index[0]).days/365.25; cagr=(e.iloc[-1]/e.iloc[0])**(1/years)-1
    dd=e/e.cummax()-1; r=e.pct_change().fillna(0); vol=r.std(ddof=0)*math.sqrt(252); ann=r.mean()*252; mdd=float(dd.min())
    annual=(1+r).groupby(r.index.year).prod()-1
    return {'cagr_pct':100*cagr,'max_drawdown_pct':100*mdd,'sharpe':ann/vol if vol else np.nan,'calmar':cagr/abs(mdd) if mdd<0 else np.nan,'final_value_10000':float(e.iloc[-1]),'worst_year_pct':100*float(annual.min()),'best_year_pct':100*float(annual.max())}


def main():
    rankings=build_rankings(); rankings.to_csv(OUT/'rankings.csv',index=False)
    print(rankings.groupby('holding_year').apply(lambda g: ', '.join(g.ticker)).to_string())
    cal,mat,world=build_prices(rankings); tg=targets(rankings)
    rows=[]; curves={}; annuals={}
    for topw in [0,.25,.5,.75,1.0]:
        label=f'World_{int((1-topw)*100)}_Top5_{int(topw*100)}'
        e,_,c1=simulate(topw,mat,world,tg,False); de,cf,c2=simulate(topw,mat,world,tg,True)
        s=stats(e); s.update({'strategy':label,'world_weight_pct':100*(1-topw),'top5_weight_pct':100*topw,'dca_xirr_pct':100*xirr(cf),'dca_contributed':-sum(v for _,v in cf if v<0),'dca_final_value':float(de.iloc[-1]),'cost_lump':c1,'cost_dca':c2})
        rows.append(s); curves[label]=e; rr=e.pct_change().fillna(0); annuals[label]=(1+rr).groupby(rr.index.year).prod()-1
    res=pd.DataFrame(rows); res.to_csv(OUT/'results.csv',index=False); pd.DataFrame(curves).to_csv(OUT/'equity.csv'); (pd.DataFrame(annuals)*100).to_csv(OUT/'annual_returns_pct.csv')
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'period':[str(cal[0].date()),str(cal[-1].date())],'benchmark':'MSCI World NETR USD via Yahoo ticker ^990100-USD-NETR','selection':'FT/Wikipedia archived global market-cap top ten for 1999-2016, filtered to mapped developed-market companies; validated frozen annual rankings for 2017-2025','cost':'10 bps one-way turnover stress','limitations':['Proxy, not official point-in-time MSCI constituent history.','FT ranking dates vary: 1999 annual, 2000-2005 some March/December, 2006+ Q4. Prior source-year list is held in the following calendar year.','Foreign ADR/local-share proxies and FX conversion can differ from MSCI free-float USD returns.','Top-ten source can miss a developed company outside the published ten if emerging-market names crowd the table.','Taxes and investor-specific FX are excluded.','Historical results are not forecasts.'],'results':res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict('records')}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('\nRESULTS\n',res.to_string(index=False))

if __name__=='__main__': main()
