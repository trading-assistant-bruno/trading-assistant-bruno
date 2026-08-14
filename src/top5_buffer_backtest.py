from __future__ import annotations

import json, math, re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import world_top5_long_backtest as bt
import world_top5_long_backtest_wiki as longsrc
import topn_long_backtest as topn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'top5_buffer'
OUT.mkdir(parents=True, exist_ok=True)

ARCHIVES = [
    'https://reference.org/facts/List_of_public_corporations_by_market_capitalization/3zIFBx5h',
    'https://www.wikizero.org/wiki/en/List_of_public_corporations_by_market_capitalization',
    'https://www.olps.co.za/kiwix/content/wikipedia_en_all_maxi_2024-01/A/List_of_public_corporations_by_market_capitalization',
]
HEAD = {'User-Agent': 'Mozilla/5.0'}


def identify(text: str):
    low = text.lower()
    for alias in sorted(bt.ALIASES, key=len, reverse=True):
        if alias.lower() in low:
            ticker, ccy = bt.ALIASES[alias]
            return alias, ticker, ccy
    return None


def parse_num(text: str) -> float:
    vals = re.findall(r'\d[\d,.]*', text.replace('\xa0', ' '))
    if not vals:
        return float('nan')
    return float(vals[-1].replace(',', ''))


def load_archive_soup():
    errors=[]
    for url in ARCHIVES:
        try:
            r=requests.get(url,timeout=40,headers=HEAD,verify=False)
            r.raise_for_status(); soup=BeautifulSoup(r.text,'html.parser')
            texts=[h.get_text(' ',strip=True) for h in soup.find_all(['h1','h2','h3','h4','h5','h6'])]
            if any(re.match(r'^\s*1999\b',t) for t in texts) and any(re.match(r'^\s*2022\b',t) for t in texts):
                print('archive source',url,'title',soup.title.get_text(' ',strip=True) if soup.title else '')
                return soup,url
            errors.append(f'{url}: headings sample={texts[:15]}')
        except Exception as e:
            errors.append(f'{url}: {type(e).__name__}: {e}')
    raise RuntimeError('No complete archive source. ' + ' | '.join(errors))


def archived_rows_1999_2022():
    soup,source_url = load_archive_soup()
    all_rows = []
    for year in range(1999, 2023):
        heading = None
        for h in soup.find_all(['h1','h2','h3','h4','h5','h6']):
            txt=h.get_text(' ', strip=True)
            if h.get('id')==str(year) or h.find(id=str(year)) is not None or re.match(rf'^\s*{year}\b', txt):
                heading = h
                break
        if heading is None:
            raise RuntimeError(f'Archive heading missing {year} from {source_url}')
        table = heading.find_next('table')
        if table is None:
            raise RuntimeError(f'Archive table missing {year}')
        trs = table.find_all('tr')
        header = [x.get_text(' ', strip=True) for x in trs[0].find_all(['th','td'])] if trs else []
        quarterly = any('Fourth quarter' in x for x in header)
        rows=[]
        for tr in trs[1:]:
            cells=[x.get_text(' ', strip=True) for x in tr.find_all(['th','td'])]
            if not cells:
                continue
            try:
                rank=int(re.sub(r'\D','',cells[0]))
            except Exception:
                continue
            if rank < 1 or rank > 10:
                continue
            target = cells[-1] if quarterly else (cells[1] + ' ' + cells[-1] if len(cells)>1 else cells[-1])
            hit=identify(target)
            if not hit:
                continue
            alias,ticker,ccy=hit
            cap=parse_num(target if quarterly else cells[-1])
            if math.isfinite(cap) and cap>0:
                rows.append({'source_year':year,'holding_year':year+1,'global_rank':rank,'name':alias,'ticker':ticker,'currency':ccy,'market_cap':cap,'source':'FT_archive'})
        if len(rows) < 5:
            raise RuntimeError(f'Only {len(rows)} eligible developed names in global Top10 for {year}: {rows}')
        all_rows.extend(sorted(rows,key=lambda x:x['global_rank']))
    return all_rows


def recent_rows_2023_2025():
    out=[]
    for year in [2023,2024,2025]:
        rows=topn.fetch_developed_ranking(year,20)
        rows=[r for r in rows if int(r['global_rank']) <= 10]
        if len(rows) < 5:
            raise RuntimeError(f'Only {len(rows)} eligible developed names in global Top10 for {year}')
        for r in rows:
            out.append({'source_year':year,'holding_year':year+1,'global_rank':int(r['global_rank']),'name':r['name_ticker'],'ticker':r['ticker'],'currency':'USD','market_cap':float(r['market_cap']),'source':'CompaniesMarketCap'})
    return out


def ranking_table():
    df=pd.DataFrame(archived_rows_1999_2022()+recent_rows_2023_2025())
    df=df.sort_values(['holding_year','global_rank']).reset_index(drop=True)
    counts=df.groupby('holding_year').size()
    if df.holding_year.min()!=2000 or df.holding_year.max()!=2026 or (counts<5).any():
        raise RuntimeError('Ranking coverage invalid')
    df.to_csv(OUT/'eligible_global_top10_by_year.csv',index=False)
    return df


def pure_top5_targets(df):
    out={}; selections=[]
    for year,g in df.groupby('holding_year'):
        s=g.sort_values('global_rank').head(5).copy()
        total=float(s.market_cap.sum())
        out[int(year)]={r.ticker:float(r.market_cap/total) for _,r in s.iterrows()}
        for _,r in s.iterrows(): selections.append({'strategy':'Top5_pure','holding_year':int(year),'ticker':r.ticker,'global_rank':int(r.global_rank),'market_cap':float(r.market_cap)})
    return out,selections


def buffer_targets(df, expanded=False):
    out={}; selections=[]; held=[]
    for year,g in df.groupby('holding_year'):
        g=g.sort_values('global_rank').copy()
        current={r.ticker:r for _,r in g.iterrows()}
        current_top5=[r.ticker for _,r in g.head(5).iterrows()]
        if not held:
            new=list(current_top5)
        elif expanded:
            survivors=[t for t in held if t in current]
            new=list(survivors)
            for t in current_top5:
                if t not in new:
                    new.append(t)
        else:
            survivors=[t for t in held if t in current]
            new=list(survivors[:5])
            for t in current_top5:
                if len(new)>=5:
                    break
                if t not in new:
                    new.append(t)
            if len(new)<5:
                for t in current:
                    if len(new)>=5:
                        break
                    if t not in new:
                        new.append(t)
        caps={t:float(current[t].market_cap) for t in new}
        total=sum(caps.values())
        out[int(year)]={t:caps[t]/total for t in new}
        label='Buffer_expanded' if expanded else 'Buffer_fixed5'
        for t in new:
            r=current[t]
            selections.append({'strategy':label,'holding_year':int(year),'ticker':t,'global_rank':int(r.global_rank),'market_cap':float(r.market_cap)})
        held=new
    return out,selections


def stats(e):
    years=(e.index[-1]-e.index[0]).days/365.25
    cagr=(e.iloc[-1]/e.iloc[0])**(1/years)-1
    r=e.pct_change().fillna(0); dd=e/e.cummax()-1
    vol=r.std(ddof=0)*math.sqrt(252); ann=r.mean()*252; mdd=float(dd.min())
    annual=(1+r).groupby(r.index.year).prod()-1
    return {'cagr_pct':100*cagr,'max_drawdown_pct':100*mdd,'sharpe':ann/vol if vol else np.nan,'calmar':cagr/abs(mdd) if mdd<0 else np.nan,'final_value_10000':float(e.iloc[-1]),'worst_year_pct':100*float(annual.min()),'best_year_pct':100*float(annual.max())}


def turnover_summary(selections):
    df=pd.DataFrame(selections)
    rows=[]
    for strategy,g in df.groupby('strategy'):
        prev=set(); changes=[]; counts=[]
        for year,yy in g.groupby('holding_year'):
            cur=set(yy.ticker); counts.append(len(cur))
            if prev:
                changes.append(len(cur-prev)+len(prev-cur))
            prev=cur
        rows.append({'strategy':strategy,'avg_holdings':float(np.mean(counts)),'min_holdings':int(min(counts)),'max_holdings':int(max(counts)),'avg_names_changed_per_year':float(np.mean(changes)) if changes else 0.0})
    return pd.DataFrame(rows)


def main():
    ranks=ranking_table()
    t_pure,s_pure=pure_top5_targets(ranks)
    t_fixed,s_fixed=buffer_targets(ranks,False)
    t_exp,s_exp=buffer_targets(ranks,True)
    sels=s_pure+s_fixed+s_exp
    pd.DataFrame(sels).to_csv(OUT/'annual_selections.csv',index=False)
    turn=turnover_summary(sels); turn.to_csv(OUT/'selection_turnover.csv',index=False)

    cal,mat,world=longsrc.build_prices(ranks[['source_year','holding_year','ticker','currency','market_cap']].copy())

    variants={
        'MSCI_World':(0.0,t_pure),
        'Top5_pure':(1.0,t_pure),
        'Top5_buffer_fixed5':(1.0,t_fixed),
        'Top5_buffer_expanded':(1.0,t_exp),
        'World50_Top5pure50':(0.5,t_pure),
        'World50_BufferFixed50':(0.5,t_fixed),
        'World50_BufferExpanded50':(0.5,t_exp),
    }
    rows=[]; curves={}; annuals={}
    for name,(topw,tg) in variants.items():
        e,_,cost=bt.simulate(topw,mat,world,tg,False)
        de,cf,dcost=bt.simulate(topw,mat,world,tg,True)
        st=stats(e)
        st.update({'strategy':name,'start':str(e.index[0].date()),'end':str(e.index[-1].date()),'dca_xirr_pct':100*bt.xirr(cf),'dca_contributed':-sum(v for _,v in cf if v<0),'dca_final_value':float(de.iloc[-1]),'lump_cost':cost,'dca_cost':dcost})
        rows.append(st); curves[name]=e
        rr=e.pct_change().fillna(0); annuals[name]=(1+rr).groupby(rr.index.year).prod()-1

    res=pd.DataFrame(rows); res.to_csv(OUT/'results.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'equity.csv')
    (pd.DataFrame(annuals)*100).to_csv(OUT/'annual_returns_pct.csv')
    payload={
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'method':'Annual market-cap weighted Top5. Fixed buffer keeps up to five incumbents while they remain in the global Top10 and fills vacancies from current eligible Top5. Expanded buffer keeps all incumbents still global Top10 and adds all current eligible Top5.',
        'source':'FT historical global Top10 1999-2022 via public mirrors; CompaniesMarketCap time machine 2023-2025; developed/unmapped exclusions follow prior Top5 experiment.',
        'important_limitation':'For 1999-2022 the buffer threshold is GLOBAL Top10, not exact developed-market Top10, because the historical public table only exposes global Top10. This is a conservative retention rule when emerging-market firms occupy global slots.',
        'results':res.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict('records'),
        'turnover':turn.to_dict('records'),
    }
    (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('\nRESULTS\n',res.to_string(index=False))
    print('\nTURNOVER\n',turn.to_string(index=False))
    print('\nSELECTIONS\n',pd.DataFrame(sels).to_string(index=False))

if __name__=='__main__': main()
