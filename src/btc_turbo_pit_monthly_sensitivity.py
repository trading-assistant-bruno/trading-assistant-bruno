from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import btc_turbo_pit_monthly as m

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'btc_turbo_pit_monthly_sensitivity'; OUT.mkdir(parents=True,exist_ok=True)
CONFIGS=[
 ('Top1_T50_rel5',1,.50,.05,0),
 ('Top2_T30_rel5',2,.30,.05,0),
 ('Top2_T50_rel5',2,.50,.05,0),
 ('Top2_T50_rel10',2,.50,.10,0),
 ('Top2_T50_rel5_RO50',2,.50,.05,.50),
]

def main():
 s,requested=m.collect(); p,r=m.panels(s)
 rows=[]; annuals={}; curves=[]
 # Common benchmark
 btc_cfg=('BTC_HOLD',0,0,0,0)
 eq,lg=m.sim(p,r,btc_cfg); met,ann=m.metrics('BTC_HOLD',eq,lg); met['universe_top_n']=0; rows.append(met); annuals['BTC_HOLD']=ann; curves.append(eq)
 for n in [10,15,20]:
  m.UNIVERSE_N=n
  for label,n_alts,aw,rel,ro in CONFIGS:
   name=f'PIT_U{n}_{label}'
   cfg=(name,n_alts,aw,rel,ro)
   eq,lg=m.sim(p,r,cfg); met,ann=m.metrics(name,eq,lg); met['universe_top_n']=n; rows.append(met); annuals[name]=ann; curves.append(eq)
 res=pd.DataFrame(rows).sort_values('cagr_pct',ascending=False)
 annual=pd.DataFrame(annuals); equity=pd.concat(curves,axis=1)
 res.to_csv(OUT/'results.csv',index=False); annual.to_csv(OUT/'annual.csv'); equity.to_csv(OUT/'equity.csv')
 payload={'generated_at':datetime.now(timezone.utc).isoformat(),'source':'CoinMarketCap historical monthly snapshots','snapshots_requested':requested,'snapshots_collected':int(s.date.nunique()),'universe_sensitivity':[10,15,20],'results':res.replace({np.nan:None}).to_dict('records')}
 (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 print('\nSENSITIVITY RESULTS\n'+res.to_string(index=False))
 print('\nTOP1 50% ONLY\n'+res[res.strategy.str.contains('Top1_T50')].to_string(index=False))
 print('\nTOP2 30% ONLY\n'+res[res.strategy.str.contains('Top2_T30')].to_string(index=False))

if __name__=='__main__': main()
