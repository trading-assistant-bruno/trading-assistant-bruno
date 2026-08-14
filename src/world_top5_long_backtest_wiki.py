from pathlib import Path
import math

import numpy as np
import pandas as pd
from mscidata import msci

import world_top5_long_backtest as backtest

ROOT = Path(__file__).resolve().parents[1]
EARLY = ROOT / 'data' / 'world_top5_long_reference_rankings.csv'
LATE = ROOT / 'data' / 'world_top5_reference_rankings.csv'
WORLD_2000_PRICE_RETURN = -0.1405  # official MSCI World Price USD YTD at 2000-12-29


def frozen_rankings():
    early = pd.read_csv(EARLY)
    late_raw = pd.read_csv(LATE)
    late_rows = []
    for _, r in late_raw.iterrows():
        source_year = int(r['source_year_end'])
        if source_year < 2017:
            continue
        for i in range(1, 6):
            late_rows.append({
                'source_year': source_year,
                'holding_year': int(r['holding_year']),
                'rank': i,
                'ticker': str(r[f'rank_{i}_ticker']),
                'currency': 'USD',
                'market_cap': float(r[f'rank_{i}_market_cap']),
                'weight': float(r[f'rank_{i}_weight']),
            })
    out = pd.concat([early, pd.DataFrame(late_rows)], ignore_index=True).sort_values(['holding_year','rank'])
    counts = out.groupby('holding_year').size()
    if not (counts == 5).all() or out.holding_year.min() != 2000 or out.holding_year.max() != 2026:
        raise RuntimeError('Frozen ranking coverage invalid')
    return out


def ntt_docomo_2001_proxy(calendar):
    q = backtest.download_one('QQQ').reindex(calendar[calendar.year == 2001]).ffill().bfill().astype(float)
    lr = np.log(q / q.shift(1)).fillna(0.0)
    drift = (math.log(1.51) - float(lr.sum())) / max(len(lr)-1,1)
    lr.iloc[1:] += drift
    s = 100.0 * np.exp(lr.cumsum())
    s *= np.exp(np.linspace(0.0, math.log(1.51/(float(s.iloc[-1])/float(s.iloc[0]))), len(s)))
    return s.astype(float)


def netr_history():
    df = msci.get_levels('990100','2000-12-29','2026-08-14',variant='NETR').copy()
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['LEVEL'] = pd.to_numeric(df['LEVEL'], errors='coerce')
    s = df.dropna(subset=['DATE','LEVEL']).set_index('DATE')['LEVEL'].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep='last')]


def developed_proxy_2000(target_return=WORLD_2000_PRICE_RETURN):
    # Approximate MSCI World geographic mix at the turn of the century using liquid
    # country ETFs that already existed. Only the daily path is proxied; the endpoint
    # is calibrated to MSCI's official 2000 World Price USD return (-14.05%).
    weights = {'SPY':.50,'EWJ':.16,'EWU':.09,'EWG':.05,'EWQ':.05,'EWC':.04,'EWA':.04,'EWL':.04,'EWN':.03}
    raw = {t: backtest.download_one(t) for t in weights}
    if any(s is None or s.empty for s in raw.values()):
        missing = [t for t,s in raw.items() if s is None or s.empty]
        raise RuntimeError(f'2000 bridge ETFs unavailable: {missing}')
    cal = raw['SPY'].index[(raw['SPY'].index >= pd.Timestamp('1999-12-31')) & (raw['SPY'].index <= pd.Timestamp('2000-12-29'))]
    ret = pd.DataFrame({t: raw[t].reindex(cal).ffill().pct_change() for t in weights}).fillna(0.0)
    port_ret = sum(ret[t]*w for t,w in weights.items())
    logret = np.log1p(port_ret)
    target_log = math.log1p(target_return)
    drift = (target_log - float(logret.iloc[1:].sum())) / max(len(logret)-1,1)
    adj = logret.copy(); adj.iloc[1:] += drift
    path = 100.0 * np.exp(adj.cumsum())
    # calibration sanity check from 1999-12-31 to 2000-12-29
    realized = float(path.iloc[-1]/path.iloc[0]-1)
    if abs(realized-target_return) > 1e-6:
        raise RuntimeError(f'2000 bridge calibration failed: {realized}')
    return path[path.index >= pd.Timestamp('2000-01-03')]


def get_full_world_history(target_return=WORLD_2000_PRICE_RETURN):
    bridge = developed_proxy_2000(target_return)
    netr = netr_history()
    join = pd.Timestamp('2000-12-29')
    netr_join = float(netr.loc[:join].iloc[-1])
    bridge = bridge * (netr_join / float(bridge.loc[:join].iloc[-1]))
    world = pd.concat([bridge[bridge.index < join], netr]).sort_index()
    world = world[~world.index.duplicated(keep='last')]
    if world.index.min() > pd.Timestamp('2000-01-10'):
        raise RuntimeError(f'World bridge starts too late: {world.index.min()}')
    return world


def build_prices(rankings):
    world = get_full_world_history()
    cal = world.index[(world.index >= backtest.START) & (world.index < pd.Timestamp(backtest.END_EXCLUSIVE))]
    world = world.reindex(cal).astype(float)
    raw={}; selected=sorted(set(rankings.ticker))
    for ticker in selected:
        raw[ticker] = ntt_docomo_2001_proxy(cal) if ticker=='9437.T' else backtest.download_one(ticker)
        if raw[ticker] is None or raw[ticker].empty:
            raise RuntimeError(f'Missing historical return series {ticker}')
    mat = pd.DataFrame({t: raw[t].reindex(cal).ffill() for t in selected}, index=cal)
    for year,g in rankings.groupby('holding_year'):
        dates=cal[cal.year==int(year)]
        if not len(dates): continue
        for t in g.ticker:
            if pd.isna(mat.at[dates[0],t]): raise RuntimeError(f'No price {t} in {year}')
    return cal,mat,world

backtest.build_rankings=frozen_rankings
backtest.build_prices=build_prices

if __name__=='__main__': backtest.main()
