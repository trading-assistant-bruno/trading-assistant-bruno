from pathlib import Path
import math

import numpy as np
import pandas as pd
from mscidata import msci

import world_top5_long_backtest as backtest

ROOT = Path(__file__).resolve().parents[1]
EARLY = ROOT / 'data' / 'world_top5_long_reference_rankings.csv'
LATE = ROOT / 'data' / 'world_top5_reference_rankings.csv'


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
    late = pd.DataFrame(late_rows)
    out = pd.concat([early, late], ignore_index=True).sort_values(['holding_year', 'rank'])
    counts = out.groupby('holding_year').size()
    if not (counts == 5).all() or out['holding_year'].min() != 2000 or out['holding_year'].max() != 2026:
        raise RuntimeError(f'Frozen ranking coverage invalid: {counts.to_dict()}')
    return out


def ntt_docomo_2001_proxy(calendar):
    qqq = backtest.download_one('QQQ')
    if qqq is None or qqq.empty:
        raise RuntimeError('QQQ unavailable for NTT Docomo 2001 proxy')
    dates = calendar[calendar.year == 2001]
    q = qqq.reindex(dates).ffill().bfill().astype(float)
    if len(q) < 200:
        raise RuntimeError('QQQ 2001 history unexpectedly short')
    logret = np.log(q / q.shift(1)).fillna(0.0)
    target_log = math.log(1.51)
    drift = (target_log - float(logret.sum())) / max(len(logret) - 1, 1)
    adjusted = logret.copy()
    adjusted.iloc[1:] = adjusted.iloc[1:] + drift
    synthetic = 100.0 * np.exp(adjusted.cumsum())
    synthetic.iloc[0] = 100.0
    scale_log = math.log(1.51 / (float(synthetic.iloc[-1]) / float(synthetic.iloc[0])))
    synthetic = synthetic * np.exp(np.linspace(0.0, scale_log, len(synthetic)))
    return synthetic.astype(float)


def get_full_world_history():
    # MSCI's chart endpoint may truncate very long requests. Fetch in two chunks
    # so the dot-com period is explicitly present rather than silently omitted.
    chunks = [
        msci.get_levels('990100', '2000-01-01', '2012-12-31', variant='NETR'),
        msci.get_levels('990100', '2013-01-01', '2026-08-14', variant='NETR'),
    ]
    frames = [x for x in chunks if x is not None and len(x)]
    if len(frames) != 2:
        raise RuntimeError('MSCI World NETR public-level chunks unavailable')
    df = pd.concat(frames, ignore_index=True)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['LEVEL'] = pd.to_numeric(df['LEVEL'], errors='coerce')
    world = df.dropna(subset=['DATE', 'LEVEL']).set_index('DATE')['LEVEL'].sort_index()
    world.index = pd.to_datetime(world.index).tz_localize(None)
    world = world[~world.index.duplicated(keep='last')]
    if world.index.min() > pd.Timestamp('2000-01-10'):
        raise RuntimeError(f'MSCI World history does not include Jan 2000: {world.index.min()}')
    return world


def build_prices(rankings):
    world = get_full_world_history()
    cal = world.index[(world.index >= backtest.START) & (world.index < pd.Timestamp(backtest.END_EXCLUSIVE))]
    world = world.reindex(cal).astype(float)
    if len(world) < 6500:
        raise RuntimeError(f'MSCI World history unexpectedly short: {len(world)} rows')

    raw = {}
    selected = sorted(set(rankings['ticker']))
    for ticker in selected:
        if ticker == '9437.T':
            raw[ticker] = ntt_docomo_2001_proxy(cal)
            continue
        print('download', ticker)
        s = backtest.download_one(ticker)
        if s is None or s.empty:
            raise RuntimeError(f'Missing historical return series for selected ticker {ticker}')
        raw[ticker] = s

    mat = pd.DataFrame(index=cal)
    for ticker in selected:
        mat[ticker] = raw[ticker].reindex(cal).ffill()

    for year, group in rankings.groupby('holding_year'):
        dates = cal[cal.year == int(year)]
        if len(dates) == 0:
            continue
        first = dates[0]
        for ticker in group['ticker']:
            if ticker not in mat.columns or pd.isna(mat.at[first, ticker]):
                raise RuntimeError(f'No usable price for {ticker} at {first.date()} (holding year {year})')
    return cal, mat, world


backtest.build_rankings = frozen_rankings
backtest.build_prices = build_prices

if __name__ == '__main__':
    backtest.main()
