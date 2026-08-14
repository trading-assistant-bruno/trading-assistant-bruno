from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math
import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
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


def parse_msci_world_archive(date: pd.Timestamp):
    ds = date.strftime('%Y%m%d')
    url = f'https://www.msci.com/eqb/esg/indexperf/asof/{ds}.html'
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            text = BeautifulSoup(r.text, 'html.parser').get_text(' ', strip=True)
            # Archive row format starts with 'MSCI World' followed by the index level.
            m = re.search(r'MSCI\s+World\s+([0-9][0-9,]*\.?[0-9]*)\s', text, flags=re.I)
            if m:
                return date, float(m.group(1).replace(',', ''))
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.4 * (attempt + 1))
    return None


def get_msci_world_price_2000():
    # Use SPY only as a trading-day calendar; the World levels themselves come from MSCI.
    spy = backtest.download_one('SPY')
    if spy is None or spy.empty:
        raise RuntimeError('SPY unavailable for 2000 trading calendar')
    dates = [pd.Timestamp(d) for d in spy.index if pd.Timestamp('2000-01-01') <= d <= pd.Timestamp('2000-12-29')]
    found = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(parse_msci_world_archive, d): d for d in dates}
        for fut in as_completed(futures):
            item = fut.result()
            if item is not None:
                found.append(item)
    if len(found) < 235:
        raise RuntimeError(f'Only {len(found)} official MSCI World archive sessions found for 2000')
    s = pd.Series({d: level for d, level in found}, dtype=float).sort_index()
    # Sanity check against MSCI's published 12/29/2000 price level 1,221.253.
    last = s.loc[:pd.Timestamp('2000-12-29')].iloc[-1]
    if abs(float(last) - 1221.253) > 0.02:
        raise RuntimeError(f'Unexpected MSCI World archive endpoint for 2000: {last}')
    return s


def get_netr_from_2000_12_29():
    # The public chart endpoint's earliest NETR observation is 2000-12-29.
    df = msci.get_levels('990100', '2000-12-29', '2026-08-14', variant='NETR')
    if df is None or len(df) == 0:
        raise RuntimeError('MSCI World NETR public-level data unavailable')
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['LEVEL'] = pd.to_numeric(df['LEVEL'], errors='coerce')
    s = df.dropna(subset=['DATE', 'LEVEL']).set_index('DATE')['LEVEL'].sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep='last')]


def get_full_world_history():
    price_2000 = get_msci_world_price_2000()
    netr = get_netr_from_2000_12_29()
    if netr.index.min() > pd.Timestamp('2000-12-31'):
        raise RuntimeError(f'MSCI NETR starts too late: {netr.index.min()}')

    # Scale the official 2000 Price index path to meet the exact NETR series at the
    # common 29-Dec-2000 observation. This preserves every official 2000 World price
    # daily return while avoiding a level discontinuity. It is conservative because
    # dividends are omitted from 2000 only; NETR is exact from the join onward.
    join = pd.Timestamp('2000-12-29')
    price_join = float(price_2000.loc[:join].iloc[-1])
    netr_join = float(netr.loc[:join].iloc[-1])
    bridge = price_2000 * (netr_join / price_join)
    bridge = bridge[bridge.index < join]
    world = pd.concat([bridge, netr]).sort_index()
    world = world[~world.index.duplicated(keep='last')]
    if world.index.min() > pd.Timestamp('2000-01-10'):
        raise RuntimeError(f'World bridge does not include Jan 2000: {world.index.min()}')
    return world


def build_prices(rankings):
    world = get_full_world_history()
    cal = world.index[(world.index >= backtest.START) & (world.index < pd.Timestamp(backtest.END_EXCLUSIVE))]
    world = world.reindex(cal).astype(float)
    if len(world) < 6650:
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
