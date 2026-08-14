from io import StringIO
from pathlib import Path
import re

import pandas as pd
import requests
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


def ntt_docomo_adjusted_close():
    frames = []
    for year in (2000, 2001):
        url = f'https://kabu.hayauma.net/kabuka/9437/{year}.html'
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        # The page contains a plain historical table. Try HTML tables first.
        parsed = []
        try:
            parsed = pd.read_html(StringIO(r.text))
        except Exception:
            parsed = []
        found = None
        for table in parsed:
            if table.shape[1] >= 7:
                candidate = table.copy()
                dates = pd.to_datetime(candidate.iloc[:, 0], errors='coerce')
                if dates.notna().sum() > 20:
                    values = pd.to_numeric(candidate.iloc[:, -1].astype(str).str.replace(',', '', regex=False), errors='coerce')
                    found = pd.Series(values.values, index=dates).dropna()
                    break
        if found is None or found.empty:
            # Fallback for the site's text-oriented markup: date + six numeric fields,
            # with adjusted close as the last number.
            rows = []
            text = re.sub(r'<[^>]+>', ' ', r.text)
            text = re.sub(r'\s+', ' ', text)
            pat = re.compile(r'(20(?:00|01)-\d{2}-\d{2})\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)')
            for m in pat.finditer(text):
                rows.append((pd.Timestamp(m.group(1)), float(m.group(7).replace(',', ''))))
            if not rows:
                raise RuntimeError(f'Could not parse NTT Docomo historical prices for {year}')
            found = pd.Series(dict(rows), dtype=float)
        frames.append(found)
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep='last')]
    return s.astype(float)


def build_prices(rankings):
    # MSCI World Net Total Return in USD from MSCI's public chart-level endpoint.
    world_df = msci.get_levels('990100', '2000-01-01', '2026-08-14', variant='NETR')
    if world_df is None or len(world_df) == 0:
        raise RuntimeError('MSCI World NETR public-level data unavailable')
    world_df = world_df.copy()
    world_df['DATE'] = pd.to_datetime(world_df['DATE'], errors='coerce')
    world_df['LEVEL'] = pd.to_numeric(world_df['LEVEL'], errors='coerce')
    world = world_df.dropna(subset=['DATE', 'LEVEL']).set_index('DATE')['LEVEL'].sort_index()
    world.index = pd.to_datetime(world.index).tz_localize(None)
    world = world[~world.index.duplicated(keep='last')]
    cal = world.index[(world.index >= backtest.START) & (world.index < pd.Timestamp(backtest.END_EXCLUSIVE))]
    world = world.reindex(cal).astype(float)
    if len(world) < 5000:
        raise RuntimeError(f'MSCI World history unexpectedly short: {len(world)} rows')

    raw = {}
    selected = sorted(set(rankings['ticker']))
    for ticker in selected:
        if ticker == '9437.T':
            raw[ticker] = ntt_docomo_adjusted_close()
        else:
            print('download', ticker)
            s = backtest.download_one(ticker)
            if s is None or s.empty:
                raise RuntimeError(f'Missing historical return series for selected ticker {ticker}')
            raw[ticker] = s

    # NTT Docomo local JPY price is converted to USD with daily USDJPY.
    if '9437.T' in raw:
        fx = backtest.download_one('JPY=X')
        if fx is None or fx.empty:
            raise RuntimeError('USDJPY unavailable for NTT Docomo conversion')
        raw['9437.T'] = raw['9437.T'].reindex(cal).ffill() / fx.reindex(cal).ffill()

    mat = pd.DataFrame(index=cal)
    for ticker in selected:
        s = raw[ticker]
        if ticker != '9437.T':
            s = s.reindex(cal).ffill()
        mat[ticker] = s

    # Validate every security only during the calendar years when it is actually held.
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
