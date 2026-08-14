from pathlib import Path

import pandas as pd

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


backtest.build_rankings = frozen_rankings

if __name__ == '__main__':
    backtest.main()
