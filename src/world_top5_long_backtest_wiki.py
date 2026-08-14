import math
import re
import requests
from bs4 import BeautifulSoup

import world_top5_long_backtest as backtest

backtest.FT_MIRROR = 'https://handwiki.org/wiki/Finance:List_of_public_corporations_by_market_capitalization'


def parse_ft_year_wiki(year: int):
    html = requests.get(backtest.FT_MIRROR, timeout=30, headers={'User-Agent': 'Mozilla/5.0'}).text
    soup = BeautifulSoup(html, 'html.parser')

    anchor = soup.find(id=str(year))
    heading = None
    if anchor is not None:
        heading = anchor if anchor.name in ('h2', 'h3', 'h4') else anchor.find_parent(['h2', 'h3', 'h4'])
    if heading is None:
        for h in soup.find_all(['h2', 'h3', 'h4']):
            txt = h.get_text(' ', strip=True)
            if re.search(rf'\b{year}\b', txt):
                heading = h
                break
    if heading is None:
        raise RuntimeError(f'FT heading missing {year}')

    table = heading.find_next('table')
    if table is None:
        raise RuntimeError(f'FT table missing {year}')
    rows = table.find_all('tr')
    header = [x.get_text(' ', strip=True) for x in rows[0].find_all(['th', 'td'])]
    quarterly = any('Fourth quarter' in x for x in header)

    ranked = []
    for tr in rows[1:]:
        cells = [x.get_text(' ', strip=True) for x in tr.find_all(['th', 'td'])]
        if not cells:
            continue
        try:
            rank = int(re.sub(r'\D', '', cells[0]))
        except Exception:
            continue
        target = cells[-1] if quarterly else (cells[1] + ' ' + cells[-1])
        hit = backtest.identify(target)
        if not hit:
            continue
        alias, ticker, ccy = hit
        cap = backtest.num(target if quarterly else cells[-1])
        if math.isfinite(cap) and cap > 0:
            ranked.append((rank, alias, ticker, ccy, cap))

    ranked = sorted(ranked, key=lambda x: x[0])[:5]
    if len(ranked) < 5:
        raise RuntimeError(f'Only {len(ranked)} developed mapped names for {year}: {ranked}')
    total = sum(x[4] for x in ranked)
    return [
        {
            'source_year': year,
            'holding_year': year + 1,
            'rank': i + 1,
            'name': x[1],
            'ticker': x[2],
            'currency': x[3],
            'market_cap': x[4],
            'weight': x[4] / total,
        }
        for i, x in enumerate(ranked)
    ]


backtest.parse_ft_year = parse_ft_year_wiki

if __name__ == '__main__':
    backtest.main()
