from __future__ import annotations

from pathlib import Path

import pandas as pd

import world_top5_annual_cmc as annual
import world_top5_blend_backtest as blend

ROOT = Path(__file__).resolve().parents[1]
RANKINGS = ROOT / "data" / "world_top5_reference_rankings.csv"


def frozen_build_prices_and_targets():
    rankings = pd.read_csv(RANKINGS)
    top5_targets = annual.target_by_year(rankings, equal_weight=False)

    tickers = list(annual.CANDIDATES) + ["URTH"]
    prices = annual.download_prices(tickers)
    if "URTH" not in prices:
        raise RuntimeError("URTH unavailable")

    calendar = annual.calendar_from_urth(prices)
    adj = annual.adj_matrix(prices, calendar)
    world = prices["URTH"]["Adj Close"].reindex(calendar).ffill().astype(float)
    return rankings, top5_targets, adj, world


if __name__ == "__main__":
    blend.build_prices_and_targets = frozen_build_prices_and_targets
    blend.main()
