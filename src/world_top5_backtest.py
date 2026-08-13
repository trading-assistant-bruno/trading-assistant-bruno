from __future__ import annotations

import shutil
from pathlib import Path

import world_top5_annual_cmc as annual

ROOT = Path(__file__).resolve().parents[1]
LEGACY_OUT = ROOT / "data" / "world_top5"
ANNUAL_OUT = ROOT / "data" / "world_top5_annual"


def main() -> None:
    # CompaniesMarketCap canonical route for Alphabet (Google).
    annual.CANDIDATES["GOOGL"] = "alphabet-google"

    # The earlier quarterly reconstruction based on Yahoo historical share counts was
    # rejected after split-basis inconsistencies were detected. Use direct published
    # end-of-year market-cap data instead; annual.py documents the remaining limits.
    annual.main()

    # Preserve the existing workflow artifact paths without changing CI configuration.
    LEGACY_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ANNUAL_OUT / "annual_top5_comparison.csv", LEGACY_OUT / "world_top5_comparison.csv")
    shutil.copy2(ANNUAL_OUT / "annual_top5_comparison.json", LEGACY_OUT / "world_top5_comparison.json")
    shutil.copy2(ANNUAL_OUT / "annual_top5_rankings.csv", LEGACY_OUT / "top5_rebalance_history.csv")
    shutil.copy2(ANNUAL_OUT / "annual_top5_equity_curves.csv", LEGACY_OUT / "top5_lump_equity_curves.csv")
    shutil.copy2(ANNUAL_OUT / "annual_top5_rankings.csv", LEGACY_OUT / "share_data_coverage.csv")


if __name__ == "__main__":
    main()
