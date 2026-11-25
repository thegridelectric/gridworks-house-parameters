"""
Export design-day cold-OAT hours from cleaned valid data (red scatter points)
to CSV, XLSX, and LaTeX. Uses the same prepared cache as plot_energy_vs_swt_grid.py.
Pass --refresh to rebuild the cache (required once after zone temp-by-hour was added).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from plot_energy_vs_swt_grid import (
    COLD_OAT_HIGHLIGHT_THRESHOLD_C,
    HOUSE_ALIASES,
    get_plot_prepared_bundle,
)
from plots import (
    DESIGN_DAY_COLD_OAT_VALID_TABLE_CAPTION,
    design_day_cold_oat_valid_hours_dataframe,
    write_design_day_zone_hours_latex_table,
)

DEFAULT_CSV = Path("results") / "design_day_cold_oat_valid_hours.csv"
DEFAULT_XLSX = Path("results") / "design_day_cold_oat_valid_hours.xlsx"
DEFAULT_TEX = Path("results") / "design_day_cold_oat_valid_hours.tex"
DEFAULT_LATEX_LABEL = "tab:design_day_cold_oat_valid"


def main() -> None:
    force_refresh = "--refresh" in sys.argv[1:]
    bundle = get_plot_prepared_bundle(HOUSE_ALIASES, force_refresh=force_refresh)
    df = design_day_cold_oat_valid_hours_dataframe(
        bundle.panels,
        oat_at_or_below_c=COLD_OAT_HIGHLIGHT_THRESHOLD_C,
        house_order=HOUSE_ALIASES,
    )
    DEFAULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_CSV, index=False)
    df.to_excel(DEFAULT_XLSX, index=False)
    write_design_day_zone_hours_latex_table(
        df,
        DEFAULT_TEX,
        caption=DESIGN_DAY_COLD_OAT_VALID_TABLE_CAPTION,
        label=DEFAULT_LATEX_LABEL,
    )
    logger.info(
        "Wrote %d valid cold-OAT hour row(s) to %s, %s, and %s",
        len(df),
        DEFAULT_CSV,
        DEFAULT_XLSX,
        DEFAULT_TEX,
    )


if __name__ == "__main__":
    main()