"""
Export red and purple highlight hours for the Maple zone 1 before/after plot.

Uses the same cache as plot_energy_vs_swt_grid.py (pass --refresh if stale).
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
    design_day_zone_scatter_highlight_hours_dataframe,
    write_design_day_zone_hours_latex_table,
)

MAPLE_ZONE1_PANEL_ALIASES = ("maple_before", "maple_after")
MAPLE_ZONE = 1

DEFAULT_CSV = Path("results") / "maple_zone1_before_after_highlight_hours.csv"
DEFAULT_XLSX = Path("results") / "maple_zone1_before_after_highlight_hours.xlsx"
DEFAULT_TEX = Path("results") / "maple_zone1_before_after_highlight_hours.tex"
LATEX_CAPTION = (
    "Hours shown as red (cleaned valid data) or purple (below-setpoint excluded) on "
    "the Maple zone~1 before/after energy--SWT plot "
    "(OAT $\\leq -20\\,^{\\circ}\\mathrm{C}$, zone~1 heat-call $\\geq 0.9$). "
    "$T_x$ and $S_x$ are zone $x$'s average temperature and setpoint over the hour."
)
LATEX_LABEL = "tab:maple_zone1_before_after_highlight_hours"


def main() -> None:
    force_refresh = "--refresh" in sys.argv[1:]
    bundle = get_plot_prepared_bundle(HOUSE_ALIASES, force_refresh=force_refresh)
    panels = [p for p in bundle.panels if p.house_alias in MAPLE_ZONE1_PANEL_ALIASES]
    missing = [
        a for a in MAPLE_ZONE1_PANEL_ALIASES if a not in {p.house_alias for p in panels}
    ]
    if missing:
        raise SystemExit(f"Missing panels: {', '.join(missing)}")
    panels.sort(key=lambda p: MAPLE_ZONE1_PANEL_ALIASES.index(p.house_alias))
    df = design_day_zone_scatter_highlight_hours_dataframe(
        panels,
        zone=MAPLE_ZONE,
        oat_at_or_below_c=COLD_OAT_HIGHLIGHT_THRESHOLD_C,
        house_order=MAPLE_ZONE1_PANEL_ALIASES,
    )
    DEFAULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_CSV, index=False)
    df.to_excel(DEFAULT_XLSX, index=False)
    write_design_day_zone_hours_latex_table(
        df,
        DEFAULT_TEX,
        caption=LATEX_CAPTION,
        label=LATEX_LABEL,
        leading_columns=("marker",),
    )
    logger.info(
        "Wrote %d Maple zone 1 highlight hour row(s) to %s, %s, and %s",
        len(df),
        DEFAULT_CSV,
        DEFAULT_XLSX,
        DEFAULT_TEX,
    )
    for marker in ("red", "purple"):
        n = int((df["marker"] == marker).sum()) if len(df) else 0
        logger.info("  %s: %d", marker, n)


if __name__ == "__main__":
    main()