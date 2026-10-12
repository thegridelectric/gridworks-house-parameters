"""
2×3 grid of SWT vs house energy plots for all houses in HOUSE_ALIASES.
Hours with max zone heat-call ≥ 0.9 are shown. Energy estimates use the
non-occupancy feature set (no weekday/weekend hour dummies).
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from house_parameters import (
    HouseEnergyParams,
    HouseEnergyParamsComputer,
    write_energy_regression_table_xlsx,
    write_maple_operating_characteristics_xlsx,
)
from plots import EnergyVsSwtPanelData, plot_estimated_energy_vs_swt_grid

START_TIME = datetime(2025, 11, 1, tzinfo=ZoneInfo("America/New_York"))
END_TIME = datetime(2026, 3, 1, tzinfo=ZoneInfo("America/New_York"))

HOUSE_ALIASES = [
    "beech",
    "oak",
    "fir",
    "maple_before",
    "maple_after",
    "elm",
]

DEFAULT_SAVEPATH = Path("results") / "all_houses_energy_vs_swt_grid.pdf"
DEFAULT_REGRESSION_XLSX = Path("results") / "all_houses_energy_regression.xlsx"
DEFAULT_MAPLE_OPERATING_XLSX = (
    Path("results") / "maple_operating_characteristics_before_after.xlsx"
)


def load_panels_and_params(
    house_aliases: list[str],
) -> tuple[
    list[EnergyVsSwtPanelData],
    dict[str, HouseEnergyParams],
    dict[str, HouseEnergyParamsComputer],
]:
    panels: list[EnergyVsSwtPanelData] = []
    params_by_alias: dict[str, HouseEnergyParams] = {}
    computers_by_alias: dict[str, HouseEnergyParamsComputer] = {}
    errors: list[str] = []
    for house in house_aliases:
        logger.info("Loading %s …", house)
        try:
            computer = HouseEnergyParamsComputer(
                house,
                START_TIME,
                END_TIME,
                include_occupancy=False,
            )
            energy_params = computer.energy_fit_on_all_cleaned_data()
            params_by_alias[house] = energy_params
            computers_by_alias[house] = computer
            panels.append(computer.energy_vs_swt_panel_data(energy_params))
        except Exception as exc:
            errors.append(f"{house}: {exc}")
            logger.exception("Failed for %s", house)
    if errors:
        raise SystemExit(
            "Could not load all houses for the grid plot:\n" + "\n".join(errors)
        )
    return panels, params_by_alias, computers_by_alias


def _load_uncleaned_oat_df(house_alias: str) -> pd.DataFrame:
    computer = HouseEnergyParamsComputer(
        house_alias,
        START_TIME,
        END_TIME,
        prepare=False,
    )
    return computer.df[["hour_start", "oat_f"]].copy()


def load_composite_uncleaned_oat(
    primary_alias: str,
    fill_aliases: list[str],
) -> tuple[pd.Series, np.ndarray]:
    """Oak-first hourly OAT; missing/non-finite hours filled from other houses."""
    primary = _load_uncleaned_oat_df(primary_alias)
    hour_start = primary["hour_start"]
    oat = pd.to_numeric(primary["oat_f"], errors="coerce").to_numpy(dtype=float).copy()
    missing = ~np.isfinite(oat)

    for alias in fill_aliases:
        if alias == primary_alias:
            continue
        try:
            other = _load_uncleaned_oat_df(alias)
        except Exception:
            logger.exception("Skipping OAT fill from %s", alias)
            continue
        other_oat = (
            other.set_index("hour_start")["oat_f"]
            .reindex(hour_start)
            .to_numpy(dtype=float)
        )
        fill = missing & np.isfinite(other_oat)
        oat[fill] = other_oat[fill]
        missing = ~np.isfinite(oat)
        if not missing.any():
            break

    n_filled = int((~np.isfinite(pd.to_numeric(primary["oat_f"], errors="coerce")) & np.isfinite(oat)).sum())
    if n_filled:
        logger.info(
            "OAT series (%s): filled %s missing hours from other houses",
            primary_alias,
            n_filled,
        )
    return hour_start, oat


def main() -> None:
    savepath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAVEPATH
    panels, params_by_alias, computers_by_alias = load_panels_and_params(
        HOUSE_ALIASES
    )
    oat_hour_start, oat_f = load_composite_uncleaned_oat("oak", HOUSE_ALIASES)
    plot_estimated_energy_vs_swt_grid(
        panels,
        savepath=savepath,
        oat_hour_start=oat_hour_start,
        oat_f=oat_f,
        maple_transition_vline=HouseEnergyParamsComputer.MAPLE_AFTER_FIRST_INCLUDED_HOUR,
    )
    write_energy_regression_table_xlsx(
        params_by_alias,
        DEFAULT_REGRESSION_XLSX,
        house_order=HOUSE_ALIASES,
    )
    write_maple_operating_characteristics_xlsx(
        computers_by_alias["maple_before"],
        computers_by_alias["maple_after"],
        DEFAULT_MAPLE_OPERATING_XLSX,
    )
    logger.info("Wrote grid plot to %s", savepath)
    logger.info("Wrote regression table to %s", DEFAULT_REGRESSION_XLSX)
    logger.info("Wrote Maple operating table to %s", DEFAULT_MAPLE_OPERATING_XLSX)


if __name__ == "__main__":
    main()
