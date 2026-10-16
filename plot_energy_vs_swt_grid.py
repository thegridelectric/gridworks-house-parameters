"""
2×3 grid of SWT vs house energy plots for all houses in HOUSE_ALIASES.
Hours with max zone heat-call ≥ 0.9 are shown. Energy estimates use the
non-occupancy feature set (no weekday/weekend hour dummies).

Prepared inputs are cached under results/plot_energy_vs_swt_grid_cache/ so
figures and tables can be regenerated quickly. Pass --refresh to rebuild.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from dataclasses import dataclass, replace
from typing import TypeVar
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
    maple_operating_stats_by_oat_bin,
    write_energy_regression_table_xlsx,
    write_maple_operating_characteristics_tables_xlsx,
)
from plots import (
    MAPLE_ZONE_PAIR_FIG_HEIGHT,
    MAPLE_ZONE_PAIR_FIG_WIDTH,
    EnergyVsSwtPanelData,
    OatCleaningHighlightPanel,
    energy_vs_swt_scatter_oat_c_for_zone,
    plot_estimated_energy_vs_swt_grid,
    plot_houses_oat_cleaning_removals,
    zone_legend_hour_percent_for_panel,
)

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
DEFAULT_MAIN_GRID_NO_OAT_SAVEPATH = (
    Path("results") / "all_houses_energy_vs_swt_grid_house_panels_only.pdf"
)
DEFAULT_MONOCHROME_COLD_OAT_SAVEPATH = (
    Path("results") / "all_houses_energy_vs_swt_grid_monochrome_cold_oat.pdf"
)
DEFAULT_MONOCHROME_COLD_OAT_NO_OAT_SAVEPATH = (
    Path("results")
    / "all_houses_energy_vs_swt_grid_monochrome_cold_oat_house_panels_only.pdf"
)
COLD_OAT_HIGHLIGHT_THRESHOLD_C = -20.6
DEFAULT_PER_ZONE_SAVEPATH = (
    Path("results") / "all_houses_energy_vs_swt_grid_per_zone_heatcall.pdf"
)
DEFAULT_CLEANING_HIGHLIGHT_SAVEPATH = (
    Path("results") / "all_houses_energy_vs_swt_grid_cleaning_excluded.pdf"
)
DEFAULT_MAPLE_ZONE1_BEFORE_AFTER_SAVEPATH = (
    Path("results") / "maple_zone1_energy_vs_swt_before_after.pdf"
)
DEFAULT_MAPLE_ZONE2_BEFORE_AFTER_SAVEPATH = (
    Path("results") / "maple_zone2_energy_vs_swt_before_after.pdf"
)
DEFAULT_REGRESSION_XLSX = Path("results") / "all_houses_energy_regression.xlsx"
DEFAULT_MAPLE_OPERATING_XLSX = (
    Path("results") / "maple_operating_characteristics_before_after.xlsx"
)
DEFAULT_OAT_CLEANING_DIR = Path("results") / "oat_cleaning_removed"
PLOT_CACHE_DIR = Path("results") / "plot_energy_vs_swt_grid_cache"
PLOT_CACHE_META_PATH = PLOT_CACHE_DIR / "meta.json"
PLOT_CACHE_BUNDLE_PATH = PLOT_CACHE_DIR / "bundle.pkl"
PLOT_CACHE_VERSION = 13

# Per-site OAT cleaning PDFs under DEFAULT_OAT_CLEANING_DIR (not the 2×3 grid figures).
MAKE_INDIVIDUAL_PLOTS = False

# Keep hours with oil boiler on in cleaned data used for grid plots and regression export.
DROP_OIL_BOILER_HOURS = False

# Elm main grid (dominant-zone coloring): omit garage from zones and legend.
ELM_MAIN_GRID_EXCLUDED_ZONES = frozenset({3})


@dataclass
class GridPlotPreparedBundle:
    panels: list[EnergyVsSwtPanelData]
    params_by_alias: dict[str, HouseEnergyParams]
    oat_grid_hour_start: pd.Series
    oat_grid_f: np.ndarray
    oat_cleaning_panels: list[OatCleaningHighlightPanel]
    maple_operating_before: pd.DataFrame
    maple_operating_after: pd.DataFrame


def _cache_source_paths(house_aliases: list[str]) -> list[Path]:
    data_dir = HouseEnergyParamsComputer.HOURLY_DATA_DIR
    paths: list[Path] = [
        data_dir / HouseEnergyParamsComputer.MAPLE_ELECTRICITY_CSV.name,
    ]
    seen: set[Path] = set()
    for alias in house_aliases:
        for name in (
            f"{alias}_input_data.csv",
            f"{alias}_house_params_data.csv",
        ):
            path = data_dir / name
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _plot_cache_fingerprint(house_aliases: list[str]) -> dict[str, object]:
    sources: dict[str, float | None] = {}
    for path in _cache_source_paths(house_aliases):
        sources[str(path)] = (
            float(path.stat().st_mtime) if path.is_file() else None
        )
    return {
        "cache_version": PLOT_CACHE_VERSION,
        "start_time": START_TIME.isoformat(),
        "end_time": END_TIME.isoformat(),
        "house_aliases": list(house_aliases),
        "include_occupancy": False,
        "drop_oil_boiler_hours": DROP_OIL_BOILER_HOURS,
        "sources": sources,
    }


def _oat_by_hour_from_computer(
    computer: HouseEnergyParamsComputer,
) -> pd.Series:
    raw = computer.loaded_raw_df()
    return pd.to_numeric(
        raw.set_index("hour_start")["oat_f"],
        errors="coerce",
    )


def composite_uncleaned_oat(
    primary_alias: str,
    fill_aliases: list[str],
    computers_by_alias: dict[str, HouseEnergyParamsComputer],
) -> tuple[pd.Series, np.ndarray]:
    """Primary-house OAT with missing hours filled from other houses (in memory)."""
    primary_oat = _oat_by_hour_from_computer(computers_by_alias[primary_alias])
    hour_start = primary_oat.index.to_series()
    oat = primary_oat.to_numpy(dtype=float).copy()
    missing = ~np.isfinite(oat)

    for alias in fill_aliases:
        if alias == primary_alias:
            continue
        try:
            other_oat = _oat_by_hour_from_computer(computers_by_alias[alias])
        except KeyError:
            logger.warning("Skipping OAT fill from %s (not loaded)", alias)
            continue
        other_vals = other_oat.reindex(primary_oat.index).to_numpy(dtype=float)
        fill = missing & np.isfinite(other_vals)
        oat[fill] = other_vals[fill]
        missing = ~np.isfinite(oat)
        if not missing.any():
            break

    n_filled = int((~np.isfinite(primary_oat.to_numpy(dtype=float)) & np.isfinite(oat)).sum())
    if n_filled:
        logger.info(
            "OAT series (%s): filled %s missing hours from other houses",
            primary_alias,
            n_filled,
        )
    return hour_start, oat


def swt_and_heatcall_for_timeline(
    computer: HouseEnergyParamsComputer,
    hour_start: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """SWT and max zone heat-call from post-load data, aligned to ``hour_start``."""
    df = computer.loaded_raw_df()
    df = df.copy()
    df["hour_start"] = pd.to_datetime(df["hour_start"])
    max_hc = computer._max_zone_heatcall_fraction(df)
    by_hour = df.set_index("hour_start")
    timeline = pd.DatetimeIndex(pd.to_datetime(hour_start))
    swt_f = by_hour["swt_average_f"].reindex(timeline).to_numpy(dtype=float)
    max_zone_heatcall = (
        pd.Series(max_hc.to_numpy(dtype=float), index=df["hour_start"])
        .reindex(timeline)
        .to_numpy(dtype=float)
    )
    return swt_f, max_zone_heatcall


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
                drop_oil_boiler_hours=DROP_OIL_BOILER_HOURS,
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


def load_oat_cleaning_panels(
    house_aliases: list[str],
    computers_by_alias: dict[str, HouseEnergyParamsComputer],
) -> list[OatCleaningHighlightPanel]:
    panels: list[OatCleaningHighlightPanel] = []
    for house in house_aliases:
        logger.info("Building OAT cleaning panel for %s …", house)
        computer = computers_by_alias[house]
        hour_start, oat_f = composite_uncleaned_oat(
            house, house_aliases, computers_by_alias
        )
        drop_reasons = computer.compute_cleaning_drop_reasons()
        hs = pd.to_datetime(hour_start)
        drop_reason_key = np.array(
            [drop_reasons.get(pd.Timestamp(h), "") for h in hs],
            dtype=object,
        )
        n_removed = int(np.sum(drop_reason_key != ""))
        logger.info("%s: %s hour(s) removed by clean data", house, n_removed)
        swt_f, max_zone_heatcall = swt_and_heatcall_for_timeline(computer, hour_start)
        panels.append(
            OatCleaningHighlightPanel(
                house_alias=house,
                hour_start=hour_start,
                oat_f=oat_f,
                drop_reason_key=drop_reason_key,
                swt_f=swt_f,
                max_zone_heatcall=max_zone_heatcall,
            )
        )
    return panels


def build_plot_prepared_bundle(house_aliases: list[str]) -> GridPlotPreparedBundle:
    panels, params_by_alias, computers_by_alias = load_panels_and_params(
        house_aliases
    )
    oat_hour_start, oat_f = composite_uncleaned_oat(
        "oak", house_aliases, computers_by_alias
    )
    oat_cleaning_panels = load_oat_cleaning_panels(
        house_aliases, computers_by_alias
    )
    maple_before = maple_operating_stats_by_oat_bin(
        computers_by_alias["maple_before"]
    )
    maple_after = maple_operating_stats_by_oat_bin(computers_by_alias["maple_after"])
    return GridPlotPreparedBundle(
        panels=panels,
        params_by_alias=params_by_alias,
        oat_grid_hour_start=oat_hour_start,
        oat_grid_f=oat_f,
        oat_cleaning_panels=oat_cleaning_panels,
        maple_operating_before=maple_before,
        maple_operating_after=maple_after,
    )


def save_plot_prepared_bundle(bundle: GridPlotPreparedBundle, house_aliases: list[str]) -> None:
    PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _plot_cache_fingerprint(house_aliases)
    PLOT_CACHE_META_PATH.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    with PLOT_CACHE_BUNDLE_PATH.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved prepared plot data to %s", PLOT_CACHE_DIR)


def load_plot_prepared_bundle(house_aliases: list[str]) -> GridPlotPreparedBundle | None:
    if not PLOT_CACHE_META_PATH.is_file() or not PLOT_CACHE_BUNDLE_PATH.is_file():
        return None
    try:
        stored = json.loads(PLOT_CACHE_META_PATH.read_text())
    except json.JSONDecodeError:
        return None
    if stored != _plot_cache_fingerprint(house_aliases):
        return None
    with PLOT_CACHE_BUNDLE_PATH.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, GridPlotPreparedBundle):
        return None
    return bundle


def get_plot_prepared_bundle(
    house_aliases: list[str],
    *,
    force_refresh: bool = False,
) -> GridPlotPreparedBundle:
    if not force_refresh:
        cached = load_plot_prepared_bundle(house_aliases)
        if cached is not None:
            logger.info("Using cached prepared plot data from %s", PLOT_CACHE_DIR)
            return cached
    logger.info("Preparing plot data from house inputs …")
    bundle = build_plot_prepared_bundle(house_aliases)
    save_plot_prepared_bundle(bundle, house_aliases)
    return bundle


_T = TypeVar("_T")


def _drop_zone_keys(d: dict[int, _T] | None, excluded: frozenset[int]) -> dict[int, _T] | None:
    if d is None:
        return None
    return {k: v for k, v in d.items() if k not in excluded}


def _dominant_zone_per_hour_excluding(
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray],
    zone_order: tuple[int, ...],
    excluded: frozenset[int],
    n_hours: int,
) -> np.ndarray:
    active_zones = [z for z in zone_order if z not in excluded]
    if not active_zones:
        raise ValueError("No zones left after exclusions")
    out = np.empty(n_hours, dtype=int)
    for i in range(n_hours):
        best_z = active_zones[0]
        best_hc = -1.0
        for z in active_zones:
            hc = zone_heatcall_fraction_by_hour.get(z)
            if hc is None or i >= len(hc):
                continue
            v = float(hc[i])
            if v > best_hc:
                best_hc = v
                best_z = z
        out[i] = best_z
    return out


def _panel_excluding_zones(
    panel: EnergyVsSwtPanelData,
    excluded: frozenset[int],
    *,
    preserve_zone_legend_hour_percent: bool = False,
) -> EnergyVsSwtPanelData:
    if not excluded:
        return panel
    zone_heatcall = panel.zone_heatcall_fraction_by_hour or {}
    new_zone_numbers = tuple(z for z in panel.zone_numbers if z not in excluded)
    dominant = _dominant_zone_per_hour_excluding(
        zone_heatcall,
        panel.zone_numbers,
        excluded,
        len(panel.estimated_kwh),
    )
    percent_override = None
    if preserve_zone_legend_hour_percent:
        percent_override = zone_legend_hour_percent_for_panel(
            panel,
            threshold=0.9,
        )
    return replace(
        panel,
        zone_numbers=new_zone_numbers,
        dominant_zone=dominant,
        zone_names=_drop_zone_keys(panel.zone_names, excluded),
        zone_avg_set_f=_drop_zone_keys(panel.zone_avg_set_f, excluded),
        coldest_hour_zone_set_f=_drop_zone_keys(panel.coldest_hour_zone_set_f, excluded),
        zone_set_f_by_hour=_drop_zone_keys(panel.zone_set_f_by_hour, excluded),
        zone_heatcall_fraction_by_hour=_drop_zone_keys(zone_heatcall, excluded),
        zone_legend_hour_percent_override=percent_override,
    )


def _panels_for_main_energy_grid(
    panels: list[EnergyVsSwtPanelData],
) -> list[EnergyVsSwtPanelData]:
    out: list[EnergyVsSwtPanelData] = []
    for panel in panels:
        if panel.house_alias == "elm":
            out.append(
                _panel_excluding_zones(
                    panel,
                    ELM_MAIN_GRID_EXCLUDED_ZONES,
                    preserve_zone_legend_hour_percent=True,
                )
            )
        else:
            out.append(panel)
    return out


def _panels_by_alias(
    bundle: GridPlotPreparedBundle,
    aliases: list[str],
) -> list[EnergyVsSwtPanelData]:
    by_alias = {p.house_alias: p for p in bundle.panels}
    missing = [a for a in aliases if a not in by_alias]
    if missing:
        raise ValueError(f"Missing panel data for: {', '.join(missing)}")
    return [by_alias[a] for a in aliases]


def render_all_outputs(
    bundle: GridPlotPreparedBundle,
    *,
    grid_savepath: Path,
) -> None:
    _main_energy_grid_panels = _panels_for_main_energy_grid(bundle.panels)
    _figure1_kw = dict(
        maple_transition_vline=HouseEnergyParamsComputer.MAPLE_AFTER_FIRST_INCLUDED_HOUR,
        zone_legend_hour_percent=True,
        show_coldest_max_swt_in_legend=False,
    )
    plot_estimated_energy_vs_swt_grid(
        _main_energy_grid_panels,
        savepath=grid_savepath,
        oat_hour_start=bundle.oat_grid_hour_start,
        oat_f=bundle.oat_grid_f,
        **_figure1_kw,
    )
    plot_estimated_energy_vs_swt_grid(
        _main_energy_grid_panels,
        savepath=DEFAULT_MAIN_GRID_NO_OAT_SAVEPATH,
        **_figure1_kw,
    )
    _monochrome_cold_oat_panels = _main_energy_grid_panels
    _monochrome_cold_oat_kw = dict(
        scatter_monochrome=True,
        highlight_oat_at_or_below_c=COLD_OAT_HIGHLIGHT_THRESHOLD_C,
    )
    plot_estimated_energy_vs_swt_grid(
        _monochrome_cold_oat_panels,
        savepath=DEFAULT_MONOCHROME_COLD_OAT_SAVEPATH,
        oat_hour_start=bundle.oat_grid_hour_start,
        oat_f=bundle.oat_grid_f,
        maple_transition_vline=HouseEnergyParamsComputer.MAPLE_AFTER_FIRST_INCLUDED_HOUR,
        **_monochrome_cold_oat_kw,
    )
    plot_estimated_energy_vs_swt_grid(
        _monochrome_cold_oat_panels,
        savepath=DEFAULT_MONOCHROME_COLD_OAT_NO_OAT_SAVEPATH,
        **_monochrome_cold_oat_kw,
    )
    plot_estimated_energy_vs_swt_grid(
        bundle.panels,
        savepath=DEFAULT_PER_ZONE_SAVEPATH,
        oat_hour_start=bundle.oat_grid_hour_start,
        oat_f=bundle.oat_grid_f,
        maple_transition_vline=HouseEnergyParamsComputer.MAPLE_AFTER_FIRST_INCLUDED_HOUR,
        scatter_per_active_zone=True,
    )
    plot_estimated_energy_vs_swt_grid(
        bundle.panels,
        savepath=DEFAULT_CLEANING_HIGHLIGHT_SAVEPATH,
        oat_hour_start=bundle.oat_grid_hour_start,
        oat_f=bundle.oat_grid_f,
        maple_transition_vline=HouseEnergyParamsComputer.MAPLE_AFTER_FIRST_INCLUDED_HOUR,
        highlight_cleaning_excluded=True,
        cleaning_highlight_oil_boiler=DROP_OIL_BOILER_HOURS,
    )
    maple_before, maple_after = _panels_by_alias(
        bundle, ["maple_before", "maple_after"]
    )
    _maple_zone_pair_kw = dict(
        nrows=1,
        ncols=2,
        scatter_per_active_zone=True,
        show_zone_legend=False,
        fig_width=MAPLE_ZONE_PAIR_FIG_WIDTH,
        fig_height=MAPLE_ZONE_PAIR_FIG_HEIGHT,
        show_coldest_oat_marker=False,
        zone_in_subplot_title=False,
        scatter_monochrome=True,
        highlight_oat_at_or_below_c=COLD_OAT_HIGHLIGHT_THRESHOLD_C,
        highlight_design_day_below_setpoint=True,
    )
    _maple_before_after_titles = ["Before", "After"]
    _maple_after_zone1_oat = energy_vs_swt_scatter_oat_c_for_zone(
        maple_after,
        1,
        show_coldest_oat_marker=False,
    )
    logger.info(
        "Maple after retrofit, zone 1 scatter OAT (°C), n=%d:",
        len(_maple_after_zone1_oat),
    )
    for _hour, _oat_c in _maple_after_zone1_oat:
        if _hour is not None:
            logger.info("  %s  %.2f°C", _hour, _oat_c)
        else:
            logger.info("  %.2f°C", _oat_c)
    plot_estimated_energy_vs_swt_grid(
        [maple_before, maple_after],
        savepath=DEFAULT_MAPLE_ZONE1_BEFORE_AFTER_SAVEPATH,
        zone_scatter_filters=[1, 1],
        fig_suptitle="Maple Zone 1 before/after retrofit",
        subplot_titles=_maple_before_after_titles,
        **_maple_zone_pair_kw,
    )
    plot_estimated_energy_vs_swt_grid(
        [maple_before, maple_after],
        savepath=DEFAULT_MAPLE_ZONE2_BEFORE_AFTER_SAVEPATH,
        zone_scatter_filters=[2, 2],
        fig_suptitle="Maple Zone 2 before/after retrofit",
        subplot_titles=_maple_before_after_titles,
        **_maple_zone_pair_kw,
    )
    if MAKE_INDIVIDUAL_PLOTS:
        energy_panels_by_alias = {p.house_alias: p for p in bundle.panels}
        oat_cleaning_paths = plot_houses_oat_cleaning_removals(
            bundle.oat_cleaning_panels,
            save_dir=DEFAULT_OAT_CLEANING_DIR,
            energy_panels_by_alias=energy_panels_by_alias,
        )
    else:
        oat_cleaning_paths = []
    write_energy_regression_table_xlsx(
        bundle.params_by_alias,
        DEFAULT_REGRESSION_XLSX,
        house_order=HOUSE_ALIASES,
    )
    write_maple_operating_characteristics_tables_xlsx(
        bundle.maple_operating_before,
        bundle.maple_operating_after,
        DEFAULT_MAPLE_OPERATING_XLSX,
    )
    logger.info("Wrote grid plot to %s", grid_savepath)
    logger.info(
        "Wrote figure 1 house-panels-only grid plot to %s",
        DEFAULT_MAIN_GRID_NO_OAT_SAVEPATH,
    )
    logger.info(
        "Wrote monochrome cold-OAT grid plot to %s",
        DEFAULT_MONOCHROME_COLD_OAT_SAVEPATH,
    )
    logger.info(
        "Wrote monochrome cold-OAT house-panels-only grid plot to %s",
        DEFAULT_MONOCHROME_COLD_OAT_NO_OAT_SAVEPATH,
    )
    logger.info("Wrote per-zone heat-call grid plot to %s", DEFAULT_PER_ZONE_SAVEPATH)
    logger.info(
        "Wrote cleaning-excluded highlight grid plot to %s",
        DEFAULT_CLEANING_HIGHLIGHT_SAVEPATH,
    )
    logger.info(
        "Wrote Maple zone 1 before/after plot to %s",
        DEFAULT_MAPLE_ZONE1_BEFORE_AFTER_SAVEPATH,
    )
    logger.info(
        "Wrote Maple zone 2 before/after plot to %s",
        DEFAULT_MAPLE_ZONE2_BEFORE_AFTER_SAVEPATH,
    )
    if MAKE_INDIVIDUAL_PLOTS:
        logger.info(
            "Wrote OAT cleaning figures (%s) to %s",
            len(oat_cleaning_paths),
            DEFAULT_OAT_CLEANING_DIR,
        )
    logger.info("Wrote regression table to %s", DEFAULT_REGRESSION_XLSX)
    logger.info("Wrote Maple operating table to %s", DEFAULT_MAPLE_OPERATING_XLSX)


def _parse_cli() -> tuple[Path, bool]:
    args = [a for a in sys.argv[1:] if a]
    force_refresh = False
    if "--refresh" in args:
        force_refresh = True
        args = [a for a in args if a != "--refresh"]
    savepath = Path(args[0]) if args else DEFAULT_SAVEPATH
    return savepath, force_refresh


def main() -> None:
    grid_savepath, force_refresh = _parse_cli()
    bundle = get_plot_prepared_bundle(HOUSE_ALIASES, force_refresh=force_refresh)
    render_all_outputs(bundle, grid_savepath=grid_savepath)


if __name__ == "__main__":
    main()
