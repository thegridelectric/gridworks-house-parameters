from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyVsSwtPanelData:
    house_alias: str
    estimated_kwh: np.ndarray
    dist_kwh: np.ndarray
    swt_f: np.ndarray
    max_zone_heatcall: np.ndarray
    dominant_zone: np.ndarray
    zone_numbers: tuple[int, ...]
    coldest_oat_estimated_kwh: float | None
    coldest_oat_dist_kwh: float | None
    coldest_oat_swt_f: float | None
    coldest_oat_panel_index: int | None = None
    coldest_oat_date_label: str | None = None
    oat_f: np.ndarray | None = None
    hour_start: np.ndarray | None = None
    zone_avg_set_f: dict[int, float] | None = None
    coldest_hour_zone_set_f: dict[int, float] | None = None
    zone_set_f_by_hour: dict[int, np.ndarray] | None = None
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray] | None = None
    zone_names: dict[int, str] | None = None
    # Hours dropped for oil boiler or below setpoint (highlight grid figure).
    cleaning_excluded_below_setpoint_estimated_kwh: np.ndarray | None = None
    cleaning_excluded_below_setpoint_swt_f: np.ndarray | None = None
    cleaning_excluded_below_setpoint_oat_f: np.ndarray | None = None
    cleaning_excluded_below_setpoint_hour_start: np.ndarray | None = None
    cleaning_excluded_below_setpoint_zone_heatcall: dict[int, np.ndarray] | None = None
    cleaning_excluded_below_setpoint_zone_was_below: dict[int, np.ndarray] | None = None
    cleaning_excluded_oil_boiler_estimated_kwh: np.ndarray | None = None
    cleaning_excluded_oil_boiler_swt_f: np.ndarray | None = None
    # Legend hour % from full zone set (e.g. Elm figure 1 keeps zone 3 in the math).
    zone_legend_hour_percent_override: dict[int, float] | None = None


@dataclass(frozen=True)
class OatCleaningHighlightPanel:
    house_alias: str
    hour_start: pd.Series | np.ndarray
    oat_f: np.ndarray
    drop_reason_key: np.ndarray
    swt_f: np.ndarray
    max_zone_heatcall: np.ndarray


class RswtLoadBucketFit(Protocol):
    rwt_intercept: float
    rwt_slope: float


def _optional_float_array(arr: np.ndarray | None) -> np.ndarray:
    if arr is None:
        return np.empty(0, dtype=float)
    return np.asarray(arr, dtype=float)


def _cleaning_excluded_scatter_xy(
    estimated_kwh: np.ndarray | None,
    swt_f: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ex_x = _optional_float_array(estimated_kwh)
    ex_swt = _optional_float_array(swt_f)
    if ex_x.size and ex_swt.size == ex_x.size:
        ex_y = _swt_f_to_c(ex_swt)
        finite = np.isfinite(ex_x) & np.isfinite(ex_y)
        return ex_x[finite], ex_y[finite]
    return np.empty(0, dtype=float), np.empty(0, dtype=float)


def _raise_panel_legend_above_data_markers(legend) -> None:
    if legend is not None:
        legend.set_zorder(10)


def _set_scatter_xy(scatter, x: np.ndarray, y: np.ndarray) -> None:
    if x.size:
        scatter.set_offsets(np.column_stack([x, y]))
        scatter.set_visible(True)
    else:
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_visible(False)


def _hourly_residual_mean_and_ci95(
    residuals: np.ndarray, hour_of_day: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    by_hour = (
        pd.DataFrame({"hour": hour_of_day, "residual": residuals})
        .groupby("hour", sort=True)["residual"]
        .agg(["mean", "std", "count"])
        .reindex(range(24))
    )
    count = np.asarray(by_hour["count"], dtype=float)
    std = np.asarray(by_hour["std"], dtype=float)
    sem = std / np.sqrt(count)
    ci95 = np.where(count >= 2, 1.96 * sem, 0.0)
    ci95 = np.where(np.isnan(ci95), 0.0, ci95)
    return np.asarray(by_hour["mean"]), ci95


def _plot_hourly_residual_bars_on_ax(
    ax,
    residuals: np.ndarray,
    hour_of_day: pd.Series,
    *,
    house_alias: str,
    panel_title: str,
) -> None:
    means, ci95 = _hourly_residual_mean_and_ci95(residuals, hour_of_day)
    hours = np.arange(24)
    ax.bar(
        hours,
        means,
        yerr=ci95,
        color="tab:blue",
        alpha=0.85,
        width=0.8,
        capsize=3,
        error_kw={"linewidth": 1, "ecolor": "0.25"},
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(hours)
    ax.set_ylabel("Mean residual (predicted − actual, kWh)")
    ax.set_title(
        f"{house_alias.capitalize()} - {panel_title} - "
        f"mean error and 95% CI on energy use prediction"
    )


def plot_pred_vs_actual(
    predicted: np.ndarray,
    actual: np.ndarray,
    oat_f: np.ndarray,
    title: str,
    *,
    oat_f_colormap_bounds: tuple[float, float],
    savepath: Path | None = None,
    baseline_mae: float | None = None,
    baseline_rmse: float | None = None,
    oos_period_label: str = "next-day",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    oat_values = np.asarray(oat_f, dtype=float)
    errors = predicted - actual

    oat_min, oat_max = oat_f_colormap_bounds
    if oat_min >= oat_max:
        oat_max = oat_min + 1.0
    norm = Normalize(vmin=oat_min, vmax=oat_max)
    cmap = plt.cm.coolwarm

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(actual, predicted, c=oat_values, cmap=cmap, norm=norm, s=8, alpha=0.6)
    hi = float(max(actual.max(), predicted.max()) * 1.05)
    limits = [0, hi]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual scaled dist_kwh (kWh)")
    ax.set_ylabel("Predicted scaled dist_kwh (kWh)")
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors**2).mean()))

    scatter_title = f"{oos_period_label.capitalize()} out-of-sample predictions"
    if baseline_mae is not None and baseline_rmse is not None:
        rmse_pct = (baseline_rmse - rmse) / baseline_rmse * 100.0
        mae_pct = (baseline_mae - mae) / baseline_mae * 100.0
        scatter_title += (
            "\n"
            + f"RMSE {abs(rmse_pct):.0f}% {'better' if rmse_pct >= 0 else 'worse'}"
            + ", "
            + f"MAE {abs(mae_pct):.0f}% {'better' if mae_pct >= 0 else 'worse'}"
            + " than αβγ"
        )
    ax.set_title(scatter_title)

    stats = f"MAE  = {mae:.3f} kWh\n"
    if baseline_mae is not None:
        stats += f"MAE αβγ = {baseline_mae:.3f} kWh\n"
    stats += f"RMSE = {rmse:.3f} kWh\n"
    if baseline_rmse is not None:
        stats += f"RMSE αβγ = {baseline_rmse:.3f} kWh"
    stats = stats.rstrip()
    ax.text(
        0.97, 0.03, stats, transform=ax.transAxes, ha="right", va="bottom",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax = axes[1]
    ax.hist(errors, bins=60, range=(-6, 6), color="tab:blue", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("Prediction error (scaled kWh)")
    ax.set_ylabel("Hours")
    ax.set_title("Error distribution")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.02)
    ticks = np.linspace(oat_min, oat_max, num=6, retstep=False).tolist()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0f}°F" for t in ticks])

    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _energy_vs_swt_regression_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[str, np.ndarray, np.ndarray]:
    if len(x) < 2:
        return f"n = {len(x)} hour(s)", np.array([]), np.array([])
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), num=100)
    y_line = slope * x_line + intercept
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    stats = (
        f"n = {len(x)} hours\n"
        f"SWT (°C) = {intercept:.2f} + {slope:.2f} × energy\n"
        f"R² = {r_squared:.3f}"
    )
    return stats, x_line, y_line


def _swt_f_to_c(swt_f: np.ndarray) -> np.ndarray:
    return (np.asarray(swt_f, dtype=float) - 32.0) * (5.0 / 9.0)


def plot_estimated_energy_vs_swt(
    estimated_kwh: np.ndarray,
    dist_kwh: np.ndarray,
    swt_f: np.ndarray,
    max_zone_heatcall: np.ndarray,
    dominant_zone: np.ndarray,
    *,
    zone_numbers: tuple[int, ...],
    house_alias: str,
    savepath: Path | None = None,
    zone_names: dict[int, str] | None = None,
    coldest_oat_estimated_kwh: float | None = None,
    coldest_oat_dist_kwh: float | None = None,
    coldest_oat_swt_f: float | None = None,
    coldest_oat_panel_index: int | None = None,
    coldest_oat_date_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.widgets import CheckButtons, Slider

    estimated_all = np.asarray(estimated_kwh, dtype=float)
    dist_kwh_all = np.asarray(dist_kwh, dtype=float)
    y_all = _swt_f_to_c(swt_f)
    heatcall_all = np.asarray(max_zone_heatcall, dtype=float)
    dominant_zone_all = np.asarray(dominant_zone, dtype=int)

    coldest_x_estimated: float | None = None
    coldest_x_measured: float | None = None
    coldest_y_c: float | None = None
    if coldest_oat_swt_f is not None and np.isfinite(coldest_oat_swt_f):
        coldest_y_c = float(_swt_f_to_c(np.array([coldest_oat_swt_f]))[0])
        if coldest_oat_estimated_kwh is not None and np.isfinite(coldest_oat_estimated_kwh):
            coldest_x_estimated = float(coldest_oat_estimated_kwh)
        if coldest_oat_dist_kwh is not None and np.isfinite(coldest_oat_dist_kwh):
            coldest_x_measured = float(coldest_oat_dist_kwh)

    sorted_zones = tuple(sorted(zone_numbers))
    zone_to_index = {zone: index for index, zone in enumerate(sorted_zones)}
    palette = plt.cm.tab10.colors
    zone_cmap = ListedColormap(
        [palette[i % len(palette)] for i in range(len(sorted_zones))]
    )

    def zone_color_indices(zone_ids: np.ndarray) -> np.ndarray:
        return np.array([zone_to_index[int(z)] for z in zone_ids], dtype=float)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_axes([0.1, 0.22, 0.82, 0.68])
    ax_slider = fig.add_axes([0.15, 0.08, 0.55, 0.03])
    ax_check = fig.add_axes([0.73, 0.06, 0.24, 0.06])
    ax_check.set_facecolor("0.95")
    ax_check.set_xticks([])
    ax_check.set_yticks([])

    threshold_init = 0.9
    use_measured_init = False

    def x_values(use_measured: bool) -> np.ndarray:
        return dist_kwh_all if use_measured else estimated_all

    def coldest_x_for_mode(use_measured: bool) -> float | None:
        return coldest_x_measured if use_measured else coldest_x_estimated

    mask = heatcall_all >= threshold_init
    x, y = x_values(use_measured_init)[mask], y_all[mask]
    zone_idx = zone_color_indices(dominant_zone_all[mask])

    sc = ax.scatter(
        x,
        y,
        c=zone_idx,
        cmap=zone_cmap,
        vmin=0,
        vmax=max(len(sorted_zones) - 1, 0),
        s=14,
        alpha=0.65,
    )
    coldest_x_init = coldest_x_for_mode(use_measured_init)
    (coldest_marker,) = ax.plot(
        [],
        [],
        marker="o",
        markersize=14,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
        linestyle="none",
        zorder=6,
        visible=False,
    )

    ax.set_xlabel(_energy_vs_swt_x_axis_label(use_measured_init))
    ax.set_ylabel(ENERGY_VS_SWT_YLABEL)
    ax.set_ylim(35, 90)
    ax.set_title(f"{house_alias.capitalize()}: SWT vs estimated house energy (cleaned data)")

    legend_handles = _energy_vs_swt_legend_handles(
        sorted_zones,
        palette,
        zone_names,
    )
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    slider = Slider(
        ax_slider,
        "Min max zone heat-call",
        0.0,
        1.0,
        valinit=threshold_init,
        valstep=0.01,
    )
    use_measured_check = CheckButtons(
        ax_check,
        ["Use measured dist_kwh"],
        [use_measured_init],
    )

    def refresh_plot() -> None:
        threshold = slider.val
        use_measured = use_measured_check.get_status()[0]
        x_all = x_values(use_measured)
        filtered = heatcall_all >= threshold
        filtered_idx = np.flatnonzero(filtered)
        xf = x_all[filtered]
        yf = y_all[filtered]
        zone_idx_filtered = zone_color_indices(dominant_zone_all[filtered])
        xf_vis, yf_vis, zone_idx_vis = xf, yf, zone_idx_filtered
        if (
            coldest_oat_panel_index is not None
            and coldest_oat_panel_index not in filtered_idx
        ):
            ci = coldest_oat_panel_index
            xf_vis = np.append(xf_vis, x_all[ci])
            yf_vis = np.append(yf_vis, y_all[ci])
            zone_idx_vis = np.append(
                zone_idx_vis,
                zone_to_index[int(dominant_zone_all[ci])],
            )
        if xf_vis.size:
            sc.set_offsets(np.column_stack([xf_vis, yf_vis]))
            sc.set_array(zone_idx_vis)
            sc.set_visible(True)
        else:
            sc.set_offsets(np.empty((0, 2)))
            sc.set_array(np.array([]))
            sc.set_visible(False)
        coldest_x = coldest_x_for_mode(use_measured)
        coldest_marker.set_visible(False)
        ax.set_xlabel(_energy_vs_swt_x_axis_label(use_measured))
        ax.set_ylim(35, 90)
        x_lim_candidates: list[float] = []
        if xf.size:
            x_lim_candidates.extend(xf.tolist())
        if coldest_x is not None and coldest_y_c is not None:
            x_lim_candidates.append(coldest_x)
        if x_lim_candidates:
            x_min = float(np.min(x_lim_candidates))
            x_max = float(np.max(x_lim_candidates))
            x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.5
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
        fig.canvas.draw_idle()

    slider.on_changed(lambda _val: refresh_plot())
    use_measured_check.on_clicked(lambda _label: refresh_plot())
    refresh_plot()

    def save_on_close(_event) -> None:
        if savepath is None:
            return
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    fig.canvas.mpl_connect("close_event", save_on_close)

    plt.show()
    plt.close(fig)


_SUBPLOT_TITLE_OVERRIDES: dict[str, str] = {
    "maple_before": "Maple before retrofit",
    "maple_after": "Maple after retrofit",
}

_COLDEST_OAT_MARKER_SUPPRESSED_HOUSE_ALIASES = frozenset({"maple_before"})


def _panel_show_coldest_oat_marker(
    panel: EnergyVsSwtPanelData,
    *,
    show_coldest_oat_marker: bool,
) -> bool:
    if not show_coldest_oat_marker:
        return False
    return panel.house_alias not in _COLDEST_OAT_MARKER_SUPPRESSED_HOUSE_ALIASES


def _house_alias_subplot_title(house_alias: str) -> str:
    override = _SUBPLOT_TITLE_OVERRIDES.get(house_alias)
    if override is not None:
        return override
    return house_alias.replace("_", " ").title()


ENERGY_VS_SWT_XLABEL_ESTIMATED = "Estimated house energy use (kWh)"
SWT_GRID_FIG_WIDTH = 14.0
SWT_GRID_FIG_HEIGHT = 8.0
MAPLE_ZONE_PAIR_FIG_WIDTH = 10.5
MAPLE_ZONE_PAIR_FIG_HEIGHT = 4.4
OAT_PANEL_FIG_HEIGHT = 3.8
ENERGY_VS_SWT_YLABEL = "Average SWT (°C)"
MULTI_ACTIVE_ZONE_SCATTER_COLOR = "#22c55e"
MULTI_ACTIVE_ZONE_LEGEND_LABEL = "Multiple zones active (≥ threshold)"
DESIGN_DAY_BELOW_SETPOINT_SCATTER_COLOR = "#A855F7"
SCATTER_MONOCHROME_GRAY = "0.72"
SCATTER_MONOCHROME_GRAY_ALPHA = 0.6
_MAPLE_MONOCHROME_PANEL_ALIASES = frozenset({"maple_before", "maple_after"})
GRID_SCATTER_MARKER_SIZE = 28
GRID_PLOT_RING_MARKER_SIZE = 16
MAPLE_MONOCHROME_SCATTER_MARKER_SIZE = GRID_SCATTER_MARKER_SIZE
MAPLE_MONOCHROME_PLOT_RING_MARKER_SIZE = GRID_PLOT_RING_MARKER_SIZE
TAB10_BLUE = "#1f77b4"
TAB10_ORANGE = "#ff7f0e"


def _maple_monochrome_enhanced_panel(panel: EnergyVsSwtPanelData, scatter_monochrome: bool) -> bool:
    return scatter_monochrome and panel.house_alias in _MAPLE_MONOCHROME_PANEL_ALIASES


def _panel_show_design_day_below_setpoint(
    panel: EnergyVsSwtPanelData,
    *,
    highlight_design_day_below_setpoint: bool,
    scatter_monochrome: bool,
) -> bool:
    if panel.house_alias != "maple_before":
        return False
    return highlight_design_day_below_setpoint or _maple_monochrome_enhanced_panel(
        panel, scatter_monochrome
    )


def _zone_scatter_facecolor(
    zone: int,
    sorted_zones: tuple[int, ...],
    palette,
    zone_color_overrides: dict[int, str] | None,
) -> str:
    if zone_color_overrides and zone in zone_color_overrides:
        return zone_color_overrides[zone]
    return palette[sorted_zones.index(zone) % len(palette)]


def _coldest_hour_legend_label(oat_f: float) -> str:
    return f"Coldest hour: {float(oat_f):.1f}°F"


def _hottest_swt_hour_legend_label(swt_c: float) -> str:
    return f"Hottest average SWT: {float(swt_c):.1f}°C"


def _oat_coldest_hour_legend_label(
    hour_start: pd.Timestamp,
    oat_c: float,
) -> str:
    when_str = pd.Timestamp(hour_start).strftime("%b %d, %Y %H:%M")
    return f"Coldest hour: {when_str}, {float(oat_c):.1f}°C"


def _oat_timeseries_legend_handles(
    *,
    show_coldest_marker: bool,
    coldest_legend_label: str | None = None,
    show_max_swt_markers: bool,
    maple_transition_label: str | None,
    highlight_oat_at_or_below_c: float | None = None,
    show_design_day_below_setpoint_line: bool = False,
) -> list:
    from matplotlib.lines import Line2D

    handles: list[Line2D] = []
    if highlight_oat_at_or_below_c is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color=SCATTER_MONOCHROME_GRAY,
                linewidth=1.5,
                label="Outside air temperature",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=1.5,
                label="Design day or colder",
            )
        )
    if show_design_day_below_setpoint_line:
        handles.append(
            Line2D(
                [0],
                [0],
                color=DESIGN_DAY_BELOW_SETPOINT_SCATTER_COLOR,
                linewidth=1.5,
                linestyle="--",
                label="Design day or colder, below setpoint",
            )
        )
    if maple_transition_label:
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                linestyle="--",
                linewidth=1.2,
                label=maple_transition_label,
            )
        )
    if show_coldest_marker:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.5,
                markersize=8,
                label=coldest_legend_label or "Coldest hour",
            )
        )
    if show_max_swt_markers:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=1.5,
                markersize=8,
                label="Highest SWT hour (per site)",
            )
        )
    return handles


def max_swt_hour_start_at_heatcall_threshold(
    panel: EnergyVsSwtPanelData,
    threshold: float,
) -> pd.Timestamp | None:
    """Hour start of the max-SWT point among hours with max zone heat-call ≥ threshold."""
    heatcall = np.asarray(panel.max_zone_heatcall, dtype=float)
    y_all = _swt_f_to_c(panel.swt_f)
    filtered = heatcall >= threshold
    if not np.any(filtered):
        return None
    yf = y_all[filtered]
    local_max_i = int(np.argmax(yf))
    filtered_idx = np.flatnonzero(filtered)
    gi = int(filtered_idx[local_max_i])
    if panel.hour_start is None or gi >= len(panel.hour_start):
        return None
    return pd.Timestamp(panel.hour_start[gi])


def coldest_hour_in_cold_oat_condition_start(
    panel: EnergyVsSwtPanelData,
    *,
    oat_at_or_below_c: float,
    heatcall_threshold: float,
) -> pd.Timestamp | None:
    """Hour start of min OAT among heat-call scatter hours with OAT ≤ threshold."""
    estimated_all = np.asarray(panel.estimated_kwh, dtype=float)
    y_all = _swt_f_to_c(panel.swt_f)
    heatcall_all = np.asarray(panel.max_zone_heatcall, dtype=float)
    dominant_zone_all = np.asarray(panel.dominant_zone, dtype=int)
    sorted_zones = tuple(sorted(panel.zone_numbers))
    zone_to_index = {zone: i for i, zone in enumerate(sorted_zones)}
    multi_zone_color_index = float(len(sorted_zones))
    zone_heatcall_by_hour = panel.zone_heatcall_fraction_by_hour or {}
    panel_oat_c = _panel_oat_c_series(panel.oat_f)
    if panel_oat_c is None or not np.asarray(panel_oat_c).size:
        return None
    pin_coldest = panel.house_alias not in _COLDEST_OAT_MARKER_SUPPRESSED_HOUSE_ALIASES
    x0, y0, _, hour_idx0 = _energy_vs_swt_scatter_xy(
        estimated_all,
        y_all,
        heatcall_all,
        dominant_zone_all,
        zone_to_index,
        threshold=heatcall_threshold,
        scatter_per_active_zone=False,
        zone_heatcall_fraction_by_hour=zone_heatcall_by_hour,
        coldest_panel_index=panel.coldest_oat_panel_index,
        multi_zone_color_index=multi_zone_color_index,
        zone_scatter_only=None,
        pin_coldest_in_scatter=pin_coldest,
    )
    _, _, _, hour_idx = _coldest_hour_in_cold_oat_condition(
        x0,
        y0,
        hour_idx0,
        np.asarray(panel_oat_c, dtype=float),
        float(oat_at_or_below_c),
    )
    if hour_idx is None or panel.hour_start is None:
        return None
    if hour_idx < 0 or hour_idx >= len(panel.hour_start):
        return None
    return pd.Timestamp(panel.hour_start[hour_idx])


def _oat_time_and_value_at_hour(
    hour: pd.Timestamp,
    oat_by_time: pd.Series,
) -> tuple[pd.Timestamp, float] | None:
    if hour not in oat_by_time.index:
        nearest = oat_by_time.index.get_indexer([hour], method="nearest")
        if not nearest.size or nearest[0] < 0:
            return None
        oat_at = float(oat_by_time.iloc[nearest[0]])
        t_at = oat_by_time.index[nearest[0]]
    else:
        oat_at = float(oat_by_time.loc[hour])
        t_at = hour
    if not np.isfinite(oat_at):
        return None
    return pd.Timestamp(t_at), oat_at


def _draw_site_max_swt_marker_on_oat(
    ax,
    hour_start: pd.Series | np.ndarray,
    oat_c: np.ndarray,
    energy_panel: EnergyVsSwtPanelData,
    *,
    heatcall_threshold: float = 0.9,
) -> bool:
    """Green circle at this site's max-SWT hour (heat-call threshold), on an OAT (°C) series."""
    hour = max_swt_hour_start_at_heatcall_threshold(energy_panel, heatcall_threshold)
    if hour is None:
        return False
    oat_times = pd.to_datetime(hour_start)
    oat_by_time = pd.Series(np.asarray(oat_c, dtype=float), index=oat_times)
    if hour not in oat_by_time.index:
        nearest = oat_by_time.index.get_indexer([hour], method="nearest")
        if not nearest.size or nearest[0] < 0:
            return False
        oat_at = float(oat_by_time.iloc[nearest[0]])
        t_at = oat_by_time.index[nearest[0]]
    else:
        oat_at = float(oat_by_time.loc[hour])
        t_at = hour
    if not np.isfinite(oat_at):
        return False
    ax.plot(
        [pd.Timestamp(t_at)],
        [oat_at],
        marker="o",
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="green",
        markeredgewidth=1.5,
        linestyle="none",
        zorder=5,
    )
    return True


def _draw_oat_timeseries_on_ax(
    ax,
    hour_start: pd.Series | np.ndarray,
    oat_f: np.ndarray,
    panels: list[EnergyVsSwtPanelData],
    *,
    heatcall_threshold: float = 0.9,
    axis_label_fontsize: float = 10,
    tick_fontsize: float = 9,
    legend_fontsize: float = 9,
    maple_transition_vline: pd.Timestamp | None = None,
    maple_transition_vline_label: str | None = "Maple retrofit",
    highlight_oat_at_or_below_c: float | None = None,
    scatter_monochrome: bool = False,
    highlight_design_day_below_setpoint: bool = False,
    zone_scatter_filters: list[int] | None = None,
) -> None:
    """Hourly OAT (°C), red = coldest hour, green = max-SWT hour per house."""
    oat_times = pd.to_datetime(hour_start)
    oat_f_arr = np.asarray(oat_f, dtype=float)
    oat_vals = _swt_f_to_c(oat_f_arr)
    purple_hours: frozenset[pd.Timestamp] = frozenset()

    if highlight_oat_at_or_below_c is not None:
        ax.plot(
            oat_times,
            oat_vals,
            color=SCATTER_MONOCHROME_GRAY,
            linewidth=1.5,
            alpha=0.9,
            zorder=1,
        )
        cold = np.isfinite(oat_vals) & (
            oat_vals <= float(highlight_oat_at_or_below_c)
        )
        y_red = np.where(cold, oat_vals, np.nan)
        ax.plot(
            oat_times,
            y_red,
            color="red",
            linewidth=1.5,
            alpha=0.95,
            zorder=2,
        )
        if scatter_monochrome:
            purple_hours = _design_day_below_setpoint_hours_for_oat_strip(
                panels,
                oat_at_or_below_c=float(highlight_oat_at_or_below_c),
                heatcall_threshold=heatcall_threshold,
                scatter_monochrome=True,
                highlight_design_day_below_setpoint=highlight_design_day_below_setpoint,
                zone_scatter_filters=zone_scatter_filters,
            )
        if purple_hours:
            purple_mask = np.asarray(
                pd.DatetimeIndex(oat_times).isin(list(purple_hours)),
                dtype=bool,
            )
            y_purple = np.where(
                purple_mask & np.isfinite(oat_vals),
                oat_vals,
                np.nan,
            )
            ax.plot(
                oat_times,
                y_purple,
                color=DESIGN_DAY_BELOW_SETPOINT_SCATTER_COLOR,
                linewidth=1.8,
                linestyle="--",
                dashes=(5, 3),
                alpha=0.95,
                zorder=3,
            )
    else:
        ax.plot(oat_times, oat_vals, color="tab:blue", linewidth=1.5, alpha=0.6, zorder=1)

    if maple_transition_vline is not None:
        ax.axvline(
            pd.Timestamp(maple_transition_vline),
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
        )

    oat_by_time = pd.Series(oat_vals, index=oat_times)
    green_markers: list[tuple[pd.Timestamp, float, str]] = []
    for panel in panels:
        hour = max_swt_hour_start_at_heatcall_threshold(panel, heatcall_threshold)
        if hour is None:
            continue
        located = _oat_time_and_value_at_hour(hour, oat_by_time)
        if located is None:
            continue
        t_at, oat_at = located
        house_label = _house_alias_subplot_title(panel.house_alias)
        green_markers.append((t_at, oat_at, house_label))

    label_stack: dict[tuple[pd.Timestamp, float], int] = {}
    for t_at, oat_at, house_label in green_markers:
        ax.plot(
            [t_at],
            [oat_at],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="green",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=5,
        )
        stack_key = (t_at, round(oat_at, 2))
        stack_i = label_stack.get(stack_key, 0)
        label_stack[stack_key] = stack_i + 1
        ax.annotate(
            house_label,
            xy=(t_at, oat_at),
            xytext=(5, 5 + stack_i * 9),
            textcoords="offset points",
            color="green",
            fontsize=7,
            ha="left",
            va="bottom",
            zorder=6,
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 1.0,
            },
        )

    transition_legend = (
        maple_transition_vline_label
        if maple_transition_vline is not None and maple_transition_vline_label
        else None
    )
    show_purple_oat_line = bool(purple_hours)
    legend_handles = _oat_timeseries_legend_handles(
        show_coldest_marker=False,
        coldest_legend_label=None,
        show_max_swt_markers=bool(green_markers),
        maple_transition_label=transition_legend,
        highlight_oat_at_or_below_c=highlight_oat_at_or_below_c,
        show_design_day_below_setpoint_line=show_purple_oat_line,
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        framealpha=0.9,
        fontsize=legend_fontsize,
    )

    ax.set_ylabel("Outside air temperature (°C)", fontsize=axis_label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)


OAT_CLEANING_HIGHLIGHT_ORDER: tuple[str, ...] = (
    "below_setpoint",
    "oil_boiler",
    "other",
)

OAT_CLEANING_HIGHLIGHT_COLORS: dict[str, str] = {
    "below_setpoint": "#f9a8d4",
    "oil_boiler": "#c4b5fd",
    "other": "#d1d5db",
}

OAT_CLEANING_HIGHLIGHT_LABELS: dict[str, str] = {
    "below_setpoint": "Below setpoint",
    "oil_boiler": "Oil boiler used",
    "other": "Other",
}

OAT_CLEANING_SWT_SUBPANEL_HEIGHT = 3.2


def _oat_cleaning_display_reason_keys(drop_reason_key: np.ndarray) -> np.ndarray:
    """Map detailed clean-data drop reasons to plot highlight categories."""
    keys = np.asarray(drop_reason_key, dtype=object)
    display = np.empty(len(keys), dtype=object)
    for i, key in enumerate(keys):
        if not key:
            display[i] = ""
        elif key == "below_setpoint":
            display[i] = "below_setpoint"
        elif key == "oil_boiler":
            display[i] = "oil_boiler"
        else:
            display[i] = "other"
    return display


def _draw_cleaning_removal_spans(
    ax,
    hour_start: pd.Series | np.ndarray,
    drop_reason_key: np.ndarray,
    *,
    reason_order: tuple[str, ...],
    reason_colors: dict[str, str],
) -> set[str]:
    """Vertical bands by first ``_clean_data`` drop reason for each hour."""
    times = pd.to_datetime(hour_start)
    keys = np.asarray(drop_reason_key, dtype=object)
    if keys.size != len(times):
        return set()
    reasons_present: set[str] = set()
    for reason in reason_order:
        color = reason_colors.get(reason)
        if color is None:
            continue
        removed = keys == reason
        if not removed.any():
            continue
        reasons_present.add(reason)
        i = 0
        n = len(removed)
        while i < n:
            if not removed[i]:
                i += 1
                continue
            j = i
            while j < n and removed[j]:
                j += 1
            t0 = pd.Timestamp(times[i])
            t1 = pd.Timestamp(times[j - 1]) + pd.Timedelta(hours=1)
            ax.axvspan(
                t0,
                t1,
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
                zorder=0,
            )
            i = j
    return reasons_present


SWT_BAR_HIGH_HEATCALL_COLOR = "green"


def _draw_hourly_swt_heatcall_bars_on_ax(
    ax,
    hour_start: pd.Series | np.ndarray,
    swt_f: np.ndarray,
    max_zone_heatcall: np.ndarray,
    *,
    heatcall_threshold: float = 0.9,
    axis_label_fontsize: float = 10,
    tick_fontsize: float = 9,
) -> bool:
    times = pd.to_datetime(hour_start)
    swt_c = _swt_f_to_c(np.asarray(swt_f, dtype=float))
    heatcall = np.asarray(max_zone_heatcall, dtype=float)
    if swt_c.size != len(times) or heatcall.size != len(times):
        return False
    bar_mask = np.isfinite(swt_c) & np.isfinite(heatcall) & (
        heatcall >= heatcall_threshold
    )
    if not np.any(bar_mask):
        ax.set_ylabel(ENERGY_VS_SWT_YLABEL, fontsize=axis_label_fontsize)
        ax.set_ylim(40, 90)
        ax.tick_params(labelsize=tick_fontsize)
        return False
    bar_times = times[bar_mask]
    bar_heights = swt_c[bar_mask]
    ax.bar(
        bar_times,
        bar_heights,
        width=pd.Timedelta(hours=1),
        align="edge",
        color=SWT_BAR_HIGH_HEATCALL_COLOR,
        linewidth=0.15,
        edgecolor="0.45",
        zorder=2,
    )
    max_bar_c = float(np.max(bar_heights))
    ax.axhline(
        max_bar_c,
        color="red",
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    ax.set_ylabel(ENERGY_VS_SWT_YLABEL, fontsize=axis_label_fontsize)
    ax.set_ylim(40, 90)
    ax.tick_params(labelsize=tick_fontsize)
    return True


def plot_oat_cleaning_removals(
    panel: OatCleaningHighlightPanel,
    *,
    energy_panel: EnergyVsSwtPanelData | None = None,
    savepath: Path | None = None,
    heatcall_threshold: float = 0.9,
) -> None:
    """
    One figure: composite-filled hourly OAT (°C) with hours removed by
    ``HouseEnergyParamsComputer._clean_data`` highlighted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    title_fontsize = 12
    axis_label_fontsize = 10
    tick_fontsize = 9
    legend_fontsize = 9

    fig_height = OAT_PANEL_FIG_HEIGHT + OAT_CLEANING_SWT_SUBPANEL_HEIGHT
    fig, (ax_oat, ax_swt) = plt.subplots(
        2,
        1,
        figsize=(SWT_GRID_FIG_WIDTH, fig_height),
        sharex=True,
        gridspec_kw={
            "height_ratios": [OAT_PANEL_FIG_HEIGHT, OAT_CLEANING_SWT_SUBPANEL_HEIGHT],
            "hspace": 0.06,
        },
    )
    times = pd.to_datetime(panel.hour_start)
    oat_c = _swt_f_to_c(np.asarray(panel.oat_f, dtype=float))

    display_keys = _oat_cleaning_display_reason_keys(panel.drop_reason_key)
    span_kwargs = {
        "reason_order": OAT_CLEANING_HIGHLIGHT_ORDER,
        "reason_colors": OAT_CLEANING_HIGHLIGHT_COLORS,
    }
    reasons_present = _draw_cleaning_removal_spans(
        ax_oat,
        times,
        display_keys,
        **span_kwargs,
    )
    _draw_cleaning_removal_spans(
        ax_swt,
        times,
        display_keys,
        **span_kwargs,
    )
    ax_oat.plot(
        times,
        oat_c,
        color="tab:blue",
        linewidth=1.2,
        alpha=0.75,
        zorder=2,
    )
    has_max_swt_marker = False
    if energy_panel is not None and energy_panel.house_alias == panel.house_alias:
        has_max_swt_marker = _draw_site_max_swt_marker_on_oat(
            ax_oat,
            times,
            oat_c,
            energy_panel,
            heatcall_threshold=heatcall_threshold,
        )
    ax_oat.set_title(
        _house_alias_subplot_title(panel.house_alias),
        fontsize=title_fontsize,
    )
    ax_oat.set_ylabel("Outside air temperature (°C)", fontsize=axis_label_fontsize)
    ax_oat.tick_params(labelsize=tick_fontsize)

    has_swt_bars = _draw_hourly_swt_heatcall_bars_on_ax(
        ax_swt,
        panel.hour_start,
        panel.swt_f,
        panel.max_zone_heatcall,
        heatcall_threshold=heatcall_threshold,
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
    )
    ax_swt.set_xlabel("Hour start", fontsize=axis_label_fontsize)

    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(
            facecolor=OAT_CLEANING_HIGHLIGHT_COLORS[reason],
            edgecolor="none",
            alpha=0.85,
            label=OAT_CLEANING_HIGHLIGHT_LABELS[reason],
        )
        for reason in OAT_CLEANING_HIGHLIGHT_ORDER
        if reason in reasons_present
    ]
    if has_swt_bars:
        legend_handles.append(
            Patch(
                facecolor=SWT_BAR_HIGH_HEATCALL_COLOR,
                edgecolor="0.45",
                alpha=1.0,
                label="Max zone heat-call ≥ 90%",
            )
        )
    if has_max_swt_marker:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=1.5,
                markersize=8,
                label="Highest SWT hour (per site)",
            )
        )
    if legend_handles:
        ax_oat.legend(
            handles=legend_handles,
            loc="upper left",
            framealpha=0.9,
            fontsize=legend_fontsize,
        )

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_houses_oat_cleaning_removals(
    panels: Sequence[OatCleaningHighlightPanel],
    *,
    save_dir: Path,
    energy_panels_by_alias: dict[str, EnergyVsSwtPanelData] | None = None,
    heatcall_threshold: float = 0.9,
) -> list[Path]:
    """Write one OAT cleaning figure PDF per site under ``save_dir``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for panel in panels:
        path = save_dir / f"{panel.house_alias}_oat_cleaning_removed.pdf"
        energy_panel = (
            energy_panels_by_alias.get(panel.house_alias)
            if energy_panels_by_alias
            else None
        )
        plot_oat_cleaning_removals(
            panel,
            energy_panel=energy_panel,
            savepath=path,
            heatcall_threshold=heatcall_threshold,
        )
        written.append(path)
    return written


def plot_oat_timeseries_with_swt_markers(
    hour_start: pd.Series | np.ndarray,
    oat_f: np.ndarray,
    panels: list[EnergyVsSwtPanelData],
    *,
    heatcall_threshold: float = 0.9,
    savepath: Path | None = None,
) -> None:
    """Standalone wide OAT figure (same drawing as the combined grid bottom panel)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(SWT_GRID_FIG_WIDTH, OAT_PANEL_FIG_HEIGHT))
    _draw_oat_timeseries_on_ax(
        ax,
        hour_start,
        oat_f,
        panels,
        heatcall_threshold=heatcall_threshold,
    )
    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _format_setpoint_f(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.1f}°F"


def _zone_legend_label_with_setpoints(
    zone: int,
    zone_names: dict[int, str] | None,
    zone_avg_set_f: dict[int, float] | None,
    coldest_hour_zone_set_f: dict[int, float] | None,
    green_hour_zone_set_f: dict[int, float] | None,
    *,
    setpoint_detail_zones: frozenset[int] | None = None,
) -> str:
    name = (zone_names or {}).get(zone)
    base = f"Zone {zone} - {name}" if name else f"Zone {zone}"
    show_setpoints = setpoint_detail_zones is None or zone in setpoint_detail_zones
    parts: list[str] = []
    if show_setpoints:
        if zone_avg_set_f and zone in zone_avg_set_f:
            parts.append(f"avg {_format_setpoint_f(zone_avg_set_f[zone])}")
        if coldest_hour_zone_set_f and zone in coldest_hour_zone_set_f:
            parts.append(f"red {_format_setpoint_f(coldest_hour_zone_set_f[zone])}")
        if green_hour_zone_set_f and zone in green_hour_zone_set_f:
            parts.append(f"green {_format_setpoint_f(green_hour_zone_set_f[zone])}")
    if parts:
        base += f" ({', '.join(parts)})"
    return base


def _energy_vs_swt_x_axis_label(use_measured: bool) -> str:
    if use_measured:
        return "Measured distribution energy dist_kwh (kWh)"
    return ENERGY_VS_SWT_XLABEL_ESTIMATED


def _zone_hour_percent_from_scatter(
    sorted_zones: tuple[int, ...],
    zone_idx_vis: np.ndarray,
    *,
    multi_zone_color_index: float | None = None,
) -> dict[int, float]:
    """Share of visible scatter points (hours) per zone, as 0–100 percentages."""
    if not sorted_zones:
        return {}
    index_to_zone = {i: z for i, z in enumerate(sorted_zones)}
    counts: dict[int, int] = {z: 0 for z in sorted_zones}
    total = int(zone_idx_vis.size)
    if total == 0:
        return {z: 0.0 for z in sorted_zones}
    multi_idx = (
        int(round(float(multi_zone_color_index)))
        if multi_zone_color_index is not None
        else len(sorted_zones)
    )
    for raw in zone_idx_vis:
        idx = int(round(float(raw)))
        if idx == multi_idx:
            continue
        zone = index_to_zone.get(idx)
        if zone is not None:
            counts[zone] += 1
    return {z: 100.0 * counts[z] / total for z in sorted_zones}


def zone_legend_hour_percent_for_panel(
    panel: EnergyVsSwtPanelData,
    *,
    threshold: float,
    scatter_per_active_zone: bool = False,
) -> dict[int, float]:
    """Hour % per zone among heat-call scatter hours (includes all panel.zone_numbers)."""
    estimated_all = np.asarray(panel.estimated_kwh, dtype=float)
    y_all = _swt_f_to_c(panel.swt_f)
    heatcall_all = np.asarray(panel.max_zone_heatcall, dtype=float)
    dominant_zone_all = np.asarray(panel.dominant_zone, dtype=int)
    sorted_zones = tuple(sorted(panel.zone_numbers))
    zone_to_index = {zone: i for i, zone in enumerate(sorted_zones)}
    multi_zone_color_index = float(len(sorted_zones))
    zone_heatcall_by_hour = panel.zone_heatcall_fraction_by_hour or {}
    pin_coldest = panel.house_alias not in _COLDEST_OAT_MARKER_SUPPRESSED_HOUSE_ALIASES
    _x, _y, zone_idx_vis, _hour_idx = _energy_vs_swt_scatter_xy(
        estimated_all,
        y_all,
        heatcall_all,
        dominant_zone_all,
        zone_to_index,
        threshold=threshold,
        scatter_per_active_zone=scatter_per_active_zone,
        zone_heatcall_fraction_by_hour=zone_heatcall_by_hour,
        coldest_panel_index=panel.coldest_oat_panel_index,
        multi_zone_color_index=multi_zone_color_index,
        zone_scatter_only=None,
        pin_coldest_in_scatter=pin_coldest,
    )
    return _zone_hour_percent_from_scatter(
        sorted_zones,
        zone_idx_vis,
        multi_zone_color_index=(
            multi_zone_color_index if scatter_per_active_zone else None
        ),
    )


def _near_constant_heatcall_zone_legend_overrides(
    legend_meta: dict,
    *,
    scatter_point_count: int,
) -> dict[int, str] | None:
    if not legend_meta.get("zone_legend_near_constant_heatcall"):
        return None
    zones_in_legend = legend_meta.get("zones_in_legend")
    if zones_in_legend is None:
        return None
    n = scatter_point_count
    return {
        int(zone): f"Hours in near-constant heat call (n={n})"
        for zone in zones_in_legend
    }


def _cold_oat_highlight_scatter_xy(
    x: np.ndarray,
    y: np.ndarray,
    hour_indices: np.ndarray,
    panel_oat_c: np.ndarray,
    threshold_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or panel_oat_c.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    oat_c = _oat_c_at_hour_indices(panel_oat_c, hour_indices)
    cold = np.isfinite(oat_c) & (oat_c <= float(threshold_c))
    return x[cold], y[cold]


def _coldest_hour_in_cold_oat_condition(
    x: np.ndarray,
    y: np.ndarray,
    hour_indices: np.ndarray,
    panel_oat_c: np.ndarray,
    threshold_c: float,
) -> tuple[float | None, float | None, float | None, int | None]:
    """Among scatter hours with OAT ≤ threshold, return (x, y, oat_c, hour_idx) at min OAT."""
    if x.size == 0 or panel_oat_c.size == 0:
        return None, None, None, None
    oat_c = _oat_c_at_hour_indices(panel_oat_c, hour_indices)
    cold = np.isfinite(oat_c) & (oat_c <= float(threshold_c))
    if not np.any(cold):
        return None, None, None, None
    cold_oat = oat_c[cold]
    cold_x = x[cold]
    cold_y = y[cold]
    cold_hours = hour_indices[cold]
    local_i = int(np.argmin(cold_oat))
    return (
        float(cold_x[local_i]),
        float(cold_y[local_i]),
        float(cold_oat[local_i]),
        int(cold_hours[local_i]),
    )


def _design_day_below_setpoint_excluded_scatter_xy(
    panel: EnergyVsSwtPanelData,
    *,
    oat_at_or_below_c: float,
    heatcall_threshold: float,
    zone_scatter_only: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    bx, by = _cleaning_excluded_scatter_xy(
        panel.cleaning_excluded_below_setpoint_estimated_kwh,
        panel.cleaning_excluded_below_setpoint_swt_f,
    )
    if bx.size == 0:
        return bx, by
    oats = _optional_float_array(panel.cleaning_excluded_below_setpoint_oat_f)
    if oats.size != bx.size:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    oat_c = _swt_f_to_c(oats)
    keep = np.isfinite(oat_c) & (oat_c <= float(oat_at_or_below_c))
    if zone_scatter_only is not None:
        z = int(zone_scatter_only)
        zhc = (panel.cleaning_excluded_below_setpoint_zone_heatcall or {}).get(z)
        zbelow = (panel.cleaning_excluded_below_setpoint_zone_was_below or {}).get(z)
        if zhc is None or zbelow is None or zhc.size != bx.size or zbelow.size != bx.size:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        keep &= (
            np.isfinite(zhc)
            & (zhc >= float(heatcall_threshold))
            & np.asarray(zbelow, dtype=bool)
        )
    return bx[keep], by[keep]


def _design_day_below_setpoint_hour_starts(
    panel: EnergyVsSwtPanelData,
    *,
    oat_at_or_below_c: float,
    heatcall_threshold: float,
    zone_scatter_only: int | None,
) -> list[pd.Timestamp]:
    hours = panel.cleaning_excluded_below_setpoint_hour_start
    oats = panel.cleaning_excluded_below_setpoint_oat_f
    if hours is None or oats is None:
        return []
    hours_arr = np.asarray(hours)
    oats_arr = np.asarray(oats, dtype=float)
    if hours_arr.size == 0 or oats_arr.size != hours_arr.size:
        return []
    oat_c = _swt_f_to_c(oats_arr)
    keep = np.isfinite(oat_c) & (oat_c <= float(oat_at_or_below_c))
    if zone_scatter_only is not None:
        z = int(zone_scatter_only)
        zhc = (panel.cleaning_excluded_below_setpoint_zone_heatcall or {}).get(z)
        zbelow = (panel.cleaning_excluded_below_setpoint_zone_was_below or {}).get(z)
        if (
            zhc is None
            or zbelow is None
            or zhc.size != hours_arr.size
            or zbelow.size != hours_arr.size
        ):
            return []
        keep &= (
            np.isfinite(zhc)
            & (zhc >= float(heatcall_threshold))
            & np.asarray(zbelow, dtype=bool)
        )
    return [pd.Timestamp(t) for t in hours_arr[keep]]


def _design_day_below_setpoint_hours_for_oat_strip(
    panels: list[EnergyVsSwtPanelData],
    *,
    oat_at_or_below_c: float,
    heatcall_threshold: float,
    scatter_monochrome: bool,
    highlight_design_day_below_setpoint: bool,
    zone_scatter_filters: list[int] | None,
) -> frozenset[pd.Timestamp]:
    hours: set[pd.Timestamp] = set()
    for index, panel in enumerate(panels):
        if not _panel_show_design_day_below_setpoint(
            panel,
            highlight_design_day_below_setpoint=highlight_design_day_below_setpoint,
            scatter_monochrome=scatter_monochrome,
        ):
            continue
        zone_scatter_only = (
            zone_scatter_filters[index] if zone_scatter_filters is not None else None
        )
        for ts in _design_day_below_setpoint_hour_starts(
            panel,
            oat_at_or_below_c=oat_at_or_below_c,
            heatcall_threshold=heatcall_threshold,
            zone_scatter_only=zone_scatter_only,
        ):
            hours.add(ts)
    return frozenset(hours)


def _design_day_colder_swt_legend_label(cold_swt_c: np.ndarray) -> str:
    if cold_swt_c.size == 0:
        return "Design day or colder"
    finite = cold_swt_c[np.isfinite(cold_swt_c)]
    if finite.size == 0:
        return "Design day or colder"
    avg = f"{float(np.mean(finite)):.1f}°C"
    return f"Design day or colder - average required: {avg}"


def _design_day_below_setpoint_max_swt_legend_label(swt_c: np.ndarray) -> str:
    if swt_c.size == 0:
        return "Design day or colder, below setpoint - max"
    finite = swt_c[np.isfinite(swt_c)]
    if finite.size == 0:
        return "Design day or colder, below setpoint - max"
    peak = f"{float(np.max(finite)):.1f}°C"
    return f"Design day or colder, below setpoint - max {peak}"


def _hottest_water_required_swt_legend_label(swt_c: float | None) -> str:
    if swt_c is None or not np.isfinite(swt_c):
        value = "—"
    else:
        value = f"{float(swt_c):.1f}°C"
    return f"Hottest water required: {value}"


def _scatter_monochrome_house_panel_legend_handles(
    *,
    cold_legend_label: str | None,
    max_swt_c: float | None,
    design_day_below_setpoint_legend_label: str | None = None,
    point_marker_size: float = 8,
    max_swt_marker_size: float = 8,
) -> list:
    from matplotlib.lines import Line2D

    handles: list[Line2D] = []
    if cold_legend_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=point_marker_size,
                label=cold_legend_label,
            )
        )
    if design_day_below_setpoint_legend_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=DESIGN_DAY_BELOW_SETPOINT_SCATTER_COLOR,
                markersize=point_marker_size,
                label=design_day_below_setpoint_legend_label,
            )
        )
    if max_swt_c is not None and np.isfinite(max_swt_c):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=2,
                markersize=max_swt_marker_size,
                label=_hottest_water_required_swt_legend_label(max_swt_c),
            )
        )
    return handles


def _energy_vs_swt_cleaning_highlight_legend_handles(
    *,
    coldest_oat_legend_label: str | None = None,
    max_swt_legend_label: str | None = None,
    coldest_marker_size: float = 10,
    max_swt_marker_size: float = 10,
    point_marker_size: float = 8,
    show_oil_boiler_excluded: bool = True,
) -> list:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.55",
            markersize=point_marker_size,
            label="Included in clean data",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=OAT_CLEANING_HIGHLIGHT_COLORS["below_setpoint"],
            markersize=point_marker_size,
            label=OAT_CLEANING_HIGHLIGHT_LABELS["below_setpoint"],
        ),
    ]
    if show_oil_boiler_excluded:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=OAT_CLEANING_HIGHLIGHT_COLORS["oil_boiler"],
                markersize=point_marker_size,
                label=OAT_CLEANING_HIGHLIGHT_LABELS["oil_boiler"],
            )
        )
    if coldest_oat_legend_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
                markersize=coldest_marker_size,
                label=coldest_oat_legend_label,
            )
        )
    if max_swt_legend_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=2,
                markersize=max_swt_marker_size,
                label=max_swt_legend_label,
            )
        )
    return handles


def _energy_vs_swt_legend_handles(
    sorted_zones: tuple[int, ...],
    palette,
    zone_names: dict[int, str] | None,
    *,
    coldest_oat_legend_label: str | None = None,
    max_swt_legend_label: str | None = None,
    zone_avg_set_f: dict[int, float] | None = None,
    coldest_hour_zone_set_f: dict[int, float] | None = None,
    green_hour_zone_set_f: dict[int, float] | None = None,
    setpoint_detail_zones: frozenset[int] | None = None,
    zone_marker_size: float = 8,
    coldest_marker_size: float = 10,
    max_swt_marker_size: float = 10,
    multi_active_zone_label: str | None = None,
    multi_active_zone_color: str | None = None,
    zones_in_legend: tuple[int, ...] | None = None,
    zone_color_overrides: dict[int, str] | None = None,
    zone_label_overrides: dict[int, str] | None = None,
    zone_hour_percent: dict[int, float] | None = None,
    zone_legend_markerfacecolor: str | None = None,
) -> list:
    from matplotlib.lines import Line2D

    legend_zones = zones_in_legend if zones_in_legend is not None else sorted_zones
    handles = []
    for zone in legend_zones:
        if zone not in sorted_zones:
            continue
        if zone_label_overrides is not None and zone in zone_label_overrides:
            label = zone_label_overrides[zone]
        else:
            label = _zone_legend_label_with_setpoints(
                zone,
                zone_names,
                zone_avg_set_f,
                coldest_hour_zone_set_f,
                green_hour_zone_set_f,
                setpoint_detail_zones=setpoint_detail_zones,
            )
            if zone_hour_percent is not None and zone in zone_hour_percent:
                label = f"{label} ({zone_hour_percent[zone]:.0f}%)"
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=(
                    zone_legend_markerfacecolor
                    if zone_legend_markerfacecolor is not None
                    else _zone_scatter_facecolor(
                        zone,
                        sorted_zones,
                        palette,
                        zone_color_overrides,
                    )
                ),
                markersize=zone_marker_size,
                label=label,
            )
        )
    if multi_active_zone_label and multi_active_zone_color:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=multi_active_zone_color,
                markersize=zone_marker_size,
                label=multi_active_zone_label,
            )
        )
    if coldest_oat_legend_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
                markersize=coldest_marker_size,
                label=coldest_oat_legend_label,
            )
        )
    if max_swt_legend_label:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=2,
                markersize=max_swt_marker_size,
                label=max_swt_legend_label,
            )
        )
    return handles


def _active_zones_at_threshold(
    hour_index: int,
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray],
    threshold: float,
) -> list[int]:
    active: list[int] = []
    for zone, hc_series in zone_heatcall_fraction_by_hour.items():
        if hour_index >= len(hc_series):
            continue
        if float(hc_series[hour_index]) >= threshold:
            active.append(int(zone))
    return active


def _energy_vs_swt_regression_mask(
    heatcall_all: np.ndarray,
    threshold: float,
    *,
    zone_scatter_only: int | None,
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray] | None,
) -> np.ndarray:
    if zone_scatter_only is not None and zone_heatcall_fraction_by_hour:
        hc = zone_heatcall_fraction_by_hour.get(zone_scatter_only)
        if hc is None:
            return np.zeros(len(heatcall_all), dtype=bool)
        n = min(len(heatcall_all), len(hc))
        mask = np.zeros(len(heatcall_all), dtype=bool)
        mask[:n] = np.asarray(hc[:n], dtype=float) >= threshold
        return mask
    return heatcall_all >= threshold


def _append_per_active_zone_scatter_point(
    xs: list[float],
    ys: list[float],
    zone_idx: list[float],
    hour_indices: list[int],
    *,
    hour_index: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray],
    zone_to_index: dict[int, int],
    threshold: float,
    multi_zone_color_index: float,
) -> None:
    active = _active_zones_at_threshold(
        hour_index, zone_heatcall_fraction_by_hour, threshold
    )
    if not active:
        return
    xs.append(float(x_all[hour_index]))
    ys.append(float(y_all[hour_index]))
    hour_indices.append(int(hour_index))
    if len(active) >= 2:
        zone_idx.append(multi_zone_color_index)
    else:
        zone_idx.append(float(zone_to_index[active[0]]))


def _energy_vs_swt_scatter_xy(
    x_all: np.ndarray,
    y_all: np.ndarray,
    heatcall_all: np.ndarray,
    dominant_zone_all: np.ndarray,
    zone_to_index: dict[int, int],
    *,
    threshold: float,
    scatter_per_active_zone: bool,
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray] | None,
    coldest_panel_index: int | None,
    multi_zone_color_index: float | None = None,
    zone_scatter_only: int | None = None,
    pin_coldest_in_scatter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scatter coordinates, zone color indices, and source hour index per point."""
    xs: list[float] = []
    ys: list[float] = []
    zone_idx: list[float] = []
    hour_indices: list[int] = []
    multi_idx = (
        float(multi_zone_color_index)
        if multi_zone_color_index is not None
        else float(len(zone_to_index))
    )

    scatter_hours: set[int] = set()
    if (
        zone_scatter_only is not None
        and scatter_per_active_zone
        and zone_heatcall_fraction_by_hour
    ):
        z = int(zone_scatter_only)
        hc = zone_heatcall_fraction_by_hour.get(z)
        if hc is not None and z in zone_to_index:
            for i in range(len(x_all)):
                if i < len(hc) and float(hc[i]) >= threshold:
                    xs.append(float(x_all[i]))
                    ys.append(float(y_all[i]))
                    zone_idx.append(float(zone_to_index[z]))
                    hour_indices.append(int(i))
                    scatter_hours.add(i)
    elif scatter_per_active_zone and zone_heatcall_fraction_by_hour:
        filtered_idx = np.flatnonzero(heatcall_all >= threshold)
        scatter_hours = set(int(i) for i in filtered_idx)
        for i in filtered_idx:
            _append_per_active_zone_scatter_point(
                xs,
                ys,
                zone_idx,
                hour_indices,
                hour_index=int(i),
                x_all=x_all,
                y_all=y_all,
                zone_heatcall_fraction_by_hour=zone_heatcall_fraction_by_hour,
                zone_to_index=zone_to_index,
                threshold=threshold,
                multi_zone_color_index=multi_idx,
            )
    else:
        filtered_idx = np.flatnonzero(heatcall_all >= threshold)
        scatter_hours = set(int(i) for i in filtered_idx)
        for i in filtered_idx:
            i = int(i)
            xs.append(float(x_all[i]))
            ys.append(float(y_all[i]))
            zone_idx.append(float(zone_to_index[int(dominant_zone_all[i])]))
            hour_indices.append(i)

    if (
        pin_coldest_in_scatter
        and coldest_panel_index is not None
        and int(coldest_panel_index) not in scatter_hours
    ):
        ci = int(coldest_panel_index)
        if 0 <= ci < len(x_all):
            if (
                zone_scatter_only is not None
                and scatter_per_active_zone
                and zone_heatcall_fraction_by_hour
            ):
                z = int(zone_scatter_only)
                if z in zone_to_index:
                    xs.append(float(x_all[ci]))
                    ys.append(float(y_all[ci]))
                    zone_idx.append(float(zone_to_index[z]))
                    hour_indices.append(ci)
            elif scatter_per_active_zone and zone_heatcall_fraction_by_hour:
                before = len(xs)
                _append_per_active_zone_scatter_point(
                    xs,
                    ys,
                    zone_idx,
                    hour_indices,
                    hour_index=ci,
                    x_all=x_all,
                    y_all=y_all,
                    zone_heatcall_fraction_by_hour=zone_heatcall_fraction_by_hour,
                    zone_to_index=zone_to_index,
                    threshold=threshold,
                    multi_zone_color_index=multi_idx,
                )
                if len(xs) == before:
                    xs.append(float(x_all[ci]))
                    ys.append(float(y_all[ci]))
                    zone_idx.append(float(zone_to_index[int(dominant_zone_all[ci])]))
                    hour_indices.append(ci)
            else:
                xs.append(float(x_all[ci]))
                ys.append(float(y_all[ci]))
                zone_idx.append(float(zone_to_index[int(dominant_zone_all[ci])]))
                hour_indices.append(ci)

    if not xs:
        return np.empty(0), np.empty(0), np.array([]), np.array([], dtype=int)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zone_idx, dtype=float),
        np.asarray(hour_indices, dtype=int),
    )


def _panel_oat_c_series(oat_f: np.ndarray | None) -> np.ndarray:
    if oat_f is None:
        return np.array([], dtype=float)
    return _swt_f_to_c(np.asarray(oat_f, dtype=float))


def _oat_c_at_hour_indices(oat_c_all: np.ndarray, hour_indices: np.ndarray) -> np.ndarray:
    if hour_indices.size == 0 or oat_c_all.size == 0:
        return np.array([], dtype=float)
    hi = np.asarray(hour_indices, dtype=int)
    out = np.full(hi.shape, np.nan, dtype=float)
    valid = (hi >= 0) & (hi < len(oat_c_all))
    out[valid] = oat_c_all[hi[valid]]
    return out


def _energy_vs_swt_panel_scatter_hour_indices(
    panel: EnergyVsSwtPanelData,
    *,
    zone_scatter_only: int | None,
    threshold: float,
    scatter_per_active_zone: bool,
    show_coldest_oat_marker: bool,
) -> np.ndarray:
    panel_show_coldest = _panel_show_coldest_oat_marker(
        panel,
        show_coldest_oat_marker=show_coldest_oat_marker,
    )
    estimated_all = np.asarray(panel.estimated_kwh, dtype=float)
    y_all = _swt_f_to_c(panel.swt_f)
    heatcall_all = np.asarray(panel.max_zone_heatcall, dtype=float)
    dominant_zone_all = np.asarray(panel.dominant_zone, dtype=int)
    sorted_zones = tuple(sorted(panel.zone_numbers))
    zone_to_index = {zone: i for i, zone in enumerate(sorted_zones)}
    zone_heatcall_by_hour = panel.zone_heatcall_fraction_by_hour or {}
    _, _, _, hour_idx = _energy_vs_swt_scatter_xy(
        estimated_all,
        y_all,
        heatcall_all,
        dominant_zone_all,
        zone_to_index,
        threshold=threshold,
        scatter_per_active_zone=scatter_per_active_zone,
        zone_heatcall_fraction_by_hour=zone_heatcall_by_hour,
        coldest_panel_index=panel.coldest_oat_panel_index,
        multi_zone_color_index=float(len(sorted_zones)),
        zone_scatter_only=zone_scatter_only,
        pin_coldest_in_scatter=panel_show_coldest,
    )
    return hour_idx


def energy_vs_swt_scatter_oat_c_for_zone(
    panel: EnergyVsSwtPanelData,
    zone: int,
    *,
    threshold: float = 0.9,
    scatter_per_active_zone: bool = True,
    show_coldest_oat_marker: bool = False,
) -> list[tuple[pd.Timestamp | None, float]]:
    """OAT (°C) at each scatter hour for one zone filter (same rules as the grid plot)."""
    panel_oat_c = _panel_oat_c_series(panel.oat_f)
    if panel_oat_c.size == 0:
        return []
    hour_idx = _energy_vs_swt_panel_scatter_hour_indices(
        panel,
        zone_scatter_only=zone,
        threshold=threshold,
        scatter_per_active_zone=scatter_per_active_zone,
        show_coldest_oat_marker=show_coldest_oat_marker,
    )
    oat_c = _oat_c_at_hour_indices(panel_oat_c, hour_idx)
    hour_start = panel.hour_start
    rows: list[tuple[pd.Timestamp | None, float]] = []
    for i, oat_val in zip(hour_idx, oat_c):
        if not np.isfinite(oat_val):
            continue
        ts: pd.Timestamp | None = None
        if hour_start is not None and 0 <= int(i) < len(hour_start):
            ts = pd.Timestamp(hour_start[int(i)])
        rows.append((ts, float(oat_val)))
    return rows


def _energy_vs_swt_scatter_oat_c_colormap_bounds(
    panels: list[EnergyVsSwtPanelData],
    *,
    threshold: float,
    scatter_per_active_zone: bool,
    zone_scatter_filters: list[int] | None,
    show_coldest_oat_marker: bool,
) -> tuple[float, float]:
    """Min/max OAT (°C) over hours shown as scatter points across ``panels``."""
    vals: list[float] = []
    for index, panel in enumerate(panels):
        panel_zone_scatter_only = (
            zone_scatter_filters[index] if zone_scatter_filters is not None else None
        )
        panel_oat_c = _panel_oat_c_series(panel.oat_f)
        if panel_oat_c.size == 0:
            continue
        hour_idx = _energy_vs_swt_panel_scatter_hour_indices(
            panel,
            zone_scatter_only=panel_zone_scatter_only,
            threshold=threshold,
            scatter_per_active_zone=scatter_per_active_zone,
            show_coldest_oat_marker=show_coldest_oat_marker,
        )
        oat_at_points = _oat_c_at_hour_indices(panel_oat_c, hour_idx)
        finite = oat_at_points[np.isfinite(oat_at_points)]
        if finite.size:
            vals.extend(finite.tolist())
    if not vals:
        return -15.0, 5.0
    oat_min = float(np.min(vals))
    oat_max = float(np.max(vals))
    if oat_min >= oat_max:
        oat_max = oat_min + 1.0
    return oat_min, oat_max


def _compute_energy_vs_swt_xlim(
    estimated_all: np.ndarray,
    dist_kwh_all: np.ndarray,
    y_all: np.ndarray,
    heatcall_all: np.ndarray,
    *,
    threshold: float,
    use_measured: bool,
    coldest_x_estimated: float | None,
    coldest_x_measured: float | None,
    coldest_y_c: float | None,
    extra_x: np.ndarray | None = None,
    include_regression_x_extent: bool = False,
) -> tuple[float, float] | None:
    x_all = dist_kwh_all if use_measured else estimated_all
    filtered = heatcall_all >= threshold
    xf = x_all[filtered]
    yf = y_all[filtered]
    coldest_x = coldest_x_measured if use_measured else coldest_x_estimated
    x_lim_candidates: list[float] = []
    if xf.size:
        x_lim_candidates.extend(xf.tolist())
    if include_regression_x_extent and len(xf) >= 2:
        _, x_line_f, _y_line_f = _energy_vs_swt_regression_stats(xf, yf)
        x_lim_candidates.extend(x_line_f.tolist())
    if coldest_x is not None and coldest_y_c is not None:
        x_lim_candidates.append(coldest_x)
    if yf.size:
        x_lim_candidates.append(float(xf[int(np.argmax(yf))]))
    if extra_x is not None and extra_x.size:
        x_lim_candidates.extend(
            float(v) for v in extra_x if np.isfinite(v)
        )
    if not x_lim_candidates:
        return None
    x_min = float(np.min(x_lim_candidates))
    x_max = float(np.max(x_lim_candidates))
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.5
    return x_min - x_pad, x_max + x_pad


def _refresh_energy_vs_swt_panel(
    ax,
    *,
    estimated_all: np.ndarray,
    dist_kwh_all: np.ndarray,
    y_all: np.ndarray,
    heatcall_all: np.ndarray,
    dominant_zone_all: np.ndarray,
    zone_to_index: dict[int, int],
    threshold: float,
    use_measured: bool,
    coldest_x_estimated: float | None,
    coldest_x_measured: float | None,
    coldest_y_c: float | None,
    coldest_panel_index: int | None = None,
    show_coldest_oat_marker: bool = True,
    scatter,
    reg_line,
    coldest_marker,
    max_swt_marker,
    set_xlabel: bool,
    highlight_above_max_y: bool = False,
    highlight_span_ref: list | None = None,
    ylim_top: float = 90,
    oat_all: np.ndarray | None = None,
    hour_start_all: np.ndarray | None = None,
    zone_set_f_by_hour: dict[int, np.ndarray] | None = None,
    zone_heatcall_fraction_by_hour: dict[int, np.ndarray] | None = None,
    scatter_per_active_zone: bool = False,
    multi_zone_color_index: float | None = None,
    zone_scatter_only: int | None = None,
    highlight_cleaning_excluded: bool = False,
    cleaning_excluded_below_scatter=None,
    cleaning_excluded_oil_scatter=None,
    cleaning_excluded_below_estimated: np.ndarray | None = None,
    cleaning_excluded_below_swt_f: np.ndarray | None = None,
    cleaning_excluded_oil_estimated: np.ndarray | None = None,
    cleaning_excluded_oil_swt_f: np.ndarray | None = None,
    cold_oat_scatter=None,
    design_day_below_setpoint_scatter=None,
    legend_meta: dict | None = None,
    fixed_xlim: tuple[float, float] | None = None,
) -> None:
    x_all = dist_kwh_all if use_measured else estimated_all
    filtered = _energy_vs_swt_regression_mask(
        heatcall_all,
        threshold,
        zone_scatter_only=zone_scatter_only,
        zone_heatcall_fraction_by_hour=zone_heatcall_fraction_by_hour,
    )
    filtered_idx = np.flatnonzero(filtered)
    xf = x_all[filtered]
    yf = y_all[filtered]
    xf_vis, yf_vis, zone_idx_vis, hour_idx_vis = _energy_vs_swt_scatter_xy(
        x_all,
        y_all,
        heatcall_all,
        dominant_zone_all,
        zone_to_index,
        threshold=threshold,
        scatter_per_active_zone=scatter_per_active_zone,
        zone_heatcall_fraction_by_hour=zone_heatcall_fraction_by_hour,
        coldest_panel_index=coldest_panel_index,
        multi_zone_color_index=multi_zone_color_index,
        zone_scatter_only=zone_scatter_only,
        pin_coldest_in_scatter=show_coldest_oat_marker,
    )
    if xf_vis.size:
        scatter.set_offsets(np.column_stack([xf_vis, yf_vis]))
        if highlight_cleaning_excluded or (
            legend_meta is not None and legend_meta.get("scatter_monochrome")
        ):
            scatter.set_visible(True)
        elif legend_meta is not None and legend_meta.get("scatter_color_by_oat"):
            oat_c_all = legend_meta.get("panel_oat_c")
            if oat_c_all is None and oat_all is not None:
                oat_c_all = _panel_oat_c_series(oat_all)
            if oat_c_all is not None and np.asarray(oat_c_all).size:
                oat_colors = _oat_c_at_hour_indices(
                    np.asarray(oat_c_all, dtype=float),
                    hour_idx_vis,
                )
                scatter.set_array(oat_colors)
            scatter.set_visible(True)
        else:
            scatter.set_array(zone_idx_vis)
            scatter.set_visible(True)
    else:
        scatter.set_offsets(np.empty((0, 2)))
        if not highlight_cleaning_excluded and not (
            legend_meta is not None and legend_meta.get("scatter_monochrome")
        ):
            scatter.set_array(np.array([]))
        scatter.set_visible(False)
    if (
        legend_meta is not None
        and legend_meta.get("scatter_monochrome")
        and cold_oat_scatter is not None
    ):
        panel_oat_c = legend_meta.get("panel_oat_c")
        threshold_c = legend_meta.get("highlight_oat_at_or_below_c")
        if (
            panel_oat_c is not None
            and threshold_c is not None
            and np.asarray(panel_oat_c).size
        ):
            cx, cy = _cold_oat_highlight_scatter_xy(
                xf_vis,
                yf_vis,
                hour_idx_vis,
                np.asarray(panel_oat_c, dtype=float),
                float(threshold_c),
            )
            _set_scatter_xy(cold_oat_scatter, cx, cy)
    design_day_below_setpoint_xy: tuple[np.ndarray, np.ndarray] = (
        np.empty(0, dtype=float),
        np.empty(0, dtype=float),
    )
    if (
        legend_meta is not None
        and legend_meta.get("highlight_design_day_below_setpoint")
        and design_day_below_setpoint_scatter is not None
    ):
        panel_data = legend_meta.get("panel_data")
        threshold_c = legend_meta.get("highlight_oat_at_or_below_c")
        if panel_data is not None and threshold_c is not None:
            design_day_below_setpoint_xy = _design_day_below_setpoint_excluded_scatter_xy(
                panel_data,
                oat_at_or_below_c=float(threshold_c),
                heatcall_threshold=threshold,
                zone_scatter_only=zone_scatter_only,
            )
            _set_scatter_xy(
                design_day_below_setpoint_scatter,
                design_day_below_setpoint_xy[0],
                design_day_below_setpoint_xy[1],
            )
    if highlight_cleaning_excluded:
        if cleaning_excluded_below_scatter is not None:
            bx, by = _cleaning_excluded_scatter_xy(
                cleaning_excluded_below_estimated,
                cleaning_excluded_below_swt_f,
            )
            _set_scatter_xy(cleaning_excluded_below_scatter, bx, by)
        if cleaning_excluded_oil_scatter is not None:
            ox, oy = _cleaning_excluded_scatter_xy(
                cleaning_excluded_oil_estimated,
                cleaning_excluded_oil_swt_f,
            )
            _set_scatter_xy(cleaning_excluded_oil_scatter, ox, oy)
    show_regression_line = (
        legend_meta.get("show_regression_line", False) if legend_meta else False
    )
    y_line_f_max: float | None = None
    if show_regression_line:
        _, x_line_f, y_line_f = _energy_vs_swt_regression_stats(xf, yf)
        if len(xf) >= 2:
            reg_line.set_data(x_line_f, y_line_f)
            reg_line.set_visible(True)
            y_line_f_max = float(np.max(y_line_f))
        else:
            reg_line.set_visible(False)
    else:
        reg_line.set_visible(False)
    coldest_x = coldest_x_measured if use_measured else coldest_x_estimated
    coldest_in_filtered = show_coldest_oat_marker and coldest_panel_index is not None
    coldest_marker.set_visible(False)
    max_swt_legend_label: str | None = None
    max_swt_c_for_legend: float | None = None
    green_hour_zone_set_f: dict[int, float] | None = None
    if yf.size:
        local_max_i = int(np.argmax(yf))
        max_swt_marker.set_data([xf[local_max_i]], [yf[local_max_i]])
        max_swt_marker.set_visible(True)
        swt_c_val = float(yf[local_max_i])
        if np.isfinite(swt_c_val):
            max_swt_c_for_legend = swt_c_val
            max_swt_legend_label = _hottest_swt_hour_legend_label(swt_c_val)
        if local_max_i < len(filtered_idx):
            gi = int(filtered_idx[local_max_i])
            if zone_set_f_by_hour:
                green_hour_zone_set_f = {}
                for zone, sets in zone_set_f_by_hour.items():
                    if gi < len(sets):
                        green_hour_zone_set_f[zone] = float(sets[gi])
    else:
        max_swt_marker.set_visible(False)
    if legend_meta is not None:
        existing = legend_meta.get("legend")
        if existing is not None:
            existing.remove()
        show_marker_legends = legend_meta.get("show_marker_legends", True)
        show_ring_legends = show_marker_legends and legend_meta.get(
            "show_coldest_max_swt_in_legend", True
        )
        if legend_meta.get("scatter_monochrome"):
            threshold_c = legend_meta.get("highlight_oat_at_or_below_c")
            panel_oat_c = legend_meta.get("panel_oat_c")
            cold_y = np.array([], dtype=float)
            if (
                threshold_c is not None
                and panel_oat_c is not None
                and np.asarray(panel_oat_c).size
            ):
                _cx, cold_y = _cold_oat_highlight_scatter_xy(
                    xf_vis,
                    yf_vis,
                    hour_idx_vis,
                    np.asarray(panel_oat_c, dtype=float),
                    float(threshold_c),
                )
            cold_legend_label = _design_day_colder_swt_legend_label(cold_y)
            panel_data = legend_meta.get("panel_data")
            if panel_data is not None and panel_data.house_alias == "maple_before":
                cold_legend_label = None
            _bd_x, _bd_y = design_day_below_setpoint_xy
            handles = _scatter_monochrome_house_panel_legend_handles(
                cold_legend_label=cold_legend_label,
                max_swt_c=max_swt_c_for_legend,
                design_day_below_setpoint_legend_label=(
                    _design_day_below_setpoint_max_swt_legend_label(_bd_y)
                    if legend_meta.get("highlight_design_day_below_setpoint")
                    and _bd_x.size > 0
                    else None
                ),
                point_marker_size=legend_meta.get("zone_marker_size", 7),
                max_swt_marker_size=legend_meta.get("max_swt_marker_size", 8),
            )
            legend_meta["legend"] = legend_meta["ax"].legend(
                handles=handles,
                loc="upper left",
                framealpha=0.9,
                fontsize=legend_meta.get("legend_fontsize", 7),
            )
            _raise_panel_legend_above_data_markers(legend_meta["legend"])
        else:
            if legend_meta.get("highlight_cleaning_excluded"):
                handles = _energy_vs_swt_cleaning_highlight_legend_handles(
                    coldest_oat_legend_label=(
                        legend_meta.get("coldest_oat_legend_label")
                        if show_ring_legends and coldest_in_filtered
                        else None
                    ),
                    max_swt_legend_label=(
                        max_swt_legend_label if show_ring_legends else None
                    ),
                    coldest_marker_size=legend_meta.get("coldest_marker_size", 7),
                    max_swt_marker_size=legend_meta.get("max_swt_marker_size", 7),
                    point_marker_size=legend_meta.get("zone_marker_size", 7),
                    show_oil_boiler_excluded=legend_meta.get(
                        "show_oil_boiler_excluded", True
                    ),
                )
            else:
                near_constant_zone_labels = (
                    _near_constant_heatcall_zone_legend_overrides(
                        legend_meta,
                        scatter_point_count=int(xf_vis.size),
                    )
                )
                zone_hour_percent = None
                if (
                    legend_meta.get("zone_legend_hour_percent")
                    and near_constant_zone_labels is None
                ):
                    percent_override = legend_meta.get(
                        "zone_legend_hour_percent_override"
                    )
                    if percent_override is not None:
                        zone_hour_percent = {
                            int(z): float(percent_override[z])
                            for z in legend_meta["sorted_zones"]
                            if z in percent_override
                        }
                    else:
                        zone_hour_percent = _zone_hour_percent_from_scatter(
                            legend_meta["sorted_zones"],
                            zone_idx_vis,
                            multi_zone_color_index=multi_zone_color_index,
                        )
                handles = _energy_vs_swt_legend_handles(
                    legend_meta["sorted_zones"],
                    legend_meta["palette"],
                    legend_meta["zone_names"],
                    coldest_oat_legend_label=(
                        legend_meta.get("coldest_oat_legend_label")
                        if show_ring_legends and coldest_in_filtered
                        else None
                    ),
                    max_swt_legend_label=(
                        max_swt_legend_label if show_ring_legends else None
                    ),
                    zone_avg_set_f=legend_meta.get("zone_avg_set_f"),
                    coldest_hour_zone_set_f=legend_meta.get("coldest_hour_zone_set_f"),
                    green_hour_zone_set_f=green_hour_zone_set_f,
                    setpoint_detail_zones=legend_meta.get("setpoint_detail_zones"),
                    zone_marker_size=legend_meta.get("zone_marker_size", 6),
                    coldest_marker_size=legend_meta.get("coldest_marker_size", 7),
                    max_swt_marker_size=legend_meta.get("max_swt_marker_size", 7),
                    multi_active_zone_label=(
                        MULTI_ACTIVE_ZONE_LEGEND_LABEL
                        if legend_meta.get("scatter_per_active_zone")
                        else None
                    ),
                    multi_active_zone_color=(
                        MULTI_ACTIVE_ZONE_SCATTER_COLOR
                        if legend_meta.get("scatter_per_active_zone")
                        and legend_meta.get("zone_scatter_only") is None
                        else None
                    ),
                    zones_in_legend=legend_meta.get("zones_in_legend"),
                    zone_color_overrides=legend_meta.get("zone_color_overrides"),
                    zone_label_overrides=near_constant_zone_labels,
                    zone_hour_percent=zone_hour_percent,
                    zone_legend_markerfacecolor=legend_meta.get(
                        "zone_legend_markerfacecolor"
                    ),
                )
            legend_meta["legend"] = legend_meta["ax"].legend(
                handles=handles,
                loc="upper left",
                framealpha=0.9,
                fontsize=legend_meta.get("legend_fontsize", 7),
            )
            _raise_panel_legend_above_data_markers(legend_meta["legend"])
    if set_xlabel:
        axis_fs = legend_meta.get("axis_label_fontsize") if legend_meta else None
        ax.set_xlabel(
            _energy_vs_swt_x_axis_label(use_measured),
            fontsize=axis_fs,
        )
    ax.set_ylim(40, ylim_top)
    if highlight_above_max_y and highlight_span_ref is not None:
        if highlight_span_ref[0] is not None:
            highlight_span_ref[0].remove()
            highlight_span_ref[0] = None
        y_peak_candidates: list[float] = []
        if yf.size:
            y_peak_candidates.append(float(np.max(yf)))
        if y_line_f_max is not None:
            y_peak_candidates.append(y_line_f_max)
        if coldest_in_filtered and coldest_x is not None and coldest_y_c is not None:
            y_peak_candidates.append(float(coldest_y_c))
        if y_peak_candidates:
            y_peak = max(y_peak_candidates)
            if y_peak < ylim_top:
                highlight_span_ref[0] = ax.axhspan(
                    y_peak,
                    ylim_top,
                    facecolor="0.88",
                    edgecolor="none",
                    zorder=0.5,
                )
    if fixed_xlim is not None:
        ax.set_xlim(fixed_xlim)
    else:
        computed = _compute_energy_vs_swt_xlim(
            estimated_all,
            dist_kwh_all,
            y_all,
            heatcall_all,
            threshold=threshold,
            use_measured=use_measured,
            coldest_x_estimated=coldest_x_estimated,
            coldest_x_measured=coldest_x_measured,
            coldest_y_c=coldest_y_c,
        )
        if computed is not None:
            ax.set_xlim(computed)


def plot_estimated_energy_vs_swt_grid(
    panels: list[EnergyVsSwtPanelData],
    *,
    savepath: Path | None = None,
    nrows: int = 2,
    ncols: int = 3,
    oat_hour_start: pd.Series | np.ndarray | None = None,
    oat_f: np.ndarray | None = None,
    maple_transition_vline: pd.Timestamp | None = None,
    maple_transition_vline_label: str | None = "Maple retrofit",
    scatter_per_active_zone: bool = False,
    highlight_cleaning_excluded: bool = False,
    cleaning_highlight_oil_boiler: bool = True,
    fig_width: float | None = None,
    show_zone_legend: bool = False,
    zone_scatter_filters: list[int] | None = None,
    zone_color_overrides: dict[int, str] | None = None,
    fig_height: float | None = None,
    show_coldest_oat_marker: bool = True,
    zone_in_subplot_title: bool = True,
    fig_suptitle: str | None = None,
    subplot_titles: list[str] | None = None,
    zone_legend_near_constant_heatcall: bool = False,
    zone_legend_hour_percent: bool = False,
    show_regression_line: bool = False,
    show_coldest_max_swt_in_legend: bool = True,
    scatter_color_by_oat: bool = False,
    scatter_monochrome: bool = False,
    highlight_oat_at_or_below_c: float | None = None,
    highlight_design_day_below_setpoint: bool = False,
    scatter_marker_size: float = GRID_SCATTER_MARKER_SIZE,
    plot_ring_marker_size: float = GRID_PLOT_RING_MARKER_SIZE,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize

    expected = nrows * ncols
    if len(panels) != expected:
        raise ValueError(f"Expected {expected} panels, got {len(panels)}")
    if zone_scatter_filters is not None and len(zone_scatter_filters) != len(panels):
        raise ValueError(
            f"zone_scatter_filters length {len(zone_scatter_filters)} "
            f"!= panels length {len(panels)}"
        )
    if subplot_titles is not None and len(subplot_titles) != len(panels):
        raise ValueError(
            f"subplot_titles length {len(subplot_titles)} != panels length {len(panels)}"
        )
    if scatter_monochrome and highlight_cleaning_excluded:
        raise ValueError("scatter_monochrome cannot be combined with highlight_cleaning_excluded")
    if scatter_monochrome and scatter_color_by_oat:
        raise ValueError("scatter_monochrome cannot be combined with scatter_color_by_oat")
    if highlight_oat_at_or_below_c is not None and not scatter_monochrome:
        raise ValueError("highlight_oat_at_or_below_c requires scatter_monochrome=True")
    if highlight_design_day_below_setpoint and (
        not scatter_monochrome or highlight_oat_at_or_below_c is None
    ):
        raise ValueError(
            "highlight_design_day_below_setpoint requires scatter_monochrome=True "
            "and highlight_oat_at_or_below_c"
        )

    if fig_width is None:
        fig_width = SWT_GRID_FIG_WIDTH * (ncols / 3.0)

    threshold_init = 0.9
    title_fontsize = 12
    axis_label_fontsize = 10
    tick_fontsize = 9
    legend_fontsize = 9
    include_oat_panel = oat_hour_start is not None and oat_f is not None
    swt_row_height = SWT_GRID_FIG_HEIGHT / nrows
    oat_height_ratio = OAT_PANEL_FIG_HEIGHT / swt_row_height

    if fig_height is None:
        fig_height = SWT_GRID_FIG_HEIGHT + (
            OAT_PANEL_FIG_HEIGHT if include_oat_panel else 0.0
        )
    compact = fig_height < SWT_GRID_FIG_HEIGHT
    gs_bottom = 0.14 if compact else 0.08
    gs_top = 0.92 if compact else 0.96
    if fig_suptitle:
        gs_top = 0.86 if compact else 0.90
    gs_hspace = 0.22 if compact else 0.28
    fig = plt.figure(figsize=(fig_width, fig_height))
    if include_oat_panel:
        gs = fig.add_gridspec(
            nrows + 1,
            ncols,
            height_ratios=[1] * nrows + [oat_height_ratio],
            left=0.07,
            right=0.98,
            top=gs_top,
            bottom=gs_bottom,
            hspace=gs_hspace,
            wspace=0.18,
        )
    else:
        gs = fig.add_gridspec(
            nrows,
            ncols,
            left=0.07,
            right=0.98,
            top=gs_top,
            bottom=gs_bottom,
            hspace=gs_hspace,
            wspace=0.18,
        )

    palette = plt.cm.tab10.colors
    oat_c_vmin, oat_c_vmax = (
        _energy_vs_swt_scatter_oat_c_colormap_bounds(
            panels,
            threshold=threshold_init,
            scatter_per_active_zone=scatter_per_active_zone,
            zone_scatter_filters=zone_scatter_filters,
            show_coldest_oat_marker=show_coldest_oat_marker,
        )
        if scatter_color_by_oat
        else (None, None)
    )
    oat_c_norm = (
        Normalize(vmin=oat_c_vmin, vmax=oat_c_vmax)
        if scatter_color_by_oat
        else None
    )
    oat_colored_axes: list = []

    for index, panel in enumerate(panels):
        row, col = divmod(index, ncols)
        panel_zone_scatter_only = (
            zone_scatter_filters[index] if zone_scatter_filters is not None else None
        )
        panel_show_coldest_oat_marker = _panel_show_coldest_oat_marker(
            panel,
            show_coldest_oat_marker=show_coldest_oat_marker,
        )
        ax = fig.add_subplot(gs[row, col])
        estimated_all = np.asarray(panel.estimated_kwh, dtype=float)
        y_all = _swt_f_to_c(panel.swt_f)
        heatcall_all = np.asarray(panel.max_zone_heatcall, dtype=float)
        dominant_zone_all = np.asarray(panel.dominant_zone, dtype=int)
        oat_all = (
            np.asarray(panel.oat_f, dtype=float)
            if panel.oat_f is not None
            else np.array([])
        )
        hour_start_all = (
            np.asarray(panel.hour_start)
            if panel.hour_start is not None
            else np.array([])
        )

        coldest_x_estimated: float | None = None
        coldest_y_c: float | None = None
        if panel.coldest_oat_swt_f is not None and np.isfinite(panel.coldest_oat_swt_f):
            coldest_y_c = float(_swt_f_to_c(np.array([panel.coldest_oat_swt_f]))[0])
            if panel.coldest_oat_estimated_kwh is not None and np.isfinite(
                panel.coldest_oat_estimated_kwh
            ):
                coldest_x_estimated = float(panel.coldest_oat_estimated_kwh)

        sorted_zones = tuple(sorted(panel.zone_numbers))
        zone_to_index = {zone: i for i, zone in enumerate(sorted_zones)}
        multi_zone_color_index = float(len(sorted_zones))
        cmap_colors = [
            _zone_scatter_facecolor(z, sorted_zones, palette, zone_color_overrides)
            for z in sorted_zones
        ]
        use_multi_zone_color = (
            scatter_per_active_zone and panel_zone_scatter_only is None
        )
        if use_multi_zone_color:
            cmap_colors.append(MULTI_ACTIVE_ZONE_SCATTER_COLOR)
        zone_cmap = ListedColormap(cmap_colors)
        scatter_vmax = (
            multi_zone_color_index
            if use_multi_zone_color
            else max(len(sorted_zones) - 1, 0)
        )

        zone_heatcall_by_hour = panel.zone_heatcall_fraction_by_hour or {}
        panel_oat_c = _panel_oat_c_series(panel.oat_f)
        x0, y0, zone_idx0, hour_idx0 = _energy_vs_swt_scatter_xy(
            estimated_all,
            y_all,
            heatcall_all,
            dominant_zone_all,
            zone_to_index,
            threshold=threshold_init,
            scatter_per_active_zone=scatter_per_active_zone,
            zone_heatcall_fraction_by_hour=zone_heatcall_by_hour,
            coldest_panel_index=panel.coldest_oat_panel_index,
            multi_zone_color_index=multi_zone_color_index,
            zone_scatter_only=panel_zone_scatter_only,
            pin_coldest_in_scatter=panel_show_coldest_oat_marker,
        )

        panel_highlight_design_day_below = _panel_show_design_day_below_setpoint(
            panel,
            highlight_design_day_below_setpoint=highlight_design_day_below_setpoint,
            scatter_monochrome=scatter_monochrome,
        )

        cold_oat_scatter = None
        design_day_below_setpoint_scatter = None
        if highlight_cleaning_excluded:
            scatter = ax.scatter(
                x0,
                y0,
                c="0.55",
                s=scatter_marker_size,
                alpha=0.65,
                zorder=2,
            )
            bx0, by0 = _cleaning_excluded_scatter_xy(
                panel.cleaning_excluded_below_setpoint_estimated_kwh,
                panel.cleaning_excluded_below_setpoint_swt_f,
            )
            cleaning_excluded_below_scatter = ax.scatter(
                bx0,
                by0,
                c=OAT_CLEANING_HIGHLIGHT_COLORS["below_setpoint"],
                s=scatter_marker_size,
                alpha=0.9,
                zorder=3,
            )
            if cleaning_highlight_oil_boiler:
                ox0, oy0 = _cleaning_excluded_scatter_xy(
                    panel.cleaning_excluded_oil_boiler_estimated_kwh,
                    panel.cleaning_excluded_oil_boiler_swt_f,
                )
                cleaning_excluded_oil_scatter = ax.scatter(
                    ox0,
                    oy0,
                    c=OAT_CLEANING_HIGHLIGHT_COLORS["oil_boiler"],
                    s=scatter_marker_size,
                    alpha=0.9,
                    zorder=3,
                )
            else:
                cleaning_excluded_oil_scatter = None
        elif scatter_monochrome:
            cleaning_excluded_below_scatter = None
            cleaning_excluded_oil_scatter = None
            scatter = ax.scatter(
                x0,
                y0,
                c=SCATTER_MONOCHROME_GRAY,
                s=scatter_marker_size,
                alpha=SCATTER_MONOCHROME_GRAY_ALPHA,
                zorder=2,
            )
            if highlight_oat_at_or_below_c is not None:
                cx0, cy0 = _cold_oat_highlight_scatter_xy(
                    x0,
                    y0,
                    hour_idx0,
                    panel_oat_c,
                    highlight_oat_at_or_below_c,
                )
                cold_oat_scatter = ax.scatter(
                    cx0,
                    cy0,
                    c="red",
                    s=scatter_marker_size,
                    alpha=0.9,
                    zorder=3,
                )
            if panel_highlight_design_day_below:
                design_day_below_setpoint_scatter = ax.scatter(
                    [],
                    [],
                    c=DESIGN_DAY_BELOW_SETPOINT_SCATTER_COLOR,
                    s=scatter_marker_size * 1.2,
                    alpha=0.95,
                    edgecolors="0.35",
                    linewidths=0.4,
                    zorder=4,
                )
        else:
            cleaning_excluded_below_scatter = None
            cleaning_excluded_oil_scatter = None
            if scatter_color_by_oat:
                scatter_colors = _oat_c_at_hour_indices(panel_oat_c, hour_idx0)
                scatter = ax.scatter(
                    x0,
                    y0,
                    c=scatter_colors,
                    cmap=plt.cm.coolwarm,
                    norm=oat_c_norm,
                    s=scatter_marker_size,
                    alpha=0.65,
                    zorder=2,
                )
            else:
                scatter = ax.scatter(
                    x0,
                    y0,
                    c=zone_idx0,
                    cmap=zone_cmap,
                    vmin=0,
                    vmax=scatter_vmax,
                    s=scatter_marker_size,
                    alpha=0.65,
                    zorder=2,
                )
        (reg_line,) = ax.plot(
            [],
            [],
            color="0.15",
            linewidth=1.5,
            visible=False,
        )
        ring_ms = plot_ring_marker_size
        (coldest_marker,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=ring_ms,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=6,
            visible=False,
        )
        (max_swt_marker,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=ring_ms,
            markerfacecolor="none",
            markeredgecolor="green",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=6,
            visible=False,
        )

        if subplot_titles is not None:
            panel_title = subplot_titles[index]
        else:
            panel_title = _house_alias_subplot_title(panel.house_alias)
        if (
            subplot_titles is None
            and zone_in_subplot_title
            and panel_zone_scatter_only is not None
        ):
            z = panel_zone_scatter_only
            zname = (panel.zone_names or {}).get(z)
            panel_title = (
                f"{panel_title} — Zone {z} ({zname})"
                if zname
                else f"{panel_title} — Zone {z}"
            )
        ax.set_title(panel_title, fontsize=title_fontsize)
        ax.set_ylim(40, 90)
        if col == 0:
            ax.set_ylabel(ENERGY_VS_SWT_YLABEL, fontsize=axis_label_fontsize)
        if row == nrows - 1:
            ax.set_xlabel(_energy_vs_swt_x_axis_label(False), fontsize=axis_label_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        zone_set_f_by_hour = panel.zone_set_f_by_hour or {}
        legend_meta: dict = {
            "ax": ax,
            "sorted_zones": sorted_zones,
            "palette": palette,
            "zone_names": panel.zone_names,
            "highlight_cleaning_excluded": highlight_cleaning_excluded,
            "show_oil_boiler_excluded": cleaning_highlight_oil_boiler,
            "scatter_per_active_zone": scatter_per_active_zone,
            "zone_scatter_only": panel_zone_scatter_only,
            "zones_in_legend": (
                ()
                if scatter_color_by_oat or scatter_monochrome
                else (
                    (panel_zone_scatter_only,)
                    if panel_zone_scatter_only is not None
                    else None
                )
            ),
            "scatter_monochrome": scatter_monochrome,
            "highlight_oat_at_or_below_c": highlight_oat_at_or_below_c,
            "highlight_design_day_below_setpoint": panel_highlight_design_day_below,
            "panel_data": panel,
            "zone_color_overrides": zone_color_overrides,
            "show_marker_legends": show_zone_legend or panel_show_coldest_oat_marker,
            "show_coldest_max_swt_in_legend": show_coldest_max_swt_in_legend,
            "coldest_oat_legend_label": (
                None if scatter_monochrome else panel.coldest_oat_date_label
            ),
            "zone_legend_near_constant_heatcall": zone_legend_near_constant_heatcall,
            "zone_legend_hour_percent": zone_legend_hour_percent,
            "zone_legend_hour_percent_override": panel.zone_legend_hour_percent_override,
            "show_regression_line": show_regression_line,
            "scatter_color_by_oat": scatter_color_by_oat,
            "panel_oat_c": panel_oat_c,
            "oat_c_vmin": oat_c_vmin,
            "oat_c_vmax": oat_c_vmax,
            "zone_avg_set_f": panel.zone_avg_set_f,
            "coldest_hour_zone_set_f": panel.coldest_hour_zone_set_f,
            "setpoint_detail_zones": frozenset(),
            "legend": None,
            "zone_marker_size": 7,
            "coldest_marker_size": 8,
            "max_swt_marker_size": 8,
            "legend_fontsize": legend_fontsize,
            "axis_label_fontsize": axis_label_fontsize,
            "tick_fontsize": tick_fontsize,
        }

        extra_xlim_x = None
        if highlight_cleaning_excluded:
            extra_parts = [
                _optional_float_array(
                    panel.cleaning_excluded_below_setpoint_estimated_kwh
                ),
            ]
            if cleaning_highlight_oil_boiler:
                extra_parts.append(
                    _optional_float_array(
                        panel.cleaning_excluded_oil_boiler_estimated_kwh
                    )
                )
            extra_xlim_x = np.concatenate(extra_parts)
        elif panel_highlight_design_day_below and highlight_oat_at_or_below_c is not None:
            yx, _yy = _design_day_below_setpoint_excluded_scatter_xy(
                panel,
                oat_at_or_below_c=float(highlight_oat_at_or_below_c),
                heatcall_threshold=0.0,
                zone_scatter_only=panel_zone_scatter_only,
            )
            extra_xlim_x = yx
        fixed_xlim = _compute_energy_vs_swt_xlim(
            estimated_all,
            estimated_all,
            y_all,
            heatcall_all,
            threshold=0.0,
            use_measured=False,
            coldest_x_estimated=(
                coldest_x_estimated if panel_show_coldest_oat_marker else None
            ),
            coldest_x_measured=None,
            coldest_y_c=coldest_y_c if panel_show_coldest_oat_marker else None,
            extra_x=extra_xlim_x,
            include_regression_x_extent=show_regression_line,
        )

        highlight_span: list = [None]

        _refresh_energy_vs_swt_panel(
            ax,
            estimated_all=estimated_all,
            dist_kwh_all=estimated_all,
            y_all=y_all,
            heatcall_all=heatcall_all,
            dominant_zone_all=dominant_zone_all,
            zone_to_index=zone_to_index,
            threshold=threshold_init,
            use_measured=False,
            coldest_x_estimated=coldest_x_estimated,
            coldest_x_measured=None,
            coldest_y_c=coldest_y_c,
            coldest_panel_index=panel.coldest_oat_panel_index,
            show_coldest_oat_marker=panel_show_coldest_oat_marker,
            scatter=scatter,
            reg_line=reg_line,
            coldest_marker=coldest_marker,
            max_swt_marker=max_swt_marker,
            set_xlabel=row == nrows - 1,
            highlight_above_max_y=True,
            highlight_span_ref=highlight_span,
            oat_all=oat_all,
            hour_start_all=hour_start_all,
            zone_set_f_by_hour=zone_set_f_by_hour,
            zone_heatcall_fraction_by_hour=zone_heatcall_by_hour,
            scatter_per_active_zone=scatter_per_active_zone,
            multi_zone_color_index=multi_zone_color_index,
            zone_scatter_only=panel_zone_scatter_only,
            highlight_cleaning_excluded=highlight_cleaning_excluded,
            cleaning_excluded_below_scatter=cleaning_excluded_below_scatter,
            cleaning_excluded_oil_scatter=cleaning_excluded_oil_scatter,
            cleaning_excluded_below_estimated=(
                panel.cleaning_excluded_below_setpoint_estimated_kwh
            ),
            cleaning_excluded_below_swt_f=(
                panel.cleaning_excluded_below_setpoint_swt_f
            ),
            cleaning_excluded_oil_estimated=(
                panel.cleaning_excluded_oil_boiler_estimated_kwh
            ),
            cleaning_excluded_oil_swt_f=panel.cleaning_excluded_oil_boiler_swt_f,
            cold_oat_scatter=cold_oat_scatter,
            design_day_below_setpoint_scatter=design_day_below_setpoint_scatter,
            legend_meta=legend_meta,
            fixed_xlim=fixed_xlim,
        )
        ax.tick_params(labelsize=tick_fontsize)
        if col == 0:
            ax.set_ylabel(ENERGY_VS_SWT_YLABEL, fontsize=axis_label_fontsize)
        if row == nrows - 1:
            ax.set_xlabel(_energy_vs_swt_x_axis_label(False), fontsize=axis_label_fontsize)
        if scatter_color_by_oat and not highlight_cleaning_excluded:
            oat_colored_axes.append(ax)

    if scatter_color_by_oat and oat_colored_axes and not highlight_cleaning_excluded:
        from matplotlib.cm import ScalarMappable

        sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=oat_c_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=oat_colored_axes, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=max(tick_fontsize - 1, 7))
        cbar.set_label(
            "Outside air temperature (°C)",
            fontsize=max(axis_label_fontsize - 1, 8),
        )

    if include_oat_panel:
        ax_oat = fig.add_subplot(gs[nrows, :])
        _draw_oat_timeseries_on_ax(
            ax_oat,
            oat_hour_start,
            oat_f,
            panels,
            heatcall_threshold=threshold_init,
            axis_label_fontsize=axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            maple_transition_vline=maple_transition_vline,
            maple_transition_vline_label=maple_transition_vline_label,
            highlight_oat_at_or_below_c=(
                highlight_oat_at_or_below_c if scatter_monochrome else None
            ),
            scatter_monochrome=scatter_monochrome,
            highlight_design_day_below_setpoint=highlight_design_day_below_setpoint,
            zone_scatter_filters=zone_scatter_filters,
        )

    if fig_suptitle:
        fig.suptitle(fig_suptitle, fontsize=title_fontsize, y=0.98)

    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_oos_rmse_by_lead(
    rmse_by_lead: list[float],
    rmse_baseline_by_lead: list[float],
    *,
    house_alias: str,
    forecast_horizon_hours: int,
    savepath: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    leads = np.arange(1, len(rmse_by_lead) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(leads, rmse_by_lead, "o-", label="Model", color="tab:blue")
    ax.plot(leads, rmse_baseline_by_lead, "s--", label="Baseline", color="tab:orange")
    ax.set_xlabel("Forecast lead (hours ahead)")
    ax.set_ylabel("RMSE (kWh)")
    ax.set_title(
        f"{house_alias.capitalize()}: out-of-sample RMSE by lead "
        f"({forecast_horizon_hours}h recursive load)"
    )
    ax.set_xticks([1, 6, 12, 24, 36, 48])
    ax.legend()
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oos_residual_by_hour_of_day(
    residuals: np.ndarray,
    hour_starts,
    *,
    house_alias: str,
    savepath: Path | None = None,
    subtitle: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    residuals = np.asarray(residuals, dtype=float)
    timestamps = pd.to_datetime(hour_starts)
    hour_of_day = timestamps.hour
    weekend_mask = np.asarray(timestamps.dayofweek >= 5, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    panels = (
        ("Weekday", ~weekend_mask),
        ("Weekend", weekend_mask),
    )
    for ax, (panel_title, mask) in zip(axes, panels):
        panel_residuals = residuals[mask]
        panel_hours = hour_of_day[mask]
        _plot_hourly_residual_bars_on_ax(
            ax,
            panel_residuals,
            panel_hours,
            house_alias=house_alias,
            panel_title=panel_title,
        )
    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylim(-1.5, 1.5)
    if subtitle:
        fig.suptitle(subtitle, fontsize=11, y=1.02)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_data_distribution(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    house_alias: str,
    required: list[str],
    zones: list[int],
    results_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    channels = [
        c
        for c in required
        if not (str(c).startswith("zone") and str(c).endswith("_heatcall_fraction"))
    ]
    n_channels = len(channels)
    col_width_in = 1.75
    fig, axes = plt.subplots(
        2,
        n_channels,
        figsize=(col_width_in * n_channels + 1.25, 8),
        squeeze=False,
        gridspec_kw={"wspace": 0.75},
    )
    row_labels = ("Before cleaning", "After cleaning")
    for row, (label, df) in enumerate(zip(row_labels, (df_before, df_after))):
        for col_idx, channel in enumerate(channels):
            ax = axes[row, col_idx]
            ax.boxplot(df[channel].dropna().to_numpy(), vert=True, widths=0.22)
            if row == 0:
                n_nans = int(np.sum(df_before[channel].isna().to_numpy()))
                nan_label = "NaN" if n_nans == 1 else "NaNs"
                ax.set_title(f"{channel}\n({n_nans} {nan_label})", fontsize=8)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.tick_params(axis="y", labelsize=8)
        axes[row, 0].set_ylabel(label)
        if row == 1:
            for z in zones:
                zone_avg_channels = [
                    c for c in channels if str(c).startswith(f"zone{z}_avg_")
                ]
                if not zone_avg_channels:
                    continue
                zone_avg_values = np.concatenate(
                    [df[channel].dropna().to_numpy() for channel in zone_avg_channels]
                )
                if not zone_avg_values.size:
                    continue
                y_min = float(zone_avg_values.min())
                y_max = float(zone_avg_values.max())
                pad = max((y_max - y_min) * 0.05, 0.25)
                shared_ylim = (y_min - pad, y_max + pad)
                zone_avg_set = set(zone_avg_channels)
                for col_idx, channel in enumerate(channels):
                    if channel in zone_avg_set:
                        axes[row, col_idx].set_ylim(shared_ylim)
    fig.suptitle(f"{house_alias.capitalize()}: channel distributions")
    fig.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    savepath = results_dir / f"{house_alias}_channel_boxplots.png"
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rswt_fit(
    swt: np.ndarray,
    rwt: np.ndarray,
    dist_kwh: np.ndarray,
    *,
    bucket_fits: list[tuple[str, RswtLoadBucketFit | None, int, float]],
    water_kg_hour_chc: float,
    water_chc_n_hours: int,
    house_alias: str,
    n_days: int,
    savepath: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    swt = np.asarray(swt, dtype=float)
    rwt = np.asarray(rwt, dtype=float)
    load_kwh = np.asarray(dist_kwh, dtype=float)
    load_max = float(np.nanmax(load_kwh)) if load_kwh.size else 1.0
    load_norm = Normalize(vmin=0.0, vmax=max(load_max, 1.0))
    n_buckets = len(bucket_fits)
    gray = np.linspace(0.78, 0.0, n_buckets) if n_buckets else np.array([])
    line_colors = [(float(g), float(g), float(g)) for g in gray]

    fig, (ax_rwt, ax_water) = plt.subplots(1, 2, figsize=(14, 6))
    sc = ax_rwt.scatter(
        swt,
        rwt,
        c=load_kwh,
        cmap=plt.cm.coolwarm,
        s=22,
        alpha=0.65,
        norm=load_norm,
        label="Hourly data",
    )

    swt_line = np.linspace(float(np.min(swt)), float(np.max(swt)), num=100)
    for color, (label, fit, n_hours, _load_mid) in zip(
        line_colors, bucket_fits, strict=True
    ):
        if fit is None:
            continue
        ax_rwt.plot(
            swt_line,
            fit.rwt_intercept + fit.rwt_slope * swt_line,
            color=color,
            linewidth=2,
            label=f"{label} (n={n_hours})",
        )

    ax_rwt.set_xlabel("Supply water temperature SWT (°F)")
    ax_rwt.set_ylabel("Return water temperature RWT (°F)")
    ax_rwt.set_title("RWT vs SWT")
    ax_rwt.legend(loc="upper left", fontsize=8)

    cbar = fig.colorbar(sc, ax=ax_rwt)
    cbar.set_label("dist_kwh")

    chc_reference_hcf = 0.9
    ax_water.scatter(
        [chc_reference_hcf],
        [water_kg_hour_chc],
        color="0.35",
        s=90,
        zorder=4,
        label=f"water_kg_hour_chc (n={water_chc_n_hours})",
    )

    hcf_line = [0.0, chc_reference_hcf]
    scaled_water = [
        water_kg_hour_chc * (hcf / chc_reference_hcf) for hcf in hcf_line
    ]
    ax_water.plot(
        hcf_line,
        scaled_water,
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        zorder=2,
        label="water_kg_hour_chc × hcf / 0.9",
    )

    ax_water.set_xlabel("Max zone heat-call fraction")
    ax_water.set_ylabel("Water flow (kg/h)")
    ax_water.set_title("CHC water flow vs heat-call")
    ax_water.set_xlim(-0.02, 1.02)
    ax_water.legend(loc="best", fontsize=8)
    ax_water.grid(True, alpha=0.3)

    fig.suptitle(
        f"{house_alias.capitalize()}: RSWT fit (last {n_days} day(s))",
        y=1.02,
    )

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


HeatCallInterval = tuple[datetime | pd.Timestamp, datetime | pd.Timestamp]
HeatCallZone = (
    tuple[str, Sequence[HeatCallInterval]]
    | tuple[str, Sequence[HeatCallInterval], int]
    | tuple[str, Sequence[HeatCallInterval], int, int]
)

_HEATCALL_ROW_FACE_EDGE: tuple[tuple[str, str], ...] = (
    ("#A8D4F0", "#4A90A4"),
    ("#F5C49A", "#D97904"),
    ("#B8E6B8", "#3D9E3D"),
    ("#F5A8A8", "#C0392B"),
)
_HEATCALL_ROW_INSET = 0.08


def _heatcall_intervals_to_broken_bar_segments(
    intervals: Sequence[HeatCallInterval],
) -> list[tuple[float, float]]:
    import matplotlib.dates as mdates

    segments: list[tuple[float, float]] = []
    for start, end in intervals:
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        if t1 <= t0:
            continue
        x = mdates.date2num(t0.to_pydatetime())
        width = mdates.date2num(t1.to_pydatetime()) - x
        segments.append((x, width))
    return segments


def heatcall_intervals_from_step_signal(
    times: pd.Series | np.ndarray | pd.DatetimeIndex,
    on: np.ndarray,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build [start, end) intervals from a step signal.

    ``on[i]`` is active on ``[times[i], times[i + 1])``. ``on`` must have length
    ``len(times) - 1``.
    """
    times = pd.DatetimeIndex(pd.to_datetime(times))
    active = np.asarray(on, dtype=bool)
    if len(active) != len(times) - 1:
        raise ValueError("on must have length len(times) - 1")

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = times[i]
        j = i
        while j < len(active) and active[j]:
            j += 1
        intervals.append((start, times[j]))
        i = j
    return intervals


def _parse_heatcall_zone(
    zone: HeatCallZone, default_row: int
) -> tuple[str, Sequence[HeatCallInterval], int, int]:
    """Return label, intervals, y-row index, and palette index."""
    if len(zone) == 4:
        label, intervals, row_idx, palette_idx = zone
    elif len(zone) == 3:
        label, intervals, row_idx = zone
        palette_idx = row_idx
    else:
        label, intervals = zone
        row_idx = default_row
        palette_idx = default_row
    return label, intervals, row_idx, palette_idx


def plot_heat_calls(
    zones: Sequence[HeatCallZone],
    *,
    title: str | None = None,
    xlim: tuple[datetime | pd.Timestamp, datetime | pd.Timestamp] | None = None,
    figsize: tuple[float, float] | None = None,
    savepath: Path | None = None,
    show: bool = False,
) -> None:
    """Gantt-style on/off heat-call rows (one horizontal band per zone)."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not zones:
        raise ValueError("plot_heat_calls requires at least one zone")

    parsed = [
        _parse_heatcall_zone(zone, default_row=i) for i, zone in enumerate(zones)
    ]
    y_max = max(row_idx for _, _, row_idx, _ in parsed) + 1
    title_pad = 0.45 if title else 0.0
    fig_h = 1.25 + 0.62 * y_max + title_pad
    if figsize is not None:
        fig_w, fig_h = figsize
    else:
        fig_w = 11.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_min: float | None = None
    x_max: float | None = None

    for label, intervals, row_idx, palette_idx in parsed:
        face, edge = _HEATCALL_ROW_FACE_EDGE[palette_idx % len(_HEATCALL_ROW_FACE_EDGE)]
        segments = _heatcall_intervals_to_broken_bar_segments(intervals)
        if segments:
            starts = [s for s, _ in segments]
            ends = [s + w for s, w in segments]
            row_min = min(starts)
            row_max = max(ends)
            x_min = row_min if x_min is None else min(x_min, row_min)
            x_max = row_max if x_max is None else max(x_max, row_max)
            bar_y = row_idx + _HEATCALL_ROW_INSET
            bar_h = 1.0 - 2.0 * _HEATCALL_ROW_INSET
            ax.broken_barh(
                segments,
                (bar_y, bar_h),
                facecolors=face,
                edgecolors="none",
                alpha=0.78,
            )
            y_bottom = bar_y
            y_top = bar_y + bar_h
            for start_x, width in segments:
                end_x = start_x + width
                ax.vlines(
                    start_x,
                    y_bottom,
                    y_top,
                    colors=edge,
                    linewidth=0.85,
                    zorder=3,
                )
                ax.vlines(
                    end_x,
                    y_bottom,
                    y_top,
                    colors=edge,
                    linewidth=0.85,
                    zorder=3,
                )

    ax.set_ylim(0, y_max)
    band_ticks = sorted(
        (row_idx + 0.5, zone_label)
        for zone_label, _, row_idx, _ in parsed
    )
    ax.set_yticks(
        [y for y, _ in band_ticks],
        [name for _, name in band_ticks],
    )
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=11, loc="center", pad=8)

    if xlim is not None:
        t0 = pd.Timestamp(xlim[0])
        t1 = pd.Timestamp(xlim[1])
        ax.set_xlim(t0, t1)
        span_hours = (t1 - t0).total_seconds() / 3600.0
    elif x_min is not None and x_max is not None:
        pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0 / (24 * 60)
        ax.set_xlim(x_min - pad, x_max + pad)
        span_hours = (x_max - x_min) * 24.0
    else:
        span_hours = 1.0

    locator = mdates.HourLocator(interval=1)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time of day")
    ax.tick_params(axis="y", pad=14)

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
