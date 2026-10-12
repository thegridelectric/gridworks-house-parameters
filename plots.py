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
    zone_names: dict[int, str] | None = None


class RswtLoadBucketFit(Protocol):
    rwt_intercept: float
    rwt_slope: float


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
    _, x_line, y_line = _energy_vs_swt_regression_stats(x, y)
    (reg_line,) = ax.plot(
        x_line,
        y_line,
        color="0.15",
        linewidth=2,
        visible=len(x) >= 2,
    )
    coldest_x_init = coldest_x_for_mode(use_measured_init)
    (coldest_marker,) = ax.plot(
        [coldest_x_init] if coldest_x_init is not None else [],
        [coldest_y_c] if coldest_x_init is not None and coldest_y_c is not None else [],
        marker="o",
        markersize=14,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
        linestyle="none",
        zorder=6,
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
        _, x_line_f, y_line_f = _energy_vs_swt_regression_stats(xf, yf)
        if len(xf) >= 2:
            reg_line.set_data(x_line_f, y_line_f)
            reg_line.set_visible(True)
        else:
            reg_line.set_visible(False)
        coldest_x = coldest_x_for_mode(use_measured)
        if coldest_x is not None and coldest_y_c is not None:
            coldest_marker.set_data([coldest_x], [coldest_y_c])
            coldest_marker.set_visible(True)
        else:
            coldest_marker.set_visible(False)
        ax.set_xlabel(_energy_vs_swt_x_axis_label(use_measured))
        ax.set_ylim(35, 90)
        x_lim_candidates: list[float] = []
        if xf.size:
            x_lim_candidates.extend(xf.tolist())
        if len(xf) >= 2:
            x_lim_candidates.extend(x_line_f.tolist())
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


def _house_alias_subplot_title(house_alias: str) -> str:
    override = _SUBPLOT_TITLE_OVERRIDES.get(house_alias)
    if override is not None:
        return override
    return house_alias.replace("_", " ").title()


ENERGY_VS_SWT_XLABEL_ESTIMATED = "Estimated house energy use (kWh)"
SWT_GRID_FIG_WIDTH = 14.0
SWT_GRID_FIG_HEIGHT = 8.0
OAT_PANEL_FIG_HEIGHT = 3.8
ENERGY_VS_SWT_YLABEL = "Average SWT (°C)"


def _coldest_hour_legend_label(oat_f: float) -> str:
    return f"Coldest hour: {float(oat_f):.1f}°F"


def _hottest_swt_hour_legend_label(swt_c: float) -> str:
    return f"Hottest avg SWT: {float(swt_c):.1f}°C"


def _oat_timeseries_legend_handles(
    *,
    show_coldest_marker: bool,
    show_max_swt_markers: bool,
    maple_transition_label: str | None,
) -> list:
    from matplotlib.lines import Line2D

    handles: list[Line2D] = []
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
                label="Coldest hour",
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
) -> None:
    """Hourly OAT (°C), red = coldest hour, green = max-SWT hour per house."""
    oat_times = pd.to_datetime(hour_start)
    oat_f_arr = np.asarray(oat_f, dtype=float)
    oat_vals = _swt_f_to_c(oat_f_arr)

    ax.plot(oat_times, oat_vals, color="tab:blue", linewidth=1.5, alpha=0.6, zorder=1)

    coldest_oat_f: float | None = None
    if np.any(np.isfinite(oat_vals)):
        coldest_i = int(np.nanargmin(oat_vals))
        coldest_oat_f = float(oat_f_arr[coldest_i])
        t_coldest = pd.Timestamp(np.asarray(oat_times)[coldest_i])
        ax.plot(
            [t_coldest],
            [float(oat_vals[coldest_i])],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=5,
        )

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
        if hour not in oat_by_time.index:
            nearest = oat_by_time.index.get_indexer([hour], method="nearest")
            if nearest.size and nearest[0] >= 0:
                oat_at = float(oat_by_time.iloc[nearest[0]])
                t_at = oat_by_time.index[nearest[0]]
            else:
                continue
        else:
            oat_at = float(oat_by_time.loc[hour])
            t_at = hour
        if not np.isfinite(oat_at):
            continue
        house_label = _house_alias_subplot_title(panel.house_alias)
        green_markers.append((pd.Timestamp(t_at), oat_at, house_label))

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
    legend_handles = _oat_timeseries_legend_handles(
        show_coldest_marker=coldest_oat_f is not None and np.isfinite(coldest_oat_f),
        show_max_swt_markers=bool(green_markers),
        maple_transition_label=transition_legend,
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        framealpha=0.9,
        fontsize=legend_fontsize,
    )

    ax.set_ylabel("Outside air temperature (°C)", fontsize=axis_label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)


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
) -> list:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[i % len(palette)],
            markersize=zone_marker_size,
            label=_zone_legend_label_with_setpoints(
                zone,
                zone_names,
                zone_avg_set_f,
                coldest_hour_zone_set_f,
                green_hour_zone_set_f,
                setpoint_detail_zones=setpoint_detail_zones,
            ),
        )
        for i, zone in enumerate(sorted_zones)
    ]
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
) -> tuple[float, float] | None:
    x_all = dist_kwh_all if use_measured else estimated_all
    filtered = heatcall_all >= threshold
    xf = x_all[filtered]
    yf = y_all[filtered]
    coldest_x = coldest_x_measured if use_measured else coldest_x_estimated
    _, x_line_f, _y_line_f = _energy_vs_swt_regression_stats(xf, yf)
    x_lim_candidates: list[float] = []
    if xf.size:
        x_lim_candidates.extend(xf.tolist())
    if len(xf) >= 2:
        x_lim_candidates.extend(x_line_f.tolist())
    if coldest_x is not None and coldest_y_c is not None:
        x_lim_candidates.append(coldest_x)
    if yf.size:
        x_lim_candidates.append(float(xf[int(np.argmax(yf))]))
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
    legend_meta: dict | None = None,
    fixed_xlim: tuple[float, float] | None = None,
) -> None:
    x_all = dist_kwh_all if use_measured else estimated_all
    filtered = heatcall_all >= threshold
    filtered_idx = np.flatnonzero(filtered)
    xf = x_all[filtered]
    yf = y_all[filtered]
    zone_idx_filtered = np.array(
        [zone_to_index[int(z)] for z in dominant_zone_all[filtered]], dtype=float
    )
    xf_vis, yf_vis, zone_idx_vis = xf, yf, zone_idx_filtered
    if coldest_panel_index is not None and coldest_panel_index not in filtered_idx:
        ci = coldest_panel_index
        xf_vis = np.append(xf_vis, x_all[ci])
        yf_vis = np.append(yf_vis, y_all[ci])
        zone_idx_vis = np.append(
            zone_idx_vis,
            zone_to_index[int(dominant_zone_all[ci])],
        )
    if xf_vis.size:
        scatter.set_offsets(np.column_stack([xf_vis, yf_vis]))
        scatter.set_array(zone_idx_vis)
        scatter.set_visible(True)
    else:
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        scatter.set_visible(False)
    _, x_line_f, y_line_f = _energy_vs_swt_regression_stats(xf, yf)
    if len(xf) >= 2:
        reg_line.set_data(x_line_f, y_line_f)
        reg_line.set_visible(True)
    else:
        reg_line.set_visible(False)
    coldest_x = coldest_x_measured if use_measured else coldest_x_estimated
    if coldest_x is not None and coldest_y_c is not None:
        coldest_marker.set_data([coldest_x], [coldest_y_c])
        coldest_marker.set_visible(True)
    else:
        coldest_marker.set_visible(False)
    max_swt_legend_label: str | None = None
    green_hour_zone_set_f: dict[int, float] | None = None
    if yf.size:
        local_max_i = int(np.argmax(yf))
        max_swt_marker.set_data([xf[local_max_i]], [yf[local_max_i]])
        max_swt_marker.set_visible(True)
        swt_c_val = float(yf[local_max_i])
        if np.isfinite(swt_c_val):
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
        handles = _energy_vs_swt_legend_handles(
            legend_meta["sorted_zones"],
            legend_meta["palette"],
            legend_meta["zone_names"],
            coldest_oat_legend_label=(
                legend_meta.get("coldest_oat_legend_label") if show_marker_legends else None
            ),
            max_swt_legend_label=max_swt_legend_label if show_marker_legends else None,
            zone_avg_set_f=legend_meta.get("zone_avg_set_f"),
            coldest_hour_zone_set_f=legend_meta.get("coldest_hour_zone_set_f"),
            green_hour_zone_set_f=green_hour_zone_set_f,
            setpoint_detail_zones=legend_meta.get("setpoint_detail_zones"),
            zone_marker_size=legend_meta.get("zone_marker_size", 6),
            coldest_marker_size=legend_meta.get("coldest_marker_size", 7),
            max_swt_marker_size=legend_meta.get("max_swt_marker_size", 7),
        )
        legend_meta["legend"] = legend_meta["ax"].legend(
            handles=handles,
            loc="upper left",
            framealpha=0.9,
            fontsize=legend_meta.get("legend_fontsize", 7),
        )
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
        if len(xf) >= 2:
            y_peak_candidates.append(float(np.max(y_line_f)))
        if coldest_x is not None and coldest_y_c is not None:
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
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    expected = nrows * ncols
    if len(panels) != expected:
        raise ValueError(f"Expected {expected} panels, got {len(panels)}")

    threshold_init = 0.9
    title_fontsize = 12
    axis_label_fontsize = 10
    tick_fontsize = 9
    legend_fontsize = 9
    include_oat_panel = oat_hour_start is not None and oat_f is not None
    swt_row_height = SWT_GRID_FIG_HEIGHT / nrows
    oat_height_ratio = OAT_PANEL_FIG_HEIGHT / swt_row_height

    fig_height = SWT_GRID_FIG_HEIGHT + (
        OAT_PANEL_FIG_HEIGHT if include_oat_panel else 0.0
    )
    fig = plt.figure(figsize=(SWT_GRID_FIG_WIDTH, fig_height))
    if include_oat_panel:
        gs = fig.add_gridspec(
            nrows + 1,
            ncols,
            height_ratios=[1] * nrows + [oat_height_ratio],
            left=0.07,
            right=0.98,
            top=0.96,
            bottom=0.08,
            hspace=0.28,
            wspace=0.18,
        )
    else:
        gs = fig.add_gridspec(
            nrows,
            ncols,
            left=0.07,
            right=0.98,
            top=0.96,
            bottom=0.08,
            hspace=0.28,
            wspace=0.18,
        )

    palette = plt.cm.tab10.colors

    for index, panel in enumerate(panels):
        row, col = divmod(index, ncols)
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
        zone_cmap = ListedColormap(
            [palette[i % len(palette)] for i in range(len(sorted_zones))]
        )

        mask = heatcall_all >= threshold_init
        x0 = estimated_all[mask]
        y0 = y_all[mask]
        zone_idx0 = np.array(
            [zone_to_index[int(z)] for z in dominant_zone_all[mask]], dtype=float
        )

        scatter = ax.scatter(
            x0,
            y0,
            c=zone_idx0,
            cmap=zone_cmap,
            vmin=0,
            vmax=max(len(sorted_zones) - 1, 0),
            s=10,
            alpha=0.65,
            zorder=2,
        )
        _, x_line, y_line = _energy_vs_swt_regression_stats(x0, y0)
        (reg_line,) = ax.plot(
            x_line,
            y_line,
            color="0.15",
            linewidth=1.5,
            visible=len(x0) >= 2,
        )
        (coldest_marker,) = ax.plot(
            [coldest_x_estimated] if coldest_x_estimated is not None else [],
            [coldest_y_c]
            if coldest_x_estimated is not None and coldest_y_c is not None
            else [],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=6,
        )
        (max_swt_marker,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="green",
            markeredgewidth=1.5,
            linestyle="none",
            zorder=6,
            visible=False,
        )

        ax.set_title(_house_alias_subplot_title(panel.house_alias), fontsize=title_fontsize)
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
            "show_marker_legends": False,
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

        fixed_xlim = _compute_energy_vs_swt_xlim(
            estimated_all,
            estimated_all,
            y_all,
            heatcall_all,
            threshold=0.0,
            use_measured=False,
            coldest_x_estimated=coldest_x_estimated,
            coldest_x_measured=None,
            coldest_y_c=coldest_y_c,
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
            legend_meta=legend_meta,
            fixed_xlim=fixed_xlim,
        )
        ax.tick_params(labelsize=tick_fontsize)
        if col == 0:
            ax.set_ylabel(ENERGY_VS_SWT_YLABEL, fontsize=axis_label_fontsize)
        if row == nrows - 1:
            ax.set_xlabel(_energy_vs_swt_x_axis_label(False), fontsize=axis_label_fontsize)

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
        )

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
    savepath: Path | None = None,
    show: bool = False,
) -> None:
    """Gantt-style on/off heat-call rows (one horizontal band per zone)."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not zones:
        raise ValueError("plot_heat_calls requires at least one zone")

    parsed = [
        _parse_heatcall_zone(zone, default_row=i) for i, zone in enumerate(zones)
    ]
    y_max = max(row_idx for _, _, row_idx, _ in parsed) + 1
    fig_h = 0.85 + 0.48 * y_max
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    legend_handles: list[Line2D] = []
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
            ax.broken_barh(
                segments,
                (row_idx, 1.0),
                facecolors=face,
                edgecolors=edge,
                linewidth=0.85,
                alpha=0.78,
            )
        legend_handles.append(
            Line2D([0], [0], color=edge, linewidth=2.8, solid_capstyle="butt", label=label)
        )

    ax.set_ylim(0, y_max)
    ax.set_yticks(range(y_max + 1))
    ax.set_yticklabels([])
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=13, loc="center", pad=10)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        framealpha=1.0,
        fontsize=9,
        handlelength=1.6,
        borderaxespad=0.6,
    )
    legend.get_frame().set_linewidth(0.8)

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
    ax.grid(which="major", axis="y", color="0.88", linewidth=0.9)
    ax.tick_params(axis="y", length=0, colors="0.35")
    ax.tick_params(axis="x", colors="0.35")
    ax.spines["bottom"].set_color("0.75")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
