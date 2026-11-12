from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from house_parameters import RswtLoadBucketFit


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
    ax: Axes,
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
    in_sample: bool = False,
    baseline_predicted: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize

    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    oat_values = np.asarray(oat_f, dtype=float)
    if baseline_predicted is not None:
        baseline_predicted = np.asarray(baseline_predicted)

    oat_min, oat_max = oat_f_colormap_bounds
    if oat_min >= oat_max:
        oat_max = oat_min + 1.0
    norm = Normalize(vmin=oat_min, vmax=oat_max)
    cmap = plt.cm.coolwarm

    hi = float(max(actual.max(), predicted.max()) * 1.05)
    if baseline_predicted is not None:
        hi = float(max(hi, baseline_predicted.max() * 1.05))
    limits = (0.0, hi)

    n_rows = 2 if baseline_predicted is not None else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    error_hist_bins = 60
    error_hist_range = (-6, 6)

    def error_hist_ymax(*error_arrays: np.ndarray) -> float:
        ymax = 0.0
        for err in error_arrays:
            counts, _ = np.histogram(err, bins=error_hist_bins, range=error_hist_range)
            ymax = max(ymax, float(counts.max()))
        return ymax * 1.05 if ymax > 0 else 1.0

    shared_hist_ylim = error_hist_ymax(
        predicted - actual,
        *( [baseline_predicted - actual] if baseline_predicted is not None else [] ),
    )

    def draw_row(
        ax_scatter: Axes,
        ax_hist: Axes,
        predicted_row: np.ndarray,
        *,
        scatter_title: str,
        compare_to_baseline: bool,
    ) -> None:
        errors = predicted_row - actual
        ax_scatter.scatter(
            actual, predicted_row, c=oat_values, cmap=cmap, norm=norm, s=8, alpha=0.6
        )
        ax_scatter.plot(limits, limits, "k--", linewidth=1)
        ax_scatter.set_xlim(limits)
        ax_scatter.set_ylim(limits)
        ax_scatter.set_xlabel("Actual scaled dist_kwh (kWh)")
        ax_scatter.set_ylabel("Predicted scaled dist_kwh (kWh)")
        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt((errors**2).mean()))

        title_text = scatter_title
        if (
            compare_to_baseline
            and baseline_mae is not None
            and baseline_rmse is not None
        ):
            rmse_pct = (baseline_rmse - rmse) / baseline_rmse * 100.0
            mae_pct = (baseline_mae - mae) / baseline_mae * 100.0
            title_text += (
                "\n"
                + f"RMSE {abs(rmse_pct):.0f}% {'better' if rmse_pct >= 0 else 'worse'}"
                + ", "
                + f"MAE {abs(mae_pct):.0f}% {'better' if mae_pct >= 0 else 'worse'}"
                + " than αβγ"
            )
        ax_scatter.set_title(title_text)

        stats = f"MAE  = {mae:.3f} kWh\nRMSE = {rmse:.3f} kWh"
        ax_scatter.text(
            0.97,
            0.03,
            stats,
            transform=ax_scatter.transAxes,
            ha="right",
            va="bottom",
            family="monospace",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax_hist.hist(
            errors, bins=error_hist_bins, range=error_hist_range, color="tab:blue", alpha=0.8
        )
        ax_hist.axvline(0, color="k", linestyle="--", linewidth=1)
        ax_hist.set_xlim(error_hist_range)
        ax_hist.set_ylim(0, shared_hist_ylim)
        ax_hist.set_xlabel("Prediction error (scaled kWh)")
        ax_hist.set_ylabel("Hours")
        ax_hist.set_title("Error distribution")

    main_scatter_title = (
        "In-sample predictions"
        if in_sample
        else f"{oos_period_label.capitalize()} out-of-sample predictions"
    )
    draw_row(
        axes[0, 0],
        axes[0, 1],
        predicted,
        scatter_title=main_scatter_title,
        compare_to_baseline=True,
    )
    if baseline_predicted is not None:
        baseline_title = (
            "αβγ baseline in-sample predictions"
            if in_sample
            else f"αβγ baseline {oos_period_label} out-of-sample predictions"
        )
        draw_row(
            axes[1, 0],
            axes[1, 1],
            baseline_predicted,
            scatter_title=baseline_title,
            compare_to_baseline=False,
        )

    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
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
    savepath = results_dir / "channel_boxplots.png"
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
    # plt.show()
    plt.close(fig)


def _rmse_vs_actual(predicted: np.ndarray, actual: np.ndarray) -> float | None:
    pred = np.asarray(predicted, dtype=float)
    act = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(act)
    if not np.any(mask):
        return None
    errors = pred[mask] - act[mask]
    return float(np.sqrt(np.mean(errors**2)))


def plot_test_period_method_comparison(
    timestamp: pd.Series | np.ndarray,
    new_method: np.ndarray,
    old_method: np.ndarray,
    *,
    actual: np.ndarray | None = None,
    baseline_method: np.ndarray | None = None,
    extra_method: np.ndarray | None = None,
    new_method_label: str = "New method",
    old_method_label: str = "Old method",
    extra_method_label: str = "Extra method",
    ylabel: str,
    title: str,
    savepath: Path | None = None,
) -> None:
    """Hourly new-model vs legacy-method lines over the test window; optional actual."""
    import matplotlib.pyplot as plt

    ts = pd.to_datetime(timestamp)
    fig, ax = plt.subplots(figsize=(12, 4))
    new_label = new_method_label
    old_label = old_method_label
    extra_label = extra_method_label
    if actual is not None:
        rmse_new = _rmse_vs_actual(new_method, actual)
        rmse_old = _rmse_vs_actual(old_method, actual)
        if rmse_new is not None:
            new_label = f"{new_method_label} (RMSE={rmse_new:.2f})"
        if rmse_old is not None:
            old_label = f"{old_method_label} (RMSE={rmse_old:.2f})"
        if extra_method is not None:
            rmse_extra = _rmse_vs_actual(extra_method, actual)
            if rmse_extra is not None:
                extra_label = f"{extra_method_label} (RMSE={rmse_extra:.2f})"
    if actual is not None:
        ax.plot(ts, actual, color="tab:red", linewidth=1.2, label="Actual")
    ax.plot(ts, new_method, color="tab:gray", linewidth=1.2, label=new_label)
    if baseline_method is not None:
        ax.plot(
            ts,
            baseline_method,
            color="tab:gray",
            linewidth=1.2,
            linestyle=":",
            label="Baseline",
        )
    ax.plot(
        ts,
        old_method,
        color="tab:gray",
        linewidth=1.2,
        linestyle="--",
        label=old_label,
    )
    if extra_method is not None:
        ax.plot(
            ts,
            extra_method,
            color="tab:blue",
            linewidth=1.2,
            linestyle="-.",
            label=extra_label,
        )
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_rswt_physics_interactive(
    timestamp: pd.Series | np.ndarray,
    q_kwh: np.ndarray,
    scada_rswt_f: np.ndarray,
    *,
    physics_rswt_fn: Callable[..., np.ndarray],
    physics_fixed_kwargs: dict[str, Any],
    initial_t_room_c: float,
    initial_n: float,
    title: str,
    t_room_f_range: tuple[float, float] | None = None,
    n_range: tuple[float, float] = (0.5, 3.0),
) -> None:
    """RSWT vs time: fixed SCADA legacy curve; physics curve with T_room and n sliders."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    ts = pd.to_datetime(timestamp)
    q = np.asarray(q_kwh, dtype=float)
    scada = np.asarray(scada_rswt_f, dtype=float)

    initial_t_room_f = initial_t_room_c * (9.0 / 5.0) + 32.0
    if t_room_f_range is None:
        t_room_f_range = (initial_t_room_f - 15.0, initial_t_room_f + 15.0)

    def _physics(t_room_c: float, n: float) -> np.ndarray:
        return np.asarray(
            physics_rswt_fn(q, t_room_c=t_room_c, n=n, **physics_fixed_kwargs),
            dtype=float,
        )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    plt.subplots_adjust(bottom=0.28)
    physics_line, = ax.plot(
        ts,
        _physics(initial_t_room_c, initial_n),
        color="tab:blue",
        linewidth=1.2,
        linestyle="-.",
        label=f"Physics (T_room={initial_t_room_f:.1f} °F, n={initial_n:.2f})",
    )
    ax.plot(
        ts,
        scada,
        color="tab:gray",
        linewidth=1.2,
        linestyle="--",
        label="SCADA legacy",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("RSWT (°F)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()

    ax_t_room = fig.add_axes([0.12, 0.14, 0.76, 0.03])
    ax_n = fig.add_axes([0.12, 0.08, 0.76, 0.03])
    slider_t_room = Slider(
        ax_t_room,
        "T_room (°F)",
        t_room_f_range[0],
        t_room_f_range[1],
        valinit=initial_t_room_f,
    )
    slider_n = Slider(ax_n, "n", n_range[0], n_range[1], valinit=initial_n)

    def _on_slider_change(_val: float) -> None:
        t_room_f = slider_t_room.val
        n_val = slider_n.val
        t_room_c = (t_room_f - 32.0) * (5.0 / 9.0)
        physics_line.set_ydata(_physics(t_room_c, n_val))
        physics_line.set_label(
            f"Physics (T_room={t_room_f:.1f} °F, n={n_val:.2f})"
        )
        ax.legend(loc="upper left")
        fig.canvas.draw_idle()

    slider_t_room.on_changed(_on_slider_change)
    slider_n.on_changed(_on_slider_change)
    plt.show()


def plot_test_period_combined_interactive(
    timestamp: pd.Series | np.ndarray,
    new_energy: np.ndarray,
    old_energy: np.ndarray,
    compare_f: np.ndarray,
    *,
    actual_energy: np.ndarray | None = None,
    new_energy_label: str = "6-feature regression",
    old_energy_label: str = "Alpha/beta/gamma",
    proposed_method_label: str = "Physics based (proposed)",
    current_method_label: str = "Quadratic (current)",
    t_room_c: float,
    water_kg_hour: float,
    cp_kwh_per_kg_k: float,
    initial_q_dd_kwh: float,
    initial_t_supply_dd_f: float,
    initial_n: float,
    title: str,
    q_dd_range: tuple[float, float] | None = None,
    q_dd_slider_max: float = 20.0,
    t_supply_dd_f_range: tuple[float, float] = (100.0, 180.0),
    n_range: tuple[float, float] = (1.0, 1.6),
    chc_swt_timestamp: pd.Series | np.ndarray | None = None,
    chc_swt_f: np.ndarray | None = None,
    chc_swt_label: str = "Measured SWT (max heat call >90%)",
) -> None:
    """Energy (top) and supply temp with sliders (bottom) for the test window."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons, RadioButtons, Slider

    from rswt_physics import supply_temp_f_from_energy_kwh_dd_anchor

    ts = pd.to_datetime(timestamp)
    q_regression = np.asarray(new_energy, dtype=float)
    q_legacy = np.asarray(old_energy, dtype=float)
    compare = np.asarray(compare_f, dtype=float)
    q_for_physics = q_regression

    q_pos = np.concatenate(
        [
            q_regression[np.isfinite(q_regression) & (q_regression > 0)],
            q_legacy[np.isfinite(q_legacy) & (q_legacy > 0)],
        ]
    )
    if q_dd_range is None:
        q_hi = float(np.max(q_pos)) if q_pos.size else initial_q_dd_kwh * 2.0
        q_dd_range = (
            max(0.5, q_hi * 0.3),
            max(q_hi * 1.5, initial_q_dd_kwh * 1.2, q_dd_slider_max),
        )

    slider_step = 0.1

    def _snap_slider(val: float) -> float:
        return round(val / slider_step) * slider_step

    def _snap_slider_range(lo: float, hi: float) -> tuple[float, float]:
        lo_s = float(np.floor(lo / slider_step) * slider_step)
        hi_s = float(np.ceil(hi / slider_step) * slider_step)
        if hi_s <= lo_s:
            hi_s = lo_s + slider_step
        return lo_s, hi_s

    q_dd_range = _snap_slider_range(q_dd_range[0], q_dd_range[1])
    t_supply_dd_f_range = _snap_slider_range(
        t_supply_dd_f_range[0], t_supply_dd_f_range[1]
    )
    n_range = _snap_slider_range(n_range[0], n_range[1])
    initial_q_dd_kwh = _snap_slider(initial_q_dd_kwh)
    initial_t_supply_dd_f = _snap_slider(initial_t_supply_dd_f)
    initial_n = _snap_slider(initial_n)

    fixed = {
        "t_room_c": t_room_c,
        "water_kg_hour": water_kg_hour,
        "cp_kwh_per_kg_k": cp_kwh_per_kg_k,
    }

    def _physics(q_dd: float, t_supply_dd_f: float, n_val: float) -> np.ndarray:
        return supply_temp_f_from_energy_kwh_dd_anchor(
            q_for_physics,
            q_dd_kwh=q_dd,
            t_supply_dd_f=t_supply_dd_f,
            n=n_val,
            **fixed,
        )

    def _update_physics_line() -> None:
        physics_line.set_ydata(
            _physics(slider_q.val, slider_t.val, slider_n.val)
        )
        fig.canvas.draw_idle()

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], top=0.88, bottom=0.24, hspace=0.20)
    ax_energy = fig.add_subplot(gs[0])
    ax_supply = fig.add_subplot(gs[1], sharex=ax_energy)

    actual_arr: np.ndarray | None = None
    new_label = new_energy_label
    old_label = old_energy_label
    actual_line = None
    if actual_energy is not None:
        actual_arr = np.asarray(actual_energy, dtype=float)
        rmse_new = _rmse_vs_actual(new_energy, actual_arr)
        rmse_old = _rmse_vs_actual(old_energy, actual_arr)
        if rmse_new is not None:
            new_label = f"{new_energy_label} (RMSE={rmse_new:.2f})"
        if rmse_old is not None:
            old_label = f"{old_energy_label} (RMSE={rmse_old:.2f})"
        actual_line, = ax_energy.plot(
            ts,
            actual_arr,
            color="tab:red",
            linewidth=1.2,
            alpha=0.55,
            label="Actual",
        )
    new_line, = ax_energy.plot(
        ts, q_regression, color="tab:gray", linewidth=1.2, label=new_label
    )
    old_line, = ax_energy.plot(
        ts,
        q_legacy,
        color="tab:gray",
        linewidth=1.2,
        linestyle="--",
        label=old_label,
    )
    ax_energy.set_ylabel("Energy use (kWh)")
    ax_energy.set_title("Energy use prediction")
    ax_energy.legend(loc="upper left", fontsize=9)
    plt.setp(ax_energy.get_xticklabels(), visible=False)

    def _refresh_energy_panel(*, show_error: bool) -> None:
        if show_error and actual_arr is not None:
            new_line.set_ydata(q_regression - actual_arr)
            old_line.set_ydata(q_legacy - actual_arr)
            if actual_line is not None:
                actual_line.set_visible(False)
            ax_energy.set_ylabel("Prediction error (kWh)")
            ax_energy.set_title("Energy prediction error (predicted − actual)")
        else:
            new_line.set_ydata(q_regression)
            old_line.set_ydata(q_legacy)
            if actual_line is not None:
                actual_line.set_ydata(actual_arr)
                actual_line.set_visible(True)
            ax_energy.set_ylabel("Energy use (kWh)")
            ax_energy.set_title("Energy use prediction")
        ax_energy.relim()
        ax_energy.autoscale_view()
        ax_energy.legend(loc="upper left", fontsize=9)
        fig.canvas.draw_idle()

    if chc_swt_f is not None and chc_swt_timestamp is not None:
        chc_ts = pd.to_datetime(chc_swt_timestamp)
        chc_swt = np.asarray(chc_swt_f, dtype=float)
        chc_ok = np.isfinite(chc_swt)
        if np.any(chc_ok):
            ax_supply.scatter(
                chc_ts[chc_ok],
                chc_swt[chc_ok],
                color="tab:red",
                s=28,
                alpha=0.55,
                zorder=5,
                label=chc_swt_label,
            )
    physics_line, = ax_supply.plot(
        ts,
        _physics(initial_q_dd_kwh, initial_t_supply_dd_f, initial_n),
        color="tab:gray",
        linewidth=1.2,
        label=proposed_method_label,
    )
    ax_supply.plot(
        ts,
        compare,
        color="tab:gray",
        linewidth=1.2,
        linestyle="--",
        label=current_method_label,
    )
    t_room_f = t_room_c * (9.0 / 5.0) + 32.0
    ax_supply.set_xlabel("Time")
    ax_supply.set_ylabel("Supply water temp (°F)")
    ax_supply.set_title(
        rf"RSWT prediction ($T_{{room}}$={t_room_f:.1f} °F, "
        rf"$\dot{{m}}$={water_kg_hour:.2f} kg/h)"
    )
    ax_supply.legend(loc="upper left", fontsize=9)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax_supply.xaxis.set_major_locator(locator)
    ax_supply.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    slider_y = 0.042
    slider_h = 0.027
    slider_w = 0.13
    slider_gap = 0.10
    n_sliders = 3
    sliders_span = n_sliders * slider_w + (n_sliders - 1) * slider_gap
    slider_x0 = (1.0 - sliders_span) / 2.0
    ax_q = fig.add_axes([slider_x0, slider_y, slider_w, slider_h])
    ax_t = fig.add_axes(
        [slider_x0 + slider_w + slider_gap, slider_y, slider_w, slider_h]
    )
    ax_n = fig.add_axes(
        [
            slider_x0 + 2 * (slider_w + slider_gap),
            slider_y,
            slider_w,
            slider_h,
        ]
    )

    def _slider_title_above(slider_ax, text: str) -> None:
        pos = slider_ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            pos.y1 + 0.025,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            transform=fig.transFigure,
        )

    slider_q = Slider(
        ax_q,
        "",
        q_dd_range[0],
        q_dd_range[1],
        valinit=initial_q_dd_kwh,
        valstep=slider_step,
    )
    slider_t = Slider(
        ax_t,
        "",
        t_supply_dd_f_range[0],
        t_supply_dd_f_range[1],
        valinit=initial_t_supply_dd_f,
        valstep=slider_step,
    )
    slider_n = Slider(
        ax_n,
        "",
        n_range[0],
        n_range[1],
        valinit=initial_n,
        valstep=slider_step,
    )
    for s in (slider_q, slider_t, slider_n):
        s.label.set_visible(False)
    _slider_title_above(ax_q, "Q_DD (kW)")
    _slider_title_above(ax_t, "T_supply,DD (°F)")
    _slider_title_above(ax_n, "n")

    if actual_arr is not None:
        energy_check_y = 0.902
        cax_energy = fig.add_axes([0.10, energy_check_y, 0.26, 0.028])
        cax_energy.set_axis_off()
        energy_mode_check = CheckButtons(
            cax_energy,
            ["Show prediction error"],
            actives=[False],
        )
        for label in energy_mode_check.labels:
            label.set_fontsize(8)

        def _on_energy_mode_toggle(_label: str) -> None:
            show_err = bool(energy_mode_check.get_status()[0])
            _refresh_energy_panel(show_error=show_err)

        energy_mode_check.on_clicked(_on_energy_mode_toggle)

    radio_row_y = 0.132
    radio_row_h = 0.045
    fig.text(
        0.10,
        radio_row_y + radio_row_h / 2,
        "Physics Q from:",
        fontsize=9,
        fontweight="bold",
        transform=fig.transFigure,
        va="center",
        ha="left",
    )
    rax = fig.add_axes([0.24, radio_row_y, 0.58, radio_row_h])
    rax.set_axis_off()
    radio_q = RadioButtons(
        rax,
        (new_energy_label, old_energy_label),
        active=0,
        layout="horizontal",
        label_props={"fontsize": [8]},
    )

    def _on_physics_q_source(choice: str) -> None:
        nonlocal q_for_physics
        q_for_physics = q_regression if choice == new_energy_label else q_legacy
        _update_physics_line()

    def _on_slider_change(_val: float) -> None:
        _update_physics_line()

    radio_q.on_clicked(_on_physics_q_source)

    slider_q.on_changed(_on_slider_change)
    slider_t.on_changed(_on_slider_change)
    slider_n.on_changed(_on_slider_change)
    plt.show()


def plot_training_hourly_avg_setpoint(
    timestamp: pd.Series | np.ndarray,
    avg_setpoint_f: pd.Series | np.ndarray,
    *,
    title: str,
    savepath: Path | None = None,
) -> None:
    """Average zone setpoint (°F) over the energy training window: time series and histogram."""
    import matplotlib.pyplot as plt

    ts = pd.to_datetime(timestamp)
    setpoints = np.asarray(avg_setpoint_f, dtype=float)
    fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title)

    ax_ts.plot(ts, setpoints, color="tab:orange", linewidth=1.2, label="Avg setpoint")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Avg setpoint (°F)")
    ax_ts.legend(loc="upper left")
    fig.autofmt_xdate()

    valid = setpoints[np.isfinite(setpoints)]
    if valid.size:
        lo = np.floor(valid.min() / 2.0) * 2.0
        hi = np.ceil(valid.max() / 2.0) * 2.0
        bins = np.arange(lo, hi + 2.0, 2.0)
    else:
        bins = 30
    ax_hist.hist(valid, bins=bins, color="tab:orange", edgecolor="white", alpha=0.85)
    ax_hist.set_xlabel("Avg setpoint (°F)")
    ax_hist.set_ylabel("Hours")
    ax_hist.set_title("Distribution")

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
