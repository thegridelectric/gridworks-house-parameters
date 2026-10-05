from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
    max_zone_heatcall_fraction: np.ndarray,
    *,
    rwt_intercept: float,
    rwt_slope: float,
    rwt_intercept_chc: float,
    rwt_slope_chc: float,
    house_alias: str,
    n_days: int,
    savepath: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    swt = np.asarray(swt, dtype=float)
    rwt = np.asarray(rwt, dtype=float)
    heatcall = np.asarray(max_zone_heatcall_fraction, dtype=float)
    heatcall_norm = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        swt,
        rwt,
        c=heatcall,
        cmap=plt.cm.coolwarm,
        s=22,
        alpha=0.65,
        norm=heatcall_norm,
        label="Hourly data",
    )

    swt_line = np.linspace(float(np.min(swt)), float(np.max(swt)), num=100)
    ax.plot(
        swt_line,
        rwt_intercept + rwt_slope * swt_line,
        color="tab:blue",
        linewidth=2,
        label="All hours (OLS)",
    )
    ax.plot(
        swt_line,
        rwt_intercept_chc + rwt_slope_chc * swt_line,
        color="tab:red",
        linewidth=2,
        label="CHC hours (OLS)",
    )

    ax.set_xlabel("Supply water temperature SWT (°F)")
    ax.set_ylabel("Return water temperature RWT (°F)")
    ax.set_title(
        f"{house_alias.capitalize()}: RWT vs SWT (last {n_days} day(s))"
    )
    ax.legend(loc="upper left")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("max zone heat-call fraction")

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
