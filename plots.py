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
    sem = by_hour["std"] / np.sqrt(by_hour["count"])
    ci95 = 1.96 * sem
    ci95 = ci95.where(by_hour["count"] >= 2, 0.0).fillna(0.0)
    return by_hour["mean"].to_numpy(), ci95.to_numpy()


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
    ticks = np.linspace(norm.vmin, norm.vmax, 6)
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
