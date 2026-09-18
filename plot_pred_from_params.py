import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from house_parameters import HouseEnergyParams, design_matrix

WIND_SPEEDS_MPH = [0, 10, 20]
OAT_RANGE_F = np.linspace(-10, 40, 200)
# Hours above this are treated as bad meter/export values and dropped from the
# predicted-vs-actual figure (not from the fit).
MAX_PLAUSIBLE_DIST_KWH = 12.0
OAT_COLOR_F = (-13.0, 70.0)


def _oat_colors(oat_f) -> tuple[np.ndarray, Normalize, object]:
    """Cold-to-warm over a fixed outdoor temperature scale, shared by every figure.

    The same -13 to 70°F range is used on every plot so colors are comparable
    across houses and figure types.
    """
    oat_f = np.asarray(oat_f, dtype=float)
    vmin, vmax = OAT_COLOR_F
    return oat_f, Normalize(vmin=vmin, vmax=vmax), plt.cm.coolwarm


def _oat_colorbar(fig, axes, norm, cmap):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    ticks = np.linspace(norm.vmin, norm.vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.0f}°F" for t in ticks])
    return cbar


def plot_curves(
    house_parameters: list[HouseEnergyParams],
    fit_days: list[pd.Timestamp],
    fit_oat_f: list[float],
    t_i_avg: float,
    previous_dist_kwh_median: float,
    title: str,
    savepath=None,
    use_legend: bool = False
):
    """One house_kwh_pred(oat) curve per fit day, in a panel per wind speed.

    Curves are colored by fit_oat_f, the mean outdoor temperature over the window
    the fit was made on, so the color says what weather shaped the parameters.

    Most features have to be pinned before a curve against outdoor temperature
    exists at all. These curves are therefore NOT unconditional predictions:
    they hold at one operating point, namely

        set_minus_temp_zone* = 0     every zone exactly at its setpoint
        previous_dist_kwh = median   sustained operation, not a cold start
        OAT_avg_6h = oat_f           envelope in equilibrium with the current weather
        solar_w_m2 = 0               no sun
        interior average             pinned at t_i_avg, the median over the dataset,
                                     which is what turns oat into deltaT and
                                     windspeed_times_deltaT

    Wind speed is already mph in the data, so the panels need no conversion.
    """
    oat_values, norm, cmap = _oat_colors(fit_oat_f)
    feature_names = list(house_parameters[0].feature_names)

    fig, axes = plt.subplots(1, len(WIND_SPEEDS_MPH), figsize=(16, 5), sharey=True)
    for ax, ws in zip(axes, WIND_SPEEDS_MPH):
        deltaT = np.maximum(t_i_avg - OAT_RANGE_F, 0.0)
        operating_point = {
            "deltaT": deltaT,
            "windspeed_times_deltaT": deltaT * ws,
            "solar_w_m2": np.zeros_like(OAT_RANGE_F),
            "previous_dist_kwh": np.full_like(OAT_RANGE_F, previous_dist_kwh_median),
            "OAT_avg_6h": OAT_RANGE_F,
        }
        for name in feature_names:
            if name.startswith("set_minus_temp_zone"):
                operating_point[name] = np.zeros_like(OAT_RANGE_F)
        op_df = pd.DataFrame(operating_point, columns=feature_names)
        X = design_matrix(op_df, feature_names)

        for p, d, o in zip(house_parameters, fit_days, oat_values):
            house_kwh_pred = X @ p.coefficients()
            ax.plot(
                OAT_RANGE_F, house_kwh_pred, color=cmap(norm(o)),
                alpha=0.9 if use_legend else 0.5,
                linewidth=1.5 if use_legend else 0.8,
                label=f"{pd.Timestamp(d).strftime('%b %d')} ({o:.0f}°F)" if use_legend else None
            )
        ax.set_title(f"Wind speed: {ws} mph")
        ax.set_xlabel("Outside air temperature (°F)")
    axes[0].set_ylabel("House energy (scaled dist kWh)")

    if use_legend:
        axes[-1].legend(title="Fit day (window mean OAT)")
    else:
        _oat_colorbar(fig, axes, norm, cmap)

    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")


def plot_pred_vs_actual(
    predicted: np.ndarray,
    actual: np.ndarray,
    oat_f: np.ndarray,
    title: str,
    savepath=None,
    baseline_mae: float | None = None,
    baseline_rmse: float | None = None,
):
    """Out-of-sample next-day predictions against what the house actually used.

    Points are colored by that hour's outdoor temperature, on the same cold-to-warm
    scale as plot_curves, so a temperature-dependent bias shows up as a color
    gradient off the 1:1 line.

    Points should already be filtered on raw dist_kwh; values are scaled energy.
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    oat_f = np.asarray(oat_f)
    errors = predicted - actual

    oat_values, norm, cmap = _oat_colors(oat_f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(actual, predicted, c=oat_values, cmap=cmap, norm=norm, s=8, alpha=0.6)
    hi = float(max(MAX_PLAUSIBLE_DIST_KWH, actual.max(), predicted.max()) * 1.05)
    limits = [0, hi]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual scaled dist_kwh (kWh)")
    ax.set_ylabel("Predicted scaled dist_kwh (kWh)")
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors**2).mean()))

    scatter_title = "Next-day out-of-sample predictions"
    if baseline_mae is not None and baseline_rmse is not None:
        def _vs_abg(new, old, name):
            pct = (old - new) / old * 100.0
            word = "better" if pct >= 0 else "worse"
            return f"{name} {abs(pct):.0f}% {word}"
        scatter_title += (
            "\n"
            + _vs_abg(rmse, baseline_rmse, "RMSE")
            + ", "
            + _vs_abg(mae, baseline_mae, "MAE")
            + " than αβγ"
        )
    ax.set_title(scatter_title)

    stats = (
        f"MAE  = {mae:.3f} kWh\n"
    )
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

    _oat_colorbar(fig, list(axes), norm, cmap)
    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
