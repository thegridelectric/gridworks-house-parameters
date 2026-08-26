import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from house_parameters import FEATURES, HouseEnergyParams, design_matrix

WIND_SPEEDS_MPH = [0, 10, 20]
OAT_RANGE_F = np.linspace(-10, 40, 200)


def _oat_colors(oat_f) -> tuple[np.ndarray, Normalize, object]:
    """Cold-to-warm over the outdoor temperature range, shared by every figure.

    Coloring by oat_f rather than by date puts fits made in comparable weather in
    comparable colors, whenever in the season they happened.
    """
    oat_f = np.asarray(oat_f, dtype=float)
    return oat_f, Normalize(vmin=oat_f.min(), vmax=oat_f.max()), plt.cm.coolwarm


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
    centers: dict[str, float],
    t_i_avg: float,
    prev_median: float,
    title: str,
    savepath=None,
    use_legend: bool = False
):
    """One house_kwh_pred(oat) curve per fit day, in a panel per wind speed.

    Curves are colored by fit_oat_f, the mean outdoor temperature over the window
    the fit was made on, so the color says what weather shaped the parameters.

    The model has seven features, so five of them have to be pinned before a
    curve against outdoor temperature exists at all. These curves are therefore
    NOT unconditional predictions: they hold at one operating point, namely

        gap1 = gap2 = 0     both rooms exactly at their setpoint
        prev = its median   sustained operation, not a cold start
        to_6h = oat_f       envelope in equilibrium with the current weather
        ghi = 0             no sun
        T_i1/T_i2 average   pinned at t_i_avg, the median over the dataset,
                            which is what turns oat into dT and wind

    Wind speed is already mph in the data, so the panels need no conversion.
    """
    oat_values, norm, cmap = _oat_colors(fit_oat_f)

    fig, axes = plt.subplots(1, len(WIND_SPEEDS_MPH), figsize=(16, 5), sharey=True)
    for ax, ws in zip(axes, WIND_SPEEDS_MPH):
        dT = np.maximum(t_i_avg - OAT_RANGE_F, 0.0)
        operating_point = pd.DataFrame({
            "dT": dT,
            "wind": dT * ws,
            "ghi": np.zeros_like(OAT_RANGE_F),
            "gap1": np.zeros_like(OAT_RANGE_F),
            "gap2": np.zeros_like(OAT_RANGE_F),
            "prev": np.full_like(OAT_RANGE_F, prev_median),
            "to_6h": OAT_RANGE_F,
        }, columns=FEATURES)
        X = design_matrix(operating_point, centers)

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
    axes[0].set_ylabel("House energy (kWh)")

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
):
    """Out-of-sample next-day predictions against what the house actually used.

    Points are colored by that hour's outdoor temperature, on the same cold-to-warm
    scale as plot_curves, so a temperature-dependent bias shows up as a color
    gradient off the 1:1 line.
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    errors = predicted - actual

    oat_values, norm, cmap = _oat_colors(oat_f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(actual, predicted, c=oat_values, cmap=cmap, norm=norm, s=8, alpha=0.6)
    limits = [0, float(max(actual.max(), predicted.max())) * 1.02]
    ax.plot(limits, limits, "k--", linewidth=1, label="1:1")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual dist_kwh (kWh)")
    ax.set_ylabel("Predicted dist_kwh (kWh)")
    ax.set_title("Next-day out-of-sample predictions")
    ax.legend(loc="upper left")

    stats = (
        f"n = {len(actual)}\n"
        f"MAE  = {np.abs(errors).mean():.3f} kWh\n"
        f"RMSE = {np.sqrt((errors**2).mean()):.3f} kWh\n"
        f"bias = {errors.mean():+.3f} kWh\n"
        f"corr = {np.corrcoef(actual, predicted)[0, 1]:.3f}"
    )
    ax.text(
        0.97, 0.03, stats, transform=ax.transAxes, ha="right", va="bottom",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax = axes[1]
    ax.hist(errors, bins=60, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction error (predicted - actual, kWh)")
    ax.set_ylabel("Hours")
    ax.set_title("Error distribution")

    _oat_colorbar(fig, list(axes), norm, cmap)
    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
