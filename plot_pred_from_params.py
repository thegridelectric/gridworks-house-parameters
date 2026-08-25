import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from house_parameters import CENTERED, FEATURES, HouseEnergyParams

WIND_SPEEDS_MPH = [0, 10, 20]
OAT_RANGE_F = np.linspace(-10, 40, 200)


def operating_point(df: pd.DataFrame, oat_f: np.ndarray, ws_mph: float) -> dict:
    """Feature values along the curve, at a representative operating point.

    The model takes eight inputs, so a curve against outdoor temperature alone
    needs the other seven pinned somewhere. The choice here is sustained
    operation: rooms sitting exactly at their switching thresholds (gap = 0),
    the loop running, no sun, the envelope in equilibrium with the weather, and
    the previous hour at its typical value. That is the same situation the
    alpha/beta/gamma curve describes.
    """
    oat_c = (oat_f - 32.0) * 5.0 / 9.0
    d_t = np.maximum(df["T_i"].median() - oat_c, 0.0)
    return {
        "dT": d_t,
        "wind": d_t * ws_mph,
        "GHI": np.zeros_like(oat_c),
        "gap1": np.zeros_like(oat_c),
        "gap2": np.zeros_like(oat_c),
        "prev": np.full_like(oat_c, df["prev"].median()),
        "call": np.ones_like(oat_c),
        "to_6h": oat_c,
    }


def predict_curve(p: HouseEnergyParams, refs: dict, point: dict) -> np.ndarray:
    y = np.full_like(point["dT"], p.coefficients["intercept"], dtype=float)
    for f in FEATURES:
        y = y + p.coefficients[f] * (point[f] - refs.get(f, 0.0))
    return y


def plot_curves(
    house_parameters: list[HouseEnergyParams],
    fit_days: list[pd.Timestamp],
    refs: dict,
    df: pd.DataFrame,
    title: str,
    savepath=None,
    use_legend: bool = False
):
    """One house_kwh_pred(oat) curve per fit day, in a panel per wind speed, colored by date."""

    ordinals = np.array([pd.Timestamp(d).toordinal() for d in fit_days])
    norm = Normalize(vmin=ordinals.min(), vmax=ordinals.max())
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, len(WIND_SPEEDS_MPH), figsize=(16, 5), sharey=True)
    for ax, ws in zip(axes, WIND_SPEEDS_MPH):
        point = operating_point(df, OAT_RANGE_F, ws)
        for p, d, o in zip(house_parameters, fit_days, ordinals):
            house_kwh_pred = predict_curve(p, refs, point)
            ax.plot(
                OAT_RANGE_F, house_kwh_pred, color=cmap(norm(o)),
                alpha=0.9 if use_legend else 0.5,
                linewidth=1.5 if use_legend else 0.8,
                label=pd.Timestamp(d).strftime("%b %d") if use_legend else None
            )
        ax.set_title(f"Wind speed: {ws} mph")
        ax.set_xlabel("Outside air temperature (°F)")
    axes[0].set_ylabel("House energy (kWh)")

    if use_legend:
        axes[-1].legend(title="Fit day")
    else:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
        tick_ordinals = np.linspace(ordinals.min(), ordinals.max(), 6)
        cbar.set_ticks(tick_ordinals)
        cbar.set_ticklabels([pd.Timestamp.fromordinal(int(o)).strftime("%b %d") for o in tick_ordinals])

    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
