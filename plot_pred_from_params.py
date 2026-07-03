import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from house_parameters import HouseParameters

WIND_SPEEDS_MPH = [0, 10, 20]
OAT_RANGE_F = np.linspace(-10, 40, 200)


def plot_curves(
    house_parameters: list[HouseParameters], 
    fit_days: list[pd.Timestamp], 
    oat_ref: float, 
    title: str, 
    use_legend: bool = False
):
    """One house_kwh_pred(oat) curve per fit day, in a panel per wind speed, colored by date."""

    ordinals = np.array([pd.Timestamp(d).toordinal() for d in fit_days])
    norm = Normalize(vmin=ordinals.min(), vmax=ordinals.max())
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, len(WIND_SPEEDS_MPH), figsize=(16, 5), sharey=True)
    for ax, ws in zip(axes, WIND_SPEEDS_MPH):
        for p, d, o in zip(house_parameters, fit_days, ordinals):
            house_kwh_pred = p.alpha + p.beta * (OAT_RANGE_F - oat_ref) + p.gamma * (65 - OAT_RANGE_F) * ws
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