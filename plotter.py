import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

    Points are colored by that hour's outdoor temperature on a fixed cold-to-warm
    scale so a temperature-dependent bias shows up as a color gradient off the 1:1 line.

    Values are scaled energy (dist_kwh × energy_ratio from each fit window).
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    oat_f = np.asarray(oat_f)
    errors = predicted - actual

    oat_values, norm, cmap = _oat_colors(oat_f)

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
