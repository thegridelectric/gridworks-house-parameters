"""
Actual vs predicted, incumbent model form against the best model found.

Every point is an out-of-fold prediction: the model that produced it was fitted
on the other five week-blocks and never saw this hour. That is stricter than
holding out one contiguous month, and it covers the whole winter rather than
just February.

The alpha/beta/gamma comparison uses the FORM, refitted per fold:

    Q = a + b * T_o + c * (T_i - T_o) * v

not the deployed coefficients, which were set for design-day sizing and so
measure how alpha was chosen rather than whether the form works.

    uv run scripts/11_compare_plot.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.simple import FEATURES, Linear, hourly_frame, score_energy
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
BLOCK, K, HORIZON = 2016, 6, 12

# validated categorical slots 1 and 2 (all-pairs CVD dE 24.7)
C_INCUMBENT = "#eb6834"
C_BEST = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

ABG = ["T_o", "wind_abg"]
BEST = FEATURES["per-zone + time of day"]


def out_of_fold(h, feats):
    pred = pd.Series(index=h.index, dtype=float)
    for f in range(K):
        tr, te = h[h.fold != f], h[h.fold == f]
        pred.loc[te.index] = Linear("m", feats).fit(tr).predict(te)
    return pred


def main() -> None:
    data = load_data(verbose=False)
    df = data.df
    cal = calibrate(df)

    h = hourly_frame(df, cal, segments=data.segments, burn=288)
    h = h[(h.i // BLOCK) == ((h.i + HORIZON - 1) // BLOCK)].reset_index(drop=True)
    h["fold"] = (h.i // BLOCK) % K
    h["wind_abg"] = (h.T_i - h.T_o) * h.v

    h["abg"] = out_of_fold(h, ABG)
    h["best"] = out_of_fold(h, BEST)

    s_abg = score_energy(h.abg, h.actual)
    s_best = score_energy(h.best, h.actual)
    print(f"  alpha/beta/gamma (refitted): MAE {s_abg['mae_kwh']:.3f} kWh, "
          f"r {s_abg['r']:.3f}")
    print(f"  best model                 : MAE {s_best['mae_kwh']:.3f} kWh, "
          f"r {s_best['r']:.3f}")
    print(f"  improvement                : "
          f"{100 * (1 - s_best['mae_kwh'] / s_abg['mae_kwh']):.0f}%")

    _plot(h, s_abg, s_best, RESULTS / "11_model_comparison.png")
    h[["t", "actual", "abg", "best", "T_o", "fold"]].to_csv(
        RESULTS / "11_out_of_fold.csv", index=False)
    print(f"  -> {RESULTS / '11_out_of_fold.csv'}")


def _plot(h, s_abg, s_best, path):
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.3)
    # clip the axes to the bulk: a handful of >10 kWh hours would otherwise
    # stretch both panels and squash the cloud that carries the comparison
    hi = float(np.quantile(h.actual, 0.995))
    n_out = int((h.actual > hi).sum())

    for k, (col, color, label, s) in enumerate([
            ("abg", C_INCUMBENT, "Current model form\n(a + b·T_o + c·ΔT·v)", s_abg),
            ("best", C_BEST, "Best model found\n(8-feature linear)", s_best)]):
        ax = fig.add_subplot(gs[0, k])
        ax.plot([0, hi], [0, hi], color=INK_2, lw=1, ls="--", zorder=1)
        ax.scatter(h.actual, h[col], s=11, alpha=0.35, color=color,
                   linewidths=0, zorder=2)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_box_aspect(1)
        ax.set_xlabel("actual next-hour energy  [kWh]", color=INK_2, fontsize=9)
        if k == 0:
            ax.set_ylabel("predicted  [kWh]", color=INK_2, fontsize=9)
        ax.set_title(label, color=INK, fontsize=10.5, loc="left", pad=10)
        # direct label instead of a legend box: one series per panel
        ax.text(0.04, 0.95, f"MAE {s['mae_kwh']:.2f} kWh\nr = {s['r']:.2f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.5)

    # error distributions, the panel that answers "how much better"
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-3, 3, 61)
    for col, color, label in (("abg", C_INCUMBENT, "current form"),
                              ("best", C_BEST, "best model")):
        ax.hist(h[col] - h.actual, bins=bins, histtype="step", lw=2,
                color=color, label=label)
    ax.axvline(0, color=INK_2, lw=1, ls="--")
    ax.set_xlabel("predicted minus actual  [kWh]", color=INK_2, fontsize=9)
    ax.set_ylabel("hours", color=INK_2, fontsize=9)
    ax.set_title("Error distribution\n(taller and narrower is better)",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_box_aspect(1)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)

    fig.suptitle(
        f"Hour-ahead loop energy, {len(h)} out-of-fold predictions "
        "(6-fold block cross-validation, Nov 2025 – Feb 2026)",
        fontsize=12, color=INK, x=0.01, ha="left", y=0.98)
    fig.text(0.01, 0.02,
             f"Axes clipped at {hi:.1f} kWh; {n_out} hours above that are "
             "off-panel but included in every statistic.",
             fontsize=8.5, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"  plot -> {path}")


if __name__ == "__main__":
    main()
