"""
Actual vs predicted, and error distributions: alpha/beta/gamma vs the linear model.

Every point is an out-of-fold prediction: the record is cut into week-blocks
dealt round-robin into 6 folds, and each hour is predicted by a model fitted
without the week it belongs to. So the scatter shows genuine held-out
performance across the whole winter, not a fit to its own training data.

The alpha/beta/gamma model is the FORM

    Q = a + b * T_o + c * (T_i - T_o) * v

refitted on each fold's training data. The deployed coefficients are not used:
alpha there is design-day sizing power, so scoring them would measure how they
were chosen rather than whether the form works.

    uv run -u scripts/11_model_comparison.py
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
from house.simple import Linear, hourly_frame, score_energy
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
BLOCK, K, H = 2016, 6, 12

# validated categorical slots 1 and 2 (see the dataviz reference palette)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

LINEAR = ["dT", "wind", "GHI", "gap1", "gap2", "prev", "call", "to_6h"]
ABG = ["T_o", "wind_abg"]


def build() -> pd.DataFrame:
    data = load_data(verbose=False)
    df = data.df
    cal = calibrate(df)
    to = df.T_o.to_numpy()

    h = hourly_frame(df, cal, segments=data.segments, burn=288)
    h = h[(h.i // BLOCK) == ((h.i + H - 1) // BLOCK)].reset_index(drop=True)
    h["fold"] = (h.i // BLOCK) % K
    h["wind_abg"] = (h.T_i - h.T_o) * h.v
    h["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in h.i]

    out = []
    for k in range(K):
        tr, te = h[h.fold != k], h[h.fold == k]
        rec = te[["t", "actual", "T_o"]].copy()
        rec["linear"] = Linear("l", LINEAR).fit(tr).predict(te)
        rec["abg"] = Linear("a", ABG).fit(tr).predict(te)
        out.append(rec)
    return pd.concat(out).sort_values("t").reset_index(drop=True)


def main() -> None:
    v = build()
    s_lin = score_energy(v.linear, v.actual)
    s_abg = score_energy(v.abg, v.actual)

    print(f"  out-of-fold hours: {len(v)}, mean actual {v.actual.mean():.2f} kWh")
    for name, s in (("alpha/beta/gamma (refitted)", s_abg),
                    ("linear (8 features)", s_lin)):
        print(f"  {name:<28s} MAE {s['mae_kwh']:.3f}  RMSE {s['rmse_kwh']:.3f}"
              f"  bias {s['bias_kwh']:+.3f}  r {s['r']:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor("#fcfcfb")

    # Crop the scatter axes at 10 kWh. The handful of larger hours are real
    # startup transients, but they stretch the axes so far that the bulk of
    # the data compresses into a corner. Scores below are still computed on
    # every point -- only the view is cropped, and the count is stated.
    hi = 10.0
    n_off = int(((v.actual > hi) | (v.linear > hi) | (v.abg > hi)).sum())

    for a, (col, colour, label, s) in zip(
            ax[:2],
            [("abg", ORANGE, "alpha/beta/gamma", s_abg),
             ("linear", BLUE, "linear model", s_lin)]):
        a.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", zorder=1)
        a.scatter(v.actual, v[col], s=11, alpha=0.35, color=colour,
                  linewidths=0, zorder=2)
        a.set_xlim(0, hi)
        a.set_ylim(0, hi)
        a.set_aspect("equal")
        a.set_xlabel("actual next-hour energy  [kWh]", color=MUTED)
        a.set_ylabel("predicted  [kWh]", color=MUTED)
        a.set_title(f"{label}\nMAE {s['mae_kwh']:.3f} kWh   r = {s['r']:.3f}",
                    color=INK, fontsize=11)
        a.text(0.5, -0.155, f"{n_off} hours above {hi:.0f} kWh not shown; "
               "scores use all data", transform=a.transAxes, ha="center",
               color=MUTED, fontsize=8)
        _clean(a)

    a = ax[2]
    # Outlines, not heavy fills: the two distributions overlap in the bulk and
    # a filled overlay hides exactly the difference we are trying to show.
    lo, hi_e = -3.0, 3.0
    bins = np.linspace(lo, hi_e, 49)
    stats = {}
    for col, colour, label in (("abg", ORANGE, "alpha/beta/gamma"),
                               ("linear", BLUE, "linear model")):
        err = (v[col] - v.actual).to_numpy()
        stats[col] = (np.quantile(err, 0.1), np.quantile(err, 0.9),
                      int((np.abs(err) > hi_e).sum()))
        a.hist(np.clip(err, lo, hi_e), bins=bins, histtype="step",
               color=colour, linewidth=2, label=label)
    a.axvline(0, color=MUTED, lw=1, ls="--")

    # middle-80% span for each model, drawn as bars above the histogram
    top = a.get_ylim()[1]
    a.set_ylim(0, top * 1.22)
    for n, (col, colour) in enumerate((("abg", ORANGE), ("linear", BLUE))):
        q10, q90, _ = stats[col]
        y = top * (1.14 - 0.07 * n)
        a.plot([q10, q90], [y, y], color=colour, lw=3, solid_capstyle="butt")
        a.text(q90 + 0.12, y, f"{q90 - q10:.2f} kWh", color=colour,
               va="center", fontsize=9)
    a.text(lo + 0.1, top * 1.19, "middle 80% of errors", color=MUTED,
           fontsize=9)

    n_out = stats["abg"][2] + stats["linear"][2]
    a.set_xlabel("predicted minus actual  [kWh]", color=MUTED)
    a.text(0.5, -0.155, f"{n_out} of {2 * len(v)} points beyond \u00b13 kWh, "
           "piled at the edges", transform=a.transAxes, ha="center",
           color=MUTED, fontsize=8)
    a.set_ylabel("hours", color=MUTED)
    a.set_title("Error distribution\nnarrower is better", color=INK, fontsize=11)
    a.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left",
             bbox_to_anchor=(0.0, 0.86))
    _clean(a)

    fig.suptitle(
        f"Hour-ahead loop energy, {len(v)} out-of-fold hours "
        f"(mean {v.actual.mean():.2f} kWh)", color=INK, fontsize=12, y=1.0)
    fig.tight_layout()
    path = RESULTS / "11_model_comparison.png"
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    print(f"\n  -> {path}")
    v.to_csv(RESULTS / "11_out_of_fold.csv", index=False)


def _clean(a):
    a.grid(True, color=GRID, lw=0.6, alpha=0.8)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)
    a.set_facecolor("#fcfcfb")


if __name__ == "__main__":
    main()
