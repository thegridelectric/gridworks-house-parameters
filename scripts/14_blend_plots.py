"""
The 50/50 blend (linear + gradient boosting) against each incumbent.

Two figures, same layout as before: a scatter per model, then their error
distributions overlaid. Every point is an out-of-fold prediction under the
6-fold week-block CV, so nothing is scored on data its model was fitted on.

Scatter axes are cropped at 10 kWh. A handful of startup-transient hours run
higher; leaving them in squeezes the bulk of the data into a corner. Scores
below use every point, and the count excluded from the view is stated.

    uv run -u scripts/14_blend_plots.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import HistGradientBoostingRegressor

from house.data import load_data
from house.simple import Linear, hourly_frame, score_energy
from house.thermostat import calibrate

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parents[1] / "results"
BLOCK, K, H, DAY = 2016, 6, 12, 288
CROP = 10.0

# validated categorical slots 1 and 2
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

ENG = ["dT", "GHI", "gap1", "gap2", "prev", "to_6h"]
RAW = ["T_i1", "T_i2", "T_o", "v", "GHI", "gap1", "gap2", "prev", "to_6h",
       "call", "hour", "dT"]
ABG = ["T_o", "wind_abg"]


def out_of_fold() -> pd.DataFrame:
    data = load_data(verbose=False)
    df = data.df
    cal = calibrate(df)
    to = df.T_o.to_numpy()

    h = hourly_frame(df, cal, segments=data.segments, burn=DAY)
    h = h[(h.i // BLOCK) == ((h.i + H - 1) // BLOCK)].reset_index(drop=True)
    h["fold"] = (h.i // BLOCK) % K
    h["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in h.i]
    h["hour"] = pd.to_datetime(h.t).dt.hour
    h["wind_abg"] = (h.T_i - h.T_o) * h.v

    out = []
    for k in range(K):
        tr, te = h[h.fold != k], h[h.fold == k]
        rec = te[["t", "actual", "T_o"]].copy()
        rec["linear"] = Linear("l", ENG).fit(tr).predict(te)
        rec["abg"] = Linear("a", ABG).fit(tr).predict(te)
        g = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=400,
            max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.1,
            random_state=0)
        g.fit(tr[RAW].to_numpy(), tr.actual.to_numpy())
        rec["gbm"] = np.clip(g.predict(te[RAW].to_numpy()), 0.0, None)
        rec["blend"] = 0.5 * (rec.linear + rec.gbm)
        out.append(rec)
    return pd.concat(out).sort_values("t").reset_index(drop=True)


def figure(v, left_col, left_label, right_col, right_label, path, title):
    s_l, s_r = score_energy(v[left_col], v.actual), score_energy(v[right_col],
                                                                 v.actual)
    n_off = int(((v.actual > CROP) | (v[left_col] > CROP)
                 | (v[right_col] > CROP)).sum())

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor(SURFACE)

    for a, (col, colour, label, s) in zip(
            ax[:2], [(left_col, ORANGE, left_label, s_l),
                     (right_col, BLUE, right_label, s_r)]):
        a.plot([0, CROP], [0, CROP], color=MUTED, lw=1, ls="--", zorder=1)
        a.scatter(v.actual, v[col], s=11, alpha=0.35, color=colour,
                  linewidths=0, zorder=2)
        a.set_xlim(0, CROP)
        a.set_ylim(0, CROP)
        a.set_aspect("equal")
        a.set_xlabel("actual next-hour energy  [kWh]", color=MUTED)
        a.set_ylabel("predicted  [kWh]", color=MUTED)
        a.set_title(f"{label}\nMAE {s['mae_kwh']:.3f} kWh   r = {s['r']:.3f}",
                    color=INK, fontsize=11)
        a.text(0.5, -0.155, f"{n_off} hours above {CROP:.0f} kWh not shown; "
               "scores use all data", transform=a.transAxes, ha="center",
               color=MUTED, fontsize=8)
        _clean(a)

    a = ax[2]
    lo, hi = -3.0, 3.0
    bins = np.linspace(lo, hi, 49)
    spans = {}
    for col, colour, label in ((left_col, ORANGE, left_label),
                               (right_col, BLUE, right_label)):
        err = (v[col] - v.actual).to_numpy()
        spans[col] = (np.quantile(err, 0.1), np.quantile(err, 0.9))
        a.hist(np.clip(err, lo, hi), bins=bins, histtype="step",
               color=colour, linewidth=2, label=label)
    a.axvline(0, color=MUTED, lw=1, ls="--")

    top = a.get_ylim()[1]
    a.set_ylim(0, top * 1.22)
    for n, (col, colour) in enumerate(((left_col, ORANGE),
                                       (right_col, BLUE))):
        q10, q90 = spans[col]
        y = top * (1.14 - 0.07 * n)
        a.plot([q10, q90], [y, y], color=colour, lw=3, solid_capstyle="butt")
        a.text(q90 + 0.12, y, f"{q90 - q10:.2f} kWh", color=colour,
               va="center", fontsize=9)
    a.text(lo + 0.1, top * 1.19, "middle 80% of errors", color=MUTED,
           fontsize=9)
    n_out = int((np.abs(v[left_col] - v.actual) > hi).sum()
                + (np.abs(v[right_col] - v.actual) > hi).sum())
    a.set_xlabel("predicted minus actual  [kWh]", color=MUTED)
    a.text(0.5, -0.155, f"{n_out} of {2 * len(v)} points beyond ±3 kWh, "
           "piled at the edges", transform=a.transAxes, ha="center",
           color=MUTED, fontsize=8)
    a.set_ylabel("hours", color=MUTED)
    a.set_title("Error distribution\nnarrower is better", color=INK,
                fontsize=11)
    a.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left",
             bbox_to_anchor=(0.0, 0.86))
    _clean(a)

    fig.suptitle(title, color=INK, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"  -> {path}", flush=True)
    return s_l, s_r


def _clean(a):
    a.grid(True, color=GRID, lw=0.6, alpha=0.8)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)
    a.set_facecolor(SURFACE)


def main() -> None:
    v = out_of_fold()
    head = (f"Hour-ahead loop energy, {len(v)} out-of-fold hours "
            f"(mean {v.actual.mean():.2f} kWh)")
    print(f"  {len(v)} out-of-fold hours\n")
    for c in ("abg", "linear", "gbm", "blend"):
        s = score_energy(v[c], v.actual)
        print(f"  {c:<8s} MAE {s['mae_kwh']:.3f}  r {s['r']:.3f}")
    print()
    figure(v, "linear", "linear model", "blend", "50/50 blend",
           RESULTS / "14_blend_vs_linear.png", head)
    figure(v, "abg", "alpha/beta/gamma", "blend", "50/50 blend",
           RESULTS / "14_blend_vs_abg.png", head)
    v.to_csv(RESULTS / "14_out_of_fold.csv", index=False)


if __name__ == "__main__":
    main()
