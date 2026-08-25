"""Distribution of wind over the season, from both available sources.

data.csv carries a wind column `v`; the hourly export carries `ws_mph`. They
are not the same quantity -- correlation 0.68 and a median ratio near 16, which
is no unit conversion -- so the range each covers matters when choosing which
wind speeds a prediction curve may honestly be drawn at.
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOUSE_ALIAS = "beech"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

df = pd.read_csv("data.csv", usecols=["timestamp", "v"], parse_dates=["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M").astype(str)

hourly = pd.read_csv(glob.glob(f"data/{HOUSE_ALIAS}_*.csv")[0],
                     usecols=["hour_start", "ws_mph"], parse_dates=["hour_start"])
hourly = hourly.dropna(subset=["ws_mph"])
hourly = hourly[hourly["hour_start"].between(df["timestamp"].min(),
                                             df["timestamp"].max())]
hourly["month"] = hourly["hour_start"].dt.to_period("M").astype(str)

months = sorted(set(df["month"]) | set(hourly["month"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#fcfcfb")

for ax, (frame, col, colour, label, unit) in zip(axes, [
        (df, "v", BLUE, "data.csv  `v`", "v (as recorded)"),
        (hourly, "ws_mph", ORANGE, "hourly export  `ws_mph`", "wind speed (mph)")]):
    data = [frame.loc[frame["month"] == m, col].to_numpy() for m in months]
    data = [d for d in data if len(d)]
    bp = ax.boxplot(data, tick_labels=[m for m, d in zip(months, data) if len(d)],
                    whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=2),
                    boxprops=dict(facecolor=colour, alpha=0.45,
                                  edgecolor=colour, linewidth=1.5),
                    whiskerprops=dict(color=colour, linewidth=1.5),
                    capprops=dict(color=colour, linewidth=1.5))
    allv = np.concatenate(data)
    ax.axhline(allv.max(), color=MUTED, lw=1, ls="--")
    ax.text(0.02, allv.max(), f" season maximum {allv.max():.2f}",
            va="bottom", color=MUTED, fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_ylabel(unit, color=MUTED)
    ax.set_title(f"{label}\nmedian {allv.mean():.2f}, p95 {np.quantile(allv, .95):.2f}, "
                 f"max {allv.max():.2f}", color=INK, fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor("#fcfcfb")

# mark where the prediction curves are being drawn
axes[1].axhline(10, color=INK, lw=1.2, ls=":")
axes[1].axhline(20, color=INK, lw=1.2, ls=":")
axes[1].text(0.02, 20, " curve drawn here (20 mph)", va="bottom", color=INK,
             fontsize=9, transform=axes[1].get_yaxis_transform())

fig.suptitle(f"{HOUSE_ALIAS.capitalize()}: wind by month, boxes 25-75% with 5-95% whiskers",
             color=INK, fontsize=12)
fig.tight_layout()
path = RESULTS_DIR / f"{HOUSE_ALIAS}_wind_by_month.png"
fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"-> {path}")
