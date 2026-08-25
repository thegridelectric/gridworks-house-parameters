"""
Is the negative intercept from the energy balance a real internal heat gain?

Estimates the implied free gain day by day, G_day = UA * dT_day - q_day, and
checks it for the structure a genuine gain would have. Run before adding a
gains term to the RC model, not after.

    uv run scripts/01b_internal_gains.py
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
from house.diagnostics import (daily_frame, energy_balance, gains_report,
                               implied_gains, ols)

RESULTS = Path(__file__).resolve().parents[1] / "results"
MIN_SAMPLES_PER_DAY = 276
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = load_data(verbose=False)

    d_all = daily_frame(data.df)
    d = d_all[(d_all.n >= MIN_SAMPLES_PER_DAY) & (d_all.n_missing == 0)] \
        .dropna(subset=["q", "dT", "GHI"])

    fits = energy_balance(d)
    ua = float(fits["simple"].coef[1])
    ua_se = float(fits["simple"].se[1])
    print(f"  UA = {ua:.1f} +/- {ua_se:.1f} W/K over {len(d)} days")

    g = implied_gains(d, ua)

    rule("IMPLIED DAILY GAIN  G = UA * dT - q")
    for line in gains_report(g, ua, ua_se):
        print("  " + line)

    rule("BY DAY OF WEEK")
    print(f"  {'day':<5s} {'n':>3s} {'mean G [W]':>11s} {'sd':>7s}")
    for i, name in enumerate(DAYS):
        sub = g[g.weekday == i].G
        if len(sub):
            print(f"  {name:<5s} {len(sub):3d} {sub.mean():11.0f} "
                  f"{sub.std():7.0f}")

    rule("BY MONTH")
    print(f"  {'month':<10s} {'n':>3s} {'mean G [W]':>11s} {'sd':>7s}")
    for m, sub in g.groupby(g.index.to_period("M")):
        print(f"  {str(m):<10s} {len(sub):3d} {sub.G.mean():11.0f} "
              f"{sub.G.std():7.0f}")

    rule("DECOMPOSE G INTO CONSTANT + SOLAR")
    f = ols(g[["GHI"]].to_numpy(), g.G.to_numpy(), ["solar [W per W/m2]"])
    print(f)
    solar_at_mean = f.coef[1] * g.GHI.mean()
    print(f"\n  constant part      : {f.coef[0]:.0f} W")
    print(f"  solar at mean GHI  : {solar_at_mean:.0f} W "
          f"(GHI mean {g.GHI.mean():.0f} W/m2)")
    if f.coef[1] > 0:
        print(f"  implied aperture   : {f.coef[1]:.1f} m2 "
              "(at unit transmittance)")
    else:
        print("  ! solar coefficient is negative, i.e. sunnier days imply "
              "LESS gain. Not physical -> solar is not separable at daily "
              "resolution here.")

    _plot(g, ua, RESULTS / "01b_internal_gains.png")


def _plot(g: pd.DataFrame, ua: float, path: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.axhline(0, color="k", lw=0.8)
    a.axhline(g.G.mean(), color="C3", lw=1.4, ls="--",
              label=f"mean {g.G.mean():.0f} W")
    a.scatter(g.index[~g.is_weekend], g.G[~g.is_weekend], s=22, label="weekday")
    a.scatter(g.index[g.is_weekend], g.G[g.is_weekend], s=30, marker="s",
              label="weekend")
    a.set_ylabel("implied gain G  [W]")
    a.set_title(f"Implied free gain per day (UA fixed at {ua:.0f} W/K)")
    a.tick_params(axis="x", rotation=30)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    a.hist(g.G, bins=30)
    a.axvline(g.G.mean(), color="C3", lw=1.4, ls="--")
    a.set_xlabel("implied gain G  [W]")
    a.set_title("Distribution (tight => constant gain is credible)")
    a.grid(alpha=0.3)

    # the decisive panel: G must NOT trend with dT
    a = ax[1, 0]
    a.scatter(g.dT, g.G, s=22)
    m, b = np.polyfit(g.dT, g.G, 1)
    xs = np.linspace(g.dT.min(), g.dT.max(), 50)
    r = float(np.corrcoef(g.G, g.dT)[0, 1])
    a.plot(xs, m * xs + b, "C3-", lw=1.6, label=f"slope {m:.0f} W/K, r={r:+.2f}")
    a.axhline(g.G.mean(), color="k", lw=0.8, ls=":")
    a.set_xlabel("daily mean dT  [K]")
    a.set_ylabel("implied gain G  [W]")
    a.set_title("G vs dT (slope => UA misspecified, not a gain)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    a.scatter(g.GHI, g.G, s=22)
    m2, b2 = np.polyfit(g.GHI, g.G, 1)
    xs = np.linspace(g.GHI.min(), g.GHI.max(), 50)
    r2 = float(np.corrcoef(g.G, g.GHI)[0, 1])
    a.plot(xs, m2 * xs + b2, "C3-", lw=1.6,
           label=f"slope {m2:.1f} m2, r={r2:+.2f}")
    a.set_xlabel("daily mean GHI  [W/m2]")
    a.set_ylabel("implied gain G  [W]")
    a.set_title("G vs solar")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")


if __name__ == "__main__":
    main()
