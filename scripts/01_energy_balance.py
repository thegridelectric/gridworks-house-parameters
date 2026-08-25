"""
FIRST TASK: daily energy-balance diagnostic.

Model-free estimate of whole-house conductance, run before any further
bound-tuning or model-structure changes.

    uv run scripts/01_energy_balance.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import DT, load_data
from house.diagnostics import daily_frame, energy_balance, interpret, ols

RESULTS = Path(__file__).resolve().parents[1] / "results"
MIN_SAMPLES_PER_DAY = 276          # 288 = full day; allow a few missing rows


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(fits: dict) -> None:
    for key in ("simple", "solar", "per_zone", "wind", "quadratic",
                "through_origin"):
        print(f"\n  [{key}]")
        print(fits[key])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    rule("DATA")
    data = load_data()
    df = data.df

    rule("DAILY AGGREGATION")
    d_all = daily_frame(df)
    full = d_all[d_all.n >= MIN_SAMPLES_PER_DAY].dropna(subset=["q", "dT", "GHI"])
    d = full[full.n_missing == 0]
    print(f"  {len(d_all)} calendar days -> {len(full)} complete days "
          f"(>= {MIN_SAMPLES_PER_DAY} of 288 samples)")
    print(f"  {len(full)} complete days -> {len(d)} with complete weather "
          f"({len(full) - len(d)} days dropped for missing T_o/GHI)")
    print(f"  dT range : {d.dT.min():.1f} .. {d.dT.max():.1f} K "
          f"(mean {d.dT.mean():.1f})")
    print(f"  q  range : {d.q.min():.0f} .. {d.q.max():.0f} W "
          f"(mean {d.q.mean():.0f})")

    # Does the storage term really cancel over a day?
    s1 = d.dT_i1_storage.abs().mean()
    s2 = d.dT_i2_storage.abs().mean()
    print(f"  day-over-day |dT_i| : zone 1 {s1:.2f} K, zone 2 {s2:.2f} K "
          "(small => storage cancels, static balance is valid)")

    mean_power = float(d.q.mean())

    rule("REGRESSION: q = a + b * (T_i_avg - T_o)   [complete-weather days]")
    fits = energy_balance(d)
    show(fits)

    rule("CONTRAST: same regression including days with weather gaps")
    contaminated = energy_balance(full)["simple"]
    print(contaminated)
    print(f"\n  slope {fits['simple'].coef[1]:.0f} -> {contaminated.coef[1]:.0f} W/K, "
          f"R2 {fits['simple'].r2:.3f} -> {contaminated.r2:.3f}")

    rule("INTERPRETATION")
    for line in interpret(fits, mean_power, float(d.dT.mean())):
        print("  " + line)

    # ---- regime check: the Nov vs Jan setpoint regimes ----
    rule("REGIME SPLIT (setpoints changed over the winter)")
    if data.has_setpoints:
        print(f"  zone 1 setpoint: {d.T_i1_set.min():.2f} .. "
              f"{d.T_i1_set.max():.2f} degC")
        print(f"  zone 2 setpoint: {d.T_i2_set.min():.2f} .. "
              f"{d.T_i2_set.max():.2f} degC")
        # zone 2 is the one that moved: low-setpoint vs high-setpoint regime
        cut = 0.5 * (d.T_i2_set.min() + d.T_i2_set.max())
        regimes = {
            f"zone2 setpoint <= {cut:.1f}": d[d.T_i2_set <= cut],
            f"zone2 setpoint >  {cut:.1f}": d[d.T_i2_set > cut],
        }
    else:
        regimes = {
            "Nov-Dec": d[d.index < pd.Timestamp("2026-01-01")],
            "Jan-Feb": d[d.index >= pd.Timestamp("2026-01-01")],
        }

    for label, sub in regimes.items():
        if len(sub) < 10:
            print(f"\n  [{label}] only {len(sub)} days, skipped")
            continue
        f = ols(sub[["dT"]].to_numpy(), sub.q.to_numpy(), ["UA [W/K]"])
        print(f"\n  [{label}]  {len(sub)} days")
        print(f)

    # ---- monthly slope, to see drift ----
    rule("MONTHLY UA (is a single constant UA defensible?)")
    print(f"  {'month':<10s} {'days':>5s} {'UA [W/K]':>12s} "
          f"{'intercept [W]':>14s} {'R2':>7s}   set")
    for label, frame in (("complete weather", d), ("all days", full)):
        print(f"\n  -- {label} --")
        for month, sub in frame.groupby(frame.index.to_period("M")):
            if len(sub) < 8:
                continue
            f = ols(sub[["dT"]].to_numpy(), sub.q.to_numpy(), ["UA"])
            print(f"  {str(month):<10s} {len(sub):5d} {f.coef[1]:12.1f} "
                  f"{f.coef[0]:14.0f} {f.r2:7.3f}")

    # ---- energy totals, restated ----
    rule("ENERGY ACCOUNTING")
    span_days = (df.timestamp.iloc[-1] - df.timestamp.iloc[0]).total_seconds() / 86400
    e_tot = float(df.Q_total.sum()) * DT / 3.6e6
    print(f"  record span        : {span_days:.1f} days")
    print(f"  total delivered    : {e_tot:.0f} kWh")
    print(f"  mean input power   : {e_tot * 1000 / (span_days * 24):.0f} W")
    ua = fits["simple"].coef[1]
    print(f"  mean dT            : {d.dT.mean():.1f} K")
    print(f"  UA * mean dT       : {ua * d.dT.mean():.0f} W  "
          "(what the fitted slope dissipates)")
    print(f"  unexplained        : {mean_power - ua * d.dT.mean():.0f} W  "
          "(= the intercept, by construction)")

    _plot(d, fits, RESULTS / "01_energy_balance.png")
    d.to_csv(RESULTS / "01_daily.csv")
    print(f"\n  daily frame -> {RESULTS / '01_daily.csv'}")


def _plot(d: pd.DataFrame, fits: dict, path: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) the headline scatter, coloured by month
    a = ax[0, 0]
    months = d.index.to_period("M")
    for m in months.unique():
        sub = d[months == m]
        a.scatter(sub.dT, sub.q, s=22, alpha=0.8, label=str(m))
    xs = np.linspace(d.dT.min(), d.dT.max(), 100)
    c = fits["simple"].coef
    a.plot(xs, c[0] + c[1] * xs, "k-", lw=2,
           label=f"UA={c[1]:.0f} W/K, a={c[0]:+.0f} W")
    q = fits["quadratic"].coef
    a.plot(xs, q[0] + q[1] * xs + q[2] * xs ** 2, "r--", lw=1.4,
           label="quadratic")
    a.set_xlabel("daily mean (T_i_avg - T_o)  [K]")
    a.set_ylabel("daily mean Q_dist  [W]")
    a.set_title("Daily energy balance")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (0,1) coloured by solar, to expose the solar confound
    a = ax[0, 1]
    sc = a.scatter(d.dT, d.q, c=d.GHI, s=26, cmap="viridis")
    fig.colorbar(sc, ax=a, label="daily mean GHI [W/m2]")
    a.plot(xs, c[0] + c[1] * xs, "k-", lw=2)
    a.set_xlabel("daily mean (T_i_avg - T_o)  [K]")
    a.set_ylabel("daily mean Q_dist  [W]")
    a.set_title("Same scatter, coloured by solar")
    a.grid(alpha=0.3)

    # (1,0) residuals over time: a trend means UA is not constant
    a = ax[1, 0]
    resid = d.q - (c[0] + c[1] * d.dT)
    a.axhline(0, color="k", lw=0.8)
    a.scatter(d.index, resid, s=20)
    a.set_ylabel("residual  [W]")
    a.set_title("Residual vs date (trend => UA not constant)")
    a.tick_params(axis="x", rotation=30)
    a.grid(alpha=0.3)

    # (1,1) residual vs dT: curvature means the straight line is wrong
    a = ax[1, 1]
    a.axhline(0, color="k", lw=0.8)
    a.scatter(d.dT, resid, s=20)
    a.set_xlabel("daily mean (T_i_avg - T_o)  [K]")
    a.set_ylabel("residual  [W]")
    a.set_title("Residual vs dT (curvature => regime dependence)")
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  plot        -> {path}")


if __name__ == "__main__":
    main()
