from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import (
    FEATURES,
    HouseEnergyParams,
    HouseEnergyParamsComputer,
    feature_centers,
    load_hourly_features,
    predict,
)
from plot_pred_from_params import plot_curves, plot_pred_vs_actual

HOUSE_ALIAS = "beech"
N = 20
SCALE_TO_HP_KWH = True   # False -> coefficients stay in distribution-kWh

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Read the hourly CSV and build the seven features. Hours whose 6-hour history is
# missing or non-contiguous are dropped there, so the count below is what is left.
df = load_hourly_features(HOUSE_ALIAS)
print(f"{len(df)} usable hours from {df['hour_start'].min()} to {df['hour_start'].max()}")

centers = feature_centers(df)
print("Centered at: " + ", ".join(f"{k}={v:.3g}" for k, v in centers.items()))
print("(the intercept is the predicted energy at those typical conditions)")

# Fit on a trailing N-day window ending on each day.
computer = HouseEnergyParamsComputer(centers=centers, scale_to_hp_kwh=SCALE_TO_HP_KWH)
days = np.sort(df["day"].unique())
fit_days = []
fit_oat_f = []
results: list[HouseEnergyParams] = []
oos_oat_f = []
oos_pred = []
oos_actual = []
for i in range(N - 1, len(days)):
    window = days[i-N+1 : i+1]
    window_df = df[df["day"].isin(window)]
    fit_days.append(days[i])
    # Mean outdoor temperature over the fit window: the weather the parameters
    # were learned in, and what the plots color by.
    fit_oat_f.append(float(window_df["oat_f"].mean()))
    params = computer.fit(window_df)
    results.append(params)

    # Score the day after the window: never in-sample, so the predicted-vs-actual
    # figure below shows what the fit is actually worth going forward.
    if i + 1 < len(days):
        next_df = df[df["day"] == days[i + 1]]
        oos_oat_f.extend(next_df["oat_f"])
        oos_pred.extend(predict(params, next_df, centers))
        oos_actual.extend(next_df["dist_kwh"])

print(
    f"Fitted {len(results)} trailing {N}-day windows "
    f"from {pd.Timestamp(fit_days[0]).date()} to {pd.Timestamp(fit_days[-1]).date()}"
)

oos_pred = np.array(oos_pred)
oos_actual = np.array(oos_actual)
errors = oos_pred - oos_actual
print(
    f"Next-day out-of-sample over {len(oos_actual)} hours: "
    f"MSE={float((errors**2).mean()):.3f}, RMSE={float(np.sqrt((errors**2).mean())):.3f}, "
    f"MAE={float(np.abs(errors).mean()):.3f} kWh"
)

# Plot all fit days, colored by date. The curves need an operating point: rooms at
# setpoint, no sun, envelope in equilibrium, and sustained operation, which is what
# the median prev stands for. See plot_curves' docstring.
plot_curves(
    results, fit_days, fit_oat_f, centers,
    t_i_avg=float((0.5 * (df["T_i1_start"] + df["T_i2_start"])).median()),
    prev_median=float(df["prev"].median()),
    title=f"{HOUSE_ALIAS.capitalize()}: house energy prediction over the year (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_yearly_N{N}.png",
)

plot_pred_vs_actual(
    oos_pred, oos_actual, oos_oat_f,
    f"{HOUSE_ALIAS.capitalize()}: next-day predicted vs actual (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_pred_vs_actual_N{N}.png",
)

# Find largest variation in each parameter across any 7-day span.
columns = ["intercept"] + FEATURES
params = pd.DataFrame(
    {name: [getattr(r, name) for r in results] for name in columns}
    | {f"std_error_{name}": [getattr(r, f"std_error_{name}") for r in results] for name in columns}
    | {
        "r_squared": [r.r_squared for r in results],
        "energy_ratio": [r.energy_ratio for r in results],
    },
    index=pd.DatetimeIndex(fit_days),
).sort_index()
params.to_csv(RESULTS_DIR / f"{HOUSE_ALIAS}_params_N{N}.csv")

weekly_range = params.rolling("7D").apply(lambda s: s.max() - s.min())

print("\nLargest variation within a single week:")
extreme_days = set()
for col in ["intercept", "dT", "gap1", "gap2"]:
    end_day = weekly_range[col].idxmax()
    week = params.loc[end_day - pd.Timedelta("6D"):end_day]
    low_day = week[col].idxmin()
    high_day = week[col].idxmax()
    extreme_days.update([low_day, high_day])
    print(f"  {col}: {weekly_range[col].max():.5g} (week ending {end_day.date()})")
    for day in (low_day, high_day):
        p = params.loc[day]
        print(
            f"    {day.date()}: intercept={p.intercept:.2f}, dT={p.dT:.4f}, "
            f"gap1={p.gap1:.3f}, gap2={p.gap2:.3f}"
        )

# Plot only the extreme-week days
result_by_day = {pd.Timestamp(d): r for d, r in zip(fit_days, results)}
oat_by_day = {pd.Timestamp(d): o for d, o in zip(fit_days, fit_oat_f)}
extreme_sorted = sorted(extreme_days)
plot_curves(
    [result_by_day[d] for d in extreme_sorted], extreme_sorted,
    [oat_by_day[d] for d in extreme_sorted], centers,
    t_i_avg=float((0.5 * (df["T_i1_start"] + df["T_i2_start"])).median()),
    prev_median=float(df["prev"].median()),
    title=f"{HOUSE_ALIAS.capitalize()}: extreme-week fit days (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_extremes_N{N}.png",
    use_legend=True,
)

plt.show()
