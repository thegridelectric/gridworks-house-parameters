import glob
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import (
    CENTERED, FEATURES, HouseEnergyParams, HouseEnergyParamsComputer,
    linear_regression, prepare_hourly,
)
from plot_pred_from_params import plot_curves

HOUSE_ALIAS = "beech"
N = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Read the 5-minute record. Unlike the hourly export it carries room
# temperatures and thermostat setpoints, which is what this model needs: how
# far each room sits below its switching threshold is the single strongest
# predictor of the next hour's heat.
df_5min = pd.read_csv("data.csv", parse_dates=["timestamp"])

# hp_kwh_th only exists in the hourly export; it is used to express the fitted
# parameters as heat-pump thermal energy rather than distribution energy.
hp = pd.read_csv(glob.glob(f"data/{HOUSE_ALIAS}_*.csv")[0],
                 usecols=["hour_start", "hp_kwh_th", "ws_mph"],
                 parse_dates=["hour_start"])
hp = hp.drop_duplicates("hour_start").set_index("hour_start")

df = prepare_hourly(df_5min, hp_kwh_th=hp["hp_kwh_th"],
                    ws_mph=hp["ws_mph"])
print(f"{len(df)} usable hourly windows "
      f"from {df.hour_start.min().date()} to {df.hour_start.max().date()}")

# Center the continuous features at their global means so the intercept is the
# predicted energy at typical conditions (inside the data cloud) instead of at
# all-zero, which decorrelates it from the slopes and stabilises it across fits.
refs = {f: float(df[f].mean()) for f in CENTERED}
print("centered at: " + ", ".join(f"{k}={v:.2f}" for k, v in refs.items()))

# Fit on a trailing N-day window ending on each day.
computer = HouseEnergyParamsComputer(
    predictor=partial(
        linear_regression, refs=refs
    )
)
days = np.sort(df["day"].unique())
fit_days = []
results: list[HouseEnergyParams] = []
for i in range(N - 1, len(days)):
    window = days[i-N+1 : i+1]
    window_df = df[df["day"].isin(window)]
    fit_days.append(days[i])
    results.append(computer.fit(window_df))

print(
    f"Fitted {len(results)} trailing {N}-day windows "
    f"from {pd.Timestamp(fit_days[0]).date()} to {pd.Timestamp(fit_days[-1]).date()}"
)

# Plot all fit days, colored by date
plot_curves(
    results, fit_days, refs, df,
    f"{HOUSE_ALIAS.capitalize()}: house energy prediction over the year (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_yearly_N{N}.png",
)

# Find largest variation in each parameter across any 7-day span.
params = pd.DataFrame(
    {
        **{name: [r.coefficients[name] for r in results]
           for name in ["intercept", *FEATURES]},
        **{f"std_error_{name}": [r.std_errors[name] for r in results]
           for name in ["intercept", *FEATURES]},
        "r_squared": [r.r_squared for r in results],
    },
    index=pd.DatetimeIndex(fit_days),
).sort_index()
params.to_csv(RESULTS_DIR / f"{HOUSE_ALIAS}_params_N{N}.csv")

weekly_range = params.rolling("7D").apply(lambda s: s.max() - s.min())

print("\nLargest variation within a single week:")
extreme_days = set()
for col in ["intercept", "gap1", "dT"]:
    end_day = weekly_range[col].idxmax()
    week = params.loc[end_day - pd.Timedelta("6D"):end_day]
    low_day = week[col].idxmin()
    high_day = week[col].idxmax()
    extreme_days.update([low_day, high_day])
    print(f"  {col}: {weekly_range[col].max():.5g} (week ending {end_day.date()})")
    for day in (low_day, high_day):
        p = params.loc[day]
        print(f"    {day.date()}: " + ", ".join(
            f"{c}={p[c]:.4g}" for c in ["intercept", "dT", "gap1", "gap2"]))

print(f"\nMean R^2 across windows: {params.r_squared.mean():.3f}")

# Plot only the extreme-week days
result_by_day = {pd.Timestamp(d): r for d, r in zip(fit_days, results)}
extreme_sorted = sorted(extreme_days)
plot_curves(
    [result_by_day[d] for d in extreme_sorted], extreme_sorted, refs, df,
    f"{HOUSE_ALIAS.capitalize()}: extreme-week fit days (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_extremes_N{N}.png",
    use_legend=True,
)

plt.show()
