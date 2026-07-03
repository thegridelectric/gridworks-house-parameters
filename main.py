import glob
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import HouseParametersComputer, HouseParametersLinear, linear_regression
from plot_pred_from_params import plot_curves

HOUSE_ALIAS = "beech"
N = 20

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Read and prepare hourly CSV
csv_path = glob.glob(f"data/{HOUSE_ALIAS}_*.csv")[0]
df = pd.read_csv(csv_path)
df = df.dropna(subset=["hp_kwh_th", "dist_kwh", "oat_f", "ws_mph"])
df["hour_start"] = pd.to_datetime(df["hour_start"])
df = df.sort_values("hour_start").reset_index(drop=True)
df["day"] = df["hour_start"].dt.normalize()

# Center oat_f at its global mean so alpha is comparable across fits and stable.
oat_ref = float(df["oat_f"].mean())
print(f"oat_f centered at {oat_ref:.1f}°F (alpha = predicted energy at that temperature)")

# Fit on a trailing N-day window ending on each day.
computer = HouseParametersComputer(
    predictor=partial(
        linear_regression, oat_ref=oat_ref
    )
)
days = np.sort(df["day"].unique())
fit_days = []
results: list[HouseParametersLinear] = []
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
    results, fit_days, oat_ref,
    f"{HOUSE_ALIAS.capitalize()}: house energy prediction over the year (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_yearly_N{N}.png",
)

# Find largest variation in each parameter across any 7-day span.
params = pd.DataFrame(
    {
        "alpha": [r.alpha for r in results],
        "beta": [r.beta for r in results],
        "gamma": [r.gamma for r in results],
    },
    index=pd.DatetimeIndex(fit_days),
).sort_index()

weekly_range = params.rolling("7D").apply(lambda s: s.max() - s.min())

print("\nLargest variation within a single week:")
extreme_days = set()
for col in ["alpha", "beta", "gamma"]:
    end_day = weekly_range[col].idxmax()
    week = params.loc[end_day - pd.Timedelta("6D"):end_day]
    low_day = week[col].idxmin()
    high_day = week[col].idxmax()
    extreme_days.update([low_day, high_day])
    print(f"  {col}: {weekly_range[col].max():.5g} (week ending {end_day.date()})")
    for day in (low_day, high_day):
        p = params.loc[day]
        print(f"    {day.date()}: alpha={p.alpha:.1f}, beta={p.beta:.2f}, gamma={p.gamma:.5f}")

# Plot only the extreme-week days
result_by_day = {pd.Timestamp(d): r for d, r in zip(fit_days, results)}
extreme_sorted = sorted(extreme_days)
plot_curves(
    [result_by_day[d] for d in extreme_sorted], extreme_sorted, oat_ref,
    f"{HOUSE_ALIAS.capitalize()}: extreme-week fit days (trailing {N}-day fits)",
    savepath=RESULTS_DIR / f"{HOUSE_ALIAS}_extremes_N{N}.png",
    use_legend=True,
)

plt.show()
