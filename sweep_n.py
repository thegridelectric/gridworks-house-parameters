from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import HouseEnergyParamsComputer

HOUSE_ALIAS = "beech"
N_VALUES = list(range(5, 26))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

computer = HouseEnergyParamsComputer(HOUSE_ALIAS)
days = np.sort(computer.df["day"].unique())


def evaluate(N):
    """For a trailing N-day window ending on each day, predict the following
    day's scaled energy (out of sample) and track the intercept. Returns
    (next-day scaled MSE, average largest weekly intercept variation)."""
    fit_days = []
    intercepts = []
    sse = 0.0
    n_points = 0
    for i in range(N - 1, len(days) - 1):
        window = days[i - N + 1 : i + 1]
        window_df = computer.df[computer.df["day"].isin(window)]
        params = computer.fit(window_df)
        intercepts.append(params.B0)
        fit_days.append(days[i])

        next_df = computer.df[computer.df["day"] == days[i + 1]]
        scaled_hat = computer.predict(params, next_df)
        actual_scaled = next_df["dist_kwh"].to_numpy() * params.energy_ratio
        sse += float(np.sum((scaled_hat - actual_scaled) ** 2))
        n_points += len(next_df)

    mse = sse / n_points
    intercept = pd.Series(intercepts, index=pd.DatetimeIndex(fit_days)).sort_index()
    weekly_intercept_range = intercept.rolling("7D").apply(lambda s: s.max() - s.min())
    return mse, float(weekly_intercept_range.mean())


mses = []
intercept_vars = []
for N in N_VALUES:
    mse, intercept_var = evaluate(N)
    mses.append(mse)
    intercept_vars.append(intercept_var)
    print(f"N={N:2d}: next-day scaled MSE={mse:.4f}, avg weekly intercept variation={intercept_var:.3f}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("N (trailing days in fit window)")
ax1.set_ylabel("Next-day scaled energy MSE", color="tab:blue")
ax1.plot(N_VALUES, mses, "o-", color="tab:blue", label="MSE")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(N_VALUES)

ax2 = ax1.twinx()
ax2.set_ylabel("Avg largest weekly intercept variation", color="tab:red")
ax2.plot(N_VALUES, intercept_vars, "s--", color="tab:red", label="intercept variation")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle(f"{HOUSE_ALIAS.capitalize()}: effect of lookback window N")
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"{HOUSE_ALIAS}_sweep_N.png", dpi=150, bbox_inches="tight")
plt.show()
