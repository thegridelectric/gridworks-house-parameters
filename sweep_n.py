import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import CENTERED, FEATURES, linear_regression, prepare_hourly

HOUSE_ALIAS = "beech"
N_VALUES = list(range(5,26))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df_5min = pd.read_csv("data.csv", parse_dates=["timestamp"])
hp = pd.read_csv(glob.glob(f"data/{HOUSE_ALIAS}_*.csv")[0],
                 usecols=["hour_start", "hp_kwh_th", "ws_mph"],
                 parse_dates=["hour_start"])
hp = hp.drop_duplicates("hour_start").set_index("hour_start")
df = prepare_hourly(df_5min, hp_kwh_th=hp["hp_kwh_th"],
                    ws_mph=hp["ws_mph"])

refs = {f: float(df[f].mean()) for f in CENTERED}
days = np.sort(df["day"].unique())


def evaluate(N):
    """For a trailing N-day window ending on each day, predict the following
    day's dist_kwh (out of sample) and track the intercept. Returns
    (next-day dist_kwh MSE, average largest weekly intercept variation)."""
    fit_days = []
    intercepts = []
    sse = 0.0
    n_points = 0
    for i in range(N - 1, len(days) - 1):
        window = days[i - N + 1 : i + 1]
        window_df = df[df["day"].isin(window)]
        next_df = df[df["day"] == days[i + 1]]
        if window_df.empty or next_df.empty:
            continue
        dist_pred, coefficients, *_ = linear_regression(window_df, refs=refs)
        energy_ratio = float(window_df["hp_kwh_th"].sum()) / float(dist_pred.sum())
        intercepts.append(coefficients[0] * energy_ratio)
        fit_days.append(days[i])

        # Predict the following day's dist_kwh with the window's fit.
        X = np.column_stack([np.ones(len(next_df))]
                            + [next_df[f].to_numpy() - refs.get(f, 0.0)
                               for f in FEATURES])
        dist_hat = X @ coefficients
        sse += float(np.sum((dist_hat - next_df["dist_kwh"].to_numpy()) ** 2))
        n_points += len(next_df)

    mse = sse / n_points
    intercept = pd.Series(intercepts, index=pd.DatetimeIndex(fit_days)).sort_index()
    weekly_range = intercept.rolling("7D").apply(lambda s: s.max() - s.min())
    return mse, float(weekly_range.mean())


mses = []
intercept_vars = []
for N in N_VALUES:
    mse, intercept_var = evaluate(N)
    mses.append(mse)
    intercept_vars.append(intercept_var)
    print(f"N={N:2d}: next-day dist_kwh MSE={mse:.4f}, avg weekly intercept variation={intercept_var:.3f}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("N (trailing days in fit window)")
ax1.set_ylabel("Next-day dist_kwh MSE", color="tab:blue")
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
