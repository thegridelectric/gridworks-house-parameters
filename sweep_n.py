import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import linear_regression

HOUSE_ALIAS = "beech"
N_VALUES = list(range(5,26))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

csv_path = glob.glob(f"data/{HOUSE_ALIAS}_*.csv")[0]
df = pd.read_csv(csv_path)
df = df.dropna(subset=["hp_kwh_th", "dist_kwh", "oat_f", "ws_mph"])
df["hour_start"] = pd.to_datetime(df["hour_start"])
df = df.sort_values("hour_start").reset_index(drop=True)
df["day"] = df["hour_start"].dt.normalize()

oat_ref = float(df["oat_f"].mean())
days = np.sort(df["day"].unique())


def evaluate(N):
    """For a trailing N-day window ending on each day, predict the following
    day's dist_kwh (out of sample) and track alpha. Returns
    (next-day dist_kwh MSE, average largest weekly alpha variation)."""
    fit_days = []
    alphas = []
    sse = 0.0
    n_points = 0
    for i in range(N - 1, len(days) - 1):
        window = days[i - N + 1 : i + 1]
        window_df = df[df["day"].isin(window)]
        dist_pred, a, b, g, *_ = linear_regression(window_df, oat_ref=oat_ref)
        energy_ratio = float(window_df["hp_kwh_th"].sum()) / float(dist_pred.sum())
        alphas.append(a * energy_ratio)
        fit_days.append(days[i])

        # Predict the following day's dist_kwh with the window's fit.
        next_df = df[df["day"] == days[i + 1]]
        oat = next_df["oat_f"].to_numpy()
        ws = next_df["ws_mph"].to_numpy()
        dist_hat = a + b * (oat - oat_ref) + g * (65 - oat) * ws
        sse += float(np.sum((dist_hat - next_df["dist_kwh"].to_numpy()) ** 2))
        n_points += len(next_df)

    mse = sse / n_points
    alpha = pd.Series(alphas, index=pd.DatetimeIndex(fit_days)).sort_index()
    weekly_alpha_range = alpha.rolling("7D").apply(lambda s: s.max() - s.min())
    return mse, float(weekly_alpha_range.mean())


mses = []
alpha_vars = []
for N in N_VALUES:
    mse, alpha_var = evaluate(N)
    mses.append(mse)
    alpha_vars.append(alpha_var)
    print(f"N={N:2d}: next-day dist_kwh MSE={mse:.4f}, avg weekly alpha variation={alpha_var:.3f}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.set_xlabel("N (trailing days in fit window)")
ax1.set_ylabel("Next-day dist_kwh MSE", color="tab:blue")
ax1.plot(N_VALUES, mses, "o-", color="tab:blue", label="MSE")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(N_VALUES)

ax2 = ax1.twinx()
ax2.set_ylabel("Avg largest weekly alpha variation", color="tab:red")
ax2.plot(N_VALUES, alpha_vars, "s--", color="tab:red", label="alpha variation")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle(f"{HOUSE_ALIAS.capitalize()}: effect of lookback window N")
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"{HOUSE_ALIAS}_sweep_N.png", dpi=150, bbox_inches="tight")
plt.show()
