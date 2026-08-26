from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from house_parameters import design_matrix, feature_centers, linear_regression, load_hourly_features

HOUSE_ALIAS = "beech"
N_VALUES = list(range(5,26))
SCALE_TO_HP_KWH = True   # False -> the tracked intercept stays in distribution-kWh

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = load_hourly_features(HOUSE_ALIAS)
centers = feature_centers(df)
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
        dist_pred, coefficients, *_ = linear_regression(window_df, centers)
        energy_ratio = 1.0
        if SCALE_TO_HP_KWH:
            energy_ratio = float(window_df["hp_kwh_th"].sum()) / float(dist_pred.sum())
        intercepts.append(coefficients[0] * energy_ratio)
        fit_days.append(days[i])

        # Predict the following day's dist_kwh with the window's fit. Coefficients
        # stay unscaled here: dist_kwh is the target, not heat-pump thermal energy.
        # Clipped at zero, which negative predictions otherwise cost ~11% of MSE.
        next_df = df[df["day"] == days[i + 1]]
        dist_hat = np.maximum(design_matrix(next_df, centers) @ coefficients, 0.0)
        sse += float(np.sum((dist_hat - next_df["dist_kwh"].to_numpy()) ** 2))
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
