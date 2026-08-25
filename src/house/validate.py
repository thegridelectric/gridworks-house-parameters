"""
Open-loop temperature validation, the common metric for the model ladder.

Filter through the holdout; every hour, freeze the state and simulate 12 steps
of 5 min with no measurement updates, using measured Q_dist and weather. Score
the predicted indoor temperature at the end of the hour.

The scoring target is T_i_avg = 0.5 (T_i1 + T_i2) for every rung, so one-zone
and two-zone models are directly comparable; per-zone errors are reported as
well where the model has zones. Persistence -- assuming the temperature does
not change over the hour -- is the baseline every rung must beat.

Note this is a unit test of the thermal physics, not the deliverable. Q_dist
is an input here, and the real target is the closed-loop Q_dist forecast with
the thermostat model supplying the heat.
"""

import numpy as np
import pandas as pd

from .fit import SIG_TI, build_inputs, measurements
from .kalman import build_bank, initial_state, wind_bins

HORIZON = 12                # 12 x 5 min = 1 h


def open_loop(model, p, df, segments, mask, nbins=12, burn=288,
              horizon=HORIZON):
    C = model.obs
    R = np.eye(C.shape[0]) * SIG_TI ** 2
    Y = measurements(model, df)
    bins_all, centres = wind_bins(df["v"].to_numpy(), nbins)
    Ad, Bd, K, Sinv, logdetS = build_bank(model, p, centres, R)
    u_all = build_inputs(model, df, p)

    t_i1 = df["T_i1"].to_numpy()
    t_i2 = df["T_i2"].to_numpy()
    i_air = model.states.index("T_i") if model.n_zones == 1 else None

    rows = []
    for a, b in segments:
        if not mask[a:b].any():
            continue
        y, u, bb = Y[a:b], u_all[a:b], bins_all[a:b]
        x = initial_state(model, y[0])
        n = b - a
        for t in range(n):
            xu = x + K[bb[t]] @ (y[t] - C @ x)

            if (t % horizon == 0) and (t + horizon < n) and t >= burn \
                    and mask[a + t]:
                xs = xu.copy()
                for k in range(horizon):
                    xs = Ad[bb[t + k]] @ xs + Bd[bb[t + k]] @ u[t + k]
                j = a + t + horizon
                if model.n_zones == 1:
                    p1 = p2 = pavg = xs[i_air]
                else:
                    p1, p2 = xs[1], xs[4]
                    pavg = 0.5 * (p1 + p2)
                rows.append({
                    "t": df["timestamp"].iloc[j],
                    "pred_avg": pavg, "pred_1": p1, "pred_2": p2,
                    "meas_avg": 0.5 * (t_i1[j] + t_i2[j]),
                    "meas_1": t_i1[j], "meas_2": t_i2[j],
                    "pers_avg": 0.5 * (t_i1[a + t] + t_i2[a + t]),
                    "pers_1": t_i1[a + t], "pers_2": t_i2[a + t],
                })

            x = Ad[bb[t]] @ xu + Bd[bb[t]] @ u[t]

    v = pd.DataFrame(rows)
    return v, score(v, model)


def score(v: pd.DataFrame, model=None) -> dict:
    if v.empty:
        return {}
    out = {"n_windows": int(len(v))}
    targets = ["avg"] if (model is not None and model.n_zones == 1) \
        else ["avg", "1", "2"]
    # one-zone models still get per-zone numbers, they just predict the same
    # value for both; reporting them keeps the ladder table uniform
    for z in ("avg", "1", "2"):
        err = v[f"pred_{z}"] - v[f"meas_{z}"]
        pers = v[f"pers_{z}"] - v[f"meas_{z}"]
        out[f"rmse_{z}"] = float(np.sqrt((err ** 2).mean()))
        out[f"mae_{z}"] = float(err.abs().mean())
        out[f"bias_{z}"] = float(err.mean())
        out[f"pers_{z}"] = float(np.sqrt((pers ** 2).mean()))
        out[f"skill_{z}"] = 1.0 - out[f"rmse_{z}"] / out[f"pers_{z}"]
    return out
