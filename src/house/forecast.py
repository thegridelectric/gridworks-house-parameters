"""
The deliverable: hour-ahead forecast of loop energy (Q_dist).

This is the closed-loop test, and it differs from the temperature validation in
one decisive way. There, Q_dist was an *input* taken from measurement. Here the
thermostat model generates it, so nothing external corrects the trajectory: an
error in temperature changes when the thermostat calls, which changes the heat,
which changes the temperature again. Process noise cannot rescue it either,
because there are no measurement updates inside the hour.

At each hour boundary:
  1. take the filtered state from the Kalman filter running on live 5-min data
  2. take the current call state (known live -- the loop is either flowing or not)
  3. simulate 12 steps of 5 min with forecast weather, the thermostat deciding
     the heat at every step
  4. sum the heat delivered -> predicted next-hour loop energy

Weather is taken from the record, i.e. a perfect forecast. That isolates the
model from weather-forecast error, which is a separate problem and not one this
repo can fix.
"""

import numpy as np
import pandas as pd

from .fit import SIG_TI, build_inputs, measurements
from .kalman import DT, build_bank, initial_state, wind_bins
from .model import GHI as GHI_COL
from .model import N_INPUTS, ONE, Q1, Q2, TO, gain_shape
from .thermostat import Controller, call_status

HORIZON = 12
J_TO_KWH = DT / 3.6e6

# Independent steady-state baseline, from the daily energy-balance diagnostic.
# Deliberately not taken from the fitted model: the point is to ask whether six
# states and a thermostat beat one line drawn through the daily data.
UA_SIMPLE = 201.0            # W/K
G_SIMPLE = 1162.0            # W


def forecast(model, p, cal, df, segments, mask, horizon=HORIZON, burn=288,
             nbins=6, power_cap=np.inf, supply=None):
    C = model.obs
    R = np.eye(C.shape[0]) * SIG_TI ** 2
    Y = measurements(model, df)
    bins_all, centres = wind_bins(df["v"].to_numpy(), nbins)
    Ad, Bd, K, _, _ = build_bank(model, p, centres, R)
    u_meas = build_inputs(model, df, p)

    z1, z2 = call_status(df)
    q_tot = df["Q_total"].to_numpy()
    t_o = df["T_o"].to_numpy()
    ghi = df["GHI"].to_numpy()
    sp1 = df["T_i1_set"].to_numpy()
    sp2 = df["T_i2_set"].to_numpy()
    shape = (gain_shape(p, df["hour"].to_numpy()) if "gp_a1" in p
             else np.ones(len(df)))
    t_i1 = df["T_i1"].to_numpy()
    t_i2 = df["T_i2"].to_numpy()

    ctrl = Controller(cal, power_cap=power_cap, supply=supply)
    rows = []

    for a, b in segments:
        if not mask[a:b].any():
            continue
        y, bb = Y[a:b], bins_all[a:b]
        x = initial_state(model, y[0])
        n = b - a

        for t in range(n):
            xu = x + K[bb[t]] @ (y[t] - C @ x)

            if (t % horizon == 0) and (t + horizon < n) and t >= burn \
                    and mask[a + t]:
                g = a + t
                xs = xu.copy()
                ctrl.reset({1: bool(z1[g]), 2: bool(z2[g])})

                energy = 0.0
                on1 = on2 = 0          # steps each zone spends calling
                for k in range(horizon):
                    j = g + k
                    heat = ctrl.step({1: xs[1], 2: xs[4]},
                                     {1: sp1[j], 2: sp2[j]},
                                     {1: xs[0], 2: xs[3]})
                    u = np.zeros(N_INPUTS)
                    u[Q1], u[Q2] = heat[1], heat[2]
                    u[TO], u[GHI_COL], u[ONE] = t_o[j], ghi[j], shape[j]
                    energy += (heat[1] + heat[2]) * J_TO_KWH
                    on1 += heat[1] > 0.0
                    on2 += heat[2] > 0.0
                    xs = Ad[bb[t + k]] @ xs + Bd[bb[t + k]] @ u

                actual = q_tot[g:g + horizon].sum() * J_TO_KWH
                prev = (q_tot[g - horizon:g].sum() * J_TO_KWH
                        if t >= horizon else np.nan)

                # steady-state baseline over the same window
                dT = 0.5 * (t_i1[g] + t_i2[g]) - t_o[g:g + horizon].mean()
                simple = max(0.0, UA_SIMPLE * dT - G_SIMPLE) * horizon * J_TO_KWH

                rows.append({
                    "t": df["timestamp"].iloc[g],
                    "pred": energy, "actual": actual,
                    "persistence": prev, "steady_state": simple,
                    "T_o": t_o[g:g + horizon].mean(),
                    "call1": bool(z1[g]), "call2": bool(z2[g]),
                    # simulated duty, for comparison with what really happened
                    "duty1_sim": on1 / horizon, "duty2_sim": on2 / horizon,
                    "duty1_act": float(z1[g:g + horizon].mean()),
                    "duty2_act": float(z2[g:g + horizon].mean()),
                })

            x = Ad[bb[t]] @ xu + Bd[bb[t]] @ u_meas[a + t]

    return pd.DataFrame(rows)


def score(v: pd.DataFrame, col="pred", ref="actual") -> dict:
    d = v.dropna(subset=[col, ref])
    if d.empty:
        return {}
    err = d[col] - d[ref]
    denom = d[ref].replace(0, np.nan)
    return {
        "n": int(len(d)),
        "mae_kwh": float(err.abs().mean()),
        "rmse_kwh": float(np.sqrt((err ** 2).mean())),
        "bias_kwh": float(err.mean()),
        "mape_pct": float((err.abs() / denom).dropna().mean() * 100),
        "r": float(np.corrcoef(d[col], d[ref])[0, 1]),
        "mean_actual": float(d[ref].mean()),
        "total_pred": float(d[col].sum()),
        "total_actual": float(d[ref].sum()),
    }
