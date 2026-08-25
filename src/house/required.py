"""
"Required energy" use of the RC model, without simulating the switch.

Simulating the thermostat is fragile here: the deadband is 0.045 K while the
temperature model is good to about 0.2 K, so individual switching instants
cannot be predicted, and an error that pins the simulated thermostat on or off
costs a whole hour of energy.

But the deliverable does not need the switching pattern. Over an hour the
thermostat holds the room near its threshold, so the heat drawn is whatever it
takes to hold that temperature, given where the envelope and the emitters
currently sit. That can be computed directly from the state equations with
T_i clamped, no on/off decisions at all:

    hold T_i fixed  ->  the air-node equation gives the heat the room needs
                    ->  the emitter equation gives the heat the loop must supply

Two corrections on top of the steady requirement:

  * an offset term, because the room does not start exactly at its threshold.
    Moving it there costs (or saves) C_i * gap.
  * an emitter term, because a cold loop must be reheated before it delivers
    anything, which is the startup transient the emitter node exists for.

Clamped at zero: the loop can add heat, never remove it.
"""

import numpy as np
import pandas as pd

from .fit import SIG_TI, build_inputs, measurements
from .kalman import DT, build_bank, initial_state, wind_bins
from .model import F1_SOLAR, _fabric
from .thermostat import call_status

HORIZON = 12
JOULE_TO_KWH = 1.0 / 3.6e6      # `need` accumulates joules, not watts


def _zone_terms(p, z, v):
    """Conductances and capacities for one zone at a given wind speed."""
    kei = p.get(f"K_ei{z}", p.get("K_ei"))
    taue = p.get(f"tau_e{z}", p.get("tau_e"))
    kim, kmo, cm = _fabric(p[f"UA_fab{z}"], p["f_im"], p[f"tau_m{z}"])
    return {
        "k_ei": kei, "c_e": taue * kei,
        "k_im": kim, "k_mo": kmo, "c_m": cm,
        "c_i": p[f"C_i{z}"],
        "ginf": p[f"g0_{z}"] + p["g1"] * v,
        "gain": p[f"G{z}"],
        "f_sol": F1_SOLAR if z == 1 else 1.0 - F1_SOLAR,
    }


def required_energy(model, p, cal, df, segments, mask, horizon=HORIZON,
                    burn=288, nbins=6):
    C = model.obs
    R = np.eye(C.shape[0]) * SIG_TI ** 2
    Y = measurements(model, df)
    bins_all, centres = wind_bins(df["v"].to_numpy(), nbins)
    Ad, Bd, K, _, _ = build_bank(model, p, centres, R)
    u_meas = build_inputs(model, df, p)

    z1c, z2c = call_status(df)
    q_tot = df["Q_total"].to_numpy()
    t_o = df["T_o"].to_numpy()
    ghi = df["GHI"].to_numpy()
    sp = {1: df["T_i1_set"].to_numpy(), 2: df["T_i2_set"].to_numpy()}
    k12 = p["K_12"]

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
                vbar = df["v"].to_numpy()[g:g + horizon].mean()
                terms = {z: _zone_terms(p, z, vbar) for z in (1, 2)}
                target = {z: float(cal[z].threshold(sp[z][g])) for z in (1, 2)}

                # state now
                t_e = {1: xu[0], 2: xu[3]}
                t_m = {1: xu[2], 2: xu[5]}
                t_i_now = {1: xu[1], 2: xu[4]}

                energy = 0.0
                for z in (1, 2):
                    tz = terms[z]
                    other = target[2 if z == 1 else 1]

                    # heat the room needs per second to hold its threshold,
                    # averaged over the hour via mean weather
                    to = t_o[g:g + horizon].mean()
                    isol = ghi[g:g + horizon].mean()
                    tm = t_m[z]
                    need = 0.0
                    tm_k = tm
                    for _ in range(horizon):
                        flow = (tz["k_im"] * (tm_k - target[z])
                                + k12 * (other - target[z])
                                + tz["ginf"] * (to - target[z])
                                + tz["f_sol"] * p["a_sol"] * isol
                                + tz["gain"])
                        need += max(0.0, -flow) * DT
                        # envelope keeps cooling while the room is held
                        dtm = ((tz["k_im"] * (target[z] - tm_k)
                                + tz["k_mo"] * (to - tm_k)) / tz["c_m"])
                        tm_k += dtm * DT

                    # bringing the room from where it is to its threshold
                    need += tz["c_i"] * (target[z] - t_i_now[z])

                    # reheating the emitter from where it is to what the
                    # steady flow requires
                    q_ss = need / (horizon * DT)
                    te_needed = target[z] + q_ss / tz["k_ei"]
                    need += tz["c_e"] * (te_needed - t_e[z])

                    energy += max(0.0, need) * JOULE_TO_KWH

                rows.append({
                    "t": df["timestamp"].iloc[g],
                    "pred": energy,
                    "actual": q_tot[g:g + horizon].sum() * DT * JOULE_TO_KWH,
                    "persistence": (q_tot[g - horizon:g].sum() * DT * JOULE_TO_KWH
                                    if t >= horizon else np.nan),
                    "T_o": t_o[g:g + horizon].mean(),
                    "call1": bool(z1c[g]), "call2": bool(z2c[g]),
                })

            x = Ad[bb[t]] @ xu + Bd[bb[t]] @ u_meas[a + t]

    return pd.DataFrame(rows)
