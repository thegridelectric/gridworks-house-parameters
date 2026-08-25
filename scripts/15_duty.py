"""
Duty-constrained energy fit: the first honest test of the 4R3C on energy.

Every previous energy fit exploited a degeneracy. Only TOTAL Q_dist is
measured, so any split between the zones that sums correctly scores the same,
and the optimiser found a solution with zone 1 permanently off (simulated duty
0.000 against an actual 0.732) and zone 2 running six times too much. Total
energy came out right through two large errors cancelling.

The per-zone call status is in the data and was never used for fitting. Adding
it closes the degeneracy:

    J = energy MAE  +  W_DUTY * ( |duty1_sim - duty1_act| + |duty2_sim - duty2_act| )

so a model cannot buy energy accuracy by getting the zones wrong. W_DUTY is set
so a full 0.1 duty error on both zones costs about as much as 0.1 kWh of energy
error -- enough to forbid the degenerate corner without drowning the objective.

Candidates, all fitted this way:
  base       two-zone 4R3C
  +occ       fixed measured occupancy shape (0.79 at midday, 1.24 at 19:00)
  +occ+bsm   the same plus the basement node (second capacitance on zone 1)

Reported against the linear regression on identical windows.

    uv run -u scripts/15_duty.py
"""

import json
import sys
import time
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.fit import bounds, pack, seed, unpack
from house.forecast import forecast, score
from house.model import R4C3x2, R4C3x2_BASEMENT, derived
from house.simple import Linear, hourly_frame, score_energy
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
NBINS, BURN, MAXFEV = 6, 288, 2000
W_DUTY = 1.0            # kWh-equivalent per unit of summed duty error
POWER_LO, POWER_HI = 2000.0, 12000.0


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def duty_err(v):
    return float((v.duty1_sim - v.duty1_act).abs().mean()
                 + (v.duty2_sim - v.duty2_act).abs().mean())


def main() -> None:
    spec = spec_from_file_location("ef", HERE / "06_energy_fit.py")
    ef = module_from_spec(spec)
    spec.loader.exec_module(ef)

    data = load_data(verbose=False)
    df = data.df
    parts, masks = ef.three_way(data)
    cal = calibrate(df[masks["fit"]].reset_index(drop=True))
    SHAPE = json.loads((RESULTS / "occupancy_shape.json").read_text())
    warm = json.loads((RESULTS / "06_energy_fit.json").read_text())[
        "tau_e bounded + free q"]["params"]

    cands = [("base", R4C3x2, False),
             ("+occ", R4C3x2, True),
             ("+occ+bsm", R4C3x2_BASEMENT, True)]

    rows, blob = [], {}
    for label, model, use_occ in cands:
        rule(f"{label}   ({model.name}, {len(model.params)} params"
             f"{', fixed occupancy shape' if use_occ else ''})")
        n_p = len(model.params)

        def unpack_full(x, model=model, use_occ=use_occ):
            p = unpack(model, x[:n_p])
            if use_occ:
                p.update({f"gp_{k}": v for k, v in SHAPE.items()})
            return p, {1: replace(cal[1], on_power=float(np.exp(x[n_p]))),
                       2: replace(cal[2], on_power=float(np.exp(x[n_p + 1])))}

        def obj(x, segs=parts["fit"], mask=masks["fit"], model=model,
                report=False):
            p, zones = unpack_full(x, model)
            try:
                v = forecast(model, p, zones, df, segs, mask,
                             nbins=NBINS, burn=BURN)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1e6
            if v.empty:
                return 1e6
            mae = float((v.pred - v.actual).abs().mean())
            j = mae + W_DUTY * duty_err(v)
            if report:
                return j, mae, duty_err(v), v
            return j if np.isfinite(j) else 1e6

        bnds = bounds(model) + [(np.log(POWER_LO), np.log(POWER_HI))] * 2
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])
        pw0 = np.log([cal[1].on_power, cal[2].on_power])
        starts = [
            ("physical seed",
             np.clip(np.r_[pack(model, seed(model)), pw0], lo, hi)),
            ("from temperature fit",
             np.clip(np.r_[pack(model, {q.name: min(max(
                 warm.get(q.name, q.seed), q.lo), q.hi)
                 for q in model.params}), pw0], lo, hi)),
        ]

        best = None
        for name, x0 in starts:
            t0 = time.time()
            res = minimize(obj, x0, method="Powell", bounds=bnds,
                           options={"maxfev": MAXFEV, "xtol": 1e-3,
                                    "ftol": 1e-4})
            s = obj(res.x, parts["sel"], masks["sel"], model)
            print(f"    {name:<22s} fit J {res.fun:.3f}  sel J {s:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if best is None or s < best[0]:
                best = (s, res.x)

        j, mae, de, v = obj(best[1], parts["test"], masks["test"], model,
                            report=True)
        p, zones = unpack_full(best[1], model)
        print(f"\n    TEST  energy MAE {mae:.3f} kWh   r {score(v)['r']:.3f}   "
              f"bias {score(v)['bias_kwh']:+.3f}")
        print(f"      duty zone1 sim {v.duty1_sim.mean():.3f} vs act "
              f"{v.duty1_act.mean():.3f}   zone2 sim {v.duty2_sim.mean():.3f} "
              f"vs act {v.duty2_act.mean():.3f}")
        d = derived(model, p)
        for k in ("UA_total [W/K]", "gains [W]", "tau_e1 [min]", "lam"):
            if k in d:
                print(f"      {k:<18s} {d[k]:10.2f}")
        print(f"      on-power {zones[1].on_power:.0f} / "
              f"{zones[2].on_power:.0f} W")

        rows.append({"model": label, "mae": mae, "r": score(v)["r"],
                     "bias": score(v)["bias_kwh"], "duty_err": de,
                     "duty1_sim": v.duty1_sim.mean(),
                     "duty2_sim": v.duty2_sim.mean()})
        blob[label] = {"params": p, "derived": d, "mae": mae,
                       "on_power": {1: zones[1].on_power, 2: zones[2].on_power}}
        idx = v.set_index("t").index

    # linear model, same windows
    hs = {k: hourly_frame(df, cal, segments=parts[k], burn=288).set_index("t")
          for k in ("fit", "test")}
    to = df.T_o.to_numpy()
    for k in hs:
        hs[k]["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in hs[k].i]
    lin = Linear("lin", ["dT", "wind", "GHI", "gap1", "gap2", "prev", "call",
                         "to_6h"]).fit(hs["fit"])
    h_te = hs["test"].reindex(idx).dropna(subset=["actual"])
    s_lin = score_energy(lin.predict(h_te), h_te.actual)
    rows.append({"model": "linear regression", "mae": s_lin["mae_kwh"],
                 "r": s_lin["r"], "bias": s_lin["bias_kwh"],
                 "duty_err": np.nan, "duty1_sim": np.nan,
                 "duty2_sim": np.nan})

    rule("CAN THE 4R3C BEAT THE LINEAR MODEL?  (test split, same windows)")
    t = pd.DataFrame(rows).sort_values("mae")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  actual duty: zone1 0.732, zone2 0.101")
    print("  reference: degenerate energy fit scored 0.684 with zone1 duty 0.000")

    t.to_csv(RESULTS / "15_duty.csv", index=False)
    with open(RESULTS / "15_duty.json", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '15_duty.csv'}", flush=True)


if __name__ == "__main__":
    main()
