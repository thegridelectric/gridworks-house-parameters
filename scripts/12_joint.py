"""
Can one parameter set predict BOTH temperature and energy?

The current model cannot. Fitted on temperature it reaches 0.141 K but only
0.853 kWh; fitted on energy it reaches 0.684 kWh but 2.383 K -- seven times
worse than assuming the temperature does not change. A model that genuinely
described the building would not force that choice, so having to choose is
evidence of missing physics rather than a bad fit.

This script fits candidates on a joint criterion

    J = 0.5 * RMSE_T / 0.20 K  +  0.5 * MAE_E / 0.70 kWh

so neither job can be sacrificed, and reports both metrics. Candidates:

  base       the current two-zone 4R3C
  +gains     internal gains follow a fitted daily shape rather than a constant.
             Measured gains run 592 W at 09:00 to 1579 W at 19:00, with
             weekday middays 500 W below weekend middays -- an occupancy
             signature a constant cannot carry.
  +basement  part of the metered heat warms pipework and the plant room before
             reaching an emitter, then leaks up into zone 1 and partly outdoors
  +both

Fit on the first 70%, choose among starts on the next 10%, report the last 20%.

    uv run -u scripts/12_joint.py
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
from house.model import (R4C3x2, R4C3x2_BASEMENT, R4C3x2_FULL, R4C3x2_GAINS,
                         derived, gain_shape)
from house.thermostat import calibrate
from house.validate import open_loop

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
NBINS, BURN = 6, 288
T_REF, E_REF = 0.20, 0.70
MAXFEV = 2500
POWER_LO, POWER_HI = 1000.0, 16000.0

CANDIDATES = [("base", R4C3x2), ("+gains", R4C3x2_GAINS),
              ("+basement", R4C3x2_BASEMENT), ("+both", R4C3x2_FULL)]


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def evaluate(model, p, zones, df, segs, mask):
    """Both metrics for one parameter set."""
    t = open_loop(model, p, df, segs, mask, nbins=NBINS, burn=BURN)[1]
    e = score(forecast(model, p, zones, df, segs, mask, nbins=NBINS, burn=BURN))
    if not t or not e:
        return None
    return t["rmse_avg"], e["mae_kwh"]


def main() -> None:
    ef_spec = spec_from_file_location("ef", HERE / "06_energy_fit.py")
    ef = module_from_spec(ef_spec)
    ef_spec.loader.exec_module(ef)

    data = load_data(verbose=False)
    df = data.df
    parts, masks = ef.three_way(data)
    cal = calibrate(df[masks["fit"]].reset_index(drop=True))

    warm = json.loads((RESULTS / "07_direct_energy.json").read_text())
    # The temperature-likelihood fit turns out to be a strong JOINT point
    # (0.141 K / 0.853 kWh, J = 0.96) -- better than anything the joint
    # optimiser reached unaided. Powell is local, so it has to be started
    # somewhere sensible; this is the sensible place.
    warm_t = json.loads((RESULTS / "06_energy_fit.json").read_text())[
        "tau_e bounded + free q"]["params"]

    rows, blob = [], {}
    for label, model in CANDIDATES:
        rule(f"{label}   ({model.name}, {model.n} states, "
             f"{len(model.params)} params)")
        n_p = len(model.params)

        def split_x(x):
            p = unpack(model, x[:n_p])
            pw = np.exp(x[n_p:])
            return p, {1: replace(cal[1], on_power=float(pw[0])),
                       2: replace(cal[2], on_power=float(pw[1]))}

        def joint(x, segs=parts["fit"], mask=masks["fit"]):
            p, zones = split_x(x)
            try:
                r = evaluate(model, p, zones, df, segs, mask)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1e6
            if r is None or not all(np.isfinite(r)):
                return 1e6
            return 0.5 * r[0] / T_REF + 0.5 * r[1] / E_REF

        bnds = bounds(model) + [(np.log(POWER_LO), np.log(POWER_HI))] * 2
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])

        x_seed = np.r_[pack(model, seed(model)),
                       np.log([cal[1].on_power, cal[2].on_power])]
        starts = [("physical seed", np.clip(x_seed, lo, hi))]
        for nm, src, pw in (("from temperature fit", warm_t,
                             (cal[1].on_power, cal[2].on_power)),
                            ("from energy fit", warm["params"],
                             (float(warm["on_power"]["1"]),
                              float(warm["on_power"]["2"])))):
            try:
                xw = np.r_[pack(model, {q.name: min(max(
                    src.get(q.name, q.seed), q.lo), q.hi)
                    for q in model.params}), np.log(pw)]
                starts.append((nm, np.clip(xw, lo, hi)))
            except (KeyError, ValueError):
                pass

        best = None
        for name, x0 in starts:
            t0 = time.time()
            res = minimize(joint, x0, method="Powell", bounds=bnds,
                           options={"maxfev": MAXFEV, "xtol": 1e-3,
                                    "ftol": 1e-4})
            p, zones = split_x(res.x)
            sel = evaluate(model, p, zones, df, parts["sel"], masks["sel"])
            j_sel = (0.5 * sel[0] / T_REF + 0.5 * sel[1] / E_REF
                     if sel else np.inf)
            print(f"    {name:<16s} fit J {res.fun:.3f}  sel J {j_sel:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if best is None or j_sel < best[0]:
                best = (j_sel, res.x)

        p, zones = split_x(best[1])
        t_rmse, e_mae = evaluate(model, p, zones, df, parts["test"],
                                 masks["test"])
        print(f"\n    TEST  temperature {t_rmse:.3f} K   energy {e_mae:.3f} kWh")
        d = derived(model, p)
        for k in ("UA_total [W/K]", "gains [W]", "tau_m1 [h]", "tau_e1 [min]"):
            if k in d:
                print(f"      {k:<18s} {d[k]:10.2f}")
        if "gp_a1" in p:
            sh = gain_shape(p, np.arange(24))
            print(f"      gain shape  min {sh.min():.2f} at {sh.argmin():02d}:00,"
                  f"  max {sh.max():.2f} at {sh.argmax():02d}:00")
        if "f_loss" in p:
            print(f"      f_loss {p['f_loss']:.3f}  K_b1 {p['K_b1']:.0f} W/K  "
                  f"K_bo {p['K_bo']:.0f} W/K  tau_b {p['tau_b'] / 3600:.1f} h")

        rows.append({"model": label, "params": len(model.params),
                     "temp_rmse_K": t_rmse, "energy_mae_kwh": e_mae,
                     "joint": 0.5 * t_rmse / T_REF + 0.5 * e_mae / E_REF})
        blob[label] = {"params": p, "derived": d, "temp_rmse": t_rmse,
                       "energy_mae": e_mae,
                       "on_power": {1: zones[1].on_power, 2: zones[2].on_power}}

    rule("CAN ONE PARAMETER SET DO BOTH?  (test split)")
    t = pd.DataFrame(rows).sort_values("joint")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  reference: temperature-only fit gave 0.141 K / 0.853 kWh;")
    print("             energy-only fit gave 2.383 K / 0.684 kWh;")
    print("             persistence 0.313 K / 1.092 kWh")

    t.to_csv(RESULTS / "12_joint.csv", index=False)
    with open(RESULTS / "12_joint.json", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '12_joint.csv'}", flush=True)


if __name__ == "__main__":
    main()
