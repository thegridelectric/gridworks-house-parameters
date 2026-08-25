"""
Fit the 4R3C model by optimising hour-ahead loop energy DIRECTLY.

Everything so far fitted the model to predict temperature one step ahead and
then hoped the energy forecast would follow. It does not: the temperature
likelihood is almost indifferent to the emitter time constant, while the
energy forecast depends on it completely. So this script optimises the thing
we actually want.

Two consequences for the method:

  * The objective is not smooth. The thermostat is a switch, so a small
    parameter change can move a switching instant and step the energy. Gradient
    methods are useless here; Powell (derivative-free, direction-set) is used
    instead, from several physically-informed starting points.

  * The controller's on-power becomes a fitted parameter rather than the
    median of the data. Measured power while calling ranges 3000-8100 W, and
    the loop keeps flowing a little after the call drops out, so the median is
    not the right effective constant. Fitting it lets the model match total
    energy instead of inheriting a bias.

Seeds come from measurements, not guesses: total conductance 201 W/K and gains
1162 W from the daily energy balance, emitter time constants from the observed
cycling (zone 1 every 25 min, zone 2 every 90 min).

Fit on the first 70%, choose among starts on the next 10%, report on the last
20%, which is touched only at the end.

    uv run -u scripts/07_direct_energy.py
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
from house.fit import bounds, pack, unpack
from house.forecast import forecast, score
from house.model import HOUR, MINUTE, R4C3x2, derived, relax
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
NBINS, BURN = 6, 288
N_STARTS = 6
MAXFEV = 2500

# Bounds informed by the diagnostics rather than by caution alone.
MODEL = relax(
    R4C3x2, "energy-fit",
    tau_e1=(5 * MINUTE, 120 * MINUTE, 20 * MINUTE),
    tau_e2=(5 * MINUTE, 120 * MINUTE, 60 * MINUTE),
    tau_m1=(5 * HOUR, 150 * HOUR, 30 * HOUR),
    tau_m2=(5 * HOUR, 150 * HOUR, 30 * HOUR),
    K_ei1=(50.0, 4000.0, 400.0), K_ei2=(50.0, 4000.0, 400.0),
    UA_fab1=(10.0, 250.0, 60.0), UA_fab2=(10.0, 250.0, 60.0),
    g0_1=(1.0, 200.0, 40.0), g0_2=(1.0, 200.0, 40.0),
    G1=(10.0, 3000.0, 800.0), G2=(10.0, 3000.0, 400.0),
    C_i1=(2e5, 2e7, 1.5e6), C_i2=(2e5, 2e7, 1.5e6),
    K_12=(5.0, 2000.0, 200.0),
    q_scale=(1e-14, 1e-6, 1e-9),
)

POWER_LO, POWER_HI = 1000.0, 16000.0


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def make_objective(model, cal, df, segs, mask):
    """Hourly-energy MAE as a function of [model params, on-power 1, 2]."""
    n_p = len(model.params)

    def unpack_all(x):
        p = unpack(model, x[:n_p])
        pw = np.exp(x[n_p:])
        zones = {1: replace(cal[1], on_power=float(pw[0])),
                 2: replace(cal[2], on_power=float(pw[1]))}
        return p, zones

    def obj(x):
        p, zones = unpack_all(x)
        try:
            v = forecast(model, p, zones, df, segs, mask,
                         nbins=NBINS, burn=BURN)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return 1e6
        if v.empty:
            return 1e6
        err = float((v.pred - v.actual).abs().mean())
        return err if np.isfinite(err) else 1e6

    obj.unpack_all = unpack_all
    return obj


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    spec = spec_from_file_location("ef", HERE / "06_energy_fit.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = load_data(verbose=False)
    df = data.df
    parts, masks = mod.three_way(data)
    cal = calibrate(df[masks["fit"]].reset_index(drop=True))

    rule("SETUP")
    for k in ("fit", "sel", "test"):
        print(f"  {k:<5s} {len(parts[k])} segs, "
              f"{sum(b - a for a, b in parts[k])} rows", flush=True)
    print(f"  measured on-power: zone1 {cal[1].on_power:.0f} W, "
          f"zone2 {cal[2].on_power:.0f} W  (now fitted, not fixed)", flush=True)

    obj_fit = make_objective(MODEL, cal, df, parts["fit"], masks["fit"])
    obj_sel = make_objective(MODEL, cal, df, parts["sel"], masks["sel"])

    bnds = bounds(MODEL) + [(np.log(POWER_LO), np.log(POWER_HI))] * 2
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])

    x_seed = np.r_[pack(MODEL, {q.name: q.seed for q in MODEL.params}),
                   np.log([cal[1].on_power, cal[2].on_power])]

    # a likelihood fit, as one of the starting points
    starts = [("physical seed", x_seed)]
    prev = RESULTS / "02_ladder.json"
    if prev.exists():
        pj = json.loads(prev.read_text())["R4C3x2"]["params"]
        try:
            xl = np.r_[pack(MODEL, {q.name: min(max(pj[q.name], q.lo), q.hi)
                                    for q in MODEL.params}),
                       np.log([cal[1].on_power, cal[2].on_power])]
            starts.append(("likelihood fit", np.clip(xl, lo, hi)))
        except KeyError:
            pass

    rng = np.random.default_rng(1)
    while len(starts) < N_STARTS:
        j = np.clip(x_seed + rng.normal(0, 0.5, x_seed.shape), lo, hi)
        starts.append((f"perturbed {len(starts) - 1}", j))

    rule("OPTIMISING ENERGY DIRECTLY (Powell)")
    best = None
    for label, x0 in starts:
        t0 = time.time()
        f0 = obj_fit(x0)
        res = minimize(obj_fit, x0, method="Powell", bounds=bnds,
                       options={"maxfev": MAXFEV, "xtol": 1e-3, "ftol": 1e-4})
        s = obj_sel(res.x)
        print(f"  {label:<16s} fit MAE {f0:.3f} -> {res.fun:.3f} kWh   "
              f"selection {s:.3f} kWh   ({time.time() - t0:.0f}s, "
              f"{res.nfev} evals)", flush=True)
        if best is None or s < best[0]:
            best = (s, res.x, label, res.fun)

    sel_mae, x_best, label, fit_mae = best
    p, zones = obj_fit.unpack_all(x_best)

    rule(f"BEST START: {label}")
    print(f"  fit MAE {fit_mae:.3f}   selection MAE {sel_mae:.3f} kWh\n",
          flush=True)
    d = derived(MODEL, p)
    for k, val in d.items():
        print(f"    {k:<20s} {val:12.3f}")
    print(f"    {'on-power zone1':<20s} {zones[1].on_power:12.0f} W")
    print(f"    {'on-power zone2':<20s} {zones[2].on_power:12.0f} W")

    v = forecast(MODEL, p, zones, df, parts["test"], masks["test"],
                 nbins=NBINS, burn=BURN)
    s = score(v)
    rule("TEST SET")
    print(f"  MAE {s['mae_kwh']:.3f} kWh   bias {s['bias_kwh']:+.3f}   "
          f"MAPE {s['mape_pct']:.1f}%   r {s['r']:.3f}   n {s['n']}")
    for lab, col in (("persistence", "persistence"),
                     ("steady-state UA line", "steady_state")):
        b = score(v, col)
        print(f"  {lab:<22s} MAE {b['mae_kwh']:.3f}   r {b['r']:.3f}")

    out = {"params": p, "derived": d, "test": s, "sel_mae": sel_mae,
           "on_power": {"1": zones[1].on_power, "2": zones[2].on_power},
           "start": label}
    with open(RESULTS / "07_direct_energy.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    v.to_csv(RESULTS / "07_direct_energy.csv", index=False)
    print(f"\n  -> {RESULTS / '07_direct_energy.json'}", flush=True)


if __name__ == "__main__":
    main()
