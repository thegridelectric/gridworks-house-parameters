"""
Energy-optimised 4R3C with a physical emitter-charging law.

Every version so far assumed the loop delivers a fixed wattage whenever the
thermostat calls. That is wrong in a way that matters. Measured power while
calling spans 3000-8100 W in zone 1, because a cold emitter pulls far more heat
out of the loop than a hot one -- the same effect that produces the 32 kW
startup spikes. A fixed number therefore carries a bias whose sign depends on
how often the loop happens to start cold, which is exactly the regime the
forecast has to get right.

So heat entering the emitter becomes

    Q = K_w * (T_w - T_e)     while calling, 0 otherwise

with supply temperature T_w and a loop conductance K_w per zone, both fitted.
This needs no new data: T_e is already a state, and the law self-limits, since
a warm emitter accepts less heat.

Optimised directly on hour-ahead energy, same three-way split as before.

    uv run -u scripts/09_supply.py
"""

import json
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.fit import bounds, pack, unpack
from house.forecast import forecast, score
from house.model import derived
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
NBINS, BURN = 6, 288
N_STARTS = 5
MAXFEV = 2500

# supply temperature [C] and per-zone loop conductance [W/K]
TW_LO, TW_HI, TW_SEED = 30.0, 75.0, 45.0
KW_LO, KW_HI, KW_SEED = 20.0, 3000.0, 300.0


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def load_module(name, fname):
    spec = spec_from_file_location(name, HERE / fname)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ef = load_module("ef", "06_energy_fit.py")
    de = load_module("de", "07_direct_energy.py")
    model = de.MODEL

    data = load_data(verbose=False)
    df = data.df
    parts, masks = ef.three_way(data)
    cal = calibrate(df[masks["fit"]].reset_index(drop=True))
    n_p = len(model.params)

    def split_x(x):
        p = unpack(model, x[:n_p])
        t_w = float(x[n_p])
        k_w = {1: float(np.exp(x[n_p + 1])), 2: float(np.exp(x[n_p + 2]))}
        return p, {"T_w": t_w, "K_w": k_w}

    def objective(segs, mask):
        def obj(x):
            p, sup = split_x(x)
            try:
                v = forecast(model, p, cal, df, segs, mask, nbins=NBINS,
                             burn=BURN, supply=sup)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1e6
            if v.empty:
                return 1e6
            e = float((v.pred - v.actual).abs().mean())
            return e if np.isfinite(e) else 1e6
        return obj

    obj_fit = objective(parts["fit"], masks["fit"])
    obj_sel = objective(parts["sel"], masks["sel"])

    bnds = bounds(model) + [(TW_LO, TW_HI),
                            (np.log(KW_LO), np.log(KW_HI)),
                            (np.log(KW_LO), np.log(KW_HI))]
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])

    base = pack(model, {q.name: q.seed for q in model.params})
    x_seed = np.r_[base, TW_SEED, np.log([KW_SEED, KW_SEED])]

    starts = [("physical seed", x_seed)]
    p7 = RESULTS / "07_direct_energy.json"
    if p7.exists():
        b7 = json.loads(p7.read_text())["params"]
        try:
            x = np.r_[pack(model, {q.name: min(max(b7[q.name], q.lo), q.hi)
                                   for q in model.params}),
                      TW_SEED, np.log([KW_SEED, KW_SEED])]
            starts.append(("from energy fit", np.clip(x, lo, hi)))
        except KeyError:
            pass

    rng = np.random.default_rng(2)
    while len(starts) < N_STARTS:
        starts.append((f"perturbed {len(starts)}",
                       np.clip(x_seed + rng.normal(0, 0.5, x_seed.shape),
                               lo, hi)))

    rule("OPTIMISING (supply-temperature emitter law)")
    best = None
    for label, x0 in starts:
        t0 = time.time()
        res = minimize(obj_fit, x0, method="Powell", bounds=bnds,
                       options={"maxfev": MAXFEV, "xtol": 1e-3, "ftol": 1e-4})
        s = obj_sel(res.x)
        print(f"  {label:<16s} fit {res.fun:.3f}  selection {s:.3f} kWh  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or s < best[0]:
            best = (s, res.x, label)

    sel_mae, x_best, label = best
    p, sup = split_x(x_best)

    rule(f"BEST: {label}   selection MAE {sel_mae:.3f} kWh")
    d = derived(model, p)
    for k, val in d.items():
        print(f"    {k:<20s} {val:12.3f}")
    print(f"    {'supply temp':<20s} {sup['T_w']:12.1f} C")
    print(f"    {'K_w zone 1':<20s} {sup['K_w'][1]:12.0f} W/K")
    print(f"    {'K_w zone 2':<20s} {sup['K_w'][2]:12.0f} W/K")

    v = forecast(model, p, cal, df, parts["test"], masks["test"],
                 nbins=NBINS, burn=BURN, supply=sup)
    s = score(v)
    rule("TEST SET")
    print(f"  MAE {s['mae_kwh']:.3f} kWh   bias {s['bias_kwh']:+.3f}   "
          f"MAPE {s['mape_pct']:.1f}%   r {s['r']:.3f}   n {s['n']}")
    for lab, col in (("persistence", "persistence"),
                     ("steady-state UA line", "steady_state")):
        b = score(v, col)
        print(f"  {lab:<22s} MAE {b['mae_kwh']:.3f}   r {b['r']:.3f}")

    with open(RESULTS / "09_supply.json", "w") as fh:
        json.dump({"params": p, "supply": sup, "derived": d, "test": s,
                   "sel_mae": sel_mae, "start": label}, fh, indent=2,
                  default=float)
    v.to_csv(RESULTS / "09_supply.csv", index=False)
    print(f"\n  -> {RESULTS / '09_supply.json'}", flush=True)


if __name__ == "__main__":
    main()
