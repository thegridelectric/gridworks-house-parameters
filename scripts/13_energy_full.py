"""
Extended 4R3C fitted on ENERGY alone: can the added physics beat the regression?

The model is the two-zone 4R3C plus the two mechanisms that showed a signal:

  * internal gains follow a fitted daily shape rather than a constant. Measured
    gains run 592 W at 09:00 to 1579 W at 19:00, and weekday middays sit ~500 W
    below weekend middays, which is occupancy rather than appliances on timers.
  * a basement node between the loop and the rooms. In the joint fit this
    settled to a small loss fraction with near-perfect coupling to zone 1, i.e.
    it behaves as a second capacitance on zone 1 rather than a separate space.

Optimised directly on hour-ahead energy, since fitting on the temperature
likelihood is what produced the earlier 0.853 kWh. Same three-way split as
every other energy result, so the number is comparable with the 0.684 kWh of
the plain energy-fitted 4R3C and the linear model on identical windows.

    uv run -u scripts/13_energy_full.py
"""

import json
import sys
import time
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.fit import bounds, pack, seed, unpack
from house.forecast import forecast, score
from house.model import R4C3x2_FULL, derived, gain_shape
from house.simple import Linear, hourly_frame, score_energy
from house.thermostat import calibrate
from house.validate import open_loop

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
NBINS, BURN, MAXFEV = 6, 288, 2500
POWER_LO, POWER_HI = 1000.0, 16000.0
MODEL = R4C3x2_FULL
LINEAR_FEATS = ["dT", "wind", "GHI", "gap1", "gap2", "prev", "call", "to_6h"]


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def main() -> None:
    spec = spec_from_file_location("ef", HERE / "06_energy_fit.py")
    ef = module_from_spec(spec)
    spec.loader.exec_module(ef)

    data = load_data(verbose=False)
    df = data.df
    parts, masks = ef.three_way(data)
    cal = calibrate(df[masks["fit"]].reset_index(drop=True))
    n_p = len(MODEL.params)

    def split_x(x):
        p = unpack(MODEL, x[:n_p])
        pw = np.exp(x[n_p:])
        return p, {1: replace(cal[1], on_power=float(pw[0])),
                   2: replace(cal[2], on_power=float(pw[1]))}

    def energy_mae(x, segs, mask):
        p, zones = split_x(x)
        try:
            v = forecast(MODEL, p, zones, df, segs, mask, nbins=NBINS, burn=BURN)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return 1e6
        if v.empty:
            return 1e6
        e = float((v.pred - v.actual).abs().mean())
        return e if np.isfinite(e) else 1e6

    bnds = bounds(MODEL) + [(np.log(POWER_LO), np.log(POWER_HI))] * 2
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])

    warm = json.loads((RESULTS / "07_direct_energy.json").read_text())
    starts = [("physical seed",
               np.clip(np.r_[pack(MODEL, seed(MODEL)),
                             np.log([cal[1].on_power, cal[2].on_power])],
                       lo, hi))]
    xw = np.r_[pack(MODEL, {q.name: min(max(warm["params"].get(q.name, q.seed),
                                            q.lo), q.hi)
                            for q in MODEL.params}),
               np.log([float(warm["on_power"]["1"]),
                       float(warm["on_power"]["2"])])]
    starts.append(("from plain energy fit", np.clip(xw, lo, hi)))
    rng = np.random.default_rng(3)
    starts.append(("perturbed", np.clip(xw + rng.normal(0, 0.3, xw.shape),
                                        lo, hi)))

    rule(f"ENERGY-ONLY FIT of {MODEL.name}  "
         f"({MODEL.n} states, {len(MODEL.params)} params)")
    best = None
    for name, x0 in starts:
        t0 = time.time()
        res = minimize(energy_mae, x0, method="Powell", bounds=bnds,
                       args=(parts["fit"], masks["fit"]),
                       options={"maxfev": MAXFEV, "xtol": 1e-3, "ftol": 1e-4})
        s = energy_mae(res.x, parts["sel"], masks["sel"])
        print(f"  {name:<22s} fit {res.fun:.3f}  selection {s:.3f} kWh  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or s < best[0]:
            best = (s, res.x, name)

    sel_mae, x_best, name = best
    p, zones = split_x(x_best)

    rule(f"BEST: {name}  (selection {sel_mae:.3f} kWh)")
    d = derived(MODEL, p)
    for k, val in d.items():
        print(f"    {k:<20s} {val:12.3f}")
    sh = gain_shape(p, np.arange(24))
    print(f"    gain shape           {sh.min():.2f} at {sh.argmin():02d}:00 "
          f"to {sh.max():.2f} at {sh.argmax():02d}:00")
    print(f"    f_loss {p['f_loss']:.3f}   K_b1 {p['K_b1']:.0f} W/K   "
          f"K_bo {p['K_bo']:.0f} W/K   tau_b {p['tau_b'] / 3600:.1f} h")
    print(f"    on-power  zone1 {zones[1].on_power:.0f} W   "
          f"zone2 {zones[2].on_power:.0f} W")

    v = forecast(MODEL, p, zones, df, parts["test"], masks["test"],
                 nbins=NBINS, burn=BURN)
    s_rc = score(v)
    t_rmse = open_loop(MODEL, p, df, parts["test"], masks["test"],
                       nbins=NBINS, burn=BURN)[1]["rmse_avg"]

    # the linear model on exactly the same windows
    hs = {k: hourly_frame(df, cal, segments=parts[k], burn=288).set_index("t")
          for k in ("fit", "test")}
    to = df.T_o.to_numpy()
    for k in hs:
        hs[k]["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in hs[k].i]
    lin = Linear("lin", LINEAR_FEATS).fit(hs["fit"])
    h_te = hs["test"].reindex(v.set_index("t").index).dropna(subset=["actual"])
    s_lin = score_energy(lin.predict(h_te), h_te.actual)

    rule("HEAD TO HEAD  (test split, identical windows)")
    print(f"  {'model':<34s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'r':>6s}")
    for lab, s in ((f"4R3C + gains + basement", s_rc),
                   ("linear regression", s_lin)):
        print(f"  {lab:<34s} {s['mae_kwh']:7.3f} {s['rmse_kwh']:7.3f} "
              f"{s['bias_kwh']:+7.3f} {s['r']:6.3f}")
    print(f"\n  for reference: plain energy-fitted 4R3C 0.684, "
          f"persistence 1.092")
    print(f"  this model's temperature RMSE: {t_rmse:.3f} K "
          f"(energy-only fit, so expected to be poor)")

    out = {"params": p, "derived": d, "test_energy": s_rc,
           "test_linear": s_lin, "temp_rmse": t_rmse, "sel_mae": sel_mae,
           "on_power": {1: zones[1].on_power, 2: zones[2].on_power}}
    with open(RESULTS / "13_energy_full.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    v.to_csv(RESULTS / "13_energy_full.csv", index=False)
    print(f"\n  -> {RESULTS / '13_energy_full.json'}", flush=True)


if __name__ == "__main__":
    main()
