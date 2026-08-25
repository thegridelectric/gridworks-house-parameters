"""
Physics imposed rather than fitted: does the occupancy shape help when it is
constrained to what was measured?

Letting the model fit its own daily gain shape failed: it chose a peak at
07:00 when the data says 19:00, and pinned on-power at its upper bound at
16 kW against a measured 6.3 kW. Four free parameters became fitting slack.

So here the shape is FIXED to the profile measured directly from the data
(0.79 at midday, 1.24 at 19:00, from results/occupancy_shape.json) and
on-power is FIXED to the calibrated value. Nothing is fitted that was already
measured. If the mechanism is real, removing that freedom should help; if the
model still loses, the mechanism is not what was missing.

Two candidates, both fitted on energy alone:

  A  two-zone 4R3C + fixed occupancy shape
  B  the same plus the basement node (a second capacitance on zone 1)

    uv run -u scripts/14_constrained.py
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
from house.fit import bounds, pack, seed, unpack
from house.forecast import forecast, score
from house.model import R4C3x2, R4C3x2_BASEMENT, derived, gain_shape
from house.simple import Linear, hourly_frame, score_energy
from house.thermostat import calibrate
from house.validate import open_loop

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
NBINS, BURN, MAXFEV = 6, 288, 2000
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

    SHAPE = json.loads((RESULTS / "occupancy_shape.json").read_text())
    sh = gain_shape({f"gp_{k}": v for k, v in SHAPE.items()}, np.arange(24))
    print(f"  fixed occupancy shape: {sh.min():.2f} at {sh.argmin():02d}:00 "
          f"to {sh.max():.2f} at {sh.argmax():02d}:00")
    print(f"  fixed on-power: zone1 {cal[1].on_power:.0f} W, "
          f"zone2 {cal[2].on_power:.0f} W")

    warm = json.loads((RESULTS / "07_direct_energy.json").read_text())["params"]
    results = {}

    for label, model in (("A  4R3C + fixed occupancy", R4C3x2),
                         ("B  A + basement node", R4C3x2_BASEMENT)):
        rule(f"{label}   ({len(model.params)} fitted params, "
             "shape and on-power fixed)")

        def unpack_full(x, model=model):
            p = unpack(model, x)
            p.update({f"gp_{k}": v for k, v in SHAPE.items()})
            return p

        def obj(x, segs=parts["fit"], mask=masks["fit"], model=model):
            p = unpack_full(x, model)
            try:
                v = forecast(model, p, cal, df, segs, mask,
                             nbins=NBINS, burn=BURN)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1e6
            if v.empty:
                return 1e6
            e = float((v.pred - v.actual).abs().mean())
            return e if np.isfinite(e) else 1e6

        bnds = bounds(model)
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])
        starts = [("physical seed", np.clip(pack(model, seed(model)), lo, hi)),
                  ("from energy fit",
                   np.clip(pack(model, {q.name: min(max(
                       warm.get(q.name, q.seed), q.lo), q.hi)
                       for q in model.params}), lo, hi))]

        best = None
        for name, x0 in starts:
            t0 = time.time()
            res = minimize(obj, x0, method="Powell", bounds=bnds,
                           options={"maxfev": MAXFEV, "xtol": 1e-3,
                                    "ftol": 1e-4})
            s = obj(res.x, parts["sel"], masks["sel"], model)
            print(f"    {name:<18s} fit {res.fun:.3f}  selection {s:.3f} kWh"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            if best is None or s < best[0]:
                best = (s, res.x)

        p = unpack_full(best[1], model)
        v = forecast(model, p, cal, df, parts["test"], masks["test"],
                     nbins=NBINS, burn=BURN)
        s = score(v)
        t_rmse = open_loop(model, p, df, parts["test"], masks["test"],
                           nbins=NBINS, burn=BURN)[1]["rmse_avg"]
        d = derived(model, p)
        print(f"\n    TEST energy MAE {s['mae_kwh']:.3f} kWh  r {s['r']:.3f}  "
              f"bias {s['bias_kwh']:+.3f}")
        print(f"    temperature RMSE {t_rmse:.3f} K")
        for k in ("UA_total [W/K]", "gains [W]", "tau_m1 [h]", "tau_e1 [min]",
                  "tau_e2 [min]"):
            if k in d:
                print(f"      {k:<18s} {d[k]:10.2f}")
        if "f_loss" in p:
            print(f"      f_loss {p['f_loss']:.3f}  K_b1 {p['K_b1']:.0f}  "
                  f"K_bo {p['K_bo']:.0f}  tau_b {p['tau_b'] / 3600:.1f} h")
        results[label] = {"params": p, "derived": d, "test": s,
                          "temp_rmse": t_rmse, "sel": best[0],
                          "windows": v.set_index("t").index}

    # linear model on the same windows
    hs = {k: hourly_frame(df, cal, segments=parts[k], burn=288).set_index("t")
          for k in ("fit", "test")}
    to = df.T_o.to_numpy()
    for k in hs:
        hs[k]["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in hs[k].i]
    lin = Linear("lin", LINEAR_FEATS).fit(hs["fit"])
    idx = next(iter(results.values()))["windows"]
    h_te = hs["test"].reindex(idx).dropna(subset=["actual"])
    s_lin = score_energy(lin.predict(h_te), h_te.actual)

    rule("HEAD TO HEAD  (test split, identical windows)")
    print(f"  {'model':<34s} {'MAE':>7s} {'r':>7s} {'temp K':>8s}")
    for lab, r in results.items():
        print(f"  {lab:<34s} {r['test']['mae_kwh']:7.3f} "
              f"{r['test']['r']:7.3f} {r['temp_rmse']:8.3f}")
    print(f"  {'linear regression':<34s} {s_lin['mae_kwh']:7.3f} "
          f"{s_lin['r']:7.3f} {'-':>8s}")
    print("\n  reference: plain energy-fitted 4R3C 0.684 (2.383 K),")
    print("             fitted-shape version 0.744 (1.333 K),")
    print("             persistence 1.092")

    with open(RESULTS / "14_constrained.json", "w") as fh:
        json.dump({k: {kk: vv for kk, vv in r.items() if kk != "windows"}
                   for k, r in results.items()} | {"linear": s_lin},
                  fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '14_constrained.json'}", flush=True)


if __name__ == "__main__":
    main()
