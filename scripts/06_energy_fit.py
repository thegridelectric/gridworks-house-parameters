"""
Fit for the job the model is actually for: hour-ahead loop energy.

Two changes over the earlier passes.

1. The emitter time constant is bounded to a physically defensible range.
   Left free it ran to 8 h, which makes the simulated thermostat call for heat
   that never reaches the room, and the energy forecast over-predicts by 31%.
   The observed cycling (zone 1 every 25 min, zone 2 every 90 min) says the
   real value is tens of minutes.

2. Candidate fits are selected by closed-loop ENERGY error, not by the
   temperature likelihood. The likelihood barely cares about the emitter time
   constant while the forecast depends on it completely, so optimising one and
   hoping for the other is what produced the earlier failure.

Selection needs its own data or it is just tuning on the test set, so the
record is split three ways: fit on the first 70%, choose among candidates on
the next 10%, report on the final 20%.

    uv run scripts/06_energy_fit.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.fit import at_bound, bounds, make_cost, pack, seed, unpack
from house.forecast import forecast, score
from house.model import HOUR, MINUTE, R4C3x2, R4C3x2_SHARED, derived, relax
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
NBINS, BURN, MAXITER, RESTARTS = 6, 288, 400, 4

TAU_E = (5 * MINUTE, 60 * MINUTE, 25 * MINUTE)     # from observed cycling
TAU_M = (5 * HOUR, 150 * HOUR, 30 * HOUR)
Q_FREE = (1e-14, 1e-4, 1e-9)
Q_TIGHT = (1e-14, 1e-9, 1e-11)


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def three_way(data, f_fit=0.70, f_sel=0.80):
    n = len(data.df)
    c1, c2 = int(f_fit * n), int(f_sel * n)
    parts = {"fit": [], "sel": [], "test": []}
    for a, b in data.segments:
        for key, lo, hi in (("fit", 0, c1), ("sel", c1, c2), ("test", c2, n)):
            s, e = max(a, lo), min(b, hi)
            if e - s >= 288:
                parts[key].append((s, e))
    masks = {}
    for key, segs in parts.items():
        m = np.zeros(n, bool)
        for a, b in segs:
            m[a:b] = True
        masks[key] = m
    return parts, masks


def variants():
    """Each entry isolates one change, so the effect can be attributed."""
    two = {"tau_e1": TAU_E, "tau_e2": TAU_E}
    one = {"tau_e": TAU_E}
    both_m = {"tau_m1": TAU_M, "tau_m2": TAU_M}
    return [
        ("free-emitter (baseline)", relax(R4C3x2, "v0", q_scale=Q_TIGHT)),
        ("tau_e bounded", relax(R4C3x2, "v1", q_scale=Q_TIGHT, **two)),
        ("tau_e bounded + free q", relax(R4C3x2, "v2", q_scale=Q_FREE, **two)),
        ("tau_e + tau_m bounded + free q",
         relax(R4C3x2, "v3", q_scale=Q_FREE, **two, **both_m)),
        ("shared emitter, all bounded",
         relax(R4C3x2_SHARED, "v4", q_scale=Q_FREE, **one, **both_m)),
    ]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = load_data(verbose=False)
    df = data.df
    parts, masks = three_way(data)

    rule("SPLIT")
    for k in ("fit", "sel", "test"):
        print(f"  {k:<5s} {len(parts[k])} segments, "
              f"{sum(b - a for a, b in parts[k])} rows")

    fit_rows = masks["fit"]
    cal = calibrate(df[fit_rows].reset_index(drop=True))
    print("\n  thermostat calibrated on fit rows only")

    rows, blob = [], {}
    for label, model in variants():
        rule(f"{label}   ({len(model.params)} params)")
        cost = make_cost(model, df, parts["fit"], nbins=NBINS, burn=BURN)
        bnds = bounds(model)
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])
        x0 = pack(model, seed(model))
        rng = np.random.default_rng(0)

        best = None
        t0 = time.time()
        for r in range(RESTARTS):
            start = x0 if r == 0 else np.clip(
                x0 + rng.normal(0, 0.4, x0.shape), lo, hi)
            res = minimize(cost, start, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": MAXITER, "maxfun": 20000,
                                    "ftol": 1e-12, "gtol": 1e-8})
            p = unpack(model, res.x)
            # SELECT ON ENERGY, not on the likelihood
            v = forecast(model, p, cal, df, parts["sel"], masks["sel"],
                         nbins=NBINS, burn=BURN)
            s = score(v)
            mae = s.get("mae_kwh", np.inf)
            print(f"      restart {r + 1}: nll={res.fun:9.4f}  "
                  f"selection MAE={mae:.3f} kWh")
            if best is None or mae < best[0]:
                best = (mae, p, res.fun)

        sel_mae, p, nll = best
        secs = time.time() - t0

        v = forecast(model, p, cal, df, parts["test"], masks["test"],
                     nbins=NBINS, burn=BURN)
        s = score(v)
        d = derived(model, p)
        hits = at_bound(model, p)

        print(f"    chosen: selection MAE={sel_mae:.3f}  ({secs:.0f}s)")
        print(f"    tau_e = "
              + ", ".join(f"{k.split()[0]}={d[k]:.1f} min"
                          for k in d if k.startswith("tau_e")))
        print(f"    at bounds ({len(hits)}): "
              f"{', '.join(hits) if hits else 'none'}")
        print(f"    TEST  MAE={s['mae_kwh']:.3f}  bias={s['bias_kwh']:+.3f}  "
              f"MAPE={s['mape_pct']:.1f}%  r={s['r']:.3f}  n={s['n']}")

        rows.append({"variant": label, "sel_mae": sel_mae,
                     "mae": s["mae_kwh"], "bias": s["bias_kwh"],
                     "mape": s["mape_pct"], "r": s["r"],
                     "n_bound": len(hits), "secs": secs,
                     **{k: d[k] for k in d if k.startswith("tau_e")}})
        blob[label] = {"params": p, "derived": d, "test": s,
                       "sel_mae": sel_mae, "at_bound": hits}

    # baselines on the same test windows
    rule("TEST SET  (final 20%, never used for fitting or selection)")
    ref = forecast(R4C3x2, blob[rows[0]["variant"]]["params"], cal, df,
                   parts["test"], masks["test"], nbins=NBINS, burn=BURN)
    for label, col in (("persistence (last hour)", "persistence"),
                       ("steady-state UA line", "steady_state")):
        s = score(ref, col)
        rows.append({"variant": label, "sel_mae": np.nan, "mae": s["mae_kwh"],
                     "bias": s["bias_kwh"], "mape": s["mape_pct"],
                     "r": s["r"], "n_bound": np.nan, "secs": np.nan})

    t = pd.DataFrame(rows).sort_values("mae")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    t.to_csv(RESULTS / "06_energy_fit.csv", index=False)
    with open(RESULTS / "06_energy_fit.json", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '06_energy_fit.csv'}")


if __name__ == "__main__":
    main()
