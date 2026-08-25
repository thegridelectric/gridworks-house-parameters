"""
Second ladder pass: move the bounds the first pass pinned against.

Three changes, each targeting one pinning pattern from pass 1:

  1. q_scale unclamped (1e-9 -> 1e-4 ceiling). It pinned high in ALL five
     rungs, including the six-parameter one, so it is not a 4R3C problem. The
     original clamp was there to stop process noise masking a broken energy
     balance; that balance now closes, so the clamp may be doing more harm
     than good.

  2. Emitter parameters shared across zones. Pass 1 showed the shared rung
     ties the free one, so the per-zone split was never identifiable, and
     zone 1's emitter had run to an 8 h time constant against its ceiling.

  3. f_im ceiling raised 0.6 -> 0.98. It pinned high, so the prior that the
     interior film is a small share of fabric resistance is contradicted.

The honest test is holdout RMSE, not cost. The open-loop score runs 12 steps
with no measurement updates, so it cannot be gamed by a large q_scale the way
the one-step likelihood can -- which is exactly why unclamping is safe to try.

    uv run scripts/03_refine.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.fit import at_bound, make_cost, optimise, unpack
from house.model import R4C3x2, R4C3x2_SHARED, derived, relax
from house.validate import open_loop

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAIN_FRAC = 0.8
NBINS = 6
BURN = 288
RESTARTS = 3
MAXITER = 400

Q_WIDE = (1e-14, 1e-4, 1e-9)
F_WIDE = (0.01, 0.98)


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    # reuse the train/holdout split from the first pass so the comparison is
    # like for like
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "ladder", Path(__file__).resolve().parent / "02_ladder.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    RESULTS.mkdir(exist_ok=True)
    rule("DATA")
    data = load_data(verbose=False)
    df = data.df
    train, hold, hold_mask = mod.split(data, TRAIN_FRAC)
    print(f"  train {len(train)} segs / {sum(b - a for a, b in train)} rows,"
          f"  hold {len(hold)} segs / {sum(b - a for a, b in hold)} rows")

    # pass-1 numbers, for a like-for-like comparison
    prev = {}
    p1 = RESULTS / "02_ladder.json"
    if p1.exists():
        prev = json.loads(p1.read_text())

    variants = [
        ("A free-q", relax(R4C3x2, "A free-q", q_scale=Q_WIDE)),
        ("B free-q+shared", relax(R4C3x2_SHARED, "B free-q+shared",
                                  q_scale=Q_WIDE)),
        ("C all three", relax(R4C3x2_SHARED, "C all three",
                              q_scale=Q_WIDE, f_im=F_WIDE)),
    ]

    rows, blob = [], {}
    for label, model in variants:
        rule(f"{label}   ({len(model.params)} params)")
        t0 = time.time()
        cost = make_cost(model, df, train, nbins=NBINS, burn=BURN)
        res = optimise(model, cost, restarts=RESTARTS, maxiter=MAXITER)
        p = unpack(model, res.x)
        secs = time.time() - t0

        hits = at_bound(model, p)
        print(f"    cost={res.fun:.5f}  ({secs:.0f}s)")
        print(f"    q_scale = {p['q_scale']:.3e}   "
              f"(implied process sd {np.sqrt(p['q_scale'] * 300):.4f} K/step "
              "vs 0.060 K measurement noise)")
        print(f"    at bounds ({len(hits)}/{len(model.params)}): "
              f"{', '.join(hits) if hits else 'none'}")
        d = derived(model, p)
        for k, val in d.items():
            print(f"      {k:<20s} {val:12.3f}")

        v, m = open_loop(model, p, df, hold, hold_mask, nbins=NBINS, burn=BURN)
        print(f"    holdout avg={m['rmse_avg']:.3f} K "
              f"(pers {m['pers_avg']:.3f}, skill {m['skill_avg']:+.1%})  "
              f"zone1={m['rmse_1']:.3f} zone2={m['rmse_2']:.3f}  "
              f"bias={m['bias_avg']:+.3f}")

        rows.append({"variant": label, "params": len(model.params),
                     "n_bound": len(hits), "q_scale": p["q_scale"],
                     "rmse_avg": m["rmse_avg"], "skill_avg": m["skill_avg"],
                     "rmse_1": m["rmse_1"], "rmse_2": m["rmse_2"],
                     "bias_avg": m["bias_avg"], "secs": secs})
        blob[label] = {"params": p, "derived": d, "metrics": m,
                       "at_bound": hits, "cost": float(res.fun)}

    rule("PASS 1 vs PASS 2  (holdout, 1-hour-ahead)")
    for name in ("R4C3x2", "R4C3x2s"):
        if name in prev:
            mm = prev[name]["metrics"]
            rows.insert(0, {
                "variant": f"pass1 {name}", "params": len(prev[name]["params"]),
                "n_bound": len(prev[name]["at_bound"]),
                "q_scale": prev[name]["params"]["q_scale"],
                "rmse_avg": mm["rmse_avg"], "skill_avg": mm["skill_avg"],
                "rmse_1": mm["rmse_1"], "rmse_2": mm["rmse_2"],
                "bias_avg": mm["bias_avg"], "secs": np.nan})

    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    t.to_csv(RESULTS / "03_refine.csv", index=False)
    with open(RESULTS / "03_refine.json", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '03_refine.csv'}")


if __name__ == "__main__":
    main()
