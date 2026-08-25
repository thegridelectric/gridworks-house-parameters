"""
PRIORITY 2: the model ladder.

Fits every rung on the same training data and scores every rung on the same
holdout with the same metric, so each structural claim can be judged on its
own. Answers "is 4R3C a good idea" rather than "does this particular 4R3C fit
look plausible".

    uv run scripts/02_ladder.py
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
from house.model import LADDER, derived
from house.validate import open_loop

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRAIN_FRAC = 0.8
NBINS = 6            # wind p95 is only 1.25 m/s, so g1 is weak regardless
BURN = 288
RESTARTS = 3
MAXITER = 400


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def split(data, frac):
    """Assign whole segments to train or holdout, keeping them contiguous."""
    n = len(data.df)
    cut = int(frac * n)
    train, hold = [], []
    for a, b in data.segments:
        if b <= cut:
            train.append((a, b))
        elif a >= cut:
            hold.append((a, b))
        else:                       # segment straddles the split, cut it
            if cut - a >= 288:
                train.append((a, cut))
            if b - cut >= 288:
                hold.append((cut, b))
    mask = np.zeros(n, bool)
    for a, b in hold:
        mask[a:b] = True
    return train, hold, mask


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    rule("DATA")
    data = load_data(verbose=True)
    df = data.df
    train, hold, hold_mask = split(data, TRAIN_FRAC)
    print(f"\n  train: {len(train)} segments, {sum(b - a for a, b in train)} rows")
    print(f"  hold : {len(hold)} segments, {sum(b - a for a, b in hold)} rows")

    table, blob = [], {}
    for model in LADDER:
        rule(f"RUNG {model.rung}: {model.name}  ({model.note})")
        print(f"  states={model.n}  params={len(model.params)}")

        t0 = time.time()
        cost = make_cost(model, df, train, nbins=NBINS, burn=BURN)
        res = optimise(model, cost, restarts=RESTARTS, maxiter=MAXITER)
        p = unpack(model, res.x)
        secs = time.time() - t0

        hits = at_bound(model, p)
        print(f"    cost={res.fun:.5f}  ({secs:.0f}s)")
        print(f"    at bounds: {', '.join(hits) if hits else 'none'}")

        print("    derived:")
        d = derived(model, p)
        for k, val in d.items():
            print(f"      {k:<20s} {val:12.3f}")

        v, m = open_loop(model, p, df, hold, hold_mask,
                         nbins=NBINS, burn=BURN)
        if m:
            print(f"    holdout 1h RMSE  avg={m['rmse_avg']:.3f} K  "
                  f"(persistence {m['pers_avg']:.3f})  skill={m['skill_avg']:+.1%}")
            if model.n_zones == 2:
                print(f"      zone1 {m['rmse_1']:.3f} vs {m['pers_1']:.3f} "
                      f"(skill {m['skill_1']:+.1%}), "
                      f"zone2 {m['rmse_2']:.3f} vs {m['pers_2']:.3f} "
                      f"(skill {m['skill_2']:+.1%})")
            print(f"      bias avg={m['bias_avg']:+.3f} K")

        table.append({
            "rung": model.rung, "model": model.name,
            "states": model.n, "params": len(model.params),
            "cost": float(res.fun), "n_bound": len(hits),
            "rmse_avg": m.get("rmse_avg", np.nan),
            "skill_avg": m.get("skill_avg", np.nan),
            "rmse_1": m.get("rmse_1", np.nan),
            "rmse_2": m.get("rmse_2", np.nan),
            "bias_avg": m.get("bias_avg", np.nan),
            "secs": secs,
        })
        blob[model.name] = {"params": p, "derived": d, "metrics": m,
                            "at_bound": hits, "cost": float(res.fun)}

    rule("LADDER SUMMARY  (holdout, 1-hour-ahead T_i_avg)")
    t = pd.DataFrame(table)
    # persistence is model-independent, so any rung's copy of it will do
    pers = next((b["metrics"].get("pers_avg") for b in blob.values()
                 if b["metrics"]), float("nan"))
    print(f"  persistence baseline RMSE = {pers:.3f} K\n")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n  Reading the ladder: a rung earns its complexity only if it "
          "beats the rung below.")
    best = t.loc[t.rmse_avg.idxmin()]
    print(f"  Best: rung {int(best.rung)} ({best.model}) at "
          f"{best.rmse_avg:.3f} K, skill {best.skill_avg:+.1%}")

    t.to_csv(RESULTS / "02_ladder.csv", index=False)
    with open(RESULTS / "02_ladder.json", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '02_ladder.csv'}")
    print(f"  -> {RESULTS / '02_ladder.json'}")


if __name__ == "__main__":
    main()
