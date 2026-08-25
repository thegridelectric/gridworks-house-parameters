"""
PRIORITY 3: score the hour-ahead loop-energy forecast. The MPC deliverable.

    uv run scripts/05_qdist.py
"""

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.forecast import forecast, score
from house.model import R4C3x2
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    spec = spec_from_file_location("ladder", HERE / "02_ladder.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = load_data(verbose=False)
    df = data.df
    train, hold, hold_mask = mod.split(data, 0.8)

    # pass 1 parameters: the physically sound fit, per the horizon test
    blob = json.loads((RESULTS / "02_ladder.json").read_text())
    p = blob["R4C3x2"]["params"]

    rule("SETUP")
    print(f"  model     : R4C3x2, pass-1 parameters (q_scale={p['q_scale']:.1e})")
    print(f"  holdout   : {len(hold)} segment(s), "
          f"{sum(b - a for a, b in hold)} rows")

    # thermostat calibrated on TRAINING data only, so the score stays honest
    train_rows = np.zeros(len(df), bool)
    for a, b in train:
        train_rows[a:b] = True
    cal = calibrate(df[train_rows].reset_index(drop=True), lam=p["lam"])
    print("  thermostat: calibrated on training rows only")
    for z in (1, 2):
        print("    " + str(cal[z]))

    v = forecast(R4C3x2, p, cal, df, hold, hold_mask)
    print(f"  windows   : {len(v)}")

    rule("HOUR-AHEAD LOOP ENERGY  [kWh]")
    methods = {
        "model + thermostat": "pred",
        "persistence (last hour)": "persistence",
        "steady-state UA line": "steady_state",
    }
    print(f"  {'method':<26s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} "
          f"{'MAPE':>7s} {'r':>6s} {'n':>5s}")
    results = {}
    for label, col in methods.items():
        s = score(v, col)
        results[label] = s
        if s:
            print(f"  {label:<26s} {s['mae_kwh']:7.3f} {s['rmse_kwh']:7.3f} "
                  f"{s['bias_kwh']:+7.3f} {s['mape_pct']:7.1f} "
                  f"{s['r']:6.3f} {s['n']:5d}")
    base = score(v)
    print(f"\n  mean actual hourly energy: {base['mean_actual']:.3f} kWh")
    print(f"  total over holdout: predicted {base['total_pred']:.0f} kWh "
          f"vs actual {base['total_actual']:.0f} kWh "
          f"({100 * (base['total_pred'] / base['total_actual'] - 1):+.1f}%)")

    rule("WHERE IT GOES WRONG")
    v2 = v.dropna(subset=["pred", "actual"]).copy()
    v2["err"] = v2.pred - v2.actual
    v2["state"] = np.select(
        [v2.call1 & v2.call2, v2.call1, v2.call2],
        ["both on", "zone 1 on", "zone 2 on"], default="both off")
    print(f"  {'at hour start':<14s} {'n':>5s} {'mean actual':>12s} "
          f"{'MAE':>7s} {'bias':>8s}")
    for st, sub in v2.groupby("state"):
        print(f"  {st:<14s} {len(sub):5d} {sub.actual.mean():12.3f} "
              f"{sub.err.abs().mean():7.3f} {sub.err.mean():+8.3f}")

    print(f"\n  {'outdoor temp':<14s} {'n':>5s} {'mean actual':>12s} "
          f"{'MAE':>7s} {'bias':>8s}")
    v2["bin"] = pd.cut(v2.T_o, bins=[-30, -15, -10, -5, 0, 5, 15])
    for bn, sub in v2.groupby("bin", observed=True):
        if len(sub) < 10:
            continue
        print(f"  {str(bn):<14s} {len(sub):5d} {sub.actual.mean():12.3f} "
              f"{sub.err.abs().mean():7.3f} {sub.err.mean():+8.3f}")

    _plot(v2, RESULTS / "05_qdist.png")
    v.to_csv(RESULTS / "05_qdist.csv", index=False)
    with open(RESULTS / "05_qdist.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n  -> {RESULTS / '05_qdist.csv'}")


def _plot(v, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    hi = max(v.actual.max(), v.pred.max())
    a.plot([0, hi], [0, hi], "k--", lw=1)
    a.scatter(v.actual, v.pred, s=14, alpha=0.5)
    a.set_xlabel("actual next-hour energy [kWh]")
    a.set_ylabel("predicted [kWh]")
    a.set_title(f"Model + thermostat (r={np.corrcoef(v.pred, v.actual)[0,1]:.3f})")
    a.grid(alpha=0.3)

    a = ax[0, 1]
    d = v.dropna(subset=["persistence"])
    a.plot([0, hi], [0, hi], "k--", lw=1)
    a.scatter(d.actual, d.persistence, s=14, alpha=0.5, color="C1")
    a.set_xlabel("actual [kWh]")
    a.set_ylabel("persistence [kWh]")
    a.set_title("Baseline: last hour repeated")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(v.t, v.actual, lw=0.9, label="actual")
    a.plot(v.t, v.pred, lw=0.9, label="forecast")
    a.set_ylabel("hourly energy [kWh]")
    a.set_title("Holdout period")
    a.tick_params(axis="x", rotation=30)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    a.hist(v.err, bins=50)
    a.axvline(0, color="k", lw=0.8)
    a.axvline(v.err.mean(), color="C3", ls="--",
              label=f"bias {v.err.mean():+.3f} kWh")
    a.set_xlabel("predicted minus actual [kWh]")
    a.set_title("Forecast error")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")


if __name__ == "__main__":
    main()
