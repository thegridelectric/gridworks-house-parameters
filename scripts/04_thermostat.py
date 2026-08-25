"""
Calibrate the thermostat model from the recorded heat calls.

    uv run scripts/04_thermostat.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import DT, load_data
from house.thermostat import (calibrate, call_status,
                              constant_setpoint_periods, edges)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = load_data(verbose=False)
    df = data.df

    z1, z2 = call_status(df)
    rule("HEAT CALLS")
    print(f"  zone 1 calling {z1.mean():.1%} of the time, "
          f"zone 2 {z2.mean():.1%}, both {np.mean(z1 & z2):.1%}, "
          f"neither {np.mean(~z1 & ~z2):.1%}")

    cal = calibrate(df)
    rule("CALIBRATED BANDS")
    for z in (1, 2):
        print("  " + str(cal[z]))

    rule("HOW SHARP ARE THE SWITCHING EDGES?")
    print("  Scatter of T_i at switch-on, measured inside periods where the")
    print("  reported setpoint is constant. Read from T_i alone: the reported")
    print("  setpoint comes from the thermostat's own sensor, which is a")
    print("  different instrument and cannot be subtracted from T_i.\n")
    for z in (1, 2):
        c = cal[z]
        print(f"  zone {z}: +/-{c.on_sd:.3f} K   deadband {c.deadband:.3f} K   "
              f"({c.n_periods} constant-setpoint periods)")

    rule("OVERSHOOT AFTER THE CALL STOPS")
    print("  Real rooms keep warming after heat is cut, because the emitters")
    print("  are still hot. Under-modelling this under-predicts energy.\n")
    for z, call in ((1, z1), (2, z2)):
        _, off_idx = edges(call)
        t_i = df[f"T_i{z}"].to_numpy()
        peaks = []
        for i in off_idx:
            j = min(i + 36, len(t_i))          # look 3 h past the cut
            if j > i:
                peaks.append(t_i[i:j].max() - t_i[max(i - 1, 0)])
        peaks = np.array(peaks)
        if peaks.size:
            print(f"  zone {z}: median {np.median(peaks):+.2f} K, "
                  f"p90 {np.quantile(peaks, 0.9):+.2f} K, "
                  f"max {peaks.max():+.2f} K  ({len(peaks)} events)")

    rule("ON-POWER STABILITY")
    print("  The controller assumes one power level while calling. If the")
    print("  actual power varies a lot, that assumption costs accuracy.\n")
    q1o = df.Q_dist1_only.to_numpy()
    q2o = df.Q_dist2_only.to_numpy()
    qb = df.Q_dist_together.to_numpy()
    for z, series, mask in ((1, q1o, z1 & ~z2), (2, q2o, z2 & ~z1)):
        v = series[mask]
        v = v[v > 1.0]
        if v.size:
            print(f"  zone {z} alone: median {np.median(v):.0f} W, "
                  f"IQR {np.quantile(v, .25):.0f}..{np.quantile(v, .75):.0f}, "
                  f"p95 {np.quantile(v, .95):.0f} W  (n={v.size})")
    vb = qb[qb > 1.0]
    if vb.size:
        print(f"  both zones   : median {np.median(vb):.0f} W, "
              f"IQR {np.quantile(vb, .25):.0f}..{np.quantile(vb, .75):.0f} "
              f"(n={vb.size})")

    rule("SETPOINT MAPPING")
    for z in (1, 2):
        c = cal[z]
        print(f"  zone {z}: T_i threshold = {c.slope:.3f} x reported "
              f"{c.intercept:+.2f} C   (r = {c.r:.4f})")
    print("\n  The offset is the difference between the thermostat's internal")
    print("  sensor and the T_i logger. Near-perfect correlation means the")
    print("  reported setpoint is usable once mapped.")

    out = {str(z): {k: (float(v) if isinstance(v, (int, float, np.floating))
                        else v)
                    for k, v in vars(cal[z]).items()} for z in (1, 2)}
    with open(RESULTS / "04_thermostat.json", "w") as fh:
        json.dump(out, fh, indent=2)
    _plot(df, cal, z1, z2, RESULTS / "04_thermostat.png")
    print(f"\n  -> {RESULTS / '04_thermostat.json'}")


def _plot(df, cal, z1, z2, path):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for r, (z, call) in enumerate(((1, z1), (2, z2))):
        t_i = df[f"T_i{z}"].to_numpy()
        sp = df[f"T_i{z}_set"].to_numpy()
        grp = constant_setpoint_periods(sp)
        on_idx, off_idx = edges(call)

        # biggest constant-setpoint period, so one band is visible cleanly
        sizes = np.bincount(grp)
        g = int(np.argmax(sizes))
        rows = np.flatnonzero(grp == g)
        a, b = rows[0], rows[-1]
        on = on_idx[(on_idx >= a) & (on_idx <= b)]
        off = off_idx[(off_idx >= a) & (off_idx <= b)]

        aa = ax[r, 0]
        if len(on) and len(off):
            lo = min(t_i[on - 1].min(), t_i[off - 1].min())
            hi = max(t_i[on - 1].max(), t_i[off - 1].max())
            bins = np.linspace(lo, hi, 40)
            aa.hist(t_i[np.maximum(on - 1, 0)], bins=bins, alpha=0.7, label="switch on")
            aa.hist(t_i[np.maximum(off - 1, 0)], bins=bins, alpha=0.7, label="switch off")
        aa.set_xlabel("room temperature  [C]")
        aa.set_title(f"Zone {z}: switching, one setpoint period "
                     f"(reported {sp[a]:.2f})")
        aa.legend(fontsize=8)
        aa.grid(alpha=0.3)

        aa = ax[r, 1]
        rep, mid = [], []
        for gg in range(grp.max() + 1):
            rr = np.flatnonzero(grp == gg)
            if len(rr) < 288:
                continue
            o = on_idx[(on_idx >= rr[0]) & (on_idx <= rr[-1])]
            f = off_idx[(off_idx >= rr[0]) & (off_idx <= rr[-1])]
            if len(o) < 8 or len(f) < 8:
                continue
            rep.append(sp[rr[0]])
            mid.append(0.5 * (np.median(t_i[o - 1]) + np.median(t_i[f - 1])))
        aa.scatter(rep, mid, s=40)
        if rep:
            xs = np.linspace(min(rep), max(rep), 10)
            aa.plot(xs, cal[z].threshold(xs), "C3-",
                    label=f"slope {cal[z].slope:.3f}, r={cal[z].r:.4f}")
            aa.legend(fontsize=8)
        aa.set_xlabel("reported setpoint  [C]")
        aa.set_ylabel("T_i threshold  [C]")
        aa.set_title(f"Zone {z}: thermostat sensor vs T_i logger")
        aa.grid(alpha=0.3)

        aa = ax[r, 2]
        if len(on_idx) > 1:
            aa.hist(np.diff(on_idx) * DT / 60.0, bins=np.linspace(0, 400, 60))
            aa.axvline(cal[z].cycle_min, color="C3", ls="--",
                       label=f"median {cal[z].cycle_min:.0f} min")
            aa.legend(fontsize=8)
        aa.set_xlabel("time between calls  [min]")
        aa.set_title(f"Zone {z}: cycle length")
        aa.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")


if __name__ == "__main__":
    main()
