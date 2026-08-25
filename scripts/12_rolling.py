"""
Rolling-window refitting: does a model do better on recent data only?

For each day, fit on the previous N days and predict that day hour by hour.
Sweeping N turns this into a diagnostic rather than a single number:

  * accuracy IMPROVES as N shrinks -> the fitted parameters are not constant,
    so the model is standing in for physics it does not contain. Drift.
  * accuracy WORSENS as N shrinks -> parameters are stable and the only thing
    a short window costs is sample size. No mismatch.

The comparison point is the same model fitted once on everything (6-fold block
CV), which is the large-N limit.

The 4R3C is expensive: it needs a numerical fit per day rather than a
least-squares solve. Each day warm-starts from the previous day's parameters,
which is both how an online version would work and what keeps this tractable.
That does make the RC path slightly history-dependent, unlike the linear fits.

    uv run -u scripts/12_rolling.py            # linear models, N sweep
    uv run -u scripts/12_rolling.py --rc 20    # add the 4R3C at N=20
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
from house.forecast import forecast
from house.simple import Linear, design, hourly_frame, score_energy
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
DAY, H = 288, 12
N_SWEEP = [10, 20, 40, 80]
LINEAR = ["dT", "GHI", "gap1", "gap2", "prev", "to_6h"]
ABG = ["T_o", "wind_abg"]
MAXFEV_DAY = 300


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def frame(data):
    df = data.df
    cal = calibrate(df)
    to = df.T_o.to_numpy()
    h = hourly_frame(df, cal, segments=data.segments, burn=DAY)
    h["day"] = h.i // DAY
    h["wind_abg"] = (h.T_i - h.T_o) * h.v
    h["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in h.i]
    return h.reset_index(drop=True), cal


def rolling_linear(h, feats, n_days) -> pd.DataFrame:
    out = []
    for d in sorted(h.day.unique()):
        tr = h[(h.day >= d - n_days) & (h.day < d)]
        te = h[h.day == d]
        # need a reasonable amount of history and something to score
        if len(tr) < 12 * min(n_days, 5) or te.empty:
            continue
        m = Linear("roll", feats).fit(tr)
        rec = te[["t", "day", "actual", "T_o"]].copy()
        rec["pred"] = m.predict(te)
        out.append(rec)
    return pd.concat(out) if out else pd.DataFrame()


def clip(segments, lo, hi, min_len=DAY + H):
    """Segments intersected with [lo, hi). Keeps the filter off irrelevant data."""
    out = []
    for a, b in segments:
        s_, e_ = max(a, lo), min(b, hi)
        if e_ - s_ >= min_len:
            out.append((s_, e_))
    return out


def rolling_rc(data, h, cal, n_days) -> pd.DataFrame:
    de_spec = spec_from_file_location("de", HERE / "07_direct_energy.py")
    de = module_from_spec(de_spec)
    de_spec.loader.exec_module(de)
    model = de.MODEL
    n_p = len(model.params)
    df = data.df
    n = len(df)

    warm = json.loads((RESULTS / "07_direct_energy.json").read_text())
    x = np.r_[pack(model, {q.name: min(max(warm["params"][q.name], q.lo), q.hi)
                           for q in model.params}),
              np.log([float(warm["on_power"]["1"]),
                      float(warm["on_power"]["2"])])]
    bnds = bounds(model) + [(np.log(1000.0), np.log(16000.0))] * 2
    x = np.clip(x, [b[0] for b in bnds], [b[1] for b in bnds])

    def split_x(xx):
        p = unpack(model, xx[:n_p])
        pw = np.exp(xx[n_p:])
        return p, {1: replace(cal[1], on_power=float(pw[0])),
                   2: replace(cal[2], on_power=float(pw[1]))}

    rows = []
    days = sorted(h.day.unique())
    t0 = time.time()
    for k, d in enumerate(days):
        tr_mask = np.zeros(n, bool)
        tr_mask[max(0, (d - n_days) * DAY):d * DAY] = True
        te_mask = np.zeros(n, bool)
        te_mask[d * DAY:(d + 1) * DAY] = True
        if tr_mask.sum() < n_days * DAY * 0.5 or te_mask.sum() == 0:
            continue

        # Only run the filter over the window in question plus one burn-in
        # day. Passing the whole record would make every objective evaluation
        # a full-record pass, which is ~5x the work for identical results.
        tr_segs = clip(data.segments, max(0, (d - n_days) * DAY - DAY), d * DAY)
        te_segs = clip(data.segments, max(0, d * DAY - DAY), (d + 1) * DAY)
        if not tr_segs or not te_segs:
            continue

        def obj(xx, tr_segs=tr_segs):
            p, zones = split_x(xx)
            try:
                v = forecast(model, p, zones, df, tr_segs, tr_mask,
                             nbins=4, burn=DAY)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1e6
            if v.empty:
                return 1e6
            e = float((v.pred - v.actual).abs().mean())
            return e if np.isfinite(e) else 1e6

        res = minimize(obj, x, method="Powell", bounds=bnds,
                       options={"maxfev": MAXFEV_DAY, "xtol": 1e-2,
                                "ftol": 1e-3})
        x = res.x                      # warm start tomorrow from today
        p, zones = split_x(x)
        v = forecast(model, p, zones, df, te_segs, te_mask,
                     nbins=4, burn=DAY)
        if not v.empty:
            rec = v[["t", "pred", "actual", "T_o"]].copy()
            rec["day"] = d
            rows.append(rec)
        if k % 10 == 0:
            print(f"    day {k}/{len(days)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    return pd.concat(rows) if rows else pd.DataFrame()


def main() -> None:
    data = load_data(verbose=False)
    h, cal = frame(data)
    rule("ROLLING WINDOW")
    print(f"  {len(h)} hourly windows over {h.day.nunique()} days")

    rows = []
    for name, feats in (("linear", LINEAR), ("alpha/beta/gamma", ABG)):
        for nd in N_SWEEP:
            v = rolling_linear(h, feats, nd)
            s = score_energy(v.pred, v.actual)
            rows.append({"model": name, "window_days": nd, **s})
            print(f"  {name:<17s} N={nd:<3d}  MAE {s['mae_kwh']:.3f}  "
                  f"r {s['r']:.3f}  n={s['n']}", flush=True)
            v.to_csv(RESULTS / f"12_roll_{name.split('/')[0]}_{nd}.csv",
                     index=False)

    if "--rc" in sys.argv:
        nd = int(sys.argv[sys.argv.index("--rc") + 1])
        rule(f"4R3C, rolling N={nd} (slow)")
        v = rolling_rc(data, h, cal, nd)
        if not v.empty:
            s = score_energy(v.pred, v.actual)
            rows.append({"model": "4R3C", "window_days": nd, **s})
            print(f"  4R3C N={nd}  MAE {s['mae_kwh']:.3f}  r {s['r']:.3f}  "
                  f"n={s['n']}")
            v.to_csv(RESULTS / f"12_roll_4R3C_{nd}.csv", index=False)

    t = pd.DataFrame(rows)
    rule("SUMMARY")
    print(t.pivot_table(index="model", columns="window_days",
                        values="mae_kwh").to_string(
        float_format=lambda x: f"{x:.3f}"))
    print("\n  reference, fitted once on everything (6-fold block CV):")
    print("    linear 0.674-0.676   alpha/beta/gamma 0.820   4R3C 0.772")
    t.to_csv(RESULTS / "12_rolling.csv", index=False)
    print(f"\n  -> {RESULTS / '12_rolling.csv'}", flush=True)


if __name__ == "__main__":
    main()
