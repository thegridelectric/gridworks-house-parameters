"""
Block cross-validation over the whole winter, not one February block.

Why blocks and not random rows: consecutive hours are strongly correlated and
one feature IS the previous hour's energy, so a random hour-level split puts
near-duplicates of every test row into training. That inflates every model and
flatters the ones that lean hardest on `prev`. Week-long blocks dealt
round-robin into folds keep train and test far apart in time while still giving
every fold a mix of November mild and January cold.

The alpha/beta/gamma comparison uses the FORM

    Q = a + b * T_o + c * (T_i - T_o) * v

refitted on each fold's training data. The deployed coefficients are not used:
they were set for design-day sizing, so scoring them measures how they were
chosen, not whether the model form is any good.

    uv run -u scripts/10_crossval.py            # simple models only (fast)
    uv run -u scripts/10_crossval.py --with-rc  # adds the 4R3C, several minutes
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
from house.simple import FEATURES, Linear, design, hourly_frame, score_energy
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
BLOCK = 2016            # 7 days at 5-min resolution
K = 6
HORIZON = 12
MAXFEV_FOLD = 1500

# your model form, refitted per fold
ABG = ["T_o", "wind_abg"]


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def fold_of(row_index: np.ndarray) -> np.ndarray:
    return (row_index // BLOCK) % K


def add_abg(h: pd.DataFrame) -> pd.DataFrame:
    """Features for Q = a + b*T_o + c*(T_i - T_o)*v, in the original form."""
    h = h.copy()
    h["wind_abg"] = (h.T_i - h.T_o) * h.v
    return h


def main() -> None:
    with_rc = "--with-rc" in sys.argv
    data = load_data(verbose=False)
    df = data.df
    cal = calibrate(df)

    h = add_abg(hourly_frame(df, cal, segments=data.segments, burn=288))
    # drop windows that straddle a block boundary, so no window is half in
    # train and half in test
    same_block = (h.i // BLOCK) == ((h.i + HORIZON - 1) // BLOCK)
    h = h[same_block].reset_index(drop=True)
    h["fold"] = fold_of(h.i.to_numpy())

    rule("BLOCK CROSS-VALIDATION")
    print(f"  {len(h)} hourly windows, {BLOCK // 288}-day blocks, {K} folds")
    print(f"  {'fold':>4s} {'n':>5s} {'first':>12s} {'last':>12s} "
          f"{'mean kWh':>9s} {'mean T_o':>9s}")
    for f, g in h.groupby("fold"):
        print(f"  {f:4d} {len(g):5d} {g.t.min():%Y-%m-%d} {g.t.max():%Y-%m-%d} "
              f"{g.actual.mean():9.2f} {g.T_o.mean():9.1f}")

    specs = dict(FEATURES)
    specs.pop("persistence", None)
    specs["alpha/beta/gamma (refitted)"] = ABG

    rows = []
    for f in range(K):
        tr, te = h[h.fold != f], h[h.fold == f]
        rows.append({"fold": f, "model": "persistence",
                     "mae": score_energy(te.prev, te.actual)["mae_kwh"],
                     "r": score_energy(te.prev, te.actual)["r"]})
        for name, feats in specs.items():
            m = Linear(name, feats).fit(tr)
            s = score_energy(m.predict(te), te.actual)
            rows.append({"fold": f, "model": name, "mae": s["mae_kwh"],
                         "r": s["r"]})

    if with_rc:
        rows += rc_folds(data, df, cal, h)

    t = pd.DataFrame(rows)
    summary = (t.groupby("model")
                 .agg(mean_mae=("mae", "mean"), sd_mae=("mae", "std"),
                      worst=("mae", "max"), mean_r=("r", "mean"))
                 .sort_values("mean_mae"))
    rule(f"RESULTS  ({K}-fold, mean actual {h.actual.mean():.2f} kWh)")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n  per-fold MAE:")
    piv = t.pivot_table(index="model", columns="fold", values="mae")
    print(piv.reindex(summary.index).to_string(
        float_format=lambda x: f"{x:.3f}"))

    t.to_csv(RESULTS / "10_crossval_folds.csv", index=False)
    summary.to_csv(RESULTS / "10_crossval.csv")
    print(f"\n  -> {RESULTS / '10_crossval.csv'}", flush=True)


def rc_folds(data, df, cal, h) -> list:
    """Refit the energy-optimised 4R3C once per fold. Slow but honest."""
    de_spec = spec_from_file_location("de", HERE / "07_direct_energy.py")
    de = module_from_spec(de_spec)
    de_spec.loader.exec_module(de)
    model = de.MODEL
    n_p = len(model.params)

    warm = json.loads((RESULTS / "07_direct_energy.json").read_text())
    x_warm = np.r_[
        pack(model, {q.name: min(max(warm["params"][q.name], q.lo), q.hi)
                     for q in model.params}),
        np.log([float(warm["on_power"]["1"]), float(warm["on_power"]["2"])])]
    bnds = bounds(model) + [(np.log(1000.0), np.log(16000.0))] * 2
    x_warm = np.clip(x_warm, [b[0] for b in bnds], [b[1] for b in bnds])

    n = len(df)
    rows = []
    for f in range(K):
        t0 = time.time()
        blocks = (np.arange(n) // BLOCK) % K
        m_te = blocks == f
        m_tr = ~m_te

        def split_x(x):
            p = unpack(model, x[:n_p])
            pw = np.exp(x[n_p:])
            return p, {1: replace(cal[1], on_power=float(pw[0])),
                       2: replace(cal[2], on_power=float(pw[1]))}

        def obj(x, mask=m_tr):
            p, zones = split_x(x)
            try:
                v = forecast(model, p, zones, df, data.segments, mask,
                             nbins=6, burn=288)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return 1e6
            if v.empty:
                return 1e6
            e = float((v.pred - v.actual).abs().mean())
            return e if np.isfinite(e) else 1e6

        res = minimize(obj, x_warm, method="Powell", bounds=bnds,
                       options={"maxfev": MAXFEV_FOLD, "xtol": 1e-3,
                                "ftol": 1e-4})
        p, zones = split_x(res.x)
        v_te = forecast(model, p, zones, df, data.segments, m_te,
                        nbins=6, burn=288).set_index("t")
        s = score_energy(v_te.pred, v_te.actual)
        rows.append({"fold": f, "model": "4R3C (energy-fitted)",
                     "mae": s["mae_kwh"], "r": s["r"]})

        # blend with the best simple model, weight chosen on TRAIN only
        tr, te = h[h.fold != f], h[h.fold == f]
        lin = Linear("per-zone", FEATURES["per-zone dT"]).fit(tr)
        v_tr = forecast(model, p, zones, df, data.segments, m_tr,
                        nbins=6, burn=288).set_index("t")
        jt = pd.DataFrame({"actual": v_tr.actual, "rc": v_tr.pred}).join(
            pd.Series(lin.predict(tr), index=tr.t, name="simple"),
            how="inner").dropna()
        w = min(np.arange(0, 1.01, 0.1),
                key=lambda ww: score_energy(ww * jt.rc + (1 - ww) * jt.simple,
                                            jt.actual)["mae_kwh"])
        je = pd.DataFrame({"actual": v_te.actual, "rc": v_te.pred}).join(
            pd.Series(lin.predict(te), index=te.t, name="simple"),
            how="inner").dropna()
        sb = score_energy(w * je.rc + (1 - w) * je.simple, je.actual)
        rows.append({"fold": f, "model": f"blend RC+linear", "mae": sb["mae_kwh"],
                     "r": sb["r"]})
        print(f"  fold {f}: RC {s['mae_kwh']:.3f}   blend(w={w:.1f}) "
              f"{sb['mae_kwh']:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    main()
