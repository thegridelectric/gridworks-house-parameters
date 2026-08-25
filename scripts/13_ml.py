"""
Does a machine-learning model beat the linear one on hour-ahead Q_dist?

Shape of the problem: ~2450 hourly samples, ~12 features, physically monotone
relationships, and a measured noise floor around 0.53 kWh. That is small-data,
low-dimensional, high-signal tabular regression. Gradient boosting is the right
default for it; a neural network is not, at this sample size.

Two things the incumbent linear model leaves on the table:

  * It is fitted by least squares, which minimises squared error, while we
    score MAE. Fitting for the metric is free accuracy.
  * It cannot express interactions. The setpoint gap plausibly matters more in
    cold weather than mild, and a tree can represent that while a plane cannot.

Honest protocol, matching everything before it: 6-fold block cross-validation
over week-blocks dealt round-robin. Hyperparameters are chosen by an INNER
split of each outer fold's training data, never on the fold being scored --
otherwise the flexible models get to peek and the comparison is rigged in
their favour.

    uv run -u scripts/13_ml.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import QuantileRegressor, Ridge

from house.data import load_data
from house.simple import Linear, hourly_frame, score_energy
from house.thermostat import calibrate

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parents[1] / "results"
BLOCK, K, H, DAY = 2016, 6, 12, 288

# what the linear model uses
ENG = ["dT", "GHI", "gap1", "gap2", "prev", "to_6h"]
# everything a tree could plausibly want, including the raw pieces
RAW = ["T_i1", "T_i2", "T_o", "v", "GHI", "gap1", "gap2", "prev", "to_6h",
       "call", "hour", "dT"]


def build():
    data = load_data(verbose=False)
    df = data.df
    cal = calibrate(df)
    to = df.T_o.to_numpy()
    h = hourly_frame(df, cal, segments=data.segments, burn=DAY)
    h = h[(h.i // BLOCK) == ((h.i + H - 1) // BLOCK)].reset_index(drop=True)
    h["fold"] = (h.i // BLOCK) % K
    h["to_6h"] = [np.nanmean(to[max(0, j - 72):j]) for j in h.i]
    h["hour"] = pd.to_datetime(h.t).dt.hour
    return h


def gbm(**kw):
    base = dict(loss="absolute_error", learning_rate=0.05, max_iter=400,
                early_stopping=False, random_state=0)
    base.update(kw)
    return HistGradientBoostingRegressor(**base)


# small grids; the inner split picks among them per outer fold
GRIDS = {
    "gradient boosting (eng)": (ENG, [
        gbm(max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=1.0),
        gbm(max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.1),
        gbm(max_leaf_nodes=31, min_samples_leaf=10, l2_regularization=0.0),
    ]),
    "gradient boosting (raw)": (RAW, [
        gbm(max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=1.0),
        gbm(max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.1),
        gbm(max_leaf_nodes=31, min_samples_leaf=10, l2_regularization=0.0),
    ]),
    "random forest (raw)": (RAW, [
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                              max_features=0.5, random_state=0, n_jobs=-1),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=15,
                              max_features=0.7, random_state=0, n_jobs=-1),
    ]),
    "ridge (eng)": (ENG, [Ridge(alpha=a) for a in (0.1, 1.0, 10.0)]),
    "median regression (eng)": (ENG, [
        QuantileRegressor(quantile=0.5, alpha=a, solver="highs")
        for a in (1e-4, 1e-2)]),
}


def fit_predict(model, tr, te, feats):
    m = model.__class__(**model.get_params())
    m.fit(tr[feats].to_numpy(), tr.actual.to_numpy())
    return np.clip(m.predict(te[feats].to_numpy()), 0.0, None)


def run(h) -> pd.DataFrame:
    rows = []

    # the incumbent, same protocol
    for k in range(K):
        tr, te = h[h.fold != k], h[h.fold == k]
        p = Linear("lin", ENG).fit(tr).predict(te)
        rows.append({"model": "linear (OLS)", "fold": k,
                     **score_energy(p, te.actual)})

    for name, (feats, candidates) in GRIDS.items():
        t0 = time.time()
        for k in range(K):
            tr, te = h[h.fold != k], h[h.fold == k]
            # inner split: hold out the training folds' last block-group
            inner_folds = sorted(tr.fold.unique())
            hold = inner_folds[-1]
            itr, ite = tr[tr.fold != hold], tr[tr.fold == hold]
            best = min(candidates,
                       key=lambda c: score_energy(
                           fit_predict(c, itr, ite, feats),
                           ite.actual)["mae_kwh"])
            p = fit_predict(best, tr, te, feats)
            rows.append({"model": name, "fold": k,
                         **score_energy(p, te.actual)})
        print(f"  {name:<26s} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    h = build()
    print(f"  {len(h)} hourly windows, {len(RAW)} candidate features\n")
    t = run(h)

    summary = (t.groupby("model")
                 .agg(mean_mae=("mae_kwh", "mean"), sd=("mae_kwh", "std"),
                      worst=("mae_kwh", "max"), mean_r=("r", "mean"))
                 .sort_values("mean_mae"))
    print("\n" + "=" * 78)
    print(f"6-FOLD BLOCK CV  (mean actual {h.actual.mean():.2f} kWh)")
    print("=" * 78)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n  per-fold MAE:")
    print(t.pivot_table(index="model", columns="fold", values="mae_kwh")
           .reindex(summary.index).to_string(
               float_format=lambda x: f"{x:.3f}"))

    base = summary.loc["linear (OLS)", "mean_mae"]
    print(f"\n  vs linear (OLS) = {base:.4f}:")
    for name, row in summary.iterrows():
        if name == "linear (OLS)":
            continue
        d = row.mean_mae - base
        print(f"    {name:<26s} {d:+.4f} kWh  ({100 * d / base:+.1f}%)")

    t.to_csv(RESULTS / "13_ml_folds.csv", index=False)
    summary.to_csv(RESULTS / "13_ml.csv")
    print(f"\n  -> {RESULTS / '13_ml.csv'}")


if __name__ == "__main__":
    main()
