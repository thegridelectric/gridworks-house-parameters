"""
Simple hour-ahead energy predictors, fitted properly.

The 4R3C model has to beat a well-built simple model, not a strawman. These
are all fitted on the same rows and scored on the same windows as the RC model,
so the comparison is like for like.

The ladder runs from "repeat the last hour" up to a linear model with the few
features that plausibly matter. Everything here is a closed-form least-squares
fit, deployable as a handful of coefficients.

Feature notes:
  * The setpoint gap uses the thermostat calibration, because the reported
    setpoint is on the thermostat's own sensor scale and must be mapped into
    T_i units before it can be compared with T_i.
  * dT is clipped at zero. Below the balance point the house needs no heat, and
    letting the term go negative lets sunny mild hours pull the fit around.
  * Last hour's energy carries the emitter state indirectly: a loop that was
    running is already hot, which is the effect the emitter node exists to
    capture.
"""

import numpy as np
import pandas as pd

HORIZON = 12
J_TO_KWH = 300.0 / 3.6e6


def hourly_frame(df: pd.DataFrame, cal=None, horizon=HORIZON,
                 segments=None, burn=0) -> pd.DataFrame:
    """
    One row per hour boundary: what is known at the boundary, plus what the
    next hour actually consumed. Weather over the coming hour is taken as
    perfectly forecast, matching how the RC model is scored.

    Windows are confined to a single contiguous segment. A window straddling a
    weather gap would average over missing outdoor temperature and silently
    produce NaN features, so segments are respected rather than cleaned up
    afterwards.

    `segments` and `burn` must match whatever the RC forecast was given, or the
    two land on different window grids and share no timestamps at all: both
    step by `horizon` from their segment start, so any offset between the two
    starts that is not a multiple of `horizon` makes the sets disjoint.
    """
    n = len(df)
    if segments is None:
        segments = [(0, n)]
    q = df["Q_total"].to_numpy()
    t_o = df["T_o"].to_numpy()
    v = df["v"].to_numpy()
    ghi = df["GHI"].to_numpy()
    t1, t2 = df["T_i1"].to_numpy(), df["T_i2"].to_numpy()
    s1, s2 = df["T_i1_set"].to_numpy(), df["T_i2_set"].to_numpy()

    starts = []
    for a, b in segments:
        first = a + max(burn, horizon)
        starts.extend(range(first, b - horizon, horizon))

    rows = []
    for g in starts:
        nxt = slice(g, g + horizon)
        prv = slice(g - horizon, g)
        r = {
            "i": g,
            "t": df["timestamp"].iloc[g],
            "actual": q[nxt].sum() * J_TO_KWH,
            "prev": q[prv].sum() * J_TO_KWH,
            "T_i1": t1[g], "T_i2": t2[g],
            "T_i": 0.5 * (t1[g] + t2[g]),
            "T_o": t_o[nxt].mean(),
            "v": v[nxt].mean(),
            "GHI": ghi[nxt].mean(),
            "call": float(q[g - 1] > 1.0),
        }
        r["dT"] = max(r["T_i"] - r["T_o"], 0.0)
        r["dT1"] = max(t1[g] - r["T_o"], 0.0)
        r["dT2"] = max(t2[g] - r["T_o"], 0.0)
        hod = df["timestamp"].iloc[g].hour
        r["sin_h"] = np.sin(2 * np.pi * hod / 24)
        r["cos_h"] = np.cos(2 * np.pi * hod / 24)
        if cal is not None:
            # how far each room sits below its switching threshold
            r["gap1"] = cal[1].threshold(s1[g]) - t1[g]
            r["gap2"] = cal[2].threshold(s2[g]) - t2[g]
        else:
            r["gap1"] = r["gap2"] = 0.0
        rows.append(r)

    out = pd.DataFrame(rows)
    cols = ["actual", "prev", "dT", "dT1", "dT2", "T_o", "v", "GHI",
            "gap1", "gap2"]
    return out.dropna(subset=[c for c in cols if c in out.columns])


FEATURES = {
    "persistence": None,                       # special-cased
    "UA line": ["dT"],
    "UA + wind + sun": ["dT", "wind", "GHI"],
    "UA + wind + sun + setpoint gap": ["dT", "wind", "GHI", "gap1", "gap2"],
    "full linear": ["dT", "wind", "GHI", "gap1", "gap2", "prev", "call"],
    "per-zone dT": ["dT1", "dT2", "wind", "GHI", "gap1", "gap2", "prev",
                    "call"],
    "per-zone + time of day": ["dT1", "dT2", "wind", "GHI", "gap1", "gap2",
                               "prev", "call", "sin_h", "cos_h"],
}


def design(h: pd.DataFrame, feats: list) -> np.ndarray:
    cols = []
    for f in feats:
        if f == "wind":
            cols.append(h.dT.to_numpy() * h.v.to_numpy())    # wind acts on dT
        else:
            cols.append(h[f].to_numpy())
    return np.column_stack(cols)


class Linear:
    """Least-squares predictor with a non-negativity clamp on the output."""

    def __init__(self, name: str, feats: list):
        self.name, self.feats = name, feats
        self.coef = None

    def fit(self, h: pd.DataFrame):
        X = np.column_stack([np.ones(len(h)), design(h, self.feats)])
        self.coef, *_ = np.linalg.lstsq(X, h.actual.to_numpy(), rcond=None)
        return self

    def predict(self, h: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([np.ones(len(h)), design(h, self.feats)])
        return np.clip(X @ self.coef, 0.0, None)

    def describe(self) -> str:
        names = ["const", *self.feats]
        return "  ".join(f"{n}={c:+.4g}" for n, c in zip(names, self.coef))


def fit_all(h_fit: pd.DataFrame) -> dict:
    return {name: Linear(name, feats).fit(h_fit)
            for name, feats in FEATURES.items() if feats is not None}


def score_energy(pred, actual) -> dict:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    m = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[m], actual[m]
    err = pred - actual
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.nanmean(np.abs(err) / np.where(actual > 0, actual, np.nan))
    return {"n": int(len(pred)), "mae_kwh": float(np.abs(err).mean()),
            "rmse_kwh": float(np.sqrt((err ** 2).mean())),
            "bias_kwh": float(err.mean()), "mape_pct": float(mape * 100),
            "r": float(np.corrcoef(pred, actual)[0, 1])}
