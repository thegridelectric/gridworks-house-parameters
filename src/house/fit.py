"""
Prediction-error fitting: Kalman innovations -> negative log-likelihood ->
L-BFGS-B in transformed space.

Positive parameters are fitted in log space (guarantees positivity and puts
parameters spanning many orders of magnitude on a comparable scale) and 0..1
parameters in logit space.

This is closed-loop identification -- Q_dist is thermostat-driven, so the
input is correlated with the output. The prediction-error method with a Kalman
filter stays consistent under closed-loop data, which is why it is used here
in preference to anything correlation-based.
"""

import numpy as np
from scipy.optimize import minimize

from .kalman import BIG_NLL, build_bank, filter_pass, initial_state, wind_bins
from .model import GHI as GHI_COL
from .model import N_INPUTS, ONE, Q1, Q2, TO, gain_shape

SIG_TI = 0.06                  # T_i sensor resolution [K]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def pack(model, p: dict) -> np.ndarray:
    x = []
    for q in model.params:
        v = p[q.name]
        if q.kind == "logit":
            v = np.clip(v, 1e-6, 1 - 1e-6)
            x.append(np.log(v / (1 - v)))
        elif q.kind == "linear":
            x.append(v)
        else:
            x.append(np.log(v))
    return np.asarray(x, float)


def unpack(model, x: np.ndarray) -> dict:
    p = {}
    for q, xi in zip(model.params, x):
        if q.kind == "logit":
            p[q.name] = 1.0 / (1.0 + np.exp(-xi))
        elif q.kind == "linear":
            p[q.name] = float(xi)
        else:
            p[q.name] = float(np.exp(xi))
    return p


def bounds(model) -> list:
    out = []
    for q in model.params:
        if q.kind == "logit":
            out.append((np.log(q.lo / (1 - q.lo)), np.log(q.hi / (1 - q.hi))))
        elif q.kind == "linear":
            out.append((q.lo, q.hi))
        else:
            out.append((np.log(q.lo), np.log(q.hi)))
    return out


def seed(model) -> dict:
    return {q.name: q.seed for q in model.params}


def at_bound(model, p: dict, tol=1e-3) -> list:
    """Which parameters have converged onto a bound, in log-space distance."""
    hits = []
    for q in model.params:
        v = p[q.name]
        if q.kind == "logit":
            t = np.log(v / (1 - v))
            lo, hi = np.log(q.lo / (1 - q.lo)), np.log(q.hi / (1 - q.hi))
        elif q.kind == "linear":
            t, lo, hi = v, q.lo, q.hi
        else:
            t, lo, hi = np.log(v), np.log(q.lo), np.log(q.hi)
        span = hi - lo
        if t - lo < tol * span:
            hits.append(f"{q.name}@lo")
        elif hi - t < tol * span:
            hits.append(f"{q.name}@hi")
    return hits


# ---------------------------------------------------------------------------
# Inputs and measurements
# ---------------------------------------------------------------------------

def build_inputs(model, df, p) -> np.ndarray:
    """
    u = [Q1, Q2, T_o, I, s(t)]; one-zone rungs carry all heat in the Q1 column.

    The last column carries internal gains. It is 1 for a constant-gain model
    and the fitted daily shape when the model has profile parameters, so gains
    enter as an ordinary input either way.
    """
    if not isinstance(p, dict):
        p = {"lam": p}                       # tolerate a bare lam, as before
    u = np.zeros((len(df), N_INPUTS))
    q1o = df["Q_dist1_only"].to_numpy()
    q2o = df["Q_dist2_only"].to_numpy()
    qb = df["Q_dist_together"].to_numpy()

    lam = p.get("lam", 0.5)
    if model.n_zones == 1:
        u[:, Q1] = q1o + q2o + qb
    else:
        u[:, Q1] = q1o + lam * qb
        u[:, Q2] = q2o + (1.0 - lam) * qb

    u[:, TO] = df["T_o"].to_numpy()
    u[:, GHI_COL] = df["GHI"].to_numpy()
    u[:, ONE] = (gain_shape(p, df["hour"].to_numpy())
                 if "gp_a1" in p else 1.0)
    return u


def measurements(model, df) -> np.ndarray:
    if model.n_zones == 1:
        return (0.5 * (df["T_i1"] + df["T_i2"])).to_numpy()[:, None]
    return df[["T_i1", "T_i2"]].to_numpy()


def make_cost(model, df, segments, nbins=12, burn=288):
    C = model.obs
    R = np.eye(C.shape[0]) * SIG_TI ** 2
    Y = measurements(model, df)
    bins_all, centres = wind_bins(df["v"].to_numpy(), nbins)
    lam_free = "lam" in model.names

    def cost(x):
        p = unpack(model, x)
        try:
            bank = build_bank(model, p, centres, R)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return BIG_NLL
        u_all = build_inputs(model, df, p)

        total, used = 0.0, 0
        for a, b in segments:
            y = Y[a:b]
            x0 = initial_state(model, y[0])
            nll, n_used = filter_pass(y, u_all[a:b], bins_all[a:b], bank,
                                      C, x0, burn)
            if nll >= BIG_NLL:
                return BIG_NLL
            total += nll
            used += n_used
        if used == 0:
            return BIG_NLL
        cost.n_used = used
        return total / used            # per-sample, keeps the scale comparable

    cost.n_used = 0
    return cost


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def optimise(model, cost, restarts=3, maxiter=600, rng_seed=0, verbose=True):
    x0 = pack(model, seed(model))
    bnds = bounds(model)
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    rng = np.random.default_rng(rng_seed)

    best = None
    for r in range(restarts):
        start = x0 if r == 0 else np.clip(
            x0 + rng.normal(0.0, 0.4, size=x0.shape), lo, hi)
        res = minimize(cost, start, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": maxiter, "maxfun": 20000,
                                "ftol": 1e-12, "gtol": 1e-8})
        if verbose:
            print(f"      restart {r + 1}/{restarts}: cost={res.fun:.5f}")
        if best is None or res.fun < best.fun:
            best = res
    return best
