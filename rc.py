"""
Two-zone 4R3C grey-box thermal model of a house, fitted from 5-minute data.

States    :  T_e1, T_i1, T_m1, T_e2, T_i2, T_m2
Measured  :  T_i1, T_i2 (+ optional T_w)

Dynamics:
    C_e1 dT_e1/dt = Q1 - (T_e1-T_i1)/R_ei1
    C_i1 dT_i1/dt = (T_e1-T_i1)/R_ei1 + (T_m1-T_i1)/R_im1 + (T_i2-T_i1)/R_12 + (T_o-T_i1)*(g0_1+g1*v) + a_sol*f1*I
    C_m1 dT_m1/dt = (T_i1-T_m1)/R_im1 + (T_o-T_m1)/R_mo1
    ... mirrored for zone 2

Heat split:
    Q1 = Q_dist1_only + lam * Q_dist_together
    Q2 = Q_dist2_only + (1-lam) * Q_dist_together

Method: 
    ZOH discretisation at dt=300 s,
    Kalman filter over the record, 
    negative log-likelihood of the innovations minimised with L-BFGS-B in log/logit space.

Discretisation:
    Wind enters only through the infiltration conductance 1/R_inf(v) = g0 + g1*v.
    Because that makes A and B depend on v, the discretisation is cached per wind
    bin (A, B are piecewise-constant over the bins) and a steady-state Kalman gain
    is solved per bin. This keeps a full pass over ~35k samples fast enough for a
    few thousand cost evaluations.

CSV columns:
    timestamp, T_o, v, GHI, T_i1, T_i2, Q_dist1_only, Q_dist2_only, Q_dist_together (+ optional T_w)
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

DT = 300 # timestep [s]
BIG_NLL = 1e12

# state indices
E1, I1, M1, E2, I2, M2 = range(6)

# ----------------------------------------------------------------------------
# Parameter handling
# ----------------------------------------------------------------------------

# 18 strictly-positive parameters fitted in log space
LOG_PARAMS = [
    "R_ei1", "R_ei2",          # emitter -> air        [K/W]
    "R_im1", "R_im2",          # air -> mass           [K/W]
    "R_mo1", "R_mo2",          # mass -> outside       [K/W]
    "R_12",                    # air1 <-> air2         [K/W]
    "C_e1", "C_e2",            # emitter mass          [J/K]
    "C_i1", "C_i2",            # air + furniture       [J/K]
    "C_m1", "C_m2",            # envelope mass         [J/K]
    "g0_1", "g0_2",            # infiltration @ v=0    [W/K]
    "g1",                      # wind sensitivity      [W/K per m/s]
    "a_sol",                   # effective aperture    [m2]
    "q_scale",                 # process noise scale   [K^2/s]
]

# 1 parameter fitted in logit space
LOGIT_PARAMS = ["lam"]

ALL_PARAMS = LOG_PARAMS + LOGIT_PARAMS

# Starting values
SEED = {
    "R_ei1": 1.0e-2, "R_ei2": 1.0e-2,
    "R_im1": 6.0e-4, "R_im2": 6.0e-4,
    "R_mo1": 1.3e-2, "R_mo2": 1.3e-2,
    "R_12": 5.0e-3,
    "C_e1": 3.0e5, "C_e2": 3.0e5,
    "C_i1": 1.0e6, "C_i2": 1.0e6,
    "C_m1": 1.5e7, "C_m2": 1.5e7,
    "g0_1": 20.0, "g0_2": 20.0,
    "g1": 5.0,
    "a_sol": 6.0,
    "q_scale": 1.0e-8,
    "lam": 0.5,
}

# Bounds in log/logit
BOUNDS = {
    "R_ei1": (1e-4, 1.0), "R_ei2": (1e-4, 1.0),
    "R_im1": (1e-5, 1e-1), "R_im2": (1e-5, 1e-1),
    "R_mo1": (1e-4, 1.0), "R_mo2": (1e-4, 1.0),
    "R_12": (1e-5, 1.0),
    "C_e1": (5e4, 2e6), "C_e2": (5e4, 2e6),
    "C_i1": (1e4, 1e8), "C_i2": (1e4, 1e8),
    "C_m1": (1e6, 1e8), "C_m2": (1e6, 1e8),
    "g0_1": (0.1, 1e3), "g0_2": (0.1, 1e3),
    "g1": (1e-3, 1e3),
    "a_sol": (1, 50),
    "q_scale": (1e-12, 1e-9),
    "lam": (0.01, 0.99),
}


def pack(p: dict) -> np.ndarray:
    """natural dict -> unconstrained vector"""
    x = [np.log(p[k]) for k in LOG_PARAMS]
    lam = np.clip(p["lam"], 1e-6, 1 - 1e-6)
    x.append(np.log(lam / (1 - lam)))
    return np.asarray(x, float)


def unpack(x: np.ndarray) -> dict:
    """unconstrained vector -> natural dict"""
    p = {k: float(np.exp(v)) for k, v in zip(LOG_PARAMS, x[: len(LOG_PARAMS)])}
    p["lam"] = float(1.0 / (1.0 + np.exp(-x[-1])))
    return p


def packed_bounds() -> list:
    b = [(np.log(BOUNDS[k][0]), np.log(BOUNDS[k][1])) for k in LOG_PARAMS]
    lo, hi = BOUNDS["lam"]
    b.append((np.log(lo / (1 - lo)), np.log(hi / (1 - hi))))
    return b


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

REQUIRED = ["timestamp", "T_o", "v", "GHI", "T_i1", "T_i2", "Q_dist1_only", "Q_dist2_only", "Q_dist_together"]
NUMERIC = ["T_o", "v", "GHI", "T_i1", "T_i2", "Q_dist1_only", "Q_dist2_only", "Q_dist_together", "T_w"]

CSV = os.path.join(os.path.dirname(__file__), "data.csv")
GHI_MAX = 1100  # W/m2; clip sensor spikes above plausible clear-sky peak

OUTDIR = "house_params_results"
F1 = 0.5
SIG_TI = 0.06
SIG_TW = 2.0
NBINS = 12
BURN = 288
RESTARTS = 3
MAXITER = 800
RANDOM_SEED = 0
NO_SE = False


@dataclass
class Dataset:
    df: pd.DataFrame
    segments: list = field(default_factory=list)   # list of (start, stop) row slices
    has_tw: bool = False


def load_data(path: str, f1: float = 0.5) -> Dataset:
    df = pd.read_csv(path)

    # Check required columns
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: CSV missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert numeric columns to float
    for c in NUMERIC:
        if c not in df.columns:
            continue
        before = df[c].isna()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n_bad = int((~before & df[c].isna()).sum())
        if n_bad:
            print(f"  ! {c}: {n_bad} non-numeric value(s) -> NaN")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Handle duplicates
    dup = df["timestamp"].duplicated().sum()
    if dup:
        print(f"  ! {dup} duplicate timestamps dropped")
        df = df.drop_duplicates("timestamp").reset_index(drop=True)

    has_tw = "T_w" in df.columns

    # slowly-varying inputs may be hourly-held; interpolate them
    for c in ["T_o", "v", "GHI"]:
        n_na = df[c].isna().sum()
        if n_na:
            print(f"  ! {c}: {n_na} NaN interpolated")
            df[c] = df[c].interpolate(limit_direction="both")

    n_ghi_clip = int((df["GHI"] > GHI_MAX).sum())
    if n_ghi_clip:
        print(f"  ! GHI: {n_ghi_clip} values clipped to {GHI_MAX} W/m2")
        df["GHI"] = df["GHI"].clip(upper=GHI_MAX)
    for c in ["T_i1", "T_i2", "Q_dist1_only", "Q_dist2_only", "Q_dist_together"]:
        n_na = df[c].isna().sum()
        if n_na:
            print(f"  ! {c}: {n_na} NaN -> those rows will break segments")

    # split into contiguous 5-min segments
    gap = df["timestamp"].diff().dt.total_seconds().to_numpy()
    bad = np.isnan(gap) | (np.abs(gap - DT) > 1.0)
    bad[0] = False
    key_cols = ["T_i1", "T_i2", "Q_dist1_only", "Q_dist2_only", "Q_dist_together"]
    bad |= df[key_cols].isna().any(axis=1).to_numpy()

    breaks = np.flatnonzero(bad)
    edges = [0, *breaks.tolist(), len(df)]
    segments = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a >= 288:                      # need >= 24 h to be useful
            segments.append((a, b))

    span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print(f"  rows={len(df)}  span={span}  segments={len(segments)} "
          f"(usable rows={sum(b - a for a, b in segments)})")
    if not segments:
        sys.exit("ERROR: no usable contiguous segment of >= 24 h found.")

    # sanity checks worth surfacing before a fit is trusted
    print(f"  T_o   : {df.T_o.min():.1f} .. {df.T_o.max():.1f} degC")
    print(f"  v     : {df.v.min():.2f} .. {df.v.max():.2f} m/s "
          f"(p95={df.v.quantile(0.95):.2f})")
    if df.v.quantile(0.95) < 2.0:
        print("    ! low wind variance -> expect g1 to be weakly identified")
    print(f"  GHI   : {df.GHI.min():.0f} .. {df.GHI.max():.0f} W/m2")
    qtot = (df.Q_dist1_only + df.Q_dist2_only + df.Q_dist_together)
    print(f"  Q_dist: max={qtot.max():.0f} W, mean={qtot.mean():.0f} W")
    e_tot = qtot.sum() * DT / 3.6e6
    e_both = df.Q_dist_together.sum() * DT / 3.6e6
    print(f"  energy: {e_tot:.0f} kWh total, {100 * e_both / max(e_tot, 1e-9):.1f}% "
          f"delivered with both zones calling (drives lam identifiability)")
    if has_tw:
        print(f"  T_w   : present ({df.T_w.min():.1f} .. {df.T_w.max():.1f} degC)")

    return Dataset(df=df, segments=segments, has_tw=has_tw)


# ----------------------------------------------------------------------------
# Model matrices
# ----------------------------------------------------------------------------

def continuous(p: dict, ginf1: float, ginf2: float, f1: float, f2: float):
    """Build A (6x6) and B (6x4) for inputs u = [Q1, Q2, T_o, I]."""
    A = np.zeros((6, 6))
    B = np.zeros((6, 4))

    ke1, ke2 = 1.0 / p["R_ei1"], 1.0 / p["R_ei2"]
    ki1, ki2 = 1.0 / p["R_im1"], 1.0 / p["R_im2"]
    ko1, ko2 = 1.0 / p["R_mo1"], 1.0 / p["R_mo2"]
    k12 = 1.0 / p["R_12"]
    Ce1, Ce2 = p["C_e1"], p["C_e2"]
    Ci1, Ci2 = p["C_i1"], p["C_i2"]
    Cm1, Cm2 = p["C_m1"], p["C_m2"]

    # emitter nodes
    A[E1, E1] = -ke1 / Ce1
    A[E1, I1] = ke1 / Ce1
    B[E1, 0] = 1.0 / Ce1

    A[E2, E2] = -ke2 / Ce2
    A[E2, I2] = ke2 / Ce2
    B[E2, 1] = 1.0 / Ce2

    # air nodes
    A[I1, E1] = ke1 / Ci1
    A[I1, I1] = -(ke1 + ki1 + k12 + ginf1) / Ci1
    A[I1, M1] = ki1 / Ci1
    A[I1, I2] = k12 / Ci1
    B[I1, 2] = ginf1 / Ci1
    B[I1, 3] = p["a_sol"] * f1 / Ci1

    A[I2, E2] = ke2 / Ci2
    A[I2, I2] = -(ke2 + ki2 + k12 + ginf2) / Ci2
    A[I2, M2] = ki2 / Ci2
    A[I2, I1] = k12 / Ci2
    B[I2, 2] = ginf2 / Ci2
    B[I2, 3] = p["a_sol"] * f2 / Ci2

    # mass nodes
    A[M1, I1] = ki1 / Cm1
    A[M1, M1] = -(ki1 + ko1) / Cm1
    B[M1, 2] = ko1 / Cm1

    A[M2, I2] = ki2 / Cm2
    A[M2, M2] = -(ki2 + ko2) / Cm2
    B[M2, 2] = ko2 / Cm2

    return A, B


def discretise(A, B, dt=DT):
    """Zero-order-hold discretisation via one matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def process_noise(q_scale: float, dt=DT):
    """Heuristic diagonal: masses drift more slowly than air/emitter nodes."""
    return q_scale * dt * np.diag([1.0, 1.0, 0.1, 1.0, 1.0, 0.1])


def obs_matrices(has_tw: bool, sig_ti: float, sig_tw: float):
    if has_tw:
        C = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0.5, 0, 0, 0.5, 0, 0],   # T_w observes a blend of both emitters
        ], float)
        R = np.diag([sig_ti ** 2, sig_ti ** 2, sig_tw ** 2])
    else:
        C = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], float)
        R = np.diag([sig_ti ** 2, sig_ti ** 2])
    return C, R


# ----------------------------------------------------------------------------
# Wind binning + per-bin steady-state Kalman gain
# ----------------------------------------------------------------------------

def make_bins(v: np.ndarray, nbins: int):
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.unique(np.quantile(v, qs))
    if len(edges) < 2:
        edges = np.array([v.min() - 1e-9, v.max() + 1e-9])
    idx = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    centres = np.array([v[idx == b].mean() if np.any(idx == b)
                        else 0.5 * (edges[b] + edges[b + 1])
                        for b in range(len(edges) - 1)])
    return idx, centres


def build_bank(p, centres, C, R, f1, f2):
    """Per wind bin: Ad, Bd, steady-state gain K, innovation covariance S."""
    nb = len(centres)
    Ad = np.empty((nb, 6, 6))
    Bd = np.empty((nb, 6, 4))
    K = np.empty((nb, 6, C.shape[0]))
    Sinv = np.empty((nb, C.shape[0], C.shape[0]))
    logdetS = np.empty(nb)

    Qd = process_noise(p["q_scale"])
    for b, vb in enumerate(centres):
        g1_ = p["g0_1"] + p["g1"] * vb
        g2_ = p["g0_2"] + p["g1"] * vb
        A, B = continuous(p, g1_, g2_, f1, f2)
        ad, bd = discretise(A, B)
        if not (np.all(np.isfinite(ad)) and np.all(np.isfinite(bd))):
            raise FloatingPointError("non-finite discretisation")
        if np.max(np.abs(np.linalg.eigvals(ad))) > 1.0 + 1e-8:
            raise FloatingPointError("unstable discrete system")
        P = solve_discrete_are(ad.T, C.T, Qd, R)
        S = C @ P @ C.T + R
        sgn, ld = np.linalg.slogdet(S)
        if sgn <= 0:
            raise FloatingPointError("non-PD innovation covariance")
        Ad[b], Bd[b] = ad, bd
        K[b] = P @ C.T @ np.linalg.inv(S)
        Sinv[b] = np.linalg.inv(S)
        logdetS[b] = ld
    return Ad, Bd, K, Sinv, logdetS


# ----------------------------------------------------------------------------
# Kalman filter pass
# ----------------------------------------------------------------------------

def _loop_numpy(y, c, F, C, Sinv, logdetS, bins, x0, burn):
    n = y.shape[0]
    x = x0.copy()
    nll = 0.0
    used = 0
    for t in range(n):
        b = bins[t]
        e = y[t] - C @ x
        if t >= burn:
            nll += 0.5 * (logdetS[b] + e @ Sinv[b] @ e)
            used += 1
        x = F[b] @ x + c[t]
    return nll, used


try:                                    # optional JIT: ~50x on the inner loop
    from numba import njit

    @njit(cache=False, fastmath=True)
    def _loop_jit(y, c, F, C, Sinv, logdetS, bins, x0, burn):
        n, ny = y.shape
        x = x0.copy()
        nll = 0.0
        used = 0
        e = np.empty(ny)
        xn = np.empty(6)
        for t in range(n):
            b = bins[t]
            for i in range(ny):
                s = 0.0
                for j in range(6):
                    s += C[i, j] * x[j]
                e[i] = y[t, i] - s
            if t >= burn:
                q = 0.0
                for i in range(ny):
                    for j in range(ny):
                        q += e[i] * Sinv[b, i, j] * e[j]
                nll += 0.5 * (logdetS[b] + q)
                used += 1
            for i in range(6):
                s = c[t, i]
                for j in range(6):
                    s += F[b, i, j] * x[j]
                xn[i] = s
            for i in range(6):
                x[i] = xn[i]
        return nll, used

    _LOOP = _loop_jit
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    _LOOP = _loop_numpy
    HAVE_NUMBA = False


def filter_pass(y, u, bins, Ad, Bd, K, Sinv, logdetS, C, x0, burn=288):
    """
    Run the steady-state filter over one contiguous segment.

    The update and predict steps are folded into a single recursion
        x_{t+1} = Ad (I - K C) x_t + Ad K y_t + Bd u_t
                = F_b x_t + c_t
    so the per-bin driving term c_t can be precomputed with a few large
    matrix products, leaving only one 6x6 mat-vec inside the time loop.
    """
    nb = Ad.shape[0]
    F = np.empty_like(Ad)
    G = np.empty((nb, 6, C.shape[0]))
    eye = np.eye(6)
    for b in range(nb):
        F[b] = Ad[b] @ (eye - K[b] @ C)
        G[b] = Ad[b] @ K[b]

    c = np.empty((y.shape[0], 6))
    for b in range(nb):
        m = bins == b
        if m.any():
            c[m] = y[m] @ G[b].T + u[m] @ Bd[b].T

    if not np.all(np.isfinite(c)):
        return BIG_NLL, 0

    nll, used = _LOOP(np.ascontiguousarray(y), c, F, C, Sinv, logdetS,
                      bins.astype(np.int64), x0.astype(float), int(burn))
    if not np.isfinite(nll):
        return BIG_NLL, 0
    return nll, used


def build_inputs(df: pd.DataFrame, lam: float):
    q1 = df["Q_dist1_only"].to_numpy() + lam * df["Q_dist_together"].to_numpy()
    q2 = df["Q_dist2_only"].to_numpy() + (1 - lam) * df["Q_dist_together"].to_numpy()
    return np.column_stack([q1, q2, df["T_o"].to_numpy(), df["GHI"].to_numpy()])


def make_cost(data: Dataset, f1, f2, sig_ti, sig_tw, nbins, burn):
    df = data.df
    C, R = obs_matrices(data.has_tw, sig_ti, sig_tw)
    if data.has_tw:
        Y = df[["T_i1", "T_i2", "T_w"]].to_numpy()
    else:
        Y = df[["T_i1", "T_i2"]].to_numpy()
    bins_all, centres = make_bins(df["v"].to_numpy(), nbins)

    def cost(x):
        p = unpack(x)
        try:
            bank = build_bank(p, centres, C, R, f1, f2)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return BIG_NLL
        Ad, Bd, K, Sinv, logdetS = bank
        u_all = build_inputs(df, p["lam"])
        total, used = 0.0, 0
        for a, b in data.segments:
            y = Y[a:b]
            x0 = np.array([y[0, 0], y[0, 0], y[0, 0],
                           y[0, 1], y[0, 1], y[0, 1]])
            nll, n_used = filter_pass(
                y, u_all[a:b], bins_all[a:b], Ad, Bd, K, Sinv, logdetS,
                C, x0, burn=burn)
            if nll >= BIG_NLL:
                return BIG_NLL
            total += nll
            used += n_used
        if used == 0:
            return BIG_NLL
        cost.n_used = used           # needed to rescale the Hessian for SEs
        return total / used          # per-sample, keeps the scale sane

    cost.n_used = 0
    return cost, C, R, bins_all, centres, Y


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------

def fit(cost, x0, bounds, maxiter=800, verbose=True):
    it = {"n": 0}

    def cb(xk):
        it["n"] += 1
        if verbose and it["n"] % 25 == 0:
            print(f"    iter {it['n']:4d}  cost={cost(xk):.6f}")

    res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                   callback=cb,
                   options={"maxiter": maxiter, "maxfun": 20000,
                            "ftol": 1e-12, "gtol": 1e-8})
    return res


def jitter(x0, rng, scale=0.35):
    return x0 + rng.normal(0.0, scale, size=x0.shape)


def standard_errors(cost, xhat, eps=1e-3):
    """
    Numerical Hessian of the *total* NLL -> SE in the transformed space.

    `cost` returns the per-sample NLL, so its Hessian must be multiplied by the
    number of scored samples before inversion, otherwise the errors come out
    inflated by sqrt(N).

    Because the parameters are fitted in log space, the resulting SE is
    directly the relative standard error of the natural-units value.
    """
    f0 = cost(xhat)
    n_obs = max(int(getattr(cost, "n_used", 1)), 1)
    n = len(xhat)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xi = xhat.copy(); xi[i] += eps
            xj = xhat.copy(); xj[j] += eps
            xij = xhat.copy(); xij[i] += eps; xij[j] += eps
            H[i, j] = H[j, i] = (cost(xij) - cost(xi) - cost(xj) + f0) / eps ** 2
    H *= n_obs

    try:
        # symmetrise, then check the curvature is a genuine minimum
        H = 0.5 * (H + H.T)
        w = np.linalg.eigvalsh(H)
        if np.min(w) <= 0:
            print(f"    ! Hessian not positive definite (min eig={np.min(w):.3g}); "
                  "some parameters are unidentified or the optimiser stopped early")
        cov = np.linalg.pinv(H)
        se_t = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se_t = np.full(n, np.nan)
    return se_t


# ----------------------------------------------------------------------------
# Validation: open-loop 1-hour temperature forecasts
# ----------------------------------------------------------------------------

def validate(data, p, f1, f2, sig_ti, sig_tw, nbins, burn, mask, horizon=12):
    """
    Filter through the held-out period; every `horizon` steps, freeze the state
    and simulate open-loop (no measurement updates) for one hour. Compare the
    predicted T_i1/T_i2 at the end of the hour with the measurements.
    Baseline = persistence (temperature stays where it is).
    """
    df = data.df
    C, R = obs_matrices(data.has_tw, sig_ti, sig_tw)
    Y = (df[["T_i1", "T_i2", "T_w"]].to_numpy() if data.has_tw
         else df[["T_i1", "T_i2"]].to_numpy())
    bins_all, centres = make_bins(df["v"].to_numpy(), nbins)
    Ad, Bd, K, Sinv, logdetS = build_bank(p, centres, C, R, f1, f2)
    u_all = build_inputs(df, p["lam"])

    rows = []
    for a, b in data.segments:
        if not mask[a:b].any():
            continue
        y = Y[a:b]
        u = u_all[a:b]
        bb = bins_all[a:b]
        x = np.array([y[0, 0], y[0, 0], y[0, 0], y[0, 1], y[0, 1], y[0, 1]])
        for t in range(b - a):
            e = y[t] - C @ x
            xu = x + K[bb[t]] @ e

            # launch an open-loop hour from here
            if (t % horizon == 0) and (t + horizon < b - a) and t >= burn \
                    and mask[a + t]:
                xs = xu.copy()
                for k in range(horizon):
                    xs = Ad[bb[t + k]] @ xs + Bd[bb[t + k]] @ u[t + k]
                rows.append({
                    "t": df["timestamp"].iloc[a + t + horizon],
                    "pred_T_i1": xs[I1], "pred_T_i2": xs[I2],
                    "meas_T_i1": y[t + horizon, 0], "meas_T_i2": y[t + horizon, 1],
                    "pers_T_i1": y[t, 0], "pers_T_i2": y[t, 1],
                })

            x = Ad[bb[t]] @ xu + Bd[bb[t]] @ u[t]

    v = pd.DataFrame(rows)
    if v.empty:
        print("  ! no validation windows produced")
        return v, {}

    out = {}
    for z in ("1", "2"):
        err = v[f"pred_T_i{z}"] - v[f"meas_T_i{z}"]
        errp = v[f"pers_T_i{z}"] - v[f"meas_T_i{z}"]
        out[f"rmse_zone{z}"] = float(np.sqrt((err ** 2).mean()))
        out[f"mae_zone{z}"] = float(err.abs().mean())
        out[f"bias_zone{z}"] = float(err.mean())
        out[f"rmse_persistence_zone{z}"] = float(np.sqrt((errp ** 2).mean()))
    out["n_windows"] = int(len(v))
    return v, out


def plot_results(v, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ! matplotlib not available, skipping plots")
        return
    if v.empty:
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for i, z in enumerate(("1", "2")):
        ax[0, i].plot(v["t"], v[f"meas_T_i{z}"], lw=0.9, label="measured")
        ax[0, i].plot(v["t"], v[f"pred_T_i{z}"], lw=0.9, label="1 h forecast")
        ax[0, i].set_title(f"Zone {z}: 1-hour-ahead open loop")
        ax[0, i].set_ylabel("degC")
        ax[0, i].legend(fontsize=8)
        ax[0, i].tick_params(axis="x", rotation=30)

        err = v[f"pred_T_i{z}"] - v[f"meas_T_i{z}"]
        ax[1, i].hist(err, bins=60)
        ax[1, i].set_title(f"Zone {z} forecast error  "
                           f"(RMSE={np.sqrt((err**2).mean()):.3f} K)")
        ax[1, i].set_xlabel("predicted - measured [K]")
    fig.tight_layout()
    path = os.path.join(outdir, "validation.png")
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}")


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

UNITS = {
    "R_ei1": "K/W", "R_ei2": "K/W", "R_im1": "K/W", "R_im2": "K/W",
    "R_mo1": "K/W", "R_mo2": "K/W", "R_12": "K/W",
    "C_e1": "J/K", "C_e2": "J/K", "C_i1": "J/K", "C_i2": "J/K",
    "C_m1": "J/K", "C_m2": "J/K",
    "g0_1": "W/K", "g0_2": "W/K", "g1": "W/K/(m/s)",
    "a_sol": "m2", "q_scale": "K2/s", "lam": "-",
}


def report(p, se_t, skipped_se=False):
    print("\n" + "=" * 74)
    print(f"{'parameter':10s} {'value':>13s} {'rel.SE':>9s} {'units':>12s}   note")
    print("-" * 74)
    for i, k in enumerate(ALL_PARAMS):
        val = p[k]
        # log-space SE is directly the relative SE; logit needs the chain rule
        if k == "lam":
            rel = se_t[i] * (1 - val) if np.isfinite(se_t[i]) else np.nan
        else:
            rel = se_t[i]
        flag = ""
        if not np.isfinite(rel):
            flag = "-" if skipped_se else "SE failed"
        elif rel > 0.5:
            flag = "NOT identified"
        elif rel > 0.2:
            flag = "weak"
        print(f"{k:10s} {val:13.4g} {rel:9.2%} {UNITS[k]:>12s}   {flag}")
    print("=" * 74)

    # derived quantities that are easier to sanity-check than raw R and C
    print("\nDerived:")
    for z in ("1", "2"):
        ua = 1.0 / (p[f"R_im{z}"] + p[f"R_mo{z}"])
        print(f"  zone {z}: fabric UA        = {ua:8.1f} W/K")
        print(f"  zone {z}: infiltration @0  = {p[f'g0_{z}']:8.1f} W/K")
        tau_e = p[f"C_e{z}"] * p[f"R_ei{z}"] / 60.0
        tau_m = p[f"C_m{z}"] * (p[f"R_im{z}"] + p[f"R_mo{z}"]) / 3600.0
        print(f"  zone {z}: emitter tau      = {tau_e:8.1f} min")
        print(f"  zone {z}: envelope tau     = {tau_m:8.1f} h")
    print(f"  inter-zone conductance = {1.0 / p['R_12']:8.1f} W/K")
    print(f"  implied glazing area   = {p['a_sol'] / 0.5:8.1f} m2 (at g=0.5)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    f1, f2 = F1, 1.0 - F1

    print("Loading data")
    data = load_data(CSV, f1)

    # train / holdout split
    df = data.df
    n = len(df)
    train_mask = np.zeros(n, bool)
    train_mask[: int(0.8 * n)] = True
    hold_mask = ~train_mask
    print(f"  train rows={train_mask.sum()}  holdout rows={hold_mask.sum()}")

    train = Dataset(
        df=df,
        segments=[(a, b) for a, b in data.segments
                  if train_mask[a:b].mean() > 0.99],
        has_tw=data.has_tw,
    )
    if not train.segments:
        # fall back: truncate segments at the split point
        train.segments = []
        cut_idx = int(np.flatnonzero(train_mask)[-1]) + 1 if train_mask.any() else 0
        for a, b in data.segments:
            bb = min(b, cut_idx)
            if bb - a >= 288:
                train.segments.append((a, bb))
    print(f"  training segments: {len(train.segments)}")

    cost, C, R, bins_all, centres, Y = make_cost(
        train, f1, f2, SIG_TI, SIG_TW, NBINS, BURN)

    x0 = pack(SEED)
    bnds = packed_bounds()
    c0 = cost(x0)
    print(f"\nSeed cost = {c0:.6f}")
    if c0 >= BIG_NLL:
        sys.exit("ERROR: seed parameters produce an invalid model. "
                 "Check units in the CSV (W, degC, m/s).")

    rng = np.random.default_rng(RANDOM_SEED)
    best = None
    for r in range(RESTARTS):
        start = x0 if r == 0 else np.clip(
            jitter(x0, rng),
            [b[0] for b in bnds], [b[1] for b in bnds])
        print(f"\nRestart {r + 1}/{RESTARTS}")
        res = fit(cost, start, bnds, maxiter=MAXITER)
        print(f"  -> cost={res.fun:.6f}  ({res.message})")
        if best is None or res.fun < best.fun:
            best = res

    p = unpack(best.x)
    print(f"\nBest cost = {best.fun:.6f}")

    if NO_SE:
        se_t = np.full(len(best.x), np.nan)
    else:
        print("Computing standard errors (numerical Hessian)")
        se_t = standard_errors(cost, best.x)

    report(p, se_t, skipped_se=NO_SE)

    print("\nValidation: rolling 1-hour open-loop forecasts on the holdout")
    v, metrics = validate(data, p, f1, f2, SIG_TI, SIG_TW,
                          NBINS, BURN, hold_mask)
    if metrics:
        for z in ("1", "2"):
            print(f"  zone {z}: RMSE={metrics[f'rmse_zone{z}']:.3f} K  "
                  f"MAE={metrics[f'mae_zone{z}']:.3f} K  "
                  f"bias={metrics[f'bias_zone{z}']:+.3f} K  "
                  f"(persistence RMSE={metrics[f'rmse_persistence_zone{z}']:.3f} K)")
        print(f"  windows: {metrics['n_windows']}")

    out = {"parameters": p,
           "rel_se_transformed": dict(zip(ALL_PARAMS, se_t.tolist())),
           "cost": float(best.fun),
           "validation": metrics,
           "config": {
               "csv": CSV, "outdir": OUTDIR,
               "f1": F1, "sig_ti": SIG_TI, "sig_tw": SIG_TW,
               "nbins": NBINS, "burn": BURN, "restarts": RESTARTS,
               "maxiter": MAXITER, "seed": RANDOM_SEED, "no_se": NO_SE,
           }}
    jp = os.path.join(OUTDIR, "fit_result.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  parameters -> {jp}")

    if not v.empty:
        vp = os.path.join(OUTDIR, "validation_windows.csv")
        v.to_csv(vp, index=False)
        print(f"  windows    -> {vp}")
        plot_results(v, OUTDIR)


if __name__ == "__main__":
    main()
