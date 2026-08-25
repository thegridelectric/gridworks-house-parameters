"""
Discretisation and steady-state Kalman filtering for any rung of the ladder.

The wind-dependent infiltration conductance makes A depend on v, so A and B
are cached per wind bin and a steady-state Kalman gain is solved once per bin.
Update and predict are folded into a single recursion

    x_{t+1} = Ad (I - K C) x_t  +  [Ad K y_t + Bd u_t]
            = F_b x_t + c_t

so the driving term c_t can be precomputed with a few large matrix products
and the time loop is one mat-vec.
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from .model import N_INPUTS

DT = 300.0
BIG_NLL = 1e12


def discretise(A, B, dt=DT):
    """Zero-order hold via one matrix exponential of the augmented system."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def process_noise(q_scale, n, dt=DT):
    """
    Diagonal process noise. Mass nodes get a tenth of the air/emitter value:
    they are physically slow, and letting them wander freely is exactly how a
    fit ends up with a decoupled envelope drifting to its bounds.
    """
    d = np.ones(n)
    if n == 2:
        d[1] = 0.1
    elif n == 3:
        d[2] = 0.1
    elif n in (6, 7):
        d[2] = d[5] = 0.1
        if n == 7:
            d[6] = 0.1          # the basement is a slow node too
    return q_scale * dt * np.diag(d)


def wind_bins(v, nbins):
    edges = np.unique(np.quantile(v, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 2:
        edges = np.array([v.min() - 1e-9, v.max() + 1e-9])
    idx = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    centres = np.array([v[idx == b].mean() if np.any(idx == b)
                        else 0.5 * (edges[b] + edges[b + 1])
                        for b in range(len(edges) - 1)])
    return idx.astype(np.int64), centres


def build_bank(model, p, centres, R):
    """Per wind bin: Ad, Bd, steady-state gain, and innovation statistics."""
    n = model.n
    C = model.obs
    ny = C.shape[0]
    nb = len(centres)

    Ad = np.empty((nb, n, n))
    Bd = np.empty((nb, n, N_INPUTS))
    K = np.empty((nb, n, ny))
    Sinv = np.empty((nb, ny, ny))
    logdetS = np.empty(nb)

    Qd = process_noise(p["q_scale"], n)
    for b, vb in enumerate(centres):
        A, B = model.build(p, vb)
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


def _loop_numpy(y, c, F, C, Sinv, logdetS, bins, x0, burn):
    x = x0.copy()
    nll = 0.0
    used = 0
    for t in range(y.shape[0]):
        b = bins[t]
        e = y[t] - C @ x
        if t >= burn:
            nll += 0.5 * (logdetS[b] + e @ Sinv[b] @ e)
            used += 1
        x = F[b] @ x + c[t]
    return nll, used


try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _loop_jit(y, c, F, C, Sinv, logdetS, bins, x0, burn):
        nt, ny = y.shape
        n = x0.shape[0]
        x = x0.copy()
        xn = np.empty(n)
        e = np.empty(ny)
        nll = 0.0
        used = 0
        for t in range(nt):
            b = bins[t]
            for i in range(ny):
                s = 0.0
                for j in range(n):
                    s += C[i, j] * x[j]
                e[i] = y[t, i] - s
            if t >= burn:
                q = 0.0
                for i in range(ny):
                    for j in range(ny):
                        q += e[i] * Sinv[b, i, j] * e[j]
                nll += 0.5 * (logdetS[b] + q)
                used += 1
            for i in range(n):
                s = c[t, i]
                for j in range(n):
                    s += F[b, i, j] * x[j]
                xn[i] = s
            for i in range(n):
                x[i] = xn[i]
        return nll, used

    _LOOP = _loop_jit
    HAVE_NUMBA = True
except Exception:                                        # pragma: no cover
    _LOOP = _loop_numpy
    HAVE_NUMBA = False


def filter_pass(y, u, bins, bank, C, x0, burn):
    Ad, Bd, K, Sinv, logdetS = bank
    nb, n = Ad.shape[0], Ad.shape[1]
    eye = np.eye(n)

    F = np.empty_like(Ad)
    G = np.empty((nb, n, C.shape[0]))
    for b in range(nb):
        F[b] = Ad[b] @ (eye - K[b] @ C)
        G[b] = Ad[b] @ K[b]

    c = np.empty((y.shape[0], n))
    for b in range(nb):
        m = bins == b
        if m.any():
            c[m] = y[m] @ G[b].T + u[m] @ Bd[b].T
    if not np.all(np.isfinite(c)):
        return BIG_NLL, 0

    nll, used = _LOOP(np.ascontiguousarray(y), c, F, C, Sinv, logdetS,
                      bins, x0.astype(float), int(burn))
    if not np.isfinite(nll):
        return BIG_NLL, 0
    return nll, used


def initial_state(model, y0):
    """Cold start: every node sits at the measured air temperature."""
    if model.n_zones == 1:
        return np.full(model.n, y0[0])
    x = np.empty(model.n)
    x[0:3] = y0[0]
    x[3:6] = y0[1]
    if model.n > 6:
        x[6:] = y0[0]           # basement starts at zone-1 temperature
    return x
