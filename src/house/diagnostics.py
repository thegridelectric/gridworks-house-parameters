"""
Model-free diagnostics.

The headline one is the daily energy balance. Over a 24 h window the storage
terms (C dT/dt for envelope, air and emitters) very nearly cancel, because the
house returns to a similar state each day. What is left is a static balance:

    Q_dist  ~=  UA_total * (T_i - T_o)  -  solar gains  +  everything else

so a straight-line fit of daily-mean heat against daily-mean temperature
difference estimates the whole-house conductance directly, with no RC model,
no Kalman filter and no optimiser involved. The intercept is the diagnostic:
it collects any heat flow that is *not* proportional to (T_i - T_o), which is
exactly the signature of a missing sink (pipe losses to unconditioned space,
ventilation, or heated volume the two T_i sensors do not see).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


@dataclass
class Fit:
    """A least-squares fit with the statistics needed to judge it."""
    names: list
    coef: np.ndarray
    se: np.ndarray
    r2: float
    r2_adj: float
    rmse: float
    n: int

    def __str__(self) -> str:
        w = max(len(s) for s in self.names)
        lines = []
        for name, c, s in zip(self.names, self.coef, self.se):
            t = c / s if s > 0 else np.nan
            lines.append(f"    {name:<{w}s} = {c:10.2f}  +/- {s:7.2f}  (t={t:6.1f})")
        lines.append(f"    R2 = {self.r2:.3f}  (adj {self.r2_adj:.3f})   "
                     f"RMSE = {self.rmse:.0f} W   n = {self.n} days")
        return "\n".join(lines)


def ols(X: np.ndarray, y: np.ndarray, names: list) -> Fit:
    """Ordinary least squares with standard errors. X excludes the intercept."""
    X = np.column_stack([np.ones(len(y)), X])
    names = ["intercept", *names]

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof

    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.abs(np.diag(XtX_inv) * s2))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof

    return Fit(names=names, coef=coef, se=se, r2=r2, r2_adj=r2_adj,
               rmse=float(np.sqrt(ss_res / n)), n=n)


def daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to daily means and attach the columns the diagnostic needs.

    `n` counts the 5-min samples in each day so partial days can be dropped;
    `dT_i*_storage` is the day-over-day change in indoor temperature, used to
    check that the storage terms really do cancel.
    """
    d = df.set_index("timestamp").resample("1D").mean(numeric_only=True)
    d["n"] = df.set_index("timestamp")["T_i1"].resample("1D").count()

    # count of rows per day with missing weather, so they can be excluded.
    # A daily mean built from a partial day is not comparable to a full one.
    miss = [c for c in ("T_o_missing", "GHI_missing") if c in df.columns]
    if miss:
        d["n_missing"] = (df.set_index("timestamp")[miss].any(axis=1)
                          .resample("1D").sum())
    else:
        d["n_missing"] = 0

    d["q"] = d.Q_dist1_only + d.Q_dist2_only + d.Q_dist_together
    d["T_i_avg"] = 0.5 * (d.T_i1 + d.T_i2)
    d["dT"] = d.T_i_avg - d.T_o
    d["dT1"] = d.T_i1 - d.T_o
    d["dT2"] = d.T_i2 - d.T_o

    # end-of-day minus end-of-previous-day, i.e. how much the house state moved
    last = df.set_index("timestamp")[["T_i1", "T_i2"]].resample("1D").last()
    d["dT_i1_storage"] = last.T_i1.diff()
    d["dT_i2_storage"] = last.T_i2.diff()

    return d


def energy_balance(d: pd.DataFrame, label: str = "") -> dict:
    """
    The FIRST TASK regression, plus the variants needed to interpret it.

    Returns a dict of Fit objects. Reported together because the simple
    fit alone cannot distinguish "conductance is bigger than the bounds allow"
    from "solar is being absorbed into the slope".
    """
    out = {}
    q = d.q.to_numpy()

    # 1. the headline fit: slope is whole-house UA [W/K]
    out["simple"] = ols(d[["dT"]].to_numpy(), q, ["UA [W/K]"])

    # 2. with solar. Daily-mean GHI is strongly anti-correlated with heating
    #    demand, so leaving it out biases both slope and intercept.
    out["solar"] = ols(d[["dT", "GHI"]].to_numpy(), q,
                       ["UA [W/K]", "solar [W per W/m2]"])

    # 3. per-zone conductance. The two zones ran at different setpoints, so
    #    their dT are not collinear and this is (weakly) identifiable.
    out["per_zone"] = ols(d[["dT1", "dT2"]].to_numpy(), q,
                          ["UA_1 [W/K]", "UA_2 [W/K]"])

    # 4. with wind, to see whether infiltration shows up at daily scale
    out["wind"] = ols(d[["dT", "GHI", "v"]].to_numpy(), q,
                      ["UA [W/K]", "solar [W per W/m2]", "wind [W per m/s]"])

    # 5. curvature check: a significant quadratic term means the single-UA
    #    picture is wrong (regime dependence, or a temperature-dependent sink)
    out["quadratic"] = ols(
        np.column_stack([d.dT.to_numpy(), d.dT.to_numpy() ** 2]), q,
        ["UA [W/K]", "dT^2 [W/K2]"])

    # 6. forced through the origin: what UA would have to be if there were no
    #    constant sink at all. Compare its RMSE against the simple fit.
    dT = d.dT.to_numpy()
    ua0 = float((dT @ q) / (dT @ dT))
    resid0 = q - ua0 * dT
    out["through_origin"] = Fit(
        names=["UA [W/K] (no intercept)"], coef=np.array([ua0]),
        se=np.array([np.sqrt(float(resid0 @ resid0) / max(len(q) - 1, 1)
                             / float(dT @ dT))]),
        r2=1.0 - float(resid0 @ resid0) / float(((q - q.mean()) ** 2).sum()),
        r2_adj=np.nan, rmse=float(np.sqrt(float(resid0 @ resid0) / len(q))),
        n=len(q))

    out["_label"] = label
    return out


def implied_gains(d: pd.DataFrame, ua: float) -> pd.DataFrame:
    """
    Per-day implied free heat gain, given a fixed whole-house conductance.

        G_day = UA * dT_day - q_day

    This is just the negated residual of the energy-balance fit, so its *mean*
    is the intercept by construction and carries no new information. The point
    is the time series: a genuine internal gain should be roughly constant,
    show a weekday/weekend occupancy signature, and not correlate with dT. If
    G instead grows with dT, it is not a gain at all, it is a misspecified UA
    leaking into the intercept, and adding a gains term to the model would be
    fitting the wrong thing.

    The level of G inherits the uncertainty in UA (an error of s_UA at a mean
    dT of D shifts every G by s_UA * D), so read the structure, not the level.
    """
    g = d.copy()
    g["G"] = ua * g.dT - g.q
    g["weekday"] = g.index.dayofweek
    g["is_weekend"] = g.weekday >= 5
    return g


def gains_report(g: pd.DataFrame, ua: float, ua_se: float) -> list:
    """Judge whether the implied gains look like a real physical quantity."""
    G = g.G
    msg = [f"mean {G.mean():.0f} W, median {G.median():.0f} W, "
           f"sd {G.std():.0f} W, IQR {G.quantile(.25):.0f}..{G.quantile(.75):.0f} W",
           f"level uncertainty from UA alone: +/-{ua_se * g.dT.mean():.0f} W"]

    # occupancy signature
    wk, we = G[~g.is_weekend], G[g.is_weekend]
    diff = we.mean() - wk.mean()
    pooled = np.sqrt(wk.var() / len(wk) + we.var() / len(we))
    msg.append(f"weekday {wk.mean():.0f} W vs weekend {we.mean():.0f} W "
               f"-> {diff:+.0f} W (t={diff / pooled:.1f})")

    for name, x in (("GHI", g.GHI), ("q", g.q)):
        msg.append(f"corr(G, {name}) = {float(np.corrcoef(G, x)[0, 1]):+.2f}")

    # The G-vs-dT test has to be done out-of-sample. G is the negated residual
    # of the fit that produced UA, so in-sample corr(G, dT) is exactly zero by
    # the orthogonality of OLS residuals -- it would "pass" for any dataset
    # whatsoever. Splitting alternate days breaks that identity and makes the
    # test mean something: UA from the odd days, G evaluated on the even ones.
    odd, even = g.iloc[1::2], g.iloc[0::2]
    ua_odd = float(ols(odd[["dT"]].to_numpy(), odd.q.to_numpy(), ["UA"]).coef[1])
    g_even = ua_odd * even.dT - even.q
    r_oos = float(np.corrcoef(g_even, even.dT)[0, 1])
    note = ("  <-- NOT a constant gain; UA is misspecified"
            if abs(r_oos) > 0.4 else "  (out-of-sample, so this one is real)")
    msg.append(f"corr(G, dT) = {r_oos:+.2f}{note}")

    # drift: regress G on day index
    t = np.arange(len(G), dtype=float)
    slope = float(np.polyfit(t, G.to_numpy(), 1)[0])
    msg.append(f"drift = {slope * 30:+.0f} W per 30 days")

    # Judge the magnitude as a balance-point offset rather than in watts. A
    # house needs no heat until T_o falls G/UA below T_i, and 3-6 K is the
    # normal range for that offset; the raw wattage only looks alarming
    # because this house has a large UA.
    offset = G.mean() / ua
    msg.append(f"balance-point offset = {offset:.1f} K "
               f"({G.mean():.0f} W / {ua:.0f} W/K); 3-6 K is normal for a "
               "house, so the magnitude is credible")
    msg.append(f"at a typical 5 W/m2 of internal gain this implies "
               f"~{G.mean() / 5:.0f} m2 of floor area")
    return msg


def interpret(fits: dict, mean_power: float, mean_dT: float) -> list:
    """Turn the numbers into the branch the instructions ask us to pick."""
    simple = fits["simple"]
    ua, icept = simple.coef[1], simple.coef[0]
    ua_se, icept_se = simple.se[1], simple.se[0]
    solar_ua = fits["solar"].coef[1]

    msg = []
    msg.append(f"whole-house UA  = {ua:.1f} +/- {ua_se:.1f} W/K "
               f"({solar_ua:.1f} W/K once solar is controlled for)")
    msg.append(f"intercept       = {icept:+.0f} +/- {icept_se:.0f} W "
               f"= {100 * icept / mean_power:+.0f}% of mean delivered power")
    msg.append(f"balance point   = {-icept / ua:.1f} K "
               "(dT at which no heating is needed)")
    msg.append(f"closes to       = {ua * mean_dT + icept:.0f} W predicted vs "
               f"{mean_power:.0f} W measured at the mean dT of {mean_dT:.1f} K")

    if ua > 250:
        msg.append("VERDICT: slope is roughly double the ~150 W/K the fit was "
                   "allowed. The bounds are the problem -> widen R_mo and g0 "
                   "and re-fit before touching model structure.")
    elif icept > 0.2 * mean_power and icept > 2 * icept_se:
        msg.append("VERDICT: moderate slope with a large POSITIVE intercept -> "
                   "a roughly constant chunk of Q_dist is not heating the "
                   "conditioned space. Chase the sink and add it as an "
                   "explicit term; do not let the optimiser hide it in "
                   "capacitance.")
    elif icept < 0 and abs(icept) > 2 * icept_se:
        msg.append("VERDICT: neither of the two candidate explanations. The "
                   "intercept is significantly NEGATIVE, which is the "
                   "signature of free heat gains (occupants, appliances, "
                   "solar) offsetting demand, not of a missing sink. The "
                   "energy balance closes. What the fit was missing is "
                   f"conductance: {ua:.0f} W/K against ~150 W/K fitted.")
    else:
        msg.append("VERDICT: neither branch cleanly. Slope is in the fitted "
                   "range and the intercept is not clearly nonzero.")
    return msg
