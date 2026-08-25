from dataclasses import dataclass, field

import pandas as pd
import numpy as np

HORIZON = 12          # 12 x 5 min = the hour being predicted
LOOKBACK_6H = 72      # 6 h of 5-min samples, for the envelope-memory feature
ON_W = 1.0            # Q_dist is exactly zero when there is no flow

# Features of the linear model, in fit order. gap1/gap2/call are left
# uncentered: gap is already meaningful at 0 (room exactly at its switching
# threshold) and call is binary.
FEATURES = ["dT", "wind", "GHI", "gap1", "gap2", "prev", "call", "to_6h"]
CENTERED = ["dT", "wind", "GHI", "prev", "to_6h"]


def setpoint_thresholds(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Map each zone's reported setpoint onto the T_i sensor scale.

    The thermostat reports a setpoint from its own internal sensor, which sits
    elsewhere in the room and reads about 1 K lower than the T_i logger. So the
    reported value cannot be compared with T_i directly. Inside a stretch where
    the reported setpoint is constant, the switch-on and switch-off
    temperatures bracket the real threshold, and their midpoint regressed on
    the reported value gives the conversion (r ~ 0.999 on this house).
    """
    both = df["Q_dist_together"].to_numpy() > ON_W
    out = {}
    for z in (1, 2):
        call = (df[f"Q_dist{z}_only"].to_numpy() > ON_W) | both
        t_i = df[f"T_i{z}"].to_numpy()
        sp = df[f"T_i{z}_set"].to_numpy()
        prev = np.r_[False, call[:-1]]
        on = np.flatnonzero(call & ~prev)
        off = np.flatnonzero(~call & prev)

        period = np.r_[0, np.cumsum(np.diff(sp) != 0)]
        reported, midpoint = [], []
        for g in range(period.max() + 1):
            rows = np.flatnonzero(period == g)
            if len(rows) < 288:                       # need >= 24 h
                continue
            a, b = rows[0], rows[-1]
            o = on[(on >= a) & (on <= b)]
            f = off[(off >= a) & (off <= b)]
            if len(o) < 8 or len(f) < 8:
                continue
            reported.append(sp[a])
            midpoint.append(0.5 * (np.median(t_i[o - 1]) + np.median(t_i[f - 1])))
        slope, intercept = np.polyfit(reported, midpoint, 1)
        out[z] = (float(slope), float(intercept))
    return out


def prepare_hourly(df: pd.DataFrame, hp_kwh_th: pd.Series | None = None,
                   ws_mph: pd.Series | None = None,
                   horizon: int = HORIZON) -> pd.DataFrame:
    """Turn the 5-minute record into one row per hour boundary.

    Each row holds what is known when the forecast is made plus the energy the
    loop actually drew over the following hour. Weather over that hour is taken
    as perfectly forecast, which is how the model would be used.

    Wind comes from the hourly export's ws_mph, not from data.csv's `v`. The
    two are not the same quantity: `v` correlates only 0.68 with ws_mph, sits
    at exactly zero a quarter of the time and never exceeds 3, so it is neither
    mph nor m/s of the same measurement. Using mph keeps the units unambiguous
    and keeps the plotted curves inside the range actually observed.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    thresholds = setpoint_thresholds(df)

    q = (df["Q_dist1_only"] + df["Q_dist2_only"] + df["Q_dist_together"]).to_numpy()
    t_o = df["T_o"].to_numpy()
    v = df["v"].to_numpy()
    ghi = df["GHI"].to_numpy()
    t1, t2 = df["T_i1"].to_numpy(), df["T_i2"].to_numpy()
    if ws_mph is not None:
        wind_mph = (df["timestamp"].map(ws_mph.reindex(
            df["timestamp"].dt.floor("h")).set_axis(df["timestamp"]))
            .ffill().bfill().to_numpy())
    else:
        wind_mph = np.zeros(len(df))
    s1, s2 = df["T_i1_set"].to_numpy(), df["T_i2_set"].to_numpy()

    # a window is only usable if the whole hour, and the hour before it, are
    # contiguous 5-minute data with no missing weather
    step = df["timestamp"].diff().dt.total_seconds().to_numpy()
    ok = np.r_[True, np.abs(step[1:] - 300.0) < 1.0]
    ok &= np.isfinite(t_o) & np.isfinite(ghi) & np.isfinite(wind_mph)

    rows = []
    for g in range(horizon, len(df) - horizon, horizon):
        if not ok[g - horizon:g + horizon].all():
            continue
        nxt, prv = slice(g, g + horizon), slice(g - horizon, g)
        t_i = 0.5 * (t1[g] + t2[g])
        to_next = t_o[nxt].mean()
        d_t = max(t_i - to_next, 0.0)
        a1, b1 = thresholds[1]
        a2, b2 = thresholds[2]
        rows.append({
            "hour_start": df["timestamp"].iloc[g],
            "dist_kwh": q[nxt].sum() * 300.0 / 3.6e6,
            "T_i": t_i,
            "dT": d_t,
            "wind": d_t * wind_mph[nxt].mean(),
            "GHI": ghi[nxt].mean(),
            "gap1": (a1 * s1[g] + b1) - t1[g],
            "gap2": (a2 * s2[g] + b2) - t2[g],
            "prev": q[prv].sum() * 300.0 / 3.6e6,
            "call": float(q[g - 1] > ON_W),
            "to_6h": t_o[max(0, g - LOOKBACK_6H):g].mean(),
            "oat_f": to_next * 9.0 / 5.0 + 32.0,
        })

    out = pd.DataFrame(rows)
    if hp_kwh_th is not None:
        out = out.merge(hp_kwh_th.rename("hp_kwh_th"), left_on="hour_start",
                        right_index=True, how="left")
    out["day"] = out["hour_start"].dt.normalize()
    return out.dropna().reset_index(drop=True)


def linear_regression(
    df: pd.DataFrame, refs: dict[str, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit dist_kwh = c0 + sum_k c_k * feature_k by MSE.

    Centering the continuous features at fixed global reference values makes c0
    the predicted energy at typical conditions rather than at all-zero, which
    decorrelates it from the slopes and makes it far more stable across the
    rolling fits. The slopes are unchanged by centering.
    """
    refs = refs or {}
    cols = [df[f].to_numpy() - refs.get(f, 0.0) for f in FEATURES]
    X = np.column_stack([np.ones(len(df))] + cols)
    dist_kwh = df["dist_kwh"].to_numpy()

    coefficients, *_ = np.linalg.lstsq(X, dist_kwh, rcond=None)
    dist_kwh_pred = X @ coefficients
    residuals = dist_kwh - dist_kwh_pred
    n, p = len(dist_kwh), X.shape[1]
    sigma2 = float(np.sum(residuals**2) / (n - p))
    std_errors = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    ss_tot = float(np.sum((dist_kwh - dist_kwh.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / ss_tot
    return dist_kwh_pred, coefficients, std_errors, r_squared


@dataclass
class HouseEnergyParams:
    coefficients: dict = field(default_factory=dict)
    std_errors: dict = field(default_factory=dict)
    r_squared: float = float("nan")


class HouseEnergyParamsComputer:
    """Fit house heating parameters from a chunk of hourly data"""
    def __init__(self, predictor=linear_regression):
        self.predictor = predictor

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df # TODO

    def fit(self, df: pd.DataFrame) -> HouseEnergyParams:
        dist_kwh_pred, coefficients, std_errors, r_squared = self.predictor(df)
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh_pred.sum())
        names = ["intercept", *FEATURES]
        return HouseEnergyParams(
            coefficients = {n: round(c * energy_ratio, 5)
                            for n, c in zip(names, coefficients)},
            std_errors = {n: round(s * energy_ratio, 5)
                          for n, s in zip(names, std_errors)},
            r_squared = round(r_squared, 2),
        )
