"""
Thermostat model, calibrated from the heat-call pattern already in the data.

The thermal model takes Q_dist as an input and predicts room temperature. To
forecast Q_dist we need the opposite direction, and nothing in the six state
equations decides when heat is called -- the thermostats do. So a controller
model is required, and it is what turns the thermal model into the deliverable.

Call status is recoverable without new logging:

    zone 1 calling  <=>  Q_dist1_only > 0  or  Q_dist_together > 0
    zone 2 calling  <=>  Q_dist2_only > 0  or  Q_dist_together > 0

At the moment a call switches OFF the room has just reached the top of the
thermostat's band; at the moment it switches ON it has just reached the bottom.

CRITICAL: the band must be read from T_i alone, never from T_i minus the
reported setpoint. These thermostats are ordinary on/off devices, but the
setpoint is a value the thermostat reports from its own internal sensor, which
is a different instrument in a different spot from the T_i logger and reads
about 1.1 K (zone 1) / 1.7 K (zone 2) lower. Subtracting one from the other
smears the switching across a whole degree and makes a sharp on/off controller
look like a modulating one -- the switching is in fact tight to about 0.1 K.

So the calibration works inside periods where the reported setpoint is
constant, takes the band edges from T_i, and then fits the mapping

    T_i threshold  =  slope * reported setpoint  +  offset

separately per zone. That regression is what makes a future reported setpoint
usable as a threshold in T_i units; it comes out at r = 0.999.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 300.0
ON_THRESHOLD = 1.0          # W; Q_dist is exactly zero when there is no flow


def call_status(df: pd.DataFrame) -> tuple:
    """Boolean per-zone heat-call series, from the three heat columns."""
    both = df["Q_dist_together"].to_numpy() > ON_THRESHOLD
    z1 = (df["Q_dist1_only"].to_numpy() > ON_THRESHOLD) | both
    z2 = (df["Q_dist2_only"].to_numpy() > ON_THRESHOLD) | both
    return z1, z2


def edges(call: np.ndarray) -> tuple:
    """Indices where a call switches on and where it switches off."""
    prev = np.r_[False, call[:-1]]
    return np.flatnonzero(call & ~prev), np.flatnonzero(~call & prev)


MIN_PERIOD = 288             # a constant-setpoint period must last >= 24 h
MIN_EVENTS = 8               # ...and contain enough switching to measure


@dataclass
class Zone:
    """One calibrated thermostat, in T_i-sensor units."""
    zone: int
    slope: float             # reported setpoint -> T_i threshold
    intercept: float
    r: float                 # correlation of that mapping
    deadband: float          # T_i at switch-off minus T_i at switch-on
    on_sd: float             # scatter of T_i at switch-on, within a period
    on_power: float          # W delivered to this zone while calling
    duty: float
    n_on: int
    n_periods: int
    cycle_min: float

    def threshold(self, reported):
        """Mid-band threshold in T_i units for a reported setpoint."""
        return self.slope * np.asarray(reported) + self.intercept

    def __str__(self):
        return (f"zone {self.zone}: T_i threshold = {self.slope:.3f} x "
                f"reported {self.intercept:+.2f} C (r={self.r:.4f}), "
                f"deadband {self.deadband:.3f} K, switch scatter "
                f"+/-{self.on_sd:.3f} K, on-power {self.on_power:.0f} W, "
                f"duty {self.duty:.1%}, cycle {self.cycle_min:.0f} min, "
                f"{self.n_on} events over {self.n_periods} periods")


def constant_setpoint_periods(sp: np.ndarray) -> np.ndarray:
    """Label each row with the index of its constant-reported-setpoint run."""
    return np.r_[0, np.cumsum(np.diff(sp) != 0)]


def calibrate_zone(df, zone, call, power) -> Zone:
    t_i = df[f"T_i{zone}"].to_numpy()
    sp = df[f"T_i{zone}_set"].to_numpy()
    grp = constant_setpoint_periods(sp)
    on_idx, off_idx = edges(call)

    reported, mid, bands, sds = [], [], [], []
    for g in range(grp.max() + 1):
        rows = np.flatnonzero(grp == g)
        if len(rows) < MIN_PERIOD:
            continue
        a, b = rows[0], rows[-1]
        on = on_idx[(on_idx >= a) & (on_idx <= b)]
        off = off_idx[(off_idx >= a) & (off_idx <= b)]
        if len(on) < MIN_EVENTS or len(off) < MIN_EVENTS:
            continue
        # one step before the edge: the last sample still on the old side
        t_on = t_i[np.maximum(on - 1, 0)]
        t_off = t_i[np.maximum(off - 1, 0)]
        reported.append(sp[a])
        mid.append(0.5 * (np.median(t_on) + np.median(t_off)))
        bands.append(np.median(t_off) - np.median(t_on))
        sds.append(np.std(t_on))

    reported = np.asarray(reported)
    mid = np.asarray(mid)
    if len(reported) >= 3:
        slope, intercept = np.polyfit(reported, mid, 1)
        r = float(np.corrcoef(reported, mid)[0, 1])
    else:                                  # not enough regimes to fit a line
        slope, intercept, r = 1.0, float(np.mean(mid - reported)), np.nan

    on_p = power[call]
    cycles = np.diff(on_idx) * DT / 60.0 if len(on_idx) > 1 else np.array([np.nan])

    return Zone(
        zone=zone, slope=float(slope), intercept=float(intercept), r=r,
        deadband=float(np.median(bands)), on_sd=float(np.median(sds)),
        on_power=float(np.median(on_p)) if on_p.size else 0.0,
        duty=float(call.mean()), n_on=len(on_idx), n_periods=len(reported),
        cycle_min=float(np.median(cycles)) if np.isfinite(cycles).any() else np.nan,
    )


def calibrate(df: pd.DataFrame, lam: float = 0.5) -> dict:
    """Calibrate both zones. lam splits the both-active heat between them."""
    z1, z2 = call_status(df)
    qb = df["Q_dist_together"].to_numpy()
    p1 = df["Q_dist1_only"].to_numpy() + lam * qb
    p2 = df["Q_dist2_only"].to_numpy() + (1.0 - lam) * qb
    return {1: calibrate_zone(df, 1, z1, p1),
            2: calibrate_zone(df, 2, z2, p2)}


class Controller:
    """
    Simulated thermostat for the forward Q_dist forecast.

    Deliberately stateful: whether heat is on right now depends on whether it
    was on a moment ago, not only on the current temperature. That hysteresis
    is the whole point -- an on/off decision taken purely from the current
    temperature would chatter every step and badly misestimate the energy.
    """

    def __init__(self, zones: dict, power_cap: float = np.inf, supply=None):
        """
        supply: optional {"T_w": float, "K_w": {zone: W/K}}. When given, the
        heat entering an emitter is K_w * (T_w - T_e) rather than a constant.

        The constant is physically wrong: measured power while calling spans
        3000-8100 W in zone 1 alone, because a cold emitter pulls far more heat
        out of the loop than a hot one. That is the startup transient behind
        the 32 kW peaks, and modelling it as a fixed number bakes in a bias
        whose sign depends on how often the loop starts cold.
        """
        self.zones = zones
        self.power_cap = power_cap
        self.supply = supply

    def reset(self, calling: dict):
        self.calling = dict(calling)

    def step(self, t_i: dict, t_set: dict, t_e: dict = None) -> dict:
        """
        Update call state, return heat delivered to each zone this step.

        t_set is the *reported* setpoint; each zone converts it to its own
        T_i-sensor units before comparing.
        """
        out = {}
        for z, cal in self.zones.items():
            mid = cal.threshold(t_set[z])
            hi = mid + 0.5 * cal.deadband
            lo = mid - 0.5 * cal.deadband
            if self.calling[z] and t_i[z] > hi:
                self.calling[z] = False
            elif not self.calling[z] and t_i[z] < lo:
                self.calling[z] = True

            if not self.calling[z]:
                out[z] = 0.0
            elif self.supply is None or t_e is None:
                out[z] = cal.on_power
            else:
                out[z] = max(0.0, self.supply["K_w"][z]
                             * (self.supply["T_w"] - t_e[z]))

        total = sum(out.values())
        if total > self.power_cap > 0:
            scale = self.power_cap / total
            out = {z: w * scale for z, w in out.items()}
        return out
