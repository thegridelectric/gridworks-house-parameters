"""
Deployable hour-ahead Q_dist predictor.

Whatever wins the comparison has to be runnable in the MPC without dragging the
research code along, so the chosen model is reduced to a small set of numbers
plus one function. This module is that artifact.

The linear form needs, at each hour boundary:

    T_i1, T_i2         room temperatures now                     [C]
    setpoint1, 2       as reported by the thermostats            [C]
    T_o                mean outdoor temperature over the hour    [C]
    T_o_6h             mean outdoor temperature, past 6 hours    [C]
    v                  mean wind over the hour                   [m/s]
    GHI                mean irradiance over the hour             [W/m2]
    prev_kwh           loop energy over the hour just finished   [kWh]
    calling            whether the loop is flowing right now     [0/1]

and returns predicted loop energy for the next hour in kWh.

Setpoints must be mapped into T_i-sensor units before use. The thermostats
report from their own internal sensors, which read about 1.1 K (zone 1) and
1.7 K (zone 2) lower than the loggers, so subtracting them from T_i directly
would be wrong by more than a whole deadband.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SetpointMap:
    """Reported setpoint -> switching threshold in T_i-sensor units."""
    slope: float
    intercept: float

    def __call__(self, reported):
        return self.slope * np.asarray(reported, float) + self.intercept


@dataclass
class LinearPredictor:
    features: list
    coef: list
    map1: SetpointMap
    map2: SetpointMap
    trained_on: str = ""
    metrics: dict = field(default_factory=dict)

    def features_from(self, T_i1, T_i2, setpoint1, setpoint2, T_o, v, GHI,
                      prev_kwh, calling, T_o_6h=None) -> dict:
        T_i = 0.5 * (T_i1 + T_i2)
        dT = max(T_i - T_o, 0.0)
        return {
            "dT": dT,
            "dT1": max(T_i1 - T_o, 0.0),
            "dT2": max(T_i2 - T_o, 0.0),
            "wind": dT * v,
            "GHI": GHI,
            "gap1": float(self.map1(setpoint1)) - T_i1,
            "gap2": float(self.map2(setpoint2)) - T_i2,
            "prev": prev_kwh,
            "call": float(calling),
            # envelope memory: the walls lag the air, so recent weather still
            # matters after current outdoor temperature is accounted for
            "to_6h": T_o if T_o_6h is None else T_o_6h,
        }

    def predict(self, **kwargs) -> float:
        f = self.features_from(**kwargs)
        x = np.r_[1.0, [f[k] for k in self.features]]
        return float(max(0.0, x @ np.asarray(self.coef, float)))

    def to_json(self, path):
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2))
        return path

    @staticmethod
    def from_json(path):
        d = json.loads(Path(path).read_text())
        d["map1"] = SetpointMap(**d["map1"])
        d["map2"] = SetpointMap(**d["map2"])
        return LinearPredictor(**d)

    def describe(self) -> str:
        lines = ["predicted_kwh = max(0,",
                 f"    {self.coef[0]:+.5f}"]
        for name, c in zip(self.features, self.coef[1:]):
            lines.append(f"  {c:+.5f} * {name}")
        lines.append(")")
        lines.append(f"gap1 = ({self.map1.slope:.4f} * setpoint1 "
                     f"{self.map1.intercept:+.3f}) - T_i1")
        lines.append(f"gap2 = ({self.map2.slope:.4f} * setpoint2 "
                     f"{self.map2.intercept:+.3f}) - T_i2")
        lines.append("wind = max(T_i_avg - T_o, 0) * v")
        return "\n".join(lines)
