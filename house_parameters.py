from dataclasses import dataclass

import pandas as pd
import numpy as np


def linear_regression(
    df: pd.DataFrame, oat_ref: float = 0.0
) -> tuple[np.ndarray, float, float, float, float, float, float, float]:
    """Fit dist_kwh = a + b*(oat_f - oat_ref) + g*(65-oat_f)*ws_mph by MSE.

    Centering oat_f at oat_ref makes a the predicted energy at oat_ref (inside
    the data cloud) instead of at 0°F, which decorrelates it from the slope b
    and makes it far more stable across fits. b and g are unchanged by centering.
    """
    oat_f = df["oat_f"].to_numpy()
    ws_mph = df["ws_mph"].to_numpy()
    dist_kwh = df["dist_kwh"].to_numpy()

    X = np.column_stack([np.ones_like(oat_f), oat_f - oat_ref, (65 - oat_f) * ws_mph])
    (a, b, g), *_ = np.linalg.lstsq(X, dist_kwh, rcond=None)
    dist_kwh_pred = X @ [a, b, g]
    residuals = dist_kwh - dist_kwh_pred
    n, p = len(dist_kwh), X.shape[1]
    sigma2 = float(np.sum(residuals**2) / (n - p))
    std_error_a, std_error_b, std_error_g = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    ss_tot = float(np.sum((dist_kwh - dist_kwh.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / ss_tot
    return dist_kwh_pred, a, b, g, std_error_a, std_error_b, std_error_g, r_squared


@dataclass
class HouseEnergyParams:
    alpha: float
    beta: float
    gamma: float
    std_error_alpha: float
    std_error_beta: float
    std_error_gamma: float
    r_squared: float


class HouseEnergyParamsComputer:
    """Fit house heating parameters from a chunk of hourly data"""
    def __init__(self, predictor=linear_regression):
        self.predictor = predictor

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df # TODO

    def fit(self, df: pd.DataFrame) -> HouseEnergyParams:
        dist_kwh_pred, a, b, g, std_error_alpha, std_error_beta, std_error_gamma, r_squared = self.predictor(df)
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh_pred.sum())
        return HouseEnergyParams(
            alpha = round(a * energy_ratio, 1),
            beta = round(b * energy_ratio, 2),
            gamma = round(g * energy_ratio, 5),
            std_error_alpha = round(std_error_alpha * energy_ratio, 1),
            std_error_beta = round(std_error_beta * energy_ratio, 2),
            std_error_gamma = round(std_error_gamma * energy_ratio, 5),
            r_squared = round(r_squared, 2),
        )


# @dataclass
# class HouseRswtParams:
#     a: float
#     b: float
#     c: float


# @dataclass
# class PowerSwtPoint:
#     power: float
#     swt: float


# class HouseRswtParamsComputer:
#     def __init__(self):
#         pass

#     def fit(self, no_power: PowerSwtPoint, intermediate: PowerSwtPoint, design_day: PowerSwtPoint) -> HouseRswtParams:
#         x0, y0 = no_power.swt, no_power.power
#         xi, yi = intermediate.swt, intermediate.power
#         xd, yd = design_day.swt, design_day.power

#         c = (xi*xd)/(xi-xd) * ((yd*x0)/(xd*(x0-xd)) - (yi*x0)/(xi*(x0-xi)))
#         b = (yi*x0)/(xi*(x0-xi)) - (x0+xi)/(x0*xi)*c
#         a = -b/x0 - c/x0/x0