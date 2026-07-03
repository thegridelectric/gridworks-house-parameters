from dataclasses import dataclass

import pandas as pd
import numpy as np


def linear_regression(df: pd.DataFrame, oat_ref: float = 0.0) -> tuple[np.ndarray, float, float, float]:
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
    return dist_kwh_pred, a, b, g


@dataclass
class HouseParametersLinear:
    alpha: float
    beta: float
    gamma: float


class HouseParametersComputer:
    """Fit house heating parameters from a chunk of hourly data"""
    def __init__(self, predictor=linear_regression):
        self.predictor = predictor

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df # TODO

    def fit(self, df: pd.DataFrame) -> HouseParametersLinear:
        dist_kwh_pred, a, b, g = self.predictor(df)
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh_pred.sum())
        return HouseParametersLinear(
            alpha = round(a * energy_ratio, 1),
            beta = round(b * energy_ratio, 2),
            gamma = round(g * energy_ratio, 5),
        )
