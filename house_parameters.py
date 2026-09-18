import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Model features in B1.. order. B0 is the intercept. Zone terms follow the base five.
#   B1 deltaT
#   B2 windspeed_times_deltaT
#   B3 solar_w_m2
#   B4 previous_dist_kwh
#   B5 OAT_avg_6h
#   B6+ set_minus_temp_zone{z} for each zone z in the export
BASE_FEATURES = [
    "deltaT",
    "windspeed_times_deltaT",
    "solar_w_m2",
    "previous_dist_kwh",
    "OAT_avg_6h",
]

BASE_REQUIRED_COLUMNS = [
    "oat_f", 
    "ws_mph",
    "solar_w_m2",
    "dist_kwh",
    "hp_kwh_th",
]


def zone_numbers(columns) -> list[int]:
    return sorted(
        int(n) for c in columns
        if (n := str(c).removeprefix("T_i").removesuffix("_start")).isdigit()
    )


def load_hourly_features(house_alias: str) -> pd.DataFrame:
    csv_path = glob.glob(f"data/{house_alias}_house_params_data.csv")[0]
    df = pd.read_csv(csv_path)

    df["hour_start"] = pd.to_datetime(df["hour_start"])
    df = df.sort_values("hour_start").reset_index(drop=True)
    df["day"] = df["hour_start"].dt.normalize()

    zones = zone_numbers(df.columns)
    if not zones:
        raise ValueError("No T_i{z}_start columns found in export")
    
    inside_temp_avg = df[[f"T_i{z}_start" for z in zones]].mean(axis=1)
    df["deltaT"] = (inside_temp_avg - df["oat_f"]).clip(lower=0)
    df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
    for z in zones:
        df[f"set_minus_temp_zone{z}"] = df[f"T_i{z}_set_start"] - df[f"T_i{z}_start"]
    df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
    df["OAT_avg_6h"] = df["oat_f"].rolling(6).mean().shift(1)

    # Making sure the 6 rows before every row are contiguous for the OAT_avg_6h feature
    history_span = df["hour_start"] - df["hour_start"].shift(6)
    df = df[history_span == pd.Timedelta(hours=6)]

    # Dealing with missing values
    features = BASE_FEATURES + [f"set_minus_temp_zone{z}" for z in zones]
    required = BASE_REQUIRED_COLUMNS + [col for z in zones for col in (f"T_i{z}_start", f"T_i{z}_set_start")]
    df = df.dropna(subset=required+features).reset_index(drop=True)

    df.attrs["zone_numbers"] = zones
    return df


def design_matrix(df: pd.DataFrame, feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or (
        BASE_FEATURES + [f"set_minus_temp_zone{z}" for z in zone_numbers(df.columns)]
    )
    columns = [np.ones(len(df))]
    for name in names:
        columns.append(df[name].to_numpy())
    return np.column_stack(columns)


def alpha_beta_gamma_design_matrix(df: pd.DataFrame) -> np.ndarray:
    oat_f = df["oat_f"].to_numpy()
    return np.column_stack([
        np.ones(len(df)),
        oat_f,
        (65.0 - oat_f) * df["ws_mph"].to_numpy(),
    ])


def fit_alpha_beta_gamma(df: pd.DataFrame) -> np.ndarray:
    """Unscaled [alpha, beta, gamma] for dist_kwh, same model as main."""
    coef, *_ = np.linalg.lstsq(
        alpha_beta_gamma_design_matrix(df), df["dist_kwh"].to_numpy(), rcond=None
    )
    return coef


def predict_alpha_beta_gamma(coef: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Predicted dist_kwh from alpha/beta/gamma, clipped at zero like predict()."""
    return np.maximum(alpha_beta_gamma_design_matrix(df) @ coef, 0.0)


def linear_regression(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit dist_kwh = B0 + B1*deltaT + ... by MSE.

    Returns (fitted values, coefficients, standard errors, R-squared), with B0
    first in both coefficient arrays.
    """
    X = design_matrix(df)
    dist_kwh = df["dist_kwh"].to_numpy()

    coefficients, *_ = np.linalg.lstsq(X, dist_kwh, rcond=None)
    dist_kwh_pred = X @ coefficients
    residuals = dist_kwh - dist_kwh_pred
    n = len(dist_kwh)
    # pinv: a zone at setpoint for the whole window makes that gap column
    # constant, so X'X is singular (elm's zone 2 is like this most of the year).
    rank = int(np.linalg.matrix_rank(X))
    sigma2 = float(np.sum(residuals**2) / max(n - rank, 1))
    std_errors = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    ss_tot = float(np.sum((dist_kwh - dist_kwh.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / ss_tot
    return dist_kwh_pred, coefficients, std_errors, r_squared


@dataclass
class HouseEnergyParams:
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    std_errors: tuple[float, ...]
    r_squared: float
    # hp_kwh_th / dist_kwh the coefficients were multiplied by
    energy_ratio: float

    @property
    def coef_names(self) -> tuple[str, ...]:
        return tuple(f"B{i}" for i in range(len(self.values)))

    def coefficients(self) -> np.ndarray:
        """[B0, B1, ...], in the unit the fit was scaled to."""
        return np.array(self.values, dtype=float)

    def dist_kwh_coefficients(self) -> np.ndarray:
        """The same coefficients back in distribution-kWh, the unit of dist_kwh."""
        return self.coefficients() / self.energy_ratio

    def __getattr__(self, name: str):
        if name.startswith("std_error_B") and name[len("std_error_B"):].isdigit():
            return self.std_errors[int(name[len("std_error_B"):])]
        if name.startswith("B") and len(name) > 1 and name[1:].isdigit():
            return self.values[int(name[1:])]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def predict(params: HouseEnergyParams, df: pd.DataFrame) -> np.ndarray:
    """Predicted dist_kwh, clipped at zero.

    About 3% of raw predictions come out negative, as low as -5 kWh, which no
    distribution loop can deliver. Clipping is not cosmetic: it is worth 11% on
    next-day MSE (0.756 -> 0.672).
    """
    features = list(params.feature_names)
    return np.maximum(design_matrix(df, features) @ params.dist_kwh_coefficients(), 0.0)


class HouseEnergyParamsComputer:
    """Fit house heating parameters from a chunk of hourly data"""
    def __init__(self, predictor=linear_regression):
        self.predictor = predictor

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df # TODO

    def fit(self, df: pd.DataFrame) -> HouseEnergyParams:
        feature_names = tuple(
            BASE_FEATURES + [f"set_minus_temp_zone{z}" for z in zone_numbers(df.columns)]
        )
        dist_kwh_pred, coefficients, std_errors, r_squared = self.predictor(df)
        # Restate the parameters as heat-pump thermal energy instead of energy
        # into the distribution loop.
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh_pred.sum())
        # Six decimals: enough for the small solar_w_m2 and
        # windspeed_times_deltaT coefficients to survive rounding, so predictions
        # rebuilt from the saved parameters match the fit.
        return HouseEnergyParams(
            feature_names=feature_names,
            values=tuple(round(c * energy_ratio, 6) for c in coefficients),
            std_errors=tuple(round(e * energy_ratio, 6) for e in std_errors),
            r_squared=round(r_squared, 3),
            energy_ratio=round(energy_ratio, 6),
        )
