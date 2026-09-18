import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Model features, in B1..B7 order. B0 is the intercept.
#   B1 deltaT
#   B2 windspeed_times_deltaT
#   B3 solar_w_m2
#   B4 previous_dist_kwh
#   B5 OAT_avg_6h
#   B6 set_minus_temp_zone1
#   B7 set_minus_temp_zone2
FEATURES = [
    "deltaT",
    "windspeed_times_deltaT",
    "solar_w_m2",
    "previous_dist_kwh",
    "OAT_avg_6h",
    "set_minus_temp_zone1",
    "set_minus_temp_zone2",
]
COEF_NAMES = [f"B{i}" for i in range(1 + len(FEATURES))]

REQUIRED_COLUMNS = [
    "oat_f", "ws_mph", "solar_w_m2", "dist_kwh", "hp_kwh_th",
    "T_i1_start", "T_i1_set_start", "T_i2_start", "T_i2_set_start",
]

# Hours of outdoor-temperature history behind the OAT_avg_6h feature.
LOOKBACK_HOURS = 6


def load_hourly_features(house_alias: str) -> pd.DataFrame:
    """Read one house's hourly export and build the seven model features.

    Every temperature in the file is Fahrenheit and stays that way; wind is
    already mph. No unit conversion anywhere.

    previous_dist_kwh and OAT_avg_6h look backwards, so an hour is only usable
    when its history is really there: the first LOOKBACK_HOURS rows of the file,
    and any hour whose preceding 6 hours are not contiguous in hour_start, are
    dropped rather than quietly averaged over the wrong hours.
    """
    # Glob the params export specifically. The older {house}_electricity_use_*.csv
    # lives in the same folder, so a looser glob would match both and pick one
    # arbitrarily.
    csv_path = glob.glob(f"data/{house_alias}_house_params_data.csv")[0]
    df = pd.read_csv(csv_path)
    df["hour_start"] = pd.to_datetime(df["hour_start"])
    df = df.sort_values("hour_start").reset_index(drop=True)

    t_i_avg = 0.5 * (df["T_i1_start"] + df["T_i2_start"])
    df["deltaT"] = (t_i_avg - df["oat_f"]).clip(lower=0)
    df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
    # solar_w_m2 is already a column in the export; no derived copy needed.
    # Plain setpoint minus room temperature: calibrating the setpoint onto the
    # room-sensor scale performs identically (the calibration slope is ~0.95 and
    # its offset is absorbed by the intercept), and splitting the gap into
    # below/above-setpoint terms is 30% worse under a rolling window.
    df["set_minus_temp_zone1"] = df["T_i1_set_start"] - df["T_i1_start"]
    df["set_minus_temp_zone2"] = df["T_i2_set_start"] - df["T_i2_start"]
    df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
    df["OAT_avg_6h"] = df["oat_f"].rolling(LOOKBACK_HOURS).mean().shift(1)

    # The 6 hours before h must be the 6 rows before it in the file.
    history_span = df["hour_start"] - df["hour_start"].shift(LOOKBACK_HOURS)
    df = df[history_span == pd.Timedelta(hours=LOOKBACK_HOURS)]

    df = df.dropna(subset=REQUIRED_COLUMNS + FEATURES).reset_index(drop=True)
    df["day"] = df["hour_start"].dt.normalize()
    return df


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    columns = [np.ones(len(df))]
    for name in FEATURES:
        columns.append(df[name].to_numpy())
    return np.column_stack(columns)


def alpha_beta_gamma_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Main-branch model: 1, oat_f, (65 - oat_f) * ws_mph."""
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
    B0: float
    B1: float
    B2: float
    B3: float
    B4: float
    B5: float
    B6: float
    B7: float
    std_error_B0: float
    std_error_B1: float
    std_error_B2: float
    std_error_B3: float
    std_error_B4: float
    std_error_B5: float
    std_error_B6: float
    std_error_B7: float
    r_squared: float
    # hp_kwh_th / dist_kwh the coefficients were multiplied by
    energy_ratio: float

    def coefficients(self) -> np.ndarray:
        """[B0, B1, ..., B7], in the unit the fit was scaled to."""
        return np.array([getattr(self, name) for name in COEF_NAMES])

    def dist_kwh_coefficients(self) -> np.ndarray:
        """The same coefficients back in distribution-kWh, the unit of dist_kwh."""
        return self.coefficients() / self.energy_ratio


def predict(params: HouseEnergyParams, df: pd.DataFrame) -> np.ndarray:
    """Predicted dist_kwh, clipped at zero.

    About 3% of raw predictions come out negative, as low as -5 kWh, which no
    distribution loop can deliver. Clipping is not cosmetic: it is worth 11% on
    next-day MSE (0.756 -> 0.672).
    """
    return np.maximum(design_matrix(df) @ params.dist_kwh_coefficients(), 0.0)


class HouseEnergyParamsComputer:
    """Fit house heating parameters from a chunk of hourly data"""
    def __init__(self, predictor=linear_regression):
        self.predictor = predictor

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df # TODO

    def fit(self, df: pd.DataFrame) -> HouseEnergyParams:
        dist_kwh_pred, coefficients, std_errors, r_squared = self.predictor(df)
        # Restate the parameters as heat-pump thermal energy instead of energy
        # into the distribution loop.
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh_pred.sum())
        # Six decimals: enough for the small solar_w_m2 and
        # windspeed_times_deltaT coefficients to survive rounding, so predictions
        # rebuilt from the saved parameters match the fit.
        return HouseEnergyParams(
            *[round(c * energy_ratio, 6) for c in coefficients],
            *[round(e * energy_ratio, 6) for e in std_errors],
            r_squared=round(r_squared, 3),
            energy_ratio=round(energy_ratio, 6),
        )
