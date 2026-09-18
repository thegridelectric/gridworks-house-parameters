import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

"""
Model features in B1.. order. B0 is the intercept. Zone terms follow the base five.
  B0
  B1 deltaT
  B2 windspeed_times_deltaT
  B3 solar_w_m2
  B4 previous_dist_kwh
  B5 OAT_avg_6h
  B6+ set_minus_temp_zone{z} for each zone z in the export
"""

FEATURES_WITHOUT_ZONES = [
    "deltaT",
    "windspeed_times_deltaT",
    "solar_w_m2",
    "previous_dist_kwh",
    "OAT_avg_6h",
]

"""
Baseline model (alpha-beta-gamma) features:
  B0
  B1 oat_f
  B2 ws_mph * (65-oat_f)
"""

FEATURES_BASELINE = [
    "oat_f",
    "windspeed_times_fixed_deltaT",
]


@dataclass
class HouseEnergyParams:
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    std_errors: tuple[float, ...]
    r_squared: float
    energy_ratio: float
    baseline: bool = False

    @property
    def coef_names(self) -> tuple[str, ...]:
        return tuple(f"B{i}" for i in range(len(self.values)))

    def coefficients(self) -> np.ndarray:
        return np.array(self.values, dtype=float)

    def __getattr__(self, name: str):
        if name.startswith("std_error_B") and name[len("std_error_B"):].isdigit():
            return self.std_errors[int(name[len("std_error_B"):])]
        if name.startswith("B") and len(name) > 1 and name[1:].isdigit():
            return self.values[int(name[1:])]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


class HouseEnergyParamsComputer:
    def __init__(self, house_alias: str):
        self.house_alias = house_alias
        self.load_data()

    def load_data(self) -> None:
        csv_path = glob.glob(f"data/{self.house_alias}_house_params_data.csv")[0]
        df = pd.read_csv(csv_path)

        df["hour_start"] = pd.to_datetime(df["hour_start"])
        df = df.sort_values("hour_start").reset_index(drop=True)
        df["day"] = df["hour_start"].dt.normalize()

        # Find the number of zones
        zones = sorted(
            int(n) for c in df.columns
            if (n := str(c).removeprefix("T_i").removesuffix("_start")).isdigit()
        )
        if not zones:
            raise ValueError("No T_i{z}_start columns found in export")

        # Calculate some of the features
        inside_temp_avg = df[[f"T_i{z}_start" for z in zones]].mean(axis=1)
        df["deltaT"] = (inside_temp_avg - df["oat_f"]).clip(lower=0)
        df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
        for z in zones:
            df[f"set_minus_temp_zone{z}"] = df[f"T_i{z}_set_start"] - df[f"T_i{z}_start"]
        df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
        df["OAT_avg_6h"] = df["oat_f"].rolling(6).mean().shift(1)

        # Keep only rows with a history span of 6 hours, necessary for the OAT_avg_6h feature
        history_span = df["hour_start"] - df["hour_start"].shift(6)
        df = df[history_span == pd.Timedelta(hours=6)]

        feature_names = tuple(
            FEATURES_WITHOUT_ZONES + [f"set_minus_temp_zone{z}" for z in zones]
        )
        required = ["oat_f", "ws_mph", "solar_w_m2", "dist_kwh", "hp_kwh_th"] + [
            col for z in zones for col in (f"T_i{z}_start", f"T_i{z}_set_start")
        ]
        df = df.dropna(subset=required + list(feature_names)).reset_index(drop=True)

        self.df = df
        self.zones = zones
        self.feature_names = feature_names

    def remove_outliers(self) -> None:
        pass

    def design_matrix(self, df: pd.DataFrame, *, baseline: bool = False, feature_names: list[str] | None = None) -> np.ndarray:
        if baseline:
            oat_f = df["oat_f"].to_numpy()
            return np.column_stack([np.ones(len(df)), oat_f, (65.0 - oat_f) * df["ws_mph"].to_numpy()])
        names = feature_names or list(self.feature_names)
        columns = [np.ones(len(df))]
        for name in names:
            columns.append(df[name].to_numpy())
        return np.column_stack(columns)

    def fit(self, df: pd.DataFrame, *, baseline: bool = False) -> HouseEnergyParams:
        X = self.design_matrix(df, baseline=baseline)
        feature_names = (FEATURES_BASELINE if baseline else self.feature_names)

        dist_kwh = df["dist_kwh"].to_numpy()
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh.sum())
        y = dist_kwh * energy_ratio

        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefficients
        n = len(dist_kwh)
        rank = int(np.linalg.matrix_rank(X))
        sigma2 = float(np.sum(residuals**2) / max(n - rank, 1))
        std_errors = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residuals**2)) / ss_tot

        return HouseEnergyParams(
            feature_names=feature_names,
            values=tuple(round(float(c), 6) for c in coefficients),
            std_errors=tuple(round(float(e), 6) for e in std_errors),
            r_squared=round(r_squared, 3),
            energy_ratio=round(energy_ratio, 6),
            baseline=baseline,
        )

    def predict(self, params: HouseEnergyParams, df: pd.DataFrame) -> np.ndarray:
        X = self.design_matrix(df, baseline=params.baseline)
        return np.maximum(X @ params.coefficients(), 0.0)
