import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")

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
    "windspeed_times_65_minus_oat",
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

    def design_matrix(self, df: pd.DataFrame, *, baseline: bool = False) -> np.ndarray:
        if baseline:
            oat_f = df["oat_f"].to_numpy()
            return np.column_stack([np.ones(len(df)), oat_f, (65.0 - oat_f) * df["ws_mph"].to_numpy()])
        columns = [np.ones(len(df))]
        for name in self.feature_names:
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
        n_params = X.shape[1]
        if rank < n_params:
            std_errors = np.full(n_params, np.nan)
        else:
            std_errors = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
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

    def trailing_n_day_fits(self, n: int) -> None:
        from plotter import plot_pred_vs_actual

        RESULTS_DIR.mkdir(exist_ok=True)
        
        days = np.sort(self.df["day"].unique())
        fit_days = []
        results: list[HouseEnergyParams] = []
        oos_oat_f = []
        oos_pred = []
        oos_pred_abg = []
        oos_actual_scaled = []

        for i in range(n - 1, len(days)):
            window = days[i - n + 1 : i + 1]
            window_df = self.df[self.df["day"].isin(window)]
            fit_days.append(days[i])
            params = self.fit(window_df)
            results.append(params)
            abg_params = self.fit(window_df, baseline=True)

            if i + 1 < len(days):
                next_df = self.df[self.df["day"] == days[i + 1]]
                oos_oat_f.extend(next_df["oat_f"])
                ratio = params.energy_ratio
                oos_pred.extend(self.predict(params, next_df))
                oos_pred_abg.extend(self.predict(abg_params, next_df))
                oos_actual_scaled.extend(next_df["dist_kwh"] * ratio)

        print(
            f"Fitted {len(results)} trailing {n}-day windows "
            f"from {pd.Timestamp(fit_days[0]).date()} to {pd.Timestamp(fit_days[-1]).date()}"
        )

        oos_pred = np.array(oos_pred)
        oos_pred_abg = np.array(oos_pred_abg)
        oos_actual_scaled = np.array(oos_actual_scaled)
        oos_oat_f = np.array(oos_oat_f)
        errors = oos_pred - oos_actual_scaled
        errors_abg = oos_pred_abg - oos_actual_scaled
        mae_abg = float(np.abs(errors_abg).mean())
        rmse_abg = float(np.sqrt((errors_abg**2).mean()))

        print(
            f"Next-day out-of-sample (scaled kWh) over {len(oos_actual_scaled)} hours: "
            f"MSE={float((errors**2).mean()):.3f}, "
            f"RMSE={float(np.sqrt((errors**2).mean())):.3f}, "
            f"MAE={float(np.abs(errors).mean()):.3f} kWh, "
            f"MAE αβγ={mae_abg:.3f} kWh, RMSE αβγ={rmse_abg:.3f} kWh"
        )

        plot_pred_vs_actual(
            oos_pred, oos_actual_scaled, oos_oat_f,
            f"{self.house_alias.capitalize()}: next-day predicted vs actual (trailing {n}-day fits)",
            savepath=RESULTS_DIR / f"{self.house_alias}_pred_vs_actual_N{n}.png",
            baseline_mae=mae_abg,
            baseline_rmse=rmse_abg,
        )

        coef_names = list(results[0].coef_names)
        params_table = pd.DataFrame(
            {name: [getattr(r, name) for r in results] for name in coef_names}
            | {
                f"std_error_{name}": [getattr(r, f"std_error_{name}") for r in results]
                for name in coef_names
            }
            | {
                "r_squared": [r.r_squared for r in results],
                "energy_ratio": [r.energy_ratio for r in results],
            },
            index=pd.DatetimeIndex(fit_days),
        ).sort_index()
        params_table.to_csv(RESULTS_DIR / f"{self.house_alias}_params_N{n}.csv")

    def sweep_n(self, min_n: int, max_n: int) -> None:
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(exist_ok=True)
        days = np.sort(self.df["day"].unique())
        mses: list[float] = []
        intercept_vars: list[float] = []

        n_values = list(range(min_n, max_n+1))

        for n in n_values:
            fit_days = []
            intercepts = []
            sse = 0.0
            n_points = 0
            for i in range(n - 1, len(days) - 1):
                window = days[i - n + 1 : i + 1]
                window_df = self.df[self.df["day"].isin(window)]
                params = self.fit(window_df)
                intercepts.append(params.B0)
                fit_days.append(days[i])

                next_df = self.df[self.df["day"] == days[i + 1]]
                scaled_hat = self.predict(params, next_df)
                actual_scaled = next_df["dist_kwh"].to_numpy() * params.energy_ratio
                sse += float(np.sum((scaled_hat - actual_scaled) ** 2))
                n_points += len(next_df)

            mse = sse / n_points
            intercept = pd.Series(intercepts, index=pd.DatetimeIndex(fit_days)).sort_index()
            weekly_intercept_range = intercept.rolling("7D").apply(lambda s: s.max() - s.min())
            intercept_var = float(weekly_intercept_range.mean())
            mses.append(mse)
            intercept_vars.append(intercept_var)
            print(
                f"N={n:2d}: next-day scaled MSE={mse:.4f}, "
                f"avg weekly intercept variation={intercept_var:.3f}"
            )

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_xlabel("N (trailing days in fit window)")
        ax1.set_ylabel("Next-day scaled energy MSE", color="tab:blue")
        ax1.plot(n_values, mses, "o-", color="tab:blue", label="MSE")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xticks(n_values)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg largest weekly intercept variation", color="tab:red")
        ax2.plot(n_values, intercept_vars, "s--", color="tab:red", label="intercept variation")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle(f"{self.house_alias.capitalize()}: effect of lookback window N")
        fig.tight_layout()
        fig.savefig(
            RESULTS_DIR / f"{self.house_alias}_sweep_N.png", dpi=150, bbox_inches="tight"
        )
        plt.show()