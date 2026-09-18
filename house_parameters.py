import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")

FEATURES_WITHOUT_ZONES = [
    "deltaT",
    "windspeed_times_deltaT",
    "solar_w_m2",
    "previous_dist_kwh",
    "OAT_avg_6h",
]

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
        self.zones = sorted(
            int(n) for c in df.columns
            if (n := str(c).removeprefix("T_i").removesuffix("_start")).isdigit()
        )
        if not self.zones:
            raise ValueError("No T_i{z}_start columns found in export")

        # Calculate some of the features
        inside_temp_avg = df[[f"T_i{z}_start" for z in self.zones]].mean(axis=1)
        df["deltaT"] = (inside_temp_avg - df["oat_f"]).clip(lower=0)
        df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
        for z in self.zones:
            df[f"set_minus_temp_zone{z}"] = df[f"T_i{z}_set_start"] - df[f"T_i{z}_start"]
        df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
        df["OAT_avg_6h"] = df["oat_f"].rolling(6).mean().shift(1)
        df['windspeed_times_65_minus_oat'] = df["ws_mph"] * (65.0 - df["oat_f"])

        # Keep only rows with a history span of 6 hours, necessary for the OAT_avg_6h feature
        history_span = df["hour_start"] - df["hour_start"].shift(6)
        self.df = df[history_span == pd.Timedelta(hours=6)]

        self.feature_names = tuple(
            FEATURES_WITHOUT_ZONES + [f"set_minus_temp_zone{z}" for z in self.zones]
        )
        self.feature_names_baseline = tuple(FEATURES_BASELINE)

        # Filter out rows with missing critical data
        required = ["oat_f", "ws_mph", "solar_w_m2", "dist_kwh", "hp_kwh_th"] + [
            col for z in self.zones for col in (f"T_i{z}_start", f"T_i{z}_set_start")
        ]
        self.df = self.df.dropna(subset=required + list(self.feature_names)).reset_index(drop=True)

        # Find the range of outdoor temperatures (for the plot)
        oat_min = float(self.df["oat_f"].min())
        oat_max = float(self.df["oat_f"].max())
        if oat_min >= oat_max:
            oat_max = oat_min + 1.0
        self.oat_color_range_f = (oat_min, oat_max)

    def remove_outliers(self) -> None:
        pass

    def design_matrix(self, df: pd.DataFrame, *, baseline: bool = False) -> np.ndarray:
        feature_names = self.feature_names_baseline if baseline else self.feature_names
        return np.column_stack(
            [np.ones(len(df))] + [df[name].to_numpy() for name in feature_names]
        )

    def fit(self, df: pd.DataFrame, *, baseline: bool = False) -> HouseEnergyParams:
        X = self.design_matrix(df, baseline=baseline)
        feature_names = (self.feature_names_baseline if baseline else self.feature_names)

        dist_kwh = df["dist_kwh"].to_numpy()
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh.sum())
        y = dist_kwh * energy_ratio

        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ coefficients
        rank = int(np.linalg.matrix_rank(X))
        sigma2 = float(np.sum(residuals**2) / max(len(dist_kwh) - rank, 1))
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
            energy_ratio=round(energy_ratio, 3),
            baseline=baseline,
        )

    def predict(self, params: HouseEnergyParams, df: pd.DataFrame) -> np.ndarray:
        X = self.design_matrix(df, baseline=params.baseline)
        return np.maximum(X @ params.coefficients(), 0.0)

    def trailing_n_day_fits(self, n: int) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        
        days = np.sort(self.df["day"].unique())
        fit_days = []
        results: list[HouseEnergyParams] = []
        oos_oat_f = []
        oos_pred = []
        oos_pred_baseline = []
        oos_actual = []

        for i in range(n - 1, len(days)):
            # Fit the model to the previous n days
            window = days[i - n + 1 : i + 1]
            window_df = self.df[self.df["day"].isin(window)]
            fit_days.append(days[i])
            params = self.fit(window_df)
            params_baseline = self.fit(window_df, baseline=True)
            results.append(params)

            # Make an out-of-sample (oos) prediction for the next day
            if i + 1 < len(days):
                next_df = self.df[self.df["day"] == days[i + 1]]
                pred = self.predict(params, next_df)
                pred_baseline = self.predict(params_baseline, next_df)
                oos_pred.extend(pred)
                oos_pred_baseline.extend(pred_baseline)
                oos_actual.extend(next_df["dist_kwh"] * params.energy_ratio)
                oos_oat_f.extend(next_df["oat_f"])

        print(
            f"Fitted {len(results)} trailing {n}-day windows "
            f"from {pd.Timestamp(fit_days[0]).date()} to {pd.Timestamp(fit_days[-1]).date()}"
        )

        oos_pred = np.array(oos_pred)
        oos_pred_baseline = np.array(oos_pred_baseline)
        oos_actual = np.array(oos_actual)
        oos_oat_f = np.array(oos_oat_f)
        errors = oos_pred - oos_actual
        errors_baseline = oos_pred_baseline - oos_actual
        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt((errors**2).mean()))
        mae_baseline = float(np.abs(errors_baseline).mean())
        rmse_baseline = float(np.sqrt((errors_baseline**2).mean()))

        # Save the parameters to a CSV file
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

        print(
            f"Next-day out-of-sample over {len(oos_actual)} hours:\n"
            f"MAE = {mae:.1f} kWh (baseline {mae_baseline:.1f} kWh)\n"
            f"RMSE = {rmse:.1f} kWh (baseline {rmse_baseline:.1f} kWh)"
        )

        self.plot_pred_vs_actual(
            oos_pred, oos_actual, oos_oat_f,
            f"{self.house_alias.capitalize()}: next-day predicted vs actual (trailing {n}-day fits)",
            savepath=RESULTS_DIR / f"{self.house_alias}_pred_vs_actual_N{n}.png",
            baseline_mae=mae_baseline,
            baseline_rmse=rmse_baseline,
        )

    def sweep_n(self, min_n: int, max_n: int) -> None:
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(exist_ok=True)

        days = np.sort(self.df["day"].unique())
        rmses: list[float] = []
        b1_weekly_ranges: list[float] = []
        n_values = list(range(min_n, max_n + 1))

        for n in n_values:
            fit_days = []
            b1_values = []
            oos_pred = []
            oos_actual = []

            for i in range(n - 1, len(days) - 1):
                # Fit the model to the previous n days
                window = days[i - n + 1 : i + 1]
                window_df = self.df[self.df["day"].isin(window)]
                fit_days.append(days[i])
                params = self.fit(window_df)
                b1_values.append(params.B1)

                # Make an out-of-sample (oos) prediction for the next day
                next_df = self.df[self.df["day"] == days[i + 1]]
                pred = self.predict(params, next_df)
                actual = next_df["dist_kwh"] * params.energy_ratio
                oos_pred.extend(pred)
                oos_actual.extend(actual)

            errors = np.array(oos_pred) - np.array(oos_actual)
            rmse = float(np.sqrt((errors**2).mean()))
            b1 = pd.Series(b1_values, index=pd.DatetimeIndex(fit_days)).sort_index()
            weekly_b1_range = b1.rolling("7D").apply(lambda s: s.max() - s.min())
            b1_weekly_range = float(weekly_b1_range.mean())
            rmses.append(rmse)
            b1_weekly_ranges.append(b1_weekly_range)
            print(
                f"N={n:2d}: next-day RMSE={rmse:.4f} kWh, "
                f"avg weekly B1 (deltaT) range={b1_weekly_range:.5g}"
            )

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_xlabel("N (trailing days in fit window)")
        ax1.set_ylabel("Next-day RMSE (kWh)", color="tab:blue")
        ax1.plot(n_values, rmses, "o-", color="tab:blue", label="RMSE")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xticks(n_values)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg largest weekly B1 (deltaT) range", color="tab:red")
        ax2.plot(n_values, b1_weekly_ranges, "s--", color="tab:red", label="B1 weekly range")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle(f"{self.house_alias.capitalize()}: effect of lookback window N")
        fig.tight_layout()
        fig.savefig(
            RESULTS_DIR / f"{self.house_alias}_sweep_N.png", dpi=150, bbox_inches="tight"
        )
        plt.show()

    def _oat_colors(self, oat_f):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        oat_f = np.asarray(oat_f, dtype=float)
        vmin, vmax = self.oat_color_range_f
        return oat_f, Normalize(vmin=vmin, vmax=vmax), plt.cm.coolwarm

    def plot_pred_vs_actual(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        oat_f: np.ndarray,
        title: str,
        savepath: Path | None = None,
        baseline_mae: float | None = None,
        baseline_rmse: float | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable

        predicted = np.asarray(predicted)
        actual = np.asarray(actual)
        oat_f = np.asarray(oat_f)
        errors = predicted - actual

        oat_values, norm, cmap = self._oat_colors(oat_f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(actual, predicted, c=oat_values, cmap=cmap, norm=norm, s=8, alpha=0.6)
        hi = float(max(actual.max(), predicted.max()) * 1.05)
        limits = [0, hi]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("Actual scaled dist_kwh (kWh)")
        ax.set_ylabel("Predicted scaled dist_kwh (kWh)")
        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt((errors**2).mean()))

        scatter_title = "Next-day out-of-sample predictions"
        if baseline_mae is not None and baseline_rmse is not None:
            rmse_pct = (baseline_rmse - rmse) / baseline_rmse * 100.0
            mae_pct = (baseline_mae - mae) / baseline_mae * 100.0
            scatter_title += (
                "\n"
                + f"RMSE {abs(rmse_pct):.0f}% {'better' if rmse_pct >= 0 else 'worse'}"
                + ", "
                + f"MAE {abs(mae_pct):.0f}% {'better' if mae_pct >= 0 else 'worse'}"
                + " than αβγ"
            )
        ax.set_title(scatter_title)

        stats = f"MAE  = {mae:.3f} kWh\n"
        if baseline_mae is not None:
            stats += f"MAE αβγ = {baseline_mae:.3f} kWh\n"
        stats += f"RMSE = {rmse:.3f} kWh\n"
        if baseline_rmse is not None:
            stats += f"RMSE αβγ = {baseline_rmse:.3f} kWh"
        stats = stats.rstrip()
        ax.text(
            0.97, 0.03, stats, transform=ax.transAxes, ha="right", va="bottom",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax = axes[1]
        ax.hist(errors, bins=60, range=(-6, 6), color="tab:blue", alpha=0.8)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xlim(-6, 6)
        ax.set_xlabel("Prediction error (scaled kWh)")
        ax.set_ylabel("Hours")
        ax.set_title("Error distribution")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.02)
        ticks = np.linspace(norm.vmin, norm.vmax, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}°F" for t in ticks])

        fig.suptitle(title)

        if savepath is not None:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
