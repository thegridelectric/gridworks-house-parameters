import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")


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
    RANGE_OF_VALID_VALUES_PER_CHANNEL = {
        "oat_f": (-128, 134),
        "ws_mph": (0, 254),
        "solar_w_m2": (0, 1361),
        "dist_kwh": (0, 30),
        "hp_kwh_th": (0, 30),
        "zone_setpoint_f": (40, 90),
    }
    MAX_ABS_RATE_OF_CHANGE_PER_CHANNEL = {
        "oat_f": 25.0,
    }
    MAX_GAP_HOURS = 4
    FEATURE_NAMES = (
        "deltaT",
        "windspeed_times_deltaT",
        "solar_w_m2",
        "previous_dist_kwh",
        "OAT_avg_6h",
    )
    FEATURE_NAMES_BASELINE = (
        "oat_f",
        "windspeed_times_65_minus_oat",
    )
    TRAINING_FREQUENCY: Literal["daily", "weekly"] = "weekly"
    GROW_WINDOW_TO_N: bool = True

    def __init__(self, house_alias: str):
        self.house_alias = house_alias
        self.load_data()

    def load_data(self) -> None:
        csv_path = glob.glob(f"data/{self.house_alias}_house_params_data.csv")[0]
        df = pd.read_csv(csv_path)
        print(f"Length of df: {len(df)}")

        df["hour_start"] = pd.to_datetime(df["hour_start"])
        df = df.sort_values("hour_start").reset_index(drop=True)
        df["day"] = df["hour_start"].dt.normalize()

        # Find the number of zones
        zones = sorted(
            int(n) for c in df.columns
            if str(c).endswith("_set_start")
            and (n := str(c).removeprefix("T_i").removesuffix("_set_start")).isdigit()
        )
        
        # Columns that are required for the model
        required = ["oat_f", "ws_mph", "solar_w_m2", "dist_kwh", "hp_kwh_th"] + [
            f"T_i{z}_set_start" for z in zones
        ]
        missing_columns = [col for col in required if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV for {self.house_alias}: {missing_columns}")

        # Calculate some of the features
        setpoint_avg = df[[f"T_i{z}_set_start" for z in zones]].mean(axis=1)
        df["deltaT"] = (setpoint_avg - df["oat_f"]).clip(lower=0)
        df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
        df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
        df["OAT_avg_6h"] = df["oat_f"].rolling(6).mean().shift(1)
        df['windspeed_times_65_minus_oat'] = df["ws_mph"] * (65.0 - df["oat_f"])

        df_before_cleaning = df.copy()

        # Replace erroneous data and outliers with NaN
        df = self._remove_erroneous_data(df)
        df = self._remove_outliers(df)

        # Decide when to interpolate and when to drop rows
        df = self._handle_missing_data(df)
        self._plot_data_distribution(df_before_cleaning, df)

        exit()

        # Keep only rows with a history span of 6 hours, necessary for the OAT_avg_6h feature
        history_span = df["hour_start"] - df["hour_start"].shift(6)
        df = df[history_span == pd.Timedelta(hours=6)]
        print(f"Length of df after removing rows without a history span of 6 hours: {len(df)}")

        self.df = df

        # Find the range of outdoor temperatures (for the plot)
        oat_min = float(self.df["oat_f"].min())
        oat_max = float(self.df["oat_f"].max())
        if oat_min >= oat_max:
            oat_max = oat_min + 1.0
        self.oat_color_range_f = (oat_min, oat_max)

    def _log_erroneous_values(self, df: pd.DataFrame, mask: pd.Series, channel: str, *,reason: str) -> None:
        for hour_start, value in df.loc[mask, ["hour_start", channel]].itertuples(index=False):
            print(f"Erroneous data ({reason}): channel={channel} hour_start={hour_start} value={value}")

    def _remove_erroneous_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace slight negative values with 0
        df.loc[df["hp_kwh_th"].between(-0.2, 0), "hp_kwh_th"] = 0

        # Remove data that is outside the range of valid values
        range_of_valid_values_per_channel = dict(self.RANGE_OF_VALID_VALUES_PER_CHANNEL)
        for col in df.columns:
            if str(col).startswith("T_i") and str(col).endswith("_set_start"):
                range_of_valid_values_per_channel[col] = range_of_valid_values_per_channel["zone_setpoint_f"]
        range_of_valid_values_per_channel.pop("zone_setpoint_f")

        nans_added_range = 0
        for channel, (min_value, max_value) in range_of_valid_values_per_channel.items():
            out_of_range = df[channel].notna() & ~df[channel].between(min_value, max_value)
            nans_added_range += int(out_of_range.sum())
            self._log_erroneous_values(
                df,
                out_of_range,
                channel,
                reason=f"valid range [{min_value}, {max_value}]",
            )
            df[channel] = df[channel].where(df[channel].between(min_value, max_value))
        print(f"Erroneous data (valid range): {nans_added_range} NaNs added")

        # Remove data that is outside the maximum absolute rate of change
        nans_added_roc = 0
        for channel, max_abs_change in self.MAX_ABS_RATE_OF_CHANGE_PER_CHANNEL.items():
            previous = df[channel].shift(1)
            abs_change = (df[channel] - previous).abs()
            roc_ok = abs_change.le(max_abs_change) | previous.isna()
            excessive_roc = df[channel].notna() & ~roc_ok
            nans_added_roc += int(excessive_roc.sum())
            self._log_erroneous_values(
                df,
                excessive_roc,
                channel,
                reason=f"rate of change > {max_abs_change}",
            )
            df[channel] = df[channel].where(roc_ok | df[channel].isna())
        print(f"Erroneous data (rate of change): {nans_added_roc} NaNs added")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Insert missing rows (need one row per hour)
        df = df.copy()
        df["hour_start"] = pd.to_datetime(df["hour_start"])
        full_hours = pd.date_range(df["hour_start"].min(), df["hour_start"].max(), freq="h")
        n_before = len(df)
        df = df.set_index("hour_start").reindex(full_hours).reset_index(names="hour_start")
        df["day"] = df["hour_start"].dt.normalize()
        print(f"Missing timestamps: inserted {len(df) - n_before} hourly rows")

        # Drop rows with long missing-data gaps
        drop_rows = pd.Series(False, index=df.index)
        for channel in df.columns:
            is_nan = df[channel].isna()
            block_id = is_nan.ne(is_nan.shift()).cumsum()
            block_len = is_nan.groupby(block_id).transform("size")
            drop_rows |= is_nan & (block_len >= self.MAX_GAP_HOURS)

        print(f"Long missing-data gaps (>={self.MAX_GAP_HOURS}h): dropped {int(drop_rows.sum())} rows")
        df = df.loc[~drop_rows].reset_index(drop=True)

        # Interpolate missing data
        non_interpolated_columns = {"hour_start", "day"}
        df = df.set_index("hour_start")
        n_filled = 0
        for channel in df.columns:
            if channel in non_interpolated_columns:
                continue
            before = df[channel].copy()
            df[channel] = df[channel].interpolate(method="time")
            n_filled += int((before.isna() & df[channel].notna()).sum())
        df = df.reset_index(names="hour_start")
        print(f"Interpolation: {n_filled} values filled")

        # If any rows still have NaNs, drop them
        remaining_nans = int(df.isna().sum().sum())
        print(f"NaNs remaining: {remaining_nans}")
        if remaining_nans > 0:
            n_before = len(df)
            df = df.dropna().reset_index(drop=True)
            print(f"Dropped {n_before - len(df)} rows with NaNs")

        return df

    def _plot_data_distribution(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        range_of_valid_values_per_channel = dict(self.RANGE_OF_VALID_VALUES_PER_CHANNEL)
        for col in df_before.columns:
            if str(col).startswith("T_i") and str(col).endswith("_set_start"):
                range_of_valid_values_per_channel[col] = range_of_valid_values_per_channel[
                    "zone_setpoint_f"
                ]
        range_of_valid_values_per_channel.pop("zone_setpoint_f")
        channels = list(range_of_valid_values_per_channel.keys())
        n_channels = len(channels)
        fig, axes = plt.subplots(
            2,
            n_channels,
            figsize=(max(2.5 * n_channels, 8), 8),
            squeeze=False,
        )
        row_labels = ("Before cleaning", "After cleaning")
        for row, (label, df) in enumerate(zip(row_labels, (df_before, df_after))):
            for col_idx, channel in enumerate(channels):
                ax = axes[row, col_idx]
                ax.boxplot(df[channel].dropna().to_numpy(), vert=True)
                if row == 0:
                    n_nans = int(df_before[channel].isna().sum())
                    nan_label = "NaN" if n_nans == 1 else "NaNs"
                    ax.set_title(f"{channel} ({n_nans} {nan_label})", fontsize=9)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            axes[row, 0].set_ylabel(label)
        fig.suptitle(f"{self.house_alias.capitalize()}: channel distributions")
        fig.tight_layout()
        RESULTS_DIR.mkdir(exist_ok=True)
        savepath = RESULTS_DIR / f"{self.house_alias}_channel_boxplots.png"
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.show()

    def design_matrix(self, df: pd.DataFrame, *, baseline: bool = False) -> np.ndarray:
        feature_names = self.FEATURE_NAMES_BASELINE if baseline else self.FEATURE_NAMES
        return np.column_stack(
            [np.ones(len(df))] + [df[name].to_numpy() for name in feature_names]
        )

    def fit(self, df: pd.DataFrame, *, baseline: bool = False) -> HouseEnergyParams:
        X = self.design_matrix(df, baseline=baseline)
        feature_names = self.FEATURE_NAMES_BASELINE if baseline else self.FEATURE_NAMES

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

    def _should_refit_trailing_window(self, day_index: int, days: np.ndarray, last_fit_index: int | None) -> bool:
        if self.TRAINING_FREQUENCY == "daily" or last_fit_index is None:
            return True
        return days[day_index] - days[last_fit_index] >= pd.Timedelta(days=7)

    def _fit_window_days(self, days: np.ndarray, day_index: int, n: int) -> np.ndarray:
        if self.GROW_WINDOW_TO_N and day_index + 1 < n:
            return days[0 : day_index + 1]
        return days[day_index - n + 1 : day_index + 1]

    def trailing_n_day_fits(self, n: int) -> None:
        RESULTS_DIR.mkdir(exist_ok=True)
        
        days = np.sort(self.df["day"].unique())
        fit_days = []
        results: list[HouseEnergyParams] = []
        oos_oat_f = []
        oos_pred = []
        oos_pred_baseline = []
        oos_actual = []

        params: HouseEnergyParams | None = None
        params_baseline: HouseEnergyParams | None = None
        last_fit_index: int | None = None

        first_fit_day_index = 0 if self.GROW_WINDOW_TO_N else n-1
        oos_future_days = 7 if self.TRAINING_FREQUENCY == "weekly" else 1
        
        for i in range(first_fit_day_index, len(days)):
            refitted = False
            if self._should_refit_trailing_window(i, days, last_fit_index):
                window = self._fit_window_days(days, i, n)
                window_df = self.df[self.df["day"].isin(window)]
                fit_days.append(days[i])
                params = self.fit(window_df)
                params_baseline = self.fit(window_df, baseline=True)
                results.append(params)
                last_fit_index = i
                refitted = True

            if params is None or params_baseline is None:
                continue
            if self.TRAINING_FREQUENCY == "weekly" and not refitted:
                continue
            if self.TRAINING_FREQUENCY == "daily" and i + 1 >= len(days):
                continue

            for j in range(i + 1, min(i + 1 + oos_future_days, len(days))):
                day_df = self.df[self.df["day"] == days[j]]
                oos_pred.extend(self.predict(params, day_df))
                oos_pred_baseline.extend(self.predict(params_baseline, day_df))
                oos_actual.extend(day_df["dist_kwh"] * params.energy_ratio)
                oos_oat_f.extend(day_df["oat_f"])

        oos_label = "next-week" if self.TRAINING_FREQUENCY == "weekly" else "next-day"
        print(
            f"Fitted {len(results)} {self.TRAINING_FREQUENCY} training, {'with' if self.GROW_WINDOW_TO_N else 'no'} growing window)"
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
        improvement_rmse_percentage = (rmse_baseline - rmse) / rmse_baseline * 100.0
        improvement_mae_percentage = (mae_baseline - mae) / mae_baseline * 100.0

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
        growing_window_label = 'growing' if self.GROW_WINDOW_TO_N else 'fixed'
        params_table.to_csv(RESULTS_DIR / f"{self.house_alias}_params_N{n}_{self.TRAINING_FREQUENCY}_{growing_window_label}.csv")

        print(
            f"{oos_label.capitalize()} out-of-sample over {len(oos_actual)} hours:\n"
            f"MAE = {mae:.2f} kWh (baseline {mae_baseline:.2f} kWh, {improvement_mae_percentage:.1f}% improvement)\n"
            f"RMSE = {rmse:.2f} kWh (baseline {rmse_baseline:.2f} kWh, {improvement_rmse_percentage:.1f}% improvement)"
        )

        self.plot_pred_vs_actual(
            oos_pred, oos_actual, oos_oat_f,
            (
                f"{self.house_alias.capitalize()}: {oos_label} predicted vs actual "
                f"({self.TRAINING_FREQUENCY} training, {'with' if self.GROW_WINDOW_TO_N else 'no'} growing window)"
            ),
            savepath=RESULTS_DIR / f"{self.house_alias}_pred_vs_actual_N{n}_{self.TRAINING_FREQUENCY}_{growing_window_label}.png",
            baseline_mae=mae_baseline,
            baseline_rmse=rmse_baseline,
            oos_period_label=oos_label,
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
            params: HouseEnergyParams | None = None
            last_fit_index: int | None = None

            first_fit_day_index = 0 if self.GROW_WINDOW_TO_N else n-1
            oos_future_days = 7 if self.TRAINING_FREQUENCY == "weekly" else 1

            for i in range(first_fit_day_index, len(days)):
                refitted = False
                if self._should_refit_trailing_window(i, days, last_fit_index):
                    window = self._fit_window_days(days, i, n)
                    window_df = self.df[self.df["day"].isin(window)]
                    fit_days.append(days[i])
                    params = self.fit(window_df)
                    b1_values.append(params.B1)
                    last_fit_index = i
                    refitted = True

                if params is None:
                    continue
                if self.TRAINING_FREQUENCY == "weekly" and not refitted:
                    continue
                if self.TRAINING_FREQUENCY == "daily" and i + 1 >= len(days):
                    continue

                for j in range(i + 1, min(i + 1 + oos_future_days, len(days))):
                    day_df = self.df[self.df["day"] == days[j]]
                    oos_pred.extend(self.predict(params, day_df))
                    oos_actual.extend(day_df["dist_kwh"] * params.energy_ratio)

            errors = np.array(oos_pred) - np.array(oos_actual)
            rmse = float(np.sqrt((errors**2).mean()))
            b1 = pd.Series(b1_values, index=pd.DatetimeIndex(fit_days)).sort_index()
            if self.TRAINING_FREQUENCY == "daily":
                b1_stability = b1.rolling("7D").apply(lambda s: s.max() - s.min())
                b1_stability_metric = float(b1_stability.mean())
            else:
                b1_stability_metric = float(b1.diff().abs().mean())
            rmses.append(rmse)
            b1_weekly_ranges.append(b1_stability_metric)
            oos_label = "next-week" if self.TRAINING_FREQUENCY == "weekly" else "next-day"
            print(
                f"N={n:2d}: {oos_label} RMSE={rmse:.4f} kWh, "
                f"B1 stability={b1_stability_metric:.5g}"
            )

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_xlabel("N (trailing days in fit window)")
        rmse_ylabel = "Next-week RMSE (kWh)" if self.TRAINING_FREQUENCY == "weekly" else "Next-day RMSE (kWh)"
        ax1.set_ylabel(rmse_ylabel, color="tab:blue")
        ax1.plot(n_values, rmses, "o-", color="tab:blue", label="RMSE")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_xticks(n_values)

        ax2 = ax1.twinx()
        b1_ylabel = (
            "Avg largest weekly B1 (deltaT) range"
            if self.TRAINING_FREQUENCY == "daily"
            else "Avg |ΔB1| between weekly refits"
        )
        ax2.set_ylabel(b1_ylabel, color="tab:red")
        ax2.plot(n_values, b1_weekly_ranges, "s--", color="tab:red", label="B1 stability")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        window_mode = "grow to N" if self.GROW_WINDOW_TO_N else "fixed trailing"
        fig.suptitle(
            f"{self.house_alias.capitalize()}: effect of lookback window N "
            f"({self.TRAINING_FREQUENCY} training, {window_mode})"
        )
        fig.tight_layout()
        fig.savefig(
            RESULTS_DIR / f"{self.house_alias}_sweep_N_{self.TRAINING_FREQUENCY}.png",
            dpi=150, bbox_inches="tight",
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
        oos_period_label: str = "next-day",
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

        scatter_title = f"{oos_period_label.capitalize()} out-of-sample predictions"
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
