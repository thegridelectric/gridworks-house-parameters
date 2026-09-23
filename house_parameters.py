import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
logger = logging.getLogger(__name__)


def _init_module_logging() -> None:
    level_name = os.environ.get("HOUSE_PARAMS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)


_init_module_logging()


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
    # Features
    FEATURE_NAMES = (
        "deltaT",
        "windspeed_times_deltaT",
        "solar_w_m2",
        "previous_dist_kwh_scaled",
        "OAT_avg_6h",
    )
    FEATURE_NAMES_BASELINE = (
        "oat_f",
        "windspeed_times_65_minus_oat",
    )

    # Training
    TRAINING_FREQUENCY: Literal["daily", "weekly"] = "daily"
    GROW_WINDOW_TO_N: bool = False
    FORECAST_HORIZON_HOURS = 48
    
    # Occupancy
    INCLUDE_OCCUPANCY: bool = True
    OCCUPANCY_INDIVIDUAL_HOURS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21)
    WEEKDAY_OCCUPANCY_FEATURE_NAMES = tuple(
        f"wd_hour_{hour}" for hour in OCCUPANCY_INDIVIDUAL_HOURS
    )
    WEEKEND_OCCUPANCY_FEATURE_NAMES = tuple(
        f"we_hour_{hour}" for hour in OCCUPANCY_INDIVIDUAL_HOURS
    )
    OCCUPANCY_FEATURE_NAMES = WEEKDAY_OCCUPANCY_FEATURE_NAMES + WEEKEND_OCCUPANCY_FEATURE_NAMES

    # Data cleaning
    RANGE_OF_VALID_VALUES_PER_CHANNEL = {
        "oat_f": (-128, 134),
        "ws_mph": (0, 254),
        "solar_w_m2": (0, 1361),
        "dist_kwh": (0, 25),
        "hp_kwh_th": (0, 25),
        "zone_set_or_temp_f": (40, 90),
        "zone_heatcall_fraction": (0, 1),
    }
    MAX_ABS_RATE_OF_CHANGE_PER_CHANNEL = {
        "oat_f": 25.0,
    }
    MAX_GAP_HOURS = 4
    BROKEN_THERMOSTAT_MIN_TEMP_SET_GAP_F = {
        'default': 3,
        'beech': {'zone2': 4.5,}
    }
    EXTERNAL_HEAT_SOURCE_MIN_TEMP_ABOVE_SET_F = 3
    OIL_BOILER_POWER_THRESHOLD = 50

    def __init__(self, house_alias: str):
        self.house_alias = house_alias
        self.results_dir = RESULTS_DIR / house_alias
        self.load_data()

    def _log_info(self, message: str) -> None:
        logger.info("[%s] %s", self.house_alias, message)

    def _log_debug(self, message: str) -> None:
        logger.debug("[%s] %s", self.house_alias, message)

    @property
    def _artifact_label(self) -> str:
        window = "_growing" if self.GROW_WINDOW_TO_N else ""
        occupancy = "_occupancy" if self.INCLUDE_OCCUPANCY else ""
        return f"{self.TRAINING_FREQUENCY}{window}{occupancy}_recursive{self.FORECAST_HORIZON_HOURS}h"

    def load_data(self) -> None:
        csv_path = glob.glob(f"data/{self.house_alias}_house_params_data.csv")[0]
        df = pd.read_csv(csv_path)
        self._log_info(f"Length of df: {len(df)} hours")

        df["hour_start"] = pd.to_datetime(df["hour_start"])
        df = df.sort_values("hour_start").reset_index(drop=True)
        df["day"] = df["hour_start"].dt.normalize()

        self.zones = sorted(
            int(str(c).removeprefix("zone").removesuffix("_avg_set"))
            for c in df.columns
            if str(c).startswith("zone") and str(c).endswith("_avg_set")
        )

        # Columns that are required for the model
        self.required = ["oat_f", "ws_mph", "solar_w_m2", "dist_kwh", "hp_kwh_th"]
        for z in self.zones:
            self.required.extend(
                [
                    f"zone{z}_avg_temp",
                    f"zone{z}_avg_set",
                    f"zone{z}_heatcall_fraction",
                ]
            )
        missing_columns = [col for col in self.required if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV for {self.house_alias}: {missing_columns}")

        # Clean the data
        df_before_cleaning = df.copy()
        df = self._remove_erroneous_data(df)
        df = self._remove_outliers(df)
        df = self._handle_missing_data(df)
        df = self._filter_out_known_bad_data(df)
        # self._plot_data_distribution(df_before_cleaning, df)
   
        # Calculate some of the features (OAT_avg_6h is built in _handle_missing_data)
        setpoint_avg = df[[f"zone{z}_avg_set" for z in self.zones]].mean(axis=1)
        df["deltaT"] = (setpoint_avg - df["oat_f"]).clip(lower=0)
        df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
        df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
        df = df.drop(0).reset_index(drop=True)
        df['windspeed_times_65_minus_oat'] = df["ws_mph"] * (65.0 - df["oat_f"])
        if self.INCLUDE_OCCUPANCY:
            hour = df["hour_start"].dt.hour
            is_weekday = df["hour_start"].dt.dayofweek < 5
            is_weekend = ~is_weekday
            for occupancy_hour in self.OCCUPANCY_INDIVIDUAL_HOURS:
                at_hour = hour == occupancy_hour
                df[f"wd_hour_{occupancy_hour}"] = (is_weekday & at_hour).astype(float)
                df[f"we_hour_{occupancy_hour}"] = (is_weekend & at_hour).astype(float)
            self.feature_names = self.FEATURE_NAMES + self.OCCUPANCY_FEATURE_NAMES
        else:
            self.feature_names = self.FEATURE_NAMES

        self.df = df

        # Find the range of outdoor temperatures (for the plot)
        oat_min = float(self.df["oat_f"].min())
        oat_max = float(self.df["oat_f"].max())
        if oat_min >= oat_max:
            oat_max = oat_min + 1.0
        self.oat_color_range_f = (oat_min, oat_max)

    def _remove_erroneous_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replace slight negative values with 0
        df.loc[df["hp_kwh_th"].between(-1, 0), "hp_kwh_th"] = 0

        # Remove data that is outside the range of valid values
        range_of_valid_values_per_channel = dict(self.RANGE_OF_VALID_VALUES_PER_CHANNEL)
        for col in df.columns:
            col_str = str(col)
            if col_str.startswith("zone") and col_str.endswith(("_avg_set", "_avg_temp")):
                range_of_valid_values_per_channel[col] = range_of_valid_values_per_channel[
                    "zone_set_or_temp_f"
                ]
            if str(col).startswith("zone") and str(col).endswith("_heatcall_fraction"):
                range_of_valid_values_per_channel[col] = range_of_valid_values_per_channel["zone_heatcall_fraction"]
        range_of_valid_values_per_channel.pop("zone_set_or_temp_f")
        range_of_valid_values_per_channel.pop("zone_heatcall_fraction")

        nans_added_range = 0
        for channel, (min_value, max_value) in range_of_valid_values_per_channel.items():
            out_of_range = df[channel].notna() & ~df[channel].between(min_value, max_value)
            nans_added_range += int(out_of_range.sum())
            reason = f"valid range [{min_value}, {max_value}]"
            for hour_start, value in df.loc[out_of_range, ["hour_start", channel]].itertuples(
                index=False
            ):
                self._log_debug(
                    f"Erroneous data ({reason}): channel={channel} hour_start={hour_start} "
                    f"value={value}"
                )
            df[channel] = df[channel].where(df[channel].between(min_value, max_value))
        self._log_info(f"Erroneous data (valid range): {nans_added_range} NaNs added")

        # Remove data that is outside the maximum absolute rate of change
        nans_added_roc = 0
        for channel, max_abs_change in self.MAX_ABS_RATE_OF_CHANGE_PER_CHANNEL.items():
            previous = df[channel].shift(1)
            abs_change = (df[channel] - previous).abs()
            roc_ok = abs_change.le(max_abs_change) | previous.isna()
            excessive_roc = df[channel].notna() & ~roc_ok
            nans_added_roc += int(excessive_roc.sum())
            reason = f"rate of change > {max_abs_change}"
            for hour_start, value in df.loc[excessive_roc, ["hour_start", channel]].itertuples(
                index=False
            ):
                self._log_debug(
                    f"Erroneous data ({reason}): channel={channel} hour_start={hour_start} "
                    f"value={value}"
                )
            df[channel] = df[channel].where(roc_ok | df[channel].isna())
        self._log_info(f"Erroneous data (rate of change): {nans_added_roc} NaNs added")

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
        self._log_info(f"Missing timestamps: inserted {len(df) - n_before} hourly rows")

        # Mark rows inside long missing-data gaps
        df = df.set_index("hour_start")
        drop_rows = pd.Series(False, index=df.index)
        for channel in df.columns:
            is_nan = df[channel].isna()
            block_id = is_nan.ne(is_nan.shift()).cumsum()
            block_len = is_nan.groupby(block_id).transform("size")
            drop_rows |= is_nan & (block_len >= self.MAX_GAP_HOURS)

        # Interpolate missing data
        n_filled = 0
        oat_f_interpolated = pd.Series(False, index=df.index)
        for channel in df.columns:
            if channel == "day":
                continue
            before = df[channel].copy()
            df[channel] = df[channel].interpolate(method="time")
            if channel != "oat_f":
                df.loc[drop_rows, channel] = before.loc[drop_rows]
            filled = before.isna() & df[channel].notna()
            n_filled += int(filled.sum())
            if channel == "oat_f":
                oat_f_interpolated |= filled
        self._log_info(f"Interpolation: {n_filled} values filled")

        # Compute average OAT over last 6 hours
        oat_mean_6h = df["oat_f"].rolling(6).mean().shift(1)
        first_oat_observed = (~oat_f_interpolated) & df["oat_f"].notna()
        endpoints_ok = first_oat_observed.shift(6) & first_oat_observed.shift(1)
        df["OAT_avg_6h"] = oat_mean_6h.where(endpoints_ok)
        df = df[df["OAT_avg_6h"].notna()]
 
        # Drop rows inside missing-data gaps
        df = df.loc[~drop_rows]
        df = df.reset_index(names="hour_start")
        self._log_info(
            f"Long missing-data gaps (>={self.MAX_GAP_HOURS}h): dropped {int(drop_rows.sum())} rows"
        )

        # If any rows still have NaNs, drop them
        remaining_nans = int(df.isna().sum().sum())
        self._log_info(f"NaNs remaining: {remaining_nans}")
        if remaining_nans > 0:
            n_before = len(df)
            df = df.dropna().reset_index(drop=True)
            self._log_info(f"Dropped {n_before - len(df)} rows with NaNs")

        return df

    def _filter_out_known_bad_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._thermostat_change(df)
        df = self._used_oil_boiler(df)
        df = self._broken_thermostat(df)
        df = self._external_heat_source(df)
        return df
    
    def _broken_thermostat(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag and drop hours with inconsistent heat call vs zone temperature.

        Per zone, uses min temp-set gap from BROKEN_THERMOSTAT_MIN_TEMP_SET_GAP_F
        (default 3 °F; optional per-house zone overrides, e.g. beech zone2).

        Triggers when either:
        - Heat off while cold: heatcall_fraction == 0 and avg_set - avg_temp >= min gap.
        - Heat on while warm: heatcall_fraction > 0.01 and avg_temp - avg_set >= min gap.

        Any hour flagged in any zone is removed from the dataframe.
        """
        gap_cfg = self.BROKEN_THERMOSTAT_MIN_TEMP_SET_GAP_F
        default_min_gap = float(gap_cfg["default"])
        house_gap_cfg = gap_cfg.get(self.house_alias)
        drop_rows = pd.Series(False, index=df.index)
        n_flags = 0
        for z in self.zones:
            heatcall_col = f"zone{z}_heatcall_fraction"
            temp_col = f"zone{z}_avg_temp"
            setpoint_col = f"zone{z}_avg_set"
            if isinstance(house_gap_cfg, dict) and f"zone{z}" in house_gap_cfg:
                min_gap = float(house_gap_cfg[f"zone{z}"])
            else:
                min_gap = default_min_gap
            temp_below_set = df[setpoint_col] - df[temp_col]
            temp_above_set = df[temp_col] - df[setpoint_col]
            heat_off_but_cold = (df[heatcall_col] == 0) & (temp_below_set >= min_gap)
            heat_on_but_warm = (df[heatcall_col] > 0.01) & (temp_above_set >= min_gap)
            broken = heat_off_but_cold | heat_on_but_warm
            drop_rows |= broken
            n_flags += int(heat_off_but_cold.sum()) + int(heat_on_but_warm.sum())
            for hour_start, temp, setpoint in df.loc[
                heat_off_but_cold, ["hour_start", temp_col, setpoint_col]
            ].itertuples(index=False):
                self._log_debug(
                    f"Broken thermostat (zone {z}, heat off while cold): hour_start={hour_start} "
                    f"{temp_col}={temp} < {setpoint_col}={setpoint}, {heatcall_col}=0"
                )
            for hour_start, temp, setpoint, heatcall in df.loc[
                heat_on_but_warm, ["hour_start", temp_col, setpoint_col, heatcall_col]
            ].itertuples(index=False):
                self._log_debug(
                    f"Broken thermostat (zone {z}, heat on while warm): hour_start={hour_start} "
                    f"{setpoint_col}={setpoint} < {temp_col}={temp}, {heatcall_col}={heatcall}"
                )
        n_drop = int(drop_rows.sum())
        self._log_info(f"Broken thermostat: {n_flags} flagged row-zone hours, dropped {n_drop} rows")
        return df.loc[~drop_rows].reset_index(drop=True)

    def _thermostat_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop hours when any zone's thermostat setpoint changed from the prior hour."""
        drop_rows = pd.Series(False, index=df.index)
        n_flags = 0
        for z in self.zones:
            setpoint_col = f"zone{z}_avg_set"
            prev_setpoint = df[setpoint_col].shift(1)
            prev_hour = df["hour_start"].shift(1)
            consecutive = (df["hour_start"] - prev_hour) == pd.Timedelta(hours=1)
            changed = (
                consecutive
                & df[setpoint_col].notna()
                & prev_setpoint.notna()
                & (df[setpoint_col] != prev_setpoint)
            )
            drop_rows |= changed
            n_flags += int(changed.sum())
            for hour_start, previous, new_set in zip(
                df.loc[changed, "hour_start"],
                prev_setpoint.loc[changed],
                df.loc[changed, setpoint_col],
                strict=True,
            ):
                self._log_debug(
                    f"Thermostat setpoint change (zone {z}): hour_start={hour_start} "
                    f"{setpoint_col} {previous} -> {new_set}"
                )
        n_drop = int(drop_rows.sum())
        self._log_info(
            f"Thermostat setpoint change: {n_flags} flagged row-zone hours, dropped {n_drop} rows"
        )
        return df.loc[~drop_rows].reset_index(drop=True)

    def _used_oil_boiler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag and drop hours when average oil boiler power was on."""
        if "oil_boiler_pwr" not in df.columns:
            return df

        threshold = self.OIL_BOILER_POWER_THRESHOLD
        used = df["oil_boiler_pwr"] > threshold
        for hour_start, pwr in df.loc[
            used, ["hour_start", "oil_boiler_pwr"]
        ].itertuples(index=False):
            self._log_debug(
                f"Oil boiler used: hour_start={hour_start} "
                f"oil_boiler_pwr={pwr}W > {threshold}W"
            )
        n_drop = int(used.sum())
        self._log_info(f"Oil boiler: {n_drop} flagged hours, dropped {n_drop} rows")
        return df.loc[~used].reset_index(drop=True)

    def _external_heat_source(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag hours that suggest heating without a heat call (e.g. sun, stove).

        Per zone, uses EXTERNAL_HEAT_SOURCE_MIN_TEMP_ABOVE_SET_F (default 3 °F).

        Triggers when:
        - heatcall_fraction == 0, and
        - avg_temp - avg_set >= that threshold, and
        - no other zone has avg_temp strictly above this zone's avg_temp (warmth
          from another zone is excluded).

        Any hour flagged in any zone is removed from the dataframe.
        """
        min_gap = self.EXTERNAL_HEAT_SOURCE_MIN_TEMP_ABOVE_SET_F
        drop_rows = pd.Series(False, index=df.index)
        n_flags = 0
        for z in self.zones:
            heatcall_col = f"zone{z}_heatcall_fraction"
            temp_col = f"zone{z}_avg_temp"
            setpoint_col = f"zone{z}_avg_set"
            temp_above_set = df[temp_col] - df[setpoint_col]
            external_heat = (df[heatcall_col] == 0) & (temp_above_set >= min_gap)
            other_zones = [other_z for other_z in self.zones if other_z != z]
            if other_zones:
                max_other_temp = df[[f"zone{other_z}_avg_temp" for other_z in other_zones]].max(
                    axis=1
                )
                external_heat &= max_other_temp <= df[temp_col]
            drop_rows |= external_heat
            n_flags += int(external_heat.sum())
            for hour_start, oat_f, temp, setpoint in df.loc[
                external_heat, ["hour_start", "oat_f", temp_col, setpoint_col]
            ].itertuples(index=False):
                self._log_debug(
                    f"External heat source (zone {z}): hour_start={hour_start} "
                    f"oat_f={oat_f}, {temp_col}={temp} >= {setpoint_col}={setpoint} + {min_gap}°F, "
                    f"{heatcall_col}=0"
                )
        n_drop = int(drop_rows.sum())
        self._log_info(f"External heat source: {n_flags} flagged row-zone hours, dropped {n_drop} rows")
        return df.loc[~drop_rows].reset_index(drop=True)

    def _plot_data_distribution(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        channels = [
            c
            for c in self.required
            if not (str(c).startswith("zone") and str(c).endswith("_heatcall_fraction"))
        ]
        n_channels = len(channels)
        col_width_in = 1.75
        fig, axes = plt.subplots(
            2,
            n_channels,
            figsize=(col_width_in * n_channels + 1.25, 8),
            squeeze=False,
            gridspec_kw={"wspace": 0.75},
        )
        row_labels = ("Before cleaning", "After cleaning")
        for row, (label, df) in enumerate(zip(row_labels, (df_before, df_after))):
            for col_idx, channel in enumerate(channels):
                ax = axes[row, col_idx]
                ax.boxplot(df[channel].dropna().to_numpy(), vert=True, widths=0.22)
                if row == 0:
                    n_nans = int(df_before[channel].isna().sum())
                    nan_label = "NaN" if n_nans == 1 else "NaNs"
                    ax.set_title(f"{channel}\n({n_nans} {nan_label})", fontsize=8)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.tick_params(axis="y", labelsize=8)
            axes[row, 0].set_ylabel(label)
            if row == 1:
                for z in self.zones:
                    zone_avg_channels = [
                        c for c in channels if str(c).startswith(f"zone{z}_avg_")
                    ]
                    if not zone_avg_channels:
                        continue
                    zone_avg_values = np.concatenate(
                        [df[channel].dropna().to_numpy() for channel in zone_avg_channels]
                    )
                    if not zone_avg_values.size:
                        continue
                    y_min = float(zone_avg_values.min())
                    y_max = float(zone_avg_values.max())
                    pad = max((y_max - y_min) * 0.05, 0.25)
                    shared_ylim = (y_min - pad, y_max + pad)
                    zone_avg_set = set(zone_avg_channels)
                    for col_idx, channel in enumerate(channels):
                        if channel in zone_avg_set:
                            axes[row, col_idx].set_ylim(shared_ylim)
        fig.suptitle(f"{self.house_alias.capitalize()}: channel distributions")
        fig.tight_layout()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        savepath = self.results_dir / f"{self.house_alias}_channel_boxplots.png"
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def design_matrix(self, df: pd.DataFrame, *, baseline: bool = False, energy_ratio: float | None = None) -> np.ndarray:
        feature_names = self.FEATURE_NAMES_BASELINE if baseline else self.feature_names
        columns = [np.ones(len(df))]
        for name in feature_names:
            if name == "previous_dist_kwh_scaled":
                if energy_ratio is None:
                    raise ValueError("energy_ratio is required for previous_dist_kwh_scaled")
                columns.append(df["previous_dist_kwh"].to_numpy(dtype=float) * energy_ratio)
            else:
                columns.append(df[name].to_numpy(dtype=float))
        return np.column_stack(columns)

    def fit(self, df: pd.DataFrame, *, baseline: bool = False) -> HouseEnergyParams:
        feature_names = self.FEATURE_NAMES_BASELINE if baseline else self.feature_names

        dist_kwh = df["dist_kwh"].to_numpy()
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh.sum())
        y = dist_kwh * energy_ratio
        X = self.design_matrix(df, baseline=baseline, energy_ratio=energy_ratio)

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
        ss_res = float(np.sum(residuals**2))
        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1.0 - ss_res / ss_tot

        return HouseEnergyParams(
            feature_names=feature_names,
            values=tuple(round(float(c), 6) for c in coefficients),
            std_errors=tuple(round(float(e), 6) for e in std_errors),
            r_squared=round(r_squared, 3),
            energy_ratio=round(energy_ratio, 3),
            baseline=baseline,
        )

    def predict(self, params: HouseEnergyParams, df: pd.DataFrame) -> np.ndarray:
        X = self.design_matrix(df, baseline=params.baseline, energy_ratio=params.energy_ratio)
        return np.maximum(X @ params.coefficients(), 0.0)

    def predict_recursive_horizon(
        self,
        params: HouseEnergyParams,
        origin_index: int,
        horizon: int | None = None,
    ) -> np.ndarray:
        if params.baseline:
            target = self.df.iloc[origin_index + 1 : origin_index + 1 + (horizon or self.FORECAST_HORIZON_HOURS)]
            return self.predict(params, target)
        if horizon is None:
            horizon = self.FORECAST_HORIZON_HOURS
        ratio = params.energy_ratio
        preds = np.empty(horizon, dtype=float)
        for step in range(horizon):
            target_idx = origin_index + 1 + step
            row = self.df.iloc[[target_idx]].copy()
            if step > 0:
                row["previous_dist_kwh"] = preds[step - 1] / ratio
            preds[step] = self.predict(params, row)[0]
        return preds

    def _fit_trailing_models(
        self, n: int
    ) -> tuple[list[pd.Timestamp], list[HouseEnergyParams], list[HouseEnergyParams]]:
        days = np.sort(self.df["day"].unique())
        fit_days: list[pd.Timestamp] = []
        results: list[HouseEnergyParams] = []
        results_baseline: list[HouseEnergyParams] = []
        last_fit_index: int | None = None
        first_fit_day_index = 0 if self.GROW_WINDOW_TO_N else n - 1

        for i in range(first_fit_day_index, len(days)):
            if self._should_refit_trailing_window(i, days, last_fit_index):
                window = self._fit_window_days(days, i, n)
                window_df = self.df[self.df["day"].isin(window)]
                fit_days.append(days[i])
                results.append(self.fit(window_df))
                results_baseline.append(self.fit(window_df, baseline=True))
                last_fit_index = i

        return fit_days, results, results_baseline

    def _params_index_for_origin_day(
        self, origin_day: pd.Timestamp, fit_days: list[pd.Timestamp]
    ) -> int | None:
        fit_idx = len(fit_days) - 1
        while fit_idx >= 0 and fit_days[fit_idx] >= origin_day:
            fit_idx -= 1
        return fit_idx if fit_idx >= 0 else None

    def _evaluate_recursive_horizon_oos(
        self,
        n: int,
        *,
        collect_pointwise: bool = False,
    ) -> dict:
        horizon = self.FORECAST_HORIZON_HOURS
        fit_days, results, results_baseline = self._fit_trailing_models(n)
        if not fit_days:
            raise ValueError(f"No trailing fits produced for N={n}")

        df = self.df
        dist_kwh = df["dist_kwh"].to_numpy(dtype=float)
        max_origin = len(df) - horizon

        errors_by_lead: list[list[float]] = [[] for _ in range(horizon)]
        errors_baseline_by_lead: list[list[float]] = [[] for _ in range(horizon)]
        oos_pred: list[float] = []
        oos_pred_baseline: list[float] = []
        oos_actual: list[float] = []
        oos_oat_f: list[float] = []
        oos_hour_start: list[pd.Timestamp] = []
        oos_lead1_errors: list[float] = []
        oos_lead1_hour_start: list[pd.Timestamp] = []

        for origin in range(max_origin):
            origin_day = df["day"].iloc[origin]
            fit_idx = self._params_index_for_origin_day(origin_day, fit_days)
            if fit_idx is None:
                continue

            params = results[fit_idx]
            params_baseline = results_baseline[fit_idx]
            pred = self.predict_recursive_horizon(params, origin, horizon)
            pred_baseline = self.predict_recursive_horizon(params_baseline, origin, horizon)
            target_start = origin + 1
            actual = dist_kwh[target_start : target_start + horizon] * params.energy_ratio

            for step in range(horizon):
                err = float(pred[step] - actual[step])
                err_baseline = float(pred_baseline[step] - actual[step])
                errors_by_lead[step].append(err)
                errors_baseline_by_lead[step].append(err_baseline)

            if collect_pointwise:
                oos_pred.extend(pred.tolist())
                oos_pred_baseline.extend(pred_baseline.tolist())
                oos_actual.extend(actual.tolist())
                oos_oat_f.extend(df["oat_f"].iloc[target_start : target_start + horizon].tolist())
                oos_hour_start.extend(
                    df["hour_start"].iloc[target_start : target_start + horizon].tolist()
                )
                oos_lead1_errors.append(float(pred[0] - actual[0]))
                oos_lead1_hour_start.append(df["hour_start"].iloc[target_start])

        rmse_by_lead = [
            float(np.sqrt(np.mean(np.square(errors))))
            if errors
            else float("nan")
            for errors in errors_by_lead
        ]
        rmse_baseline_by_lead = [
            float(np.sqrt(np.mean(np.square(errors))))
            if errors
            else float("nan")
            for errors in errors_baseline_by_lead
        ]
        all_errors = [err for lead in errors_by_lead for err in lead]
        all_errors_baseline = [err for lead in errors_baseline_by_lead for err in lead]
        rmse = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else float("nan")
        rmse_baseline = (
            float(np.sqrt(np.mean(np.square(all_errors_baseline))))
            if all_errors_baseline
            else float("nan")
        )
        mae = float(np.mean(np.abs(all_errors))) if all_errors else float("nan")
        mae_baseline = (
            float(np.mean(np.abs(all_errors_baseline))) if all_errors_baseline else float("nan")
        )

        return {
            "fit_days": fit_days,
            "results": results,
            "results_baseline": results_baseline,
            "rmse": rmse,
            "rmse_baseline": rmse_baseline,
            "mae": mae,
            "mae_baseline": mae_baseline,
            "rmse_by_lead": rmse_by_lead,
            "rmse_baseline_by_lead": rmse_baseline_by_lead,
            "n_forecast_points": len(all_errors),
            "oos_pred": np.array(oos_pred),
            "oos_pred_baseline": np.array(oos_pred_baseline),
            "oos_actual": np.array(oos_actual),
            "oos_oat_f": np.array(oos_oat_f),
            "oos_hour_start": oos_hour_start,
            "errors": np.array(all_errors),
            "oos_lead1_errors": np.array(oos_lead1_errors),
            "oos_lead1_hour_start": oos_lead1_hour_start,
        }

    def _should_refit_trailing_window(self, day_index: int, days: np.ndarray, last_fit_index: int | None) -> bool:
        if self.TRAINING_FREQUENCY == "daily" or last_fit_index is None:
            return True
        return days[day_index] - days[last_fit_index] >= pd.Timedelta(days=7)

    def _fit_window_days(self, days: np.ndarray, day_index: int, n: int) -> np.ndarray:
        if self.GROW_WINDOW_TO_N and day_index + 1 < n:
            return days[0 : day_index + 1]
        return days[day_index - n + 1 : day_index + 1]

    def trailing_n_day_fits(self, n: int) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        oos = self._evaluate_recursive_horizon_oos(n, collect_pointwise=True)
        fit_days = oos["fit_days"]
        results = oos["results"]
        oos_pred = oos["oos_pred"]
        oos_pred_baseline = oos["oos_pred_baseline"]
        oos_actual = oos["oos_actual"]
        oos_oat_f = oos["oos_oat_f"]
        oos_hour_start = oos["oos_hour_start"]
        errors = oos["errors"]
        mae = oos["mae"]
        rmse = oos["rmse"]
        mae_baseline = oos["mae_baseline"]
        rmse_baseline = oos["rmse_baseline"]
        rmse_by_lead = oos["rmse_by_lead"]
        rmse_baseline_by_lead = oos["rmse_baseline_by_lead"]

        oos_label = f"recursive {self.FORECAST_HORIZON_HOURS}h"
        self._log_info(
            f"Fitted {len(results)} {self.TRAINING_FREQUENCY} training, "
            f"{'with' if self.GROW_WINDOW_TO_N else 'no'} growing window "
            f"from {pd.Timestamp(fit_days[0]).date()} to {pd.Timestamp(fit_days[-1]).date()}"
        )

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
        params_table.to_csv(
            self.results_dir / f"{self.house_alias}_params_N{n}_{self._artifact_label}.csv"
        )

        self._log_info(
            f"{oos_label.capitalize()} out-of-sample over {oos['n_forecast_points']} "
            f"origin×lead points: MAE = {mae:.2f} kWh (baseline {mae_baseline:.2f} kWh, "
            f"{improvement_mae_percentage:.1f}% improvement), "
            f"RMSE = {rmse:.2f} kWh (baseline {rmse_baseline:.2f} kWh, "
            f"{improvement_rmse_percentage:.1f}% improvement)"
        )
        self._log_info(
            "RMSE by lead (kWh): "
            + ", ".join(f"{h}h={rmse_by_lead[h - 1]:.2f}" for h in (1, 6, 12, 24, 48))
        )

        rmse_by_lead_df = pd.DataFrame(
            {
                "lead_hour": range(1, self.FORECAST_HORIZON_HOURS + 1),
                "rmse_kwh": rmse_by_lead,
                "rmse_baseline_kwh": rmse_baseline_by_lead,
            }
        )
        rmse_by_lead_df.to_csv(
            self.results_dir
            / f"{self.house_alias}_rmse_by_lead_N{n}_{self._artifact_label}.csv",
            index=False,
        )

        self.plot_pred_vs_actual(
            oos_pred, oos_actual, oos_oat_f,
            (
                f"{self.house_alias.capitalize()}: {oos_label} predicted vs actual "
                f"({self.TRAINING_FREQUENCY} training, {'with' if self.GROW_WINDOW_TO_N else 'no'} growing window)"
            ),
            savepath=self.results_dir
            / f"{self.house_alias}_pred_vs_actual_N{n}_{self._artifact_label}.png",
            baseline_mae=mae_baseline,
            baseline_rmse=rmse_baseline,
            oos_period_label=oos_label,
        )
        self.plot_oos_residual_by_hour_of_day(
            oos["oos_lead1_errors"],
            oos["oos_lead1_hour_start"],
            savepath=self.results_dir
            / f"{self.house_alias}_oos_residual_by_hour_N{n}_{self._artifact_label}.png",
            subtitle="first forecast hour (lead 1) only",
        )
        self.plot_oos_rmse_by_lead(
            rmse_by_lead,
            rmse_baseline_by_lead,
            savepath=self.results_dir
            / f"{self.house_alias}_rmse_by_lead_N{n}_{self._artifact_label}.png",
        )

    def sweep_n(self, min_n: int, max_n: int) -> None:
        import matplotlib.pyplot as plt

        self.results_dir.mkdir(parents=True, exist_ok=True)

        rmses: list[float] = []
        b1_weekly_ranges: list[float] = []
        n_values = list(range(min_n, max_n + 1))

        for n in n_values:
            oos = self._evaluate_recursive_horizon_oos(n, collect_pointwise=False)
            rmse = oos["rmse"]
            fit_days = oos["fit_days"]
            b1_values = [r.B1 for r in oos["results"]]
            b1 = pd.Series(b1_values, index=pd.DatetimeIndex(fit_days)).sort_index()
            if self.TRAINING_FREQUENCY == "daily":
                b1_stability = b1.rolling("7D").apply(lambda s: s.max() - s.min())
                b1_stability_metric = float(b1_stability.mean())
            else:
                b1_stability_metric = float(b1.diff().abs().mean())
            rmses.append(rmse)
            b1_weekly_ranges.append(b1_stability_metric)
            self._log_info(
                f"N={n:2d}: recursive {self.FORECAST_HORIZON_HOURS}h RMSE={rmse:.4f} kWh, "
                f"B1 stability={b1_stability_metric:.5g}"
            )

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_xlabel("N (trailing days in fit window)")
        rmse_ylabel = f"Recursive {self.FORECAST_HORIZON_HOURS}h RMSE (kWh)"
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
            self.results_dir / f"{self.house_alias}_sweep_N_{self.TRAINING_FREQUENCY}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

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
        plt.close(fig)

    @staticmethod
    def _hourly_residual_mean_and_ci95(
        residuals: np.ndarray, hour_of_day: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        by_hour = (
            pd.DataFrame({"hour": hour_of_day, "residual": residuals})
            .groupby("hour", sort=True)["residual"]
            .agg(["mean", "std", "count"])
            .reindex(range(24))
        )
        sem = by_hour["std"] / np.sqrt(by_hour["count"])
        ci95 = 1.96 * sem
        ci95 = ci95.where(by_hour["count"] >= 2, 0.0).fillna(0.0)
        return by_hour["mean"].to_numpy(), ci95.to_numpy()

    def _plot_hourly_residual_bars_on_ax(
        self,
        ax,
        residuals: np.ndarray,
        hour_of_day: pd.Series,
        *,
        house_alias: str,
        panel_title: str,
    ) -> None:
        means, ci95 = self._hourly_residual_mean_and_ci95(residuals, hour_of_day)
        hours = np.arange(24)
        ax.bar(
            hours,
            means,
            yerr=ci95,
            color="tab:blue",
            alpha=0.85,
            width=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "ecolor": "0.25"},
        )
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(hours)
        ax.set_ylabel("Mean residual (predicted − actual, kWh)")
        ax.set_title(
            f"{house_alias.capitalize()} - {panel_title} - "
            f"mean error and 95% CI on energy use prediction"
        )

    def plot_oos_rmse_by_lead(
        self,
        rmse_by_lead: list[float],
        rmse_baseline_by_lead: list[float],
        savepath: Path | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        leads = np.arange(1, len(rmse_by_lead) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(leads, rmse_by_lead, "o-", label="Model", color="tab:blue")
        ax.plot(leads, rmse_baseline_by_lead, "s--", label="Baseline", color="tab:orange")
        ax.set_xlabel("Forecast lead (hours ahead)")
        ax.set_ylabel("RMSE (kWh)")
        ax.set_title(
            f"{self.house_alias.capitalize()}: out-of-sample RMSE by lead "
            f"({self.FORECAST_HORIZON_HOURS}h recursive load)"
        )
        ax.set_xticks([1, 6, 12, 24, 36, 48])
        ax.legend()
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_oos_residual_by_hour_of_day(
        self,
        residuals: np.ndarray,
        hour_starts,
        savepath: Path | None = None,
        subtitle: str | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        residuals = np.asarray(residuals, dtype=float)
        timestamps = pd.to_datetime(hour_starts)
        hour_of_day = timestamps.hour
        weekend_mask = np.asarray(timestamps.dayofweek >= 5, dtype=bool)

        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        panels = (
            ("Weekday", ~weekend_mask),
            ("Weekend", weekend_mask),
        )
        for ax, (panel_title, mask) in zip(axes, panels):
            panel_residuals = residuals[mask]
            panel_hours = hour_of_day[mask]
            self._plot_hourly_residual_bars_on_ax(
                ax,
                panel_residuals,
                panel_hours,
                house_alias=self.house_alias,
                panel_title=panel_title,
            )
        for ax in axes:
            ax.set_xlabel("Hour of day")
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_ylim(-1.5, 1.5)
        if subtitle:
            fig.suptitle(subtitle, fontsize=11, y=1.02)
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
