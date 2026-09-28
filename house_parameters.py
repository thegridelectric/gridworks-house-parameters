import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from plots import (
    plot_oos_residual_by_hour_of_day,
    plot_oos_rmse_by_lead,
    plot_pred_vs_actual,
)

logger = logging.getLogger(__name__)


@dataclass
class HouseEnergyParams:
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    std_errors: tuple[float, ...]
    r_squared: float
    energy_ratio: float
    is_baseline: bool = False


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

    # Occupancy features
    INCLUDE_OCCUPANCY: bool = True
    OCCUPANCY_INDIVIDUAL_HOURS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21)
    WEEKDAY_OCCUPANCY_FEATURE_NAMES = tuple(f"wd_hour_{hour}" for hour in OCCUPANCY_INDIVIDUAL_HOURS)
    WEEKEND_OCCUPANCY_FEATURE_NAMES = tuple(f"we_hour_{hour}" for hour in OCCUPANCY_INDIVIDUAL_HOURS)
    OCCUPANCY_FEATURE_NAMES = WEEKDAY_OCCUPANCY_FEATURE_NAMES + WEEKEND_OCCUPANCY_FEATURE_NAMES

    # Training
    TRAINING_FREQUENCY: Literal["daily", "weekly"] = "daily"
    GROW_WINDOW_TO_N: bool = False
    MIN_FIT_WINDOW_DAYS: int = 10
    FORECAST_HORIZON_HOURS = 48
    
    # Data cleaning - erroneous data
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

    # Data cleaning - missing data
    MAX_DATA_GAP_HOURS = 4

    # Data cleaning - known bad data
    OIL_BOILER_POWER_THRESHOLD_WATTS = 50
    BROKEN_THERMOSTAT_MIN_TEMP_SET_GAP_F = {
        'default': 3,
        'beech': {'zone2': 4.5,}
    }
    BELOW_SETPOINT_MIN_TEMP_SET_GAP_F = 1
    EXTERNAL_HEAT_SOURCE_MIN_TEMP_ABOVE_SET_F = 3

    def __init__(self, house_alias: str):
        self.house_alias = house_alias
        self.results_dir = Path("results") / house_alias
        self.prepare_data()
        
    def _log_info(self, message: str) -> None:
        logger.info("[%s] %s", self.house_alias, message)

    def _log_debug(self, message: str) -> None:
        logger.debug("[%s] %s", self.house_alias, message)

    # --------------------------------
    # Data preparation 
    # --------------------------------

    def prepare_data(self) -> None:
        """
        - Loads the data
        - Cleans the data
        - Engineers the features
        """
        self._load_data()
        self._clean_data()
        self._engineer_features()

    def _load_data(self) -> None:
        """
        Loads the data from the CSV file and prepares it for the analysis.
        - Sorts by hour_start, and adds a day column.
        - Finds the number of zones
        - Checks that all required columns are present
        """
        csv_path = Path("data") / f"{self.house_alias}_house_params_data.csv"
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
        
        self.df = df

    def _clean_data(self) -> None:
        """
        - Removes erroneous data
        - Handles missing data
        - Filters out known bad data
        """
        self.df = self._remove_erroneous_data(self.df)
        self.df = self._handle_missing_data(self.df)
        self.df = self._filter_out_known_bad_data(self.df)

    def _engineer_features(self) -> None:
        """
        Adds engineered columns to the dataframe and sets self.feature_names.
        (OAT_avg_6h is built during missing-data handling in _clean_data.)
        """
        df = self.df.copy()
        setpoint_avg = df[[f"zone{z}_avg_set" for z in self.zones]].mean(axis=1)
        df["deltaT"] = (setpoint_avg - df["oat_f"]).clip(lower=0)
        df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
        df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
        df = df.drop(0).reset_index(drop=True)
        df["windspeed_times_65_minus_oat"] = df["ws_mph"] * (65.0 - df["oat_f"])
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

    def _remove_erroneous_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes erroneous data:
        - Data outside the range of valid values
        - Data outside the maximum absolute rate of change
        """
        # Replace slight negative HP thermal power values with 0
        df.loc[df["hp_kwh_th"].between(-1, 0), "hp_kwh_th"] = 0

        # Remove data that is outside the range of valid values
        range_of_valid_values_per_channel = dict(self.RANGE_OF_VALID_VALUES_PER_CHANNEL)
        for col in df.columns:
            if str(col).startswith("zone") and str(col).endswith(("_avg_set", "_avg_temp")):
                if "zone_set_or_temp_f" in range_of_valid_values_per_channel:
                    range_of_valid_values_per_channel[col] = range_of_valid_values_per_channel["zone_set_or_temp_f"]
            if str(col).startswith("zone") and str(col).endswith("_heatcall_fraction"):
                if "zone_heatcall_fraction" in range_of_valid_values_per_channel:
                    range_of_valid_values_per_channel[col] = range_of_valid_values_per_channel["zone_heatcall_fraction"]
        range_of_valid_values_per_channel.pop("zone_set_or_temp_f")
        range_of_valid_values_per_channel.pop("zone_heatcall_fraction")

        nans_added_range = 0
        for channel, (min_value, max_value) in range_of_valid_values_per_channel.items():
            out_of_range = df[channel].notna() & ~df[channel].between(min_value, max_value)
            nans_added_range += int(out_of_range.sum())
            reason = f"valid range [{min_value}, {max_value}]"
            for hour_start, value in df.loc[out_of_range, ["hour_start", channel]].itertuples(index=False):
                self._log_debug(f"Erroneous data ({reason}): channel={channel} hour_start={hour_start} value={value}")
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
            for hour_start, value in df.loc[excessive_roc, ["hour_start", channel]].itertuples(index=False):
                self._log_debug(f"Erroneous data ({reason}): channel={channel} hour_start={hour_start} value={value}")
            df[channel] = df[channel].where(roc_ok | df[channel].isna())
        self._log_info(f"Erroneous data (rate of change): {nans_added_roc} NaNs added")

        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing data:
        - Ensures there is one row per hour
        - Marks rows inside long missing-data gaps
        - Interpolates missing data
        - Computes average OAT over last 6 hours when the start and end value are not interpolated, otherwise drops the row
        - Drops rows inside missing-data gaps
        - Drops remaining rows that contain NaNs
        """
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
            drop_rows |= is_nan & (block_len >= self.MAX_DATA_GAP_HOURS)

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
        self._log_info(f"Long missing-data gaps (>={self.MAX_DATA_GAP_HOURS}h): dropped {int(drop_rows.sum())} rows")

        # If any rows still have NaNs, drop them
        remaining_nans = int(df.isna().sum().sum())
        self._log_info(f"NaNs remaining: {remaining_nans}")
        if remaining_nans > 0:
            n_before = len(df)
            df = df.dropna().reset_index(drop=True)
            self._log_info(f"Dropped {n_before - len(df)} rows with NaNs")

        return df

    def _filter_out_known_bad_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out hours which include known-bad data:
        - Used oil boiler
        - Zone below setpoint
        - Zone setpoint change
        - Broken thermostat
        - External heat source (voluntarily excluded for now)
        """
        df = self._used_oil_boiler(df)
        df = self._zone_below_setpoint(df)
        df = self._setpoint_change(df)
        df = self._broken_thermostat(df)
        # df = self._external_heat_source(df)
        return df
    
    def _used_oil_boiler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop hours when the oil boiler power indicates it was used."""
        threshold = self.OIL_BOILER_POWER_THRESHOLD_WATTS
        if "oil_boiler_pwr" not in df.columns:
            return df
        used = df["oil_boiler_pwr"] > threshold
        for hour_start, pwr in df.loc[used, ["hour_start", "oil_boiler_pwr"]].itertuples(index=False):
            self._log_debug(f"Oil boiler used: hour_start={hour_start} oil_boiler_pwr={pwr}W > {threshold}W")
        self._log_info(f"Oil boiler: dropped {int(used.sum())} rows")
        return df.loc[~used].reset_index(drop=True)
    
    def _zone_below_setpoint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop hours when any zone's average temperature is well below its average setpoint."""
        min_gap = self.BELOW_SETPOINT_MIN_TEMP_SET_GAP_F
        drop_rows = pd.Series(False, index=df.index)
        for z in self.zones:
            temp_col = f"zone{z}_avg_temp"
            setpoint_col = f"zone{z}_avg_set"
            below_setpoint = (df[setpoint_col] - df[temp_col]) >= min_gap
            drop_rows |= below_setpoint
            for hour_start, temp, setpoint in df.loc[below_setpoint, ["hour_start", temp_col, setpoint_col]].itertuples(index=False):
                self._log_debug(f"Below setpoint (zone {z}): hour_start={hour_start} set={setpoint} - temp={temp} >= {min_gap}°F")
        self._log_info(f"Below setpoint: dropped {int(drop_rows.sum())} rows")
        return df.loc[~drop_rows].reset_index(drop=True)

    def _setpoint_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop hours when any zone's setpoint changed from the prior hour."""
        drop_rows = pd.Series(False, index=df.index)
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
            for hour_start, previous, new_set in zip(
                df.loc[changed, "hour_start"],
                prev_setpoint.loc[changed],
                df.loc[changed, setpoint_col],
                strict=True,
            ):
                self._log_debug(f"Setpoint change (zone {z}): hour_start={hour_start} {setpoint_col} {previous} -> {new_set}")
        self._log_info(f"Setpoint change: dropped {int(drop_rows.sum())} rows")
        return df.loc[~drop_rows].reset_index(drop=True)
    
    def _broken_thermostat(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop hours when the thermostat is broken (one of the following conditions is met):
        - Heat off while cold: heatcall_fraction == 0 and avg_set - avg_temp >= min gap.
        - Heat on while warm: heatcall_fraction > 0.01 and avg_temp - avg_set >= min gap.
        """
        gap_cfg = self.BROKEN_THERMOSTAT_MIN_TEMP_SET_GAP_F
        default_min_gap = float(gap_cfg["default"])
        house_gap_cfg = gap_cfg.get(self.house_alias)
        drop_rows = pd.Series(False, index=df.index)
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
            for hour_start, temp, setpoint in df.loc[
                heat_off_but_cold, ["hour_start", temp_col, setpoint_col]
            ].itertuples(index=False):
                self._log_debug(f"Broken thermostat (zone {z}): hour_start={hour_start} temp={temp} < set={setpoint}, heatcall=0")
            for hour_start, temp, setpoint, heatcall in df.loc[
                heat_on_but_warm, ["hour_start", temp_col, setpoint_col, heatcall_col]
            ].itertuples(index=False):
                self._log_debug(f"Broken thermostat (zone {z}): hour_start={hour_start} set={setpoint} < temp={temp}, heatcall={heatcall}")
        self._log_info(f"Broken thermostat: dropped {int(drop_rows.sum())} rows")
        return df.loc[~drop_rows].reset_index(drop=True)

    def _external_heat_source(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop hours that suggest heating without a heat call (e.g. sun, fireplace):
        - heatcall_fraction == 0, and
        - avg_temp - avg_set >= threshold, and
        - no other zone has avg_temp strictly above this zone's avg_temp (warmth from another zone is excluded).
        """
        min_gap = self.EXTERNAL_HEAT_SOURCE_MIN_TEMP_ABOVE_SET_F
        drop_rows = pd.Series(False, index=df.index)
        for z in self.zones:
            heatcall_col = f"zone{z}_heatcall_fraction"
            temp_col = f"zone{z}_avg_temp"
            setpoint_col = f"zone{z}_avg_set"
            temp_above_set = df[temp_col] - df[setpoint_col]
            external_heat = (df[heatcall_col] == 0) & (temp_above_set >= min_gap)
            other_zones = [other_z for other_z in self.zones if other_z != z]
            if other_zones:
                max_other_temp = df[[f"zone{other_z}_avg_temp" for other_z in other_zones]].max(axis=1)
                external_heat &= max_other_temp <= df[temp_col]
            drop_rows |= external_heat
            for hour_start, oat_f, temp, setpoint in df.loc[
                external_heat, ["hour_start", "oat_f", temp_col, setpoint_col]
            ].itertuples(index=False):
                self._log_debug(
                    f"External heat source (zone {z}): hour_start={hour_start} "
                    f"oat_f={oat_f}, {temp_col}={temp} >= {setpoint_col}={setpoint} + {min_gap}°F, "
                    f"{heatcall_col}=0"
                )
        self._log_info(f"External heat source: dropped {int(drop_rows.sum())} rows")
        return df.loc[~drop_rows].reset_index(drop=True)

    # --------------------------------
    # Model: fitting and predicting
    # --------------------------------

    def fit(self, df: pd.DataFrame, *, baseline: bool = False) -> HouseEnergyParams:
        """
        Fits the linear regression model to the data
        - Calculates the energy ratio from the data and scales the distribution kWh by it.
        - Fits the model and returns the parameters.
        - Can be called for the baseline model (alpha/beta/gamma) too.
        """
        dist_kwh = df["dist_kwh"].to_numpy()
        energy_ratio = float(df["hp_kwh_th"].sum()) / float(dist_kwh.sum())
        dist_kwh_scaled = dist_kwh * energy_ratio
        y = dist_kwh_scaled
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
            feature_names=self.FEATURE_NAMES_BASELINE if baseline else self.feature_names,
            values=tuple(round(float(c), 6) for c in coefficients),
            std_errors=tuple(round(float(e), 6) for e in std_errors),
            r_squared=round(r_squared, 3),
            energy_ratio=round(energy_ratio, 3),
            is_baseline=baseline,
        )

    def predict(self, params: HouseEnergyParams, df: pd.DataFrame) -> np.ndarray:
        """Predicts the scaled distribution kWh using a given set of parameters and data."""
        X = self.design_matrix(df, baseline=params.is_baseline, energy_ratio=params.energy_ratio)
        return np.maximum(X @ np.asarray(params.values, dtype=float), 0.0)

    def design_matrix(self, df: pd.DataFrame, *, baseline: bool = False, energy_ratio: float) -> np.ndarray:
        """
        Constructs the matrix of features for the linear regression model.
        - Can be used for baseline (alpha/beta/gamma) or non-baseline model.
        - Scales the previous distribution kWh (by the energy ratio) since the model predicts scaled distribution kWh.
        """
        feature_names = self.FEATURE_NAMES_BASELINE if baseline else self.feature_names
        columns = [np.ones(len(df))]
        for name in feature_names:
            if name == "previous_dist_kwh_scaled":
                columns.append(df["previous_dist_kwh"].to_numpy(dtype=float) * energy_ratio)
            else:
                columns.append(df[name].to_numpy(dtype=float))
        return np.column_stack(columns)

    # ---------------------------------------------------------
    # Evaluate the accuracy of the recursive horizon forecasts
    # ---------------------------------------------------------

    def trailing_n_day_fits(self, n: int) -> None:
        """
        Repeatedly (daily or weekly, depending on ``TRAINING_FREQUENCY``) refits the house model 
        on the last ``n`` days of data. When ``GROW_WINDOW_TO_N`` is True and there is less 
        than ``n`` previous days available, it uses all previous data (but not until at least
        ``MIN_FIT_WINDOW_DAYS`` calendar days are available).
        
        For every hour, it takes the latest fit whose training window ends before that hour's day, 
        runs a recursive ``FORECAST_HORIZON_HOURS``-step ahead prediction, and scores errors against
        actual scaled_dist_kwh. 
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        oos = self._evaluate_recursive_horizon_oos(n, collect_pointwise=True)

        # Collect the results
        fit_days = oos["fit_days"]
        results = oos["results"]
        oos_pred = oos["oos_pred"]
        oos_actual = oos["oos_actual"]
        oos_oat_f = oos["oos_oat_f"]
        mae = oos["mae"]
        rmse = oos["rmse"]
        mae_baseline = oos["mae_baseline"]
        rmse_baseline = oos["rmse_baseline"]
        rmse_by_lead = oos["rmse_by_lead"]
        rmse_baseline_by_lead = oos["rmse_baseline_by_lead"]
        improvement_rmse_percentage = (rmse_baseline - rmse) / rmse_baseline * 100.0
        improvement_mae_percentage = (mae_baseline - mae) / mae_baseline * 100.0

        oos_label = f"recursive {self.FORECAST_HORIZON_HOURS}h"
        self._log_info(
            f"Fitted {len(results)} {self.TRAINING_FREQUENCY} training, "
            f"{'with' if self.GROW_WINDOW_TO_N else 'no'} growing window "
            f"from {pd.Timestamp(fit_days[0]).date()} to {pd.Timestamp(fit_days[-1]).date()}"
        )

        # Save the learned model parameters to a CSV file
        n_coef = len(results[0].values)
        coef_names = [f"B{i}" for i in range(n_coef)]
        params_table = pd.DataFrame(
            {name: [r.values[i] for r in results] for i, name in enumerate(coef_names)}
            | {
                f"std_error_{name}": [r.std_errors[i] for r in results]
                for i, name in enumerate(coef_names)
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
        self._log_info("RMSE by lead (kWh): " + ", ".join(f"{h}h={rmse_by_lead[h - 1]:.2f}" for h in (1, 6, 12, 24, 48)))

        # Save the RMSE by lead (number of hours ahead of the forecast origin) to a CSV
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

        # Plots
        plot_pred_vs_actual(
            oos_pred, oos_actual, oos_oat_f,
            (
                f"{self.house_alias.capitalize()}: {oos_label} predicted vs actual "
                f"({self.TRAINING_FREQUENCY} training, {'with' if self.GROW_WINDOW_TO_N else 'no'} growing window)"
            ),
            oat_f_colormap_bounds=(float(self.df["oat_f"].min()), float(self.df["oat_f"].max())),
            savepath=self.results_dir / f"{self.house_alias}_pred_vs_actual_N{n}_{self._artifact_label}.png",
            baseline_mae=mae_baseline,
            baseline_rmse=rmse_baseline,
            oos_period_label=oos_label,
        )
        plot_oos_residual_by_hour_of_day(
            oos["oos_lead1_errors"],
            oos["oos_lead1_hour_start"],
            house_alias=self.house_alias,
            savepath=self.results_dir / f"{self.house_alias}_oos_residual_by_hour_N{n}_{self._artifact_label}.png",
            subtitle="first forecast hour (lead 1) only",
        )
        plot_oos_rmse_by_lead(
            rmse_by_lead,
            rmse_baseline_by_lead,
            house_alias=self.house_alias,
            forecast_horizon_hours=self.FORECAST_HORIZON_HOURS,
            savepath=self.results_dir / f"{self.house_alias}_rmse_by_lead_N{n}_{self._artifact_label}.png",
        )

    def predict_recursive_horizon(self, params: HouseEnergyParams, origin_index: int, horizon: int) -> np.ndarray:
        """
        Forecast scaled_dist_kwh for the next ``horizon`` hours after a forecast origin.

        For the main (non-baseline) model,the lagged load feature uses the observed 
        previous hour for lead 1, then the prior step's prediction for leads 2 onward.

        Returns a length-``horizon`` array of scaled_dist_kwh.
        """
        if params.is_baseline:
            target = self.df.iloc[origin_index + 1 : origin_index + 1 + (horizon or self.FORECAST_HORIZON_HOURS)]
            return self.predict(params, target)
        preds = np.empty(horizon, dtype=float)
        for step in range(horizon):
            target_idx = origin_index + 1 + step
            row = self.df.iloc[[target_idx]].copy()
            if step > 0:
                row["previous_dist_kwh"] = preds[step - 1] / params.energy_ratio
            preds[step] = self.predict(params, row)[0]
        return preds

    def _evaluate_recursive_horizon_oos(self, n: int, *, collect_pointwise: bool = False) -> dict:
        """
        Evaluates the recursive horizon out-of-sample predictions.
        - Fits the trailing (or growing-to-N) models.
        - Scores recursive multi-step forecasts at each origin hour.
        """
        horizon = self.FORECAST_HORIZON_HOURS
        days = np.sort(self.df["day"].unique())
        fit_days: list[pd.Timestamp] = []
        results: list[HouseEnergyParams] = []
        results_baseline: list[HouseEnergyParams] = []
        last_fit_index: int | None = None
        if self.GROW_WINDOW_TO_N:
            first_fit_day_index = max(self.MIN_FIT_WINDOW_DAYS - 1, 0)
        else:
            first_fit_day_index = n - 1

        # Fit the models (baseline and non-baseline)
        for i in range(first_fit_day_index, len(days)):
            should_refit = (
                self.TRAINING_FREQUENCY == "daily"
                or last_fit_index is None
                or days[i] - days[last_fit_index] >= pd.Timedelta(days=7)
            )
            if should_refit:
                if self.GROW_WINDOW_TO_N and i + 1 < n:
                    window = days[0 : i + 1]
                else:
                    window = days[i - n + 1 : i + 1]
                window_df = self.df[self.df["day"].isin(window)]
                fit_days.append(days[i])
                results.append(self.fit(window_df))
                results_baseline.append(self.fit(window_df, baseline=True))
                last_fit_index = i

        if not fit_days:
            raise ValueError(f"No trailing fits produced for N={n}")

        df = self.df
        dist_kwh = df["dist_kwh"].to_numpy(dtype=float)
        max_origin = len(df) - horizon

        errors_by_lead: list[list[float]] = [[] for _ in range(horizon)]
        errors_baseline_by_lead: list[list[float]] = [[] for _ in range(horizon)]
        oos_pred: list[float] = []
        oos_actual: list[float] = []
        oos_oat_f: list[float] = []
        oos_lead1_errors: list[float] = []
        oos_lead1_hour_start: list[pd.Timestamp] = []

        # Score the recursive horizon forecasts
        for origin in range(max_origin):
            self._log_debug(f"Scoring recursive horizon forecasts for origin {origin} / {max_origin}")
            origin_day = df["day"].iloc[origin]
            fit_idx = len(fit_days) - 1
            while fit_idx >= 0 and fit_days[fit_idx] >= origin_day:
                fit_idx -= 1
            if fit_idx < 0:
                continue

            # Get the predicted and actual distribution kWh for the next ``horizon`` hours
            params = results[fit_idx]
            params_baseline = results_baseline[fit_idx]
            pred = self.predict_recursive_horizon(params, origin, horizon)
            pred_baseline = self.predict_recursive_horizon(params_baseline, origin, horizon)
            target_start = origin + 1
            actual = dist_kwh[target_start : target_start + horizon] * params.energy_ratio

            # Collect the errors by lead
            for step in range(horizon):
                err = float(pred[step] - actual[step])
                err_baseline = float(pred_baseline[step] - actual[step])
                errors_by_lead[step].append(err)
                errors_baseline_by_lead[step].append(err_baseline)

            # Gathering of individual prediction and target values for every origin×lead in the recursive forecast evaluation.
            # This is used by the plotting functions in trailing_n_day_fits() to plot the predicted vs actual values.
            if collect_pointwise:
                oos_pred.extend(pred.tolist())
                oos_actual.extend(actual.tolist())
                oos_oat_f.extend(df["oat_f"].iloc[target_start : target_start + horizon].tolist())
                oos_lead1_errors.append(float(pred[0] - actual[0]))
                oos_lead1_hour_start.append(df["hour_start"].iloc[target_start])

        # Calculate the RMSE by lead
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

        # Calculate the overall RMSE and MAE
        all_errors = [err for lead in errors_by_lead for err in lead]
        all_errors_baseline = [err for lead in errors_baseline_by_lead for err in lead]
        rmse = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else float("nan")
        rmse_baseline = (
            float(np.sqrt(np.mean(np.square(all_errors_baseline))))
            if all_errors_baseline else float("nan")
        )
        mae = float(np.mean(np.abs(all_errors))) if all_errors else float("nan")
        mae_baseline = float(np.mean(np.abs(all_errors_baseline))) if all_errors_baseline else float("nan")

        return {
            "fit_days": fit_days,
            "results": results,
            "rmse": rmse,
            "rmse_baseline": rmse_baseline,
            "mae": mae,
            "mae_baseline": mae_baseline,
            "rmse_by_lead": rmse_by_lead,
            "rmse_baseline_by_lead": rmse_baseline_by_lead,
            "n_forecast_points": len(all_errors),
            "oos_pred": np.array(oos_pred),
            "oos_actual": np.array(oos_actual),
            "oos_oat_f": np.array(oos_oat_f),
            "oos_lead1_errors": np.array(oos_lead1_errors),
            "oos_lead1_hour_start": oos_lead1_hour_start,
        }

    @property
    def _artifact_label(self) -> str:
        window = "_growing" if self.GROW_WINDOW_TO_N else ""
        min_window = (
            f"_min{self.MIN_FIT_WINDOW_DAYS}"
            if self.GROW_WINDOW_TO_N and self.MIN_FIT_WINDOW_DAYS > 1
            else ""
        )
        occupancy = "_occupancy" if self.INCLUDE_OCCUPANCY else ""
        return f"{self.TRAINING_FREQUENCY}{window}{min_window}{occupancy}_recursive{self.FORECAST_HORIZON_HOURS}h"

    # ---------------------------------------------------------
    # Sweep the number of trailing days
    # ---------------------------------------------------------

    def sweep_n(self, min_n: int, max_n: int) -> None:
        """
        Sweeps the number of trailing days and plots the RMSE and B1 stability.
        """
        import matplotlib.pyplot as plt

        self.results_dir.mkdir(parents=True, exist_ok=True)

        rmses: list[float] = []
        b1_weekly_ranges: list[float] = []
        n_values = list(range(min_n, max_n + 1))

        # Evaluate the recursive horizon forecasts for each number of trailing days
        for n in n_values:
            oos = self._evaluate_recursive_horizon_oos(n, collect_pointwise=False)
            rmse = oos["rmse"]
            fit_days = oos["fit_days"]
            delta_t_coef_index = 1 + self.FEATURE_NAMES.index("deltaT")
            b1_values = [r.values[delta_t_coef_index] for r in oos["results"]]
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

        # Plot the RMSE and B1 stability
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
