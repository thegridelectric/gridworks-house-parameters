import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from get_data import HouseParamsDataFetcher
from plots import (
    EnergyVsSwtPanelData,
    _coldest_hour_legend_label,
    plot_estimated_energy_vs_swt as render_estimated_energy_vs_swt_plot,
    plot_oos_residual_by_hour_of_day,
    plot_oos_rmse_by_lead,
    plot_pred_vs_actual,
    plot_rswt_fit,
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


def _house_table_column_title(house_alias: str) -> str:
    return house_alias.replace("_", " ").title()


_ENERGY_COEFFICIENT_DESCRIPTIONS: dict[str, str] = {
    "deltaT": "Delta T (setpoint − OAT)",
    "windspeed_times_deltaT": "Wind speed × delta T",
    "solar_w_m2": "Solar irradiance (W/m²)",
    "previous_dist_kwh_scaled": "Previous hour scaled distribution kWh",
    "OAT_avg_4h": "Average OAT (4 h)",
    "dist_kwh_scaled_avg_4h": "Average scaled distribution kWh (4 h)",
    "oat_f": "Outdoor air temperature (OAT)",
    "windspeed_times_65_minus_oat": "Wind speed × (65°F − OAT)",
}


def _energy_regression_term_description(
    feature_names: tuple[str, ...], b_index: int
) -> str:
    if b_index == 0:
        return "Intercept"
    name = feature_names[b_index - 1]
    if name in _ENERGY_COEFFICIENT_DESCRIPTIONS:
        return _ENERGY_COEFFICIENT_DESCRIPTIONS[name]
    if name.startswith("wd_hour_"):
        hour = name.removeprefix("wd_hour_")
        return f"Weekday occupancy hour {hour}"
    if name.startswith("we_hour_"):
        hour = name.removeprefix("we_hour_")
        return f"Weekend occupancy hour {hour}"
    return name


def write_energy_regression_table_xlsx(
    params_by_house_alias: dict[str, HouseEnergyParams],
    path: Path,
    *,
    house_order: list[str] | None = None,
) -> None:
    """
    Write one column per house: B0 (intercept), B1…Bn (features), then R².
    Each coefficient has two rows (Coefficient, Std Error), matching the grid energy fit.
    """
    aliases = house_order or sorted(params_by_house_alias.keys())
    columns: dict[str, list[float | None]] = {}
    index_tuples: list[tuple[str, str]] = []
    term_descriptions: list[str] = []
    n_params: int | None = None
    reference_feature_names: tuple[str, ...] = ()

    for alias in aliases:
        params = params_by_house_alias.get(alias)
        if params is None or params.is_baseline:
            continue
        if n_params is None:
            n_params = len(params.values)
            reference_feature_names = params.feature_names
            for i in range(n_params):
                index_tuples.append((f"B{i}", "Coefficient"))
                index_tuples.append((f"B{i}", "Std Error"))
                term_descriptions.append(
                    _energy_regression_term_description(reference_feature_names, i)
                )
                term_descriptions.append("")
            index_tuples.append(("R²", "Coefficient"))
            index_tuples.append(("R²", "Std Error"))
            term_descriptions.append("R²")
            term_descriptions.append("")

        col_title = _house_table_column_title(alias)
        cells: list[float | None] = []
        for value, stderr in zip(params.values, params.std_errors, strict=True):
            cells.append(float(value))
            cells.append(float(stderr))
        cells.append(float(params.r_squared))
        cells.append(None)
        columns[col_title] = cells

    if not columns or n_params is None:
        raise ValueError("No non-baseline energy regression results to export")

    index = pd.MultiIndex.from_tuples(index_tuples, names=("Parameter", "Statistic"))
    table = pd.DataFrame(columns, index=index)
    table.insert(0, "Term", term_descriptions)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, merge_cells=False)


MAPLE_OAT_OPERATING_BINS_C: tuple[tuple[float, float, str], ...] = (
    (-25.0, -15.0, "-25 to -15"),
    (-15.0, -10.0, "-15 to -10"),
    (-10.0, -5.0, "-10 to -5"),
)


def _fahrenheit_to_celsius(values: np.ndarray | pd.Series) -> np.ndarray:
    return (np.asarray(values, dtype=float) - 32.0) * 5.0 / 9.0


def maple_operating_stats_by_oat_bin(
    computer: "HouseEnergyParamsComputer",
) -> pd.DataFrame:
    """
    Operating metrics by outside-air temperature bin (°C), on cleaned hourly data.
    """
    df = computer.df
    oat_c = _fahrenheit_to_celsius(df["oat_f"])
    max_hc = computer._max_zone_heatcall_fraction(df)
    rows: list[dict[str, object]] = []
    for lo, hi, label in MAPLE_OAT_OPERATING_BINS_C:
        mask = (oat_c >= lo) & (oat_c < hi)
        sub = df.loc[mask]
        n_hours = int(mask.sum())
        swt_c = _fahrenheit_to_celsius(sub["swt_average_f"].dropna())
        mean_swt_c = float(swt_c.mean()) if len(swt_c) else math.nan
        hc_sub = max_hc.loc[mask]
        if n_hours:
            building_heat_call_pct = float(hc_sub.mean() * 100.0)
            hours_ge_80_pct_heat_call_pct = float((hc_sub >= 0.8).mean() * 100.0)
            heat_delivered_kwh = float(sub["dist_kwh"].mean())
        else:
            building_heat_call_pct = math.nan
            hours_ge_80_pct_heat_call_pct = math.nan
            heat_delivered_kwh = math.nan
        rows.append(
            {
                "OAT (°C)": label,
                "OAT lower (°C)": lo,
                "OAT upper (°C)": hi,
                "n_hours": n_hours,
                "Mean SWT (°C)": mean_swt_c,
                "Building heat-call time (%)": building_heat_call_pct,
                "Hours ≥ 80% heat call (%)": hours_ge_80_pct_heat_call_pct,
                "Heat delivered (kWh)": heat_delivered_kwh,
            }
        )
    return pd.DataFrame(rows)


def _before_after_numeric(before: float, after: float, *, decimals: int) -> str:
    if not (math.isfinite(before) and math.isfinite(after)):
        return ""
    return f"{before:.{decimals}f} → {after:.{decimals}f}"


def _before_after_percent(before: float, after: float) -> str:
    if not (math.isfinite(before) and math.isfinite(after)):
        return ""
    return f"{round(before)}% → {round(after)}%"


def write_maple_operating_characteristics_xlsx(
    before_computer: "HouseEnergyParamsComputer",
    after_computer: "HouseEnergyParamsComputer",
    path: Path,
) -> None:
    """
    Compare maple_before and maple_after operating metrics by OAT bin (Table 8 style).
    """
    before = maple_operating_stats_by_oat_bin(before_computer)
    after = maple_operating_stats_by_oat_bin(after_computer)
    comparison = pd.DataFrame(
        {
            "OAT (°C)": before["OAT (°C)"],
            "Mean SWT (°C)": [
                _before_after_numeric(b, a, decimals=1)
                for b, a in zip(
                    before["Mean SWT (°C)"],
                    after["Mean SWT (°C)"],
                    strict=True,
                )
            ],
            "Building heat-call time": [
                _before_after_percent(b, a)
                for b, a in zip(
                    before["Building heat-call time (%)"],
                    after["Building heat-call time (%)"],
                    strict=True,
                )
            ],
            "Hours ≥ 80% heat call": [
                _before_after_percent(b, a)
                for b, a in zip(
                    before["Hours ≥ 80% heat call (%)"],
                    after["Hours ≥ 80% heat call (%)"],
                    strict=True,
                )
            ],
            "Heat delivered (kWh)": [
                _before_after_numeric(b, a, decimals=2)
                for b, a in zip(
                    before["Heat delivered (kWh)"],
                    after["Heat delivered (kWh)"],
                    strict=True,
                )
            ],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="Before vs after", index=False)
        before.to_excel(writer, sheet_name="Maple before", index=False)
        after.to_excel(writer, sheet_name="Maple after", index=False)


@dataclass(frozen=True)
class RswtLoadBucketFit:
    rwt_intercept: float
    rwt_slope: float


@dataclass(frozen=True)
class HouseRswtParams:
    rwt_min: float
    water_kg_hour_chc: float
    load_0_2: RswtLoadBucketFit | None
    load_2_4: RswtLoadBucketFit | None
    load_4_6: RswtLoadBucketFit | None
    load_6_8: RswtLoadBucketFit | None
    load_8_plus: RswtLoadBucketFit | None


class HouseEnergyParamsComputer:
    # Data source
    DATA_SOURCE: Literal["database", "csv"] = "database"
    HOURLY_DATA_DIR = Path("data")
    FORCE_DATABASE_REFRESH: bool = False
    MAPLE_ELECTRICITY_HOUSE_ALIASES = frozenset({"maple_before", "maple_after"})
    MAPLE_ELECTRICITY_CSV = (
        HOURLY_DATA_DIR / "maple_electricity_use_2025-11-01-04-00-2026-03-01-05-00.csv"
    )
    # Pre/post retrofit windows (hour_start is naive local time).
    MAPLE_BEFORE_LAST_INCLUDED_HOUR = pd.Timestamp("2026-01-02 23:00:00")
    MAPLE_AFTER_FIRST_INCLUDED_HOUR = pd.Timestamp("2026-01-05 00:00:00")

    # Data cleaning - erroneous data
    RANGE_OF_VALID_VALUES_PER_CHANNEL = {
        "oat_f": (-128, 134),
        "ws_mph": (0, 254),
        "solar_w_m2": (0, 1361),
        "dist_kwh": (0, 25),
        "hp_kwh_th": (0, 25),
        "swt_average_f": (40, 200),
        "rwt_average_f": (40, 200),
        "dist_water_kg": (0, int(3.785 * 60 * 10)),
        "dist_flow_fraction": (0, 1),
        "zone_set_or_temp_f": (-10, 110),
        "zone_heatcall_fraction": (0, 1),
    }
    MAX_ABS_RATE_OF_CHANGE_PER_CHANNEL = {
        "oat_f": 25.0,
    }

    # Data cleaning - missing data
    MAX_DATA_GAP_HOURS = 4
    NON_INTERPOLATED_OPTIONAL_COLUMNS = (
        "swt_average_f",
        "rwt_average_f",
        "dist_water_kg",
        "dist_flow_fraction",
    )

    # Data cleaning - known bad data
    OIL_BOILER_POWER_THRESHOLD_WATTS = 50
    # Max zone heat-call may exceed dist-flow on-time by at most this much (0–1 scale).
    HEATCALL_DIST_FLOW_TOLERANCE = 0.25
    # Only apply heat-call vs dist-flow check when max zone heat-call is at least this high.
    MIN_HEATCALL_FOR_DIST_FLOW_CHECK = 0.05
    REQUIRED_INPUT_CACHE_COLUMNS = ("dist_flow_fraction",)
    BROKEN_THERMOSTAT_MIN_TEMP_SET_GAP_F = {
        'default': 3,
        'beech': {'zone2': 4.5,}
    }
    BELOW_SETPOINT_MIN_TEMP_SET_GAP_F = 1
    EXTERNAL_HEAT_SOURCE_MIN_TEMP_ABOVE_SET_F = 3

    # SWT / energy plot legend labels (override DB channel display names).
    ZONE_LEGEND_NAME_OVERRIDES: dict[str, dict[int, str]] = {
        "beech": {1: "Downstairs", 2: "Upstairs"},
        "oak": {1: "Living room", 3: "Gear room"},
        "fir": {2: "Master bedroom", 3: "Upstairs bedroom"},
        "maple_before": {1: "Living room", 2: "Bedroom"},
        "maple_after": {1: "Living room", 2: "Bedroom"},
        "elm": {4: "Master bedroom"},
    }

    # Regression features
    FEATURE_NAMES = (
        "deltaT",
        "windspeed_times_deltaT",
        "solar_w_m2",
        "previous_dist_kwh_scaled",
        "OAT_avg_4h",
        "dist_kwh_scaled_avg_4h",
    )

    # Baseline regression features (alpha beta gamma model)
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
    TRAINING_FREQUENCY: Literal["daily", "weekly"] = "weekly"
    GROW_WINDOW_TO_N: bool = False
    MIN_FIT_WINDOW_DAYS: int = 10
    FORECAST_HORIZON_HOURS = 48
    ROLLING_AVG_HOURS = 4
    RSWT_LOAD_BUCKET_BOUNDS: tuple[tuple[float, float, str], ...] = (
        (0.0, 2.0, "0-2 kWh"),
        (2.0, 4.0, "2-4 kWh"),
        (4.0, 6.0, "4-6 kWh"),
        (6.0, 8.0, "6-8 kWh"),
        (8.0, math.inf, "8+ kWh"),
    )
    RSWT_CHC_HEATCALL_FRACTION_THRESHOLD = 0.8
    RSWT_CHC_HEATCALL_FLOOR = 0.7
    RSWT_BUCKET_MIN_FIT_ROWS = 20

    def __init__(
        self,
        house_alias: str,
        start_time: datetime,
        end_time: datetime,
        timezone: ZoneInfo = ZoneInfo("America/New_York"),
        *,
        include_occupancy: bool | None = None,
        prepare: bool = True,
    ) -> None:
        self.house_alias = house_alias
        self.start_time = start_time
        self.end_time = end_time
        self.timezone = timezone
        self.results_dir = Path("results") / house_alias
        if include_occupancy is not None:
            self.INCLUDE_OCCUPANCY = include_occupancy
        if prepare:
            self.prepare_data()
        else:
            self._load_data()
        
    def _log_info(self, message: str) -> None:
        logger.info(f"[{self.house_alias}] {message}")

    def _log_debug(self, message: str) -> None:
        logger.debug(f"[{self.house_alias}] {message}")

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

    def _apply_maple_data_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maple DB/cache exports are often incomplete. Merge house_params_data (zones, weather)
        and the shared electricity CSV (dist/hp kWh, heat-call, OAT, SWT/RWT).
        """
        if self.house_alias not in self.MAPLE_ELECTRICITY_HOUSE_ALIASES:
            return df

        df["hour_start"] = pd.to_datetime(df["hour_start"])

        params_path = self.HOURLY_DATA_DIR / f"{self.house_alias}_house_params_data.csv"
        if params_path.is_file():
            params = pd.read_csv(params_path)
            params["hour_start"] = pd.to_datetime(params["hour_start"])
            params = params.drop(columns=["dist_kwh", "hp_kwh_th"], errors="ignore")
            params = params.drop_duplicates(subset=["hour_start"], keep="last")
            params_cols = [c for c in params.columns if c != "hour_start"]
            df = df.drop(columns=[c for c in params_cols if c in df.columns], errors="ignore")
            df = df.merge(params, on="hour_start", how="left")
            self._log_info(f"Merged zone/weather columns from {params_path}")
        else:
            self._log_info(f"No house params CSV at {params_path}; using electricity export only")

        elec_path = self.MAPLE_ELECTRICITY_CSV
        if not elec_path.is_file():
            raise FileNotFoundError(
                f"{self.house_alias}: maple electricity CSV not found at {elec_path}"
            )
        elec_usecols = [
            "hour_start",
            "dist_kwh",
            "hp_kwh_th",
            "oat_f",
            "ws_mph",
            "dist_avg_swt",
            "dist_avg_rwt",
            "zone1_heatcall_fraction",
            "zone2_heatcall_fraction",
            "zone3_heatcall_fraction",
            "zone4_heatcall_fraction",
        ]
        elec = pd.read_csv(elec_path, usecols=elec_usecols)
        elec["hour_start"] = pd.to_datetime(elec["hour_start"])
        elec = elec.rename(
            columns={"dist_avg_swt": "swt_average_f", "dist_avg_rwt": "rwt_average_f"}
        )
        elec = elec.drop_duplicates(subset=["hour_start"], keep="last")
        elec = elec.rename(
            columns={
                c: f"{c}__elec"
                for c in elec.columns
                if c != "hour_start"
            }
        )
        df = df.merge(elec, on="hour_start", how="left")

        df["dist_kwh"] = df["dist_kwh__elec"]
        df["hp_kwh_th"] = df["hp_kwh_th__elec"]

        oat_lo, oat_hi = self.RANGE_OF_VALID_VALUES_PER_CHANNEL["oat_f"]
        params_oat = df["oat_f"] if "oat_f" in df.columns else pd.Series(np.nan, index=df.index)
        valid_oat = params_oat.notna() & params_oat.between(oat_lo, oat_hi)
        df["oat_f"] = np.where(valid_oat, params_oat, df["oat_f__elec"])

        ws_lo, ws_hi = self.RANGE_OF_VALID_VALUES_PER_CHANNEL["ws_mph"]
        params_ws = df["ws_mph"] if "ws_mph" in df.columns else pd.Series(np.nan, index=df.index)
        valid_ws = params_ws.notna() & params_ws.between(ws_lo, ws_hi)
        df["ws_mph"] = np.where(valid_ws, params_ws, df["ws_mph__elec"])

        swt_lo, swt_hi = self.RANGE_OF_VALID_VALUES_PER_CHANNEL["swt_average_f"]
        for target, elec_col in (
            ("swt_average_f", "swt_average_f__elec"),
            ("rwt_average_f", "rwt_average_f__elec"),
        ):
            base = df[target] if target in df.columns else pd.Series(np.nan, index=df.index)
            valid = base.notna() & base.between(swt_lo, swt_hi)
            from_elec = df[elec_col]
            df[target] = np.where(valid, base, from_elec)

        for col in df.columns:
            if col.endswith("__elec"):
                continue
            elec_col = f"{col}__elec"
            if elec_col not in df.columns:
                continue
            if col.endswith("_heatcall_fraction"):
                if col in df.columns:
                    df[col] = df[col].fillna(df[elec_col])
                else:
                    df[col] = df[elec_col]

        df = df.drop(columns=[c for c in df.columns if c.endswith("__elec")])

        for col in df.columns:
            if str(col).startswith("zone") and str(col).endswith(("_avg_set", "_avg_temp")):
                df[col] = df[col].ffill().bfill()

        df["dist_kwh"] = df["dist_kwh"].fillna(0.0)
        df["hp_kwh_th"] = df["hp_kwh_th"].fillna(0.0)
        df["oat_f"] = df["oat_f"].fillna(0.0)
        df["ws_mph"] = df["ws_mph"].fillna(0.0)
        if "solar_w_m2" not in df.columns:
            df["solar_w_m2"] = 0.0
        else:
            df["solar_w_m2"] = df["solar_w_m2"].fillna(0.0)
        if "oil_boiler_pwr" not in df.columns:
            df["oil_boiler_pwr"] = 0.0

        self._log_info(f"Overrode dist_kwh, hp_kwh_th, and filled gaps from {elec_path}")
        return df

    def _filter_maple_analysis_window(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.house_alias == "maple_before":
            mask = df["hour_start"] <= self.MAPLE_BEFORE_LAST_INCLUDED_HOUR
            label = f"hour_start <= {self.MAPLE_BEFORE_LAST_INCLUDED_HOUR} (Jan 3, 2026 excluded)"
        elif self.house_alias == "maple_after":
            mask = df["hour_start"] >= self.MAPLE_AFTER_FIRST_INCLUDED_HOUR
            label = f"hour_start >= {self.MAPLE_AFTER_FIRST_INCLUDED_HOUR} (Jan 4, 2026 excluded)"
        else:
            return df
        n_before = len(df)
        df = df.loc[mask].reset_index(drop=True)
        self._log_info(
            f"{self.house_alias}: kept {len(df)} of {n_before} hours ({label})"
        )
        return df

    def _load_data(self) -> None:
        """
        Loads house params data from CSV or the database and prepares it for analysis.
        - Sorts by hour_start, and adds a day column.
        - Finds the number of zones
        - Checks that all required columns are present
        """
        zone_names_from_channels: dict[int, str] = {}

        if self.DATA_SOURCE == "csv":
            csv_path = self.HOURLY_DATA_DIR / f"{self.house_alias}_house_params_data.csv"
            df = pd.read_csv(csv_path)
            df["hour_start"] = pd.to_datetime(df["hour_start"])
            df = self._apply_maple_data_overrides(df)
            self._log_info(f"Loaded {csv_path}")
            zone_names_from_channels = HouseParamsDataFetcher.load_zone_display_names(
                self.house_alias
            )

        elif self.DATA_SOURCE == "database":
            range_start = pd.Timestamp(
                self.start_time.astimezone(self.timezone).replace(tzinfo=None)
            )
            range_end = pd.Timestamp(
                self.end_time.astimezone(self.timezone).replace(tzinfo=None)
            )
            cache_path = self.HOURLY_DATA_DIR / f"{self.house_alias}_input_data.csv"
            df = None
            cache_rejected_missing_columns = False
            if not self.FORCE_DATABASE_REFRESH and cache_path.is_file():
                cached = pd.read_csv(cache_path)
                cached["hour_start"] = pd.to_datetime(cached["hour_start"])
                expected_hours = pd.date_range(
                    start=range_start,
                    end=range_end,
                    freq="h",
                    inclusive="left",
                )
                in_range = cached.loc[
                    (cached["hour_start"] >= range_start)
                    & (cached["hour_start"] < range_end),
                    "hour_start",
                ]
                cache_ok = expected_hours.empty or (
                    len(in_range) == len(expected_hours)
                    and set(in_range) == set(expected_hours)
                )
                if cache_ok:
                    missing_cache_cols = [
                        col
                        for col in self.REQUIRED_INPUT_CACHE_COLUMNS
                        if col not in cached.columns
                    ]
                    if missing_cache_cols:
                        cache_ok = False
                        cache_rejected_missing_columns = True
                        self._log_info(
                            f"Cache {cache_path} missing columns {missing_cache_cols}; "
                            f"querying database"
                        )
                if cache_ok:
                    df = cached
                    self._log_info(f"Loaded hourly data from cache {cache_path}")
                    zone_names_from_channels = (
                        HouseParamsDataFetcher.load_zone_display_names(self.house_alias)
                    )
                elif not cache_rejected_missing_columns:
                    self._log_info(
                        f"Cache {cache_path} missing hours for "
                        f"{range_start.isoformat()}–{range_end.isoformat()}; "
                        f"querying database"
                    )
            loaded_from_cache = df is not None
            if df is None:
                self.HOURLY_DATA_DIR.mkdir(parents=True, exist_ok=True)
                fetcher = HouseParamsDataFetcher(
                    self.house_alias,
                    self.start_time,
                    self.end_time,
                    self.timezone,
                )
                (
                    channel_data_by_hour_start,
                    csv_fieldnames,
                    zone_numbers,
                    zone_names_from_channels,
                ) = fetcher.fetch()
                rows: list[dict[str, object]] = []
                for hour in sorted(channel_data_by_hour_start.keys()):
                    data = channel_data_by_hour_start[hour]
                    row: dict[str, object] = {
                        "hour_start": hour,
                        "oat_f": data.get("oat_f", np.nan),
                        "ws_mph": data.get("ws_mph", np.nan),
                        "solar_w_m2": data.get("solar_w_m2", np.nan),
                        "dist_kwh": data.get("dist_kwh", 0.0),
                        "hp_kwh_th": data.get("hp_kwh_th", 0.0),
                        "swt_average_f": data.get("swt_average_f", np.nan),
                        "rwt_average_f": data.get("rwt_average_f", np.nan),
                        "dist_water_kg": data.get("dist_water_kg", np.nan),
                        "dist_flow_fraction": data.get("dist_flow_fraction", np.nan),
                        "oil_boiler_pwr": data.get("oil_boiler_pwr", 0.0),
                    }
                    for zone in zone_numbers:
                        row[f"zone{zone}_heatcall_fraction"] = data.get(
                            f"zone{zone}_heatcall_fraction", 0.0
                        )
                    for zone in zone_numbers:
                        row[f"zone{zone}_avg_set"] = data.get(
                            f"zone{zone}_avg_set", np.nan
                        )
                        row[f"zone{zone}_avg_temp"] = data.get(
                            f"zone{zone}_avg_temp", np.nan
                        )
                    rows.append(row)
                df = pd.DataFrame(rows, columns=csv_fieldnames)
                self._log_info(f"Loaded data from database ({len(df)} rows)")

            weather_csv_path = (
                self.HOURLY_DATA_DIR / f"{self.house_alias}_house_params_data.csv"
            )
            weather_df = pd.read_csv(
                weather_csv_path,
                usecols=["hour_start", "oat_f", "ws_mph", "solar_w_m2"],
            )
            weather_df["hour_start"] = pd.to_datetime(weather_df["hour_start"])
            df["hour_start"] = pd.to_datetime(df["hour_start"])
            df = df.drop(
                columns=["oat_f", "ws_mph", "solar_w_m2"], errors="ignore"
            ).merge(weather_df, on="hour_start", how="left")
            self._log_info(f"Weather from {weather_csv_path}")
            df = self._apply_maple_data_overrides(df)
            if not loaded_from_cache:
                df.to_csv(cache_path, index=False)
                self._log_info(f"Wrote hourly data cache to {cache_path}")

        else:
            raise ValueError(f"Unsupported DATA_SOURCE: {self.DATA_SOURCE!r}")

        df["hour_start"] = pd.to_datetime(df["hour_start"])
        range_start = pd.Timestamp(self.start_time.astimezone(self.timezone).replace(tzinfo=None))
        range_end = pd.Timestamp(self.end_time.astimezone(self.timezone).replace(tzinfo=None))
        df = df.loc[(df["hour_start"] >= range_start) & (df["hour_start"] < range_end)]
        self._log_info(
            f"Cropped datafrom {range_start.isoformat()} to "
            f"{range_end.isoformat()} ({self.timezone})"
        )
        df = self._filter_maple_analysis_window(df)
        df = df.sort_values("hour_start").reset_index(drop=True)
        df["day"] = df["hour_start"].dt.normalize()

        self.zones = sorted(
            int(str(c).removeprefix("zone").removesuffix("_avg_set"))
            for c in df.columns
            if str(c).startswith("zone") and str(c).endswith("_avg_set")
        )
        self.zone_names = {
            z: zone_names_from_channels[z]
            for z in self.zones
            if z in zone_names_from_channels
        }
        for zone, display in self.ZONE_LEGEND_NAME_OVERRIDES.get(
            self.house_alias, {}
        ).items():
            if zone in self.zones:
                self.zone_names[zone] = display

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
            raise ValueError(f"Missing required columns for {self.house_alias}: {missing_columns}")
        
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
        (OAT_avg_4h and dist_kwh_avg_4h are built during missing-data handling in _clean_data.)
        """
        df = self.df.copy()
        if self.zones:
            set_cols = [f"zone{z}_avg_set" for z in self.zones]
            setpoint_avg = df[set_cols].mean(axis=1)
            for col in set_cols:
                df[col] = df[col].ffill().bfill()
            setpoint_avg = setpoint_avg.fillna(df[set_cols].mean(axis=1))
        else:
            setpoint_avg = pd.Series(68.0, index=df.index)
        df["deltaT"] = (setpoint_avg - df["oat_f"]).clip(lower=0)
        df["windspeed_times_deltaT"] = df["deltaT"] * df["ws_mph"]
        df["previous_dist_kwh"] = df["dist_kwh"].shift(1)
        df = df.iloc[1:].reset_index(drop=True)
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
            if channel not in df.columns:
                continue
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
        - Computes average OAT over the last ROLLING_AVG_HOURS when the start and end value are not interpolated, otherwise drops the row
        - Computes average dist_kwh over the last ROLLING_AVG_HOURS only when all hours in the window are not interpolated, otherwise drops the row
        - Drops rows inside missing-data gaps
        - Drops remaining rows that contain NaNs (except dist SWT/RWT averages)
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
            if channel == "day" or channel in self.NON_INTERPOLATED_OPTIONAL_COLUMNS:
                continue
            is_nan = df[channel].isna()
            block_id = is_nan.ne(is_nan.shift()).cumsum()
            block_len = is_nan.groupby(block_id).transform("size")
            drop_rows |= is_nan & (block_len >= self.MAX_DATA_GAP_HOURS)

        # Interpolate missing data
        n_filled = 0
        oat_f_interpolated = pd.Series(False, index=df.index)
        dist_kwh_interpolated = pd.Series(False, index=df.index)
        for channel in df.columns:
            if channel == "day" or channel in self.NON_INTERPOLATED_OPTIONAL_COLUMNS:
                continue
            before = df[channel].copy()
            df[channel] = df[channel].interpolate(method="time")
            if channel != "oat_f":
                df.loc[drop_rows, channel] = before.loc[drop_rows]
            filled = before.isna() & df[channel].notna()
            n_filled += int(filled.sum())
            if channel == "oat_f":
                oat_f_interpolated |= filled
            if channel == "dist_kwh":
                dist_kwh_interpolated |= filled
        self._log_info(f"Interpolation: {n_filled} values filled")

        # Compute average OAT and dist_kwh over the last ROLLING_AVG_HOURS
        n = self.ROLLING_AVG_HOURS
        oat_mean = pd.Series(df["oat_f"].rolling(n).mean(), index=df.index).shift(1)
        first_oat_observed = (~oat_f_interpolated) & df["oat_f"].notna()
        oat_endpoints_ok = first_oat_observed.shift(n) & first_oat_observed.shift(1)
        df["OAT_avg_4h"] = oat_mean.where(oat_endpoints_ok)

        dist_kwh_mean = pd.Series(df["dist_kwh"].rolling(n).mean(), index=df.index).shift(1)
        first_dist_kwh_observed = (~dist_kwh_interpolated) & df["dist_kwh"].notna()
        dist_kwh_window_ok = first_dist_kwh_observed.shift(1)
        for lag in range(2, n + 1):
            dist_kwh_window_ok &= first_dist_kwh_observed.shift(lag)
        df["dist_kwh_avg_4h"] = dist_kwh_mean.where(dist_kwh_window_ok)

        df = df.loc[df["OAT_avg_4h"].notna() & df["dist_kwh_avg_4h"].notna()]
 
        # Drop rows inside missing-data gaps
        df = df.loc[~drop_rows]
        df = df.reset_index(names="hour_start")
        self._log_info(f"Long missing-data gaps (>={self.MAX_DATA_GAP_HOURS}h): dropped {int(drop_rows.sum())} rows")

        # If any rows still have NaNs (except in NON_INTERPOLATED_OPTIONAL_COLUMNS), drop them
        required_for_dropna = [c for c in df.columns if c not in self.NON_INTERPOLATED_OPTIONAL_COLUMNS]
        remaining_nans = int(df[required_for_dropna].isna().sum().sum())
        self._log_info(f"NaNs remaining (excl. dist SWT/RWT): {remaining_nans}")
        if remaining_nans > 0:
            n_before = len(df)
            df = df.dropna(subset=required_for_dropna).reset_index(drop=True)
            self._log_info(f"Dropped {n_before - len(df)} rows with NaNs")

        return df

    def _filter_out_known_bad_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out hours which include known-bad data:
        - Zone below setpoint (and the prior consecutive hour)
        - Used oil boiler
        - Zone setpoint change
        - Broken thermostat
        - Zone heat-call fraction higher than dist-flow on-time (shared pump)
        - External heat source (voluntarily excluded for now)
        """
        df = self._zone_below_setpoint(df)
        df = self._used_oil_boiler(df)
        df = self._setpoint_change(df)
        df = self._broken_thermostat(df)
        df = self._heat_call_exceeds_dist_flow(df)
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
        """
        Drop hours when any zone's average temperature is well below its average setpoint,
        and the hour immediately before such an hour when it is consecutive in time.
        """
        min_gap = self.BELOW_SETPOINT_MIN_TEMP_SET_GAP_F
        drop_rows = pd.Series(False, index=df.index)
        next_hour = df["hour_start"].shift(-1)
        consecutive_to_next = (next_hour - df["hour_start"]) == pd.Timedelta(hours=1)
        for z in self.zones:
            temp_col = f"zone{z}_avg_temp"
            setpoint_col = f"zone{z}_avg_set"
            below_setpoint = (df[setpoint_col] - df[temp_col]) >= min_gap
            below_next = below_setpoint.shift(-1).fillna(False).astype(bool)
            hour_before_below = below_next & consecutive_to_next
            drop_rows |= below_setpoint | hour_before_below
            for hour_start, temp, setpoint in df.loc[below_setpoint, ["hour_start", temp_col, setpoint_col]].itertuples(index=False):
                self._log_debug(f"Below setpoint (zone {z}): hour_start={hour_start} set={setpoint} - temp={temp} >= {min_gap}°F")
            for hour_start, temp, setpoint in df.loc[
                hour_before_below & ~below_setpoint,
                ["hour_start", temp_col, setpoint_col],
            ].itertuples(index=False):
                self._log_debug(
                    f"Before below setpoint (zone {z}): hour_start={hour_start} "
                    f"(next consecutive hour below setpoint); set={setpoint} temp={temp}"
                )
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

    def _heat_call_exceeds_dist_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop hours when max zone heat-call fraction exceeds dist-flow on-time fraction
        (beyond tolerance). Several zones share one dist pump, so flow on-time may be
        much higher than any single zone's heat-call; the reverse is not physical.
        """
        if "dist_flow_fraction" not in df.columns:
            raise ValueError(
                f"{self.house_alias}: dist_flow_fraction is required for heat-call vs "
                f"dist-flow filtering; refresh input data from the database "
                f"(delete {self.HOURLY_DATA_DIR / f'{self.house_alias}_input_data.csv'} "
                f"or set FORCE_DATABASE_REFRESH)"
            )
        max_hc = self._max_zone_heatcall_fraction(df)
        flow_on = df["dist_flow_fraction"]
        tolerance = self.HEATCALL_DIST_FLOW_TOLERANCE
        min_hc = self.MIN_HEATCALL_FOR_DIST_FLOW_CHECK
        bad = (
            (max_hc >= min_hc)
            & flow_on.notna()
            & (max_hc > flow_on + tolerance)
        )
        for hour_start, hc, flow_frac in zip(
            df.loc[bad, "hour_start"],
            max_hc.loc[bad],
            flow_on.loc[bad],
            strict=True,
        ):
            self._log_debug(
                f"Heat-call > dist-flow: hour_start={hour_start} "
                f"max_zone_heatcall={hc:.2f} > dist_flow_fraction={flow_frac:.2f} "
                f"+ tolerance={tolerance:.2f}"
            )
        self._log_info(f"Heat-call > dist-flow: dropped {int(bad.sum())} rows")
        return df.loc[~bad].reset_index(drop=True)

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
        if len(df) == 0:
            raise ValueError(f"{self.house_alias}: no rows to fit energy model")
        dist_kwh = df["dist_kwh"].to_numpy()
        dist_kwh_total = float(np.sum(dist_kwh))
        if dist_kwh_total <= 0:
            raise ValueError(
                f"{self.house_alias}: total dist_kwh is {dist_kwh_total}; "
                "cannot compute energy ratio (check cached/database energy data)"
            )
        energy_ratio = float(np.sum(df["hp_kwh_th"].to_numpy())) / dist_kwh_total
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
            if name in ("previous_dist_kwh_scaled", "dist_kwh_scaled_avg_4h"):
                raw_col = "previous_dist_kwh" if name == "previous_dist_kwh_scaled" else "dist_kwh_avg_4h"
                columns.append(df[raw_col].to_numpy(dtype=float) * energy_ratio)
            else:
                columns.append(df[name].to_numpy(dtype=float))
        return np.column_stack(columns)

    # --------------------------------------------------
    # Get energy parameters for the last N days of data
    # --------------------------------------------------

    def _df_for_last_n_calendar_days(self, n: int) -> pd.DataFrame:
        days = np.unique(self.df["day"].to_numpy())
        window_days = (days[-n:] if len(days) >= n else days).tolist()
        return self.df.loc[self.df["day"].isin(window_days)]

    def _max_zone_heatcall_fraction(self, df: pd.DataFrame) -> pd.Series:
        """Max hourly heat-call duty cycle across zones (each zone is one fraction in [0, 1])."""
        heatcall_cols = [f"zone{z}_heatcall_fraction" for z in self.zones]
        return df[heatcall_cols].max(axis=1)

    def _dominant_zone_at_max_heatcall(self, df: pd.DataFrame) -> pd.Series:
        """Zone index (per ``self.zones``) with the highest heat-call fraction each hour (ties: first zone)."""
        heatcall_cols = [f"zone{z}_heatcall_fraction" for z in self.zones]
        argmax = np.nanargmax(df[heatcall_cols].to_numpy(dtype=float), axis=1)
        return pd.Series([self.zones[i] for i in argmax], index=df.index)

    def energy_fit_on_last_n_days(self, n: int) -> HouseEnergyParams:
        """
        Fits the model on the last ``n`` calendar days of data.
        """
        return self.fit(self._df_for_last_n_calendar_days(n))

    def energy_fit_on_all_cleaned_data(self) -> HouseEnergyParams:
        """Fits the model on every hour in ``self.df`` (post clean + feature engineering)."""
        return self.fit(self.df)

    # ----------------------------------------------------
    # Get rswt parameters for the last N days of data
    # ----------------------------------------------------

    def rswt_fit_on_last_n_days(self, n: int) -> HouseRswtParams:
        """
        Fits return-side water temperature (RSWT) line parameters on the last ``n`` calendar days.

        Uses ``rwt_average_f ~ intercept + slope * swt_average_f`` separately for each hourly
        ``dist_kwh`` band (see ``RSWT_LOAD_BUCKET_BOUNDS``). ``water_kg_hour_chc`` is mean
        ``dist_water_kg`` on hours with max zone heat-call above
        ``RSWT_CHC_HEATCALL_FRACTION_THRESHOLD``, or top-k by heat-call if too few.
        Load bands with fewer than ``RSWT_BUCKET_MIN_FIT_ROWS`` hours are omitted (``None``).
        """
        days = np.unique(self.df["day"].to_numpy())
        window_days = (days[-n:] if len(days) >= n else days).tolist()
        window_df = self.df.loc[self.df["day"].isin(window_days)]

        heatcall_cols = [f"zone{z}_heatcall_fraction" for z in self.zones]
        rwt_cols = ["swt_average_f", "rwt_average_f", "dist_kwh", *heatcall_cols]
        rwt_df = window_df[rwt_cols].dropna()
        if rwt_df.empty:
            raise ValueError(
                f"{self.house_alias}: no rows with swt_average_f, rwt_average_f, dist_kwh, "
                f"and zone heat-call data in the last {n} day(s)"
            )

        if "dist_water_kg" not in window_df.columns:
            raise ValueError(f"{self.house_alias}: dist_water_kg column is missing")
        rwt_df = rwt_df.copy()
        rwt_df["dist_water_kg"] = window_df.loc[rwt_df.index, "dist_water_kg"]
        rwt_df["max_zone_heatcall_fraction"] = self._max_zone_heatcall_fraction(rwt_df)
        rwt_df = rwt_df.drop(columns=heatcall_cols)

        bucket_fits: list[RswtLoadBucketFit | None] = []
        bucket_n_hours: list[int] = []
        bucket_mid_load_kwh: list[float] = []
        min_rows = self.RSWT_BUCKET_MIN_FIT_ROWS
        bounds = self.RSWT_LOAD_BUCKET_BOUNDS
        high_lo, high_hi, _high_load_label = bounds[-1]
        prev_lo = bounds[-2][0]
        chc_reference_load_kwh = (
            high_lo + (high_lo - prev_lo) / 2
            if not math.isfinite(high_hi)
            else (high_lo + high_hi) / 2
        )
        for idx, (lo, hi, label) in enumerate(bounds):
            is_last = idx == len(bounds) - 1
            if is_last and not math.isfinite(hi):
                mid_load = chc_reference_load_kwh
            else:
                mid_load = (lo + hi) / 2
            bucket_mid_load_kwh.append(mid_load)

            load = rwt_df["dist_kwh"]
            if is_last:
                mask = load >= lo
            else:
                mask = (load >= lo) & (load < hi)
            bucket_df = rwt_df.loc[mask]
            n_bucket = len(bucket_df)
            bucket_n_hours.append(n_bucket)
            if n_bucket < min_rows:
                self._log_info(
                    f"RSWT {label}: skipped ({n_bucket} hour(s), need {min_rows})"
                )
                bucket_fits.append(None)
                continue

            intercept, slope = self._fit_rwt_vs_swt(bucket_df)
            self._log_info(
                f"RSWT {label}: {n_bucket} hour(s), "
                f"intercept={intercept:.4f}, slope={slope:.4f}"
            )
            bucket_fits.append(
                RswtLoadBucketFit(
                    rwt_intercept=round(intercept, 6),
                    rwt_slope=round(slope, 6),
                )
            )

        if all(f is None for f in bucket_fits):
            raise ValueError(
                f"{self.house_alias}: no load band had at least {min_rows} hour(s) "
                f"for RSWT fit in the last {n} day(s)"
            )

        threshold = self.RSWT_CHC_HEATCALL_FRACTION_THRESHOLD
        floor = self.RSWT_CHC_HEATCALL_FLOOR
        hc = rwt_df["max_zone_heatcall_fraction"]
        above_threshold = rwt_df.loc[hc > threshold]
        if len(above_threshold) >= min_rows:
            chc_df = above_threshold
            self._log_info(
                f"CHC water: {len(chc_df)} hour(s) with max zone heat-call > {threshold}"
            )
        else:
            n_above = len(above_threshold)
            k = min(min_rows, len(rwt_df))
            floored = rwt_df.loc[hc >= floor]
            if len(floored) >= k:
                pool = floored
                pool_label = f"heat-call >= {floor}"
            else:
                pool = rwt_df
                pool_label = "all hours"
            chc_df = pool.nlargest(k, "max_zone_heatcall_fraction")
            min_hc = float(chc_df["max_zone_heatcall_fraction"].min())
            max_hc = float(chc_df["max_zone_heatcall_fraction"].max())
            self._log_info(
                f"CHC water: only {n_above} hour(s) above {threshold}, "
                f"using top {len(chc_df)} by heat-call from {pool_label} "
                f"(heat-call {min_hc:.2f}–{max_hc:.2f})"
            )

        chc_dist_water = chc_df["dist_water_kg"].dropna()
        if chc_dist_water.empty:
            raise ValueError(
                f"{self.house_alias}: no dist_water_kg on CHC training rows "
                f"in the last {n} day(s)"
            )
        water_kg_hour_chc = round(float(chc_dist_water.mean()), 2)
        water_chc_n_hours = len(chc_dist_water)
        self._log_info(
            f"water_kg_hour_chc={water_kg_hour_chc:.2f} kg "
            f"(mean over {water_chc_n_hours} CHC training hour(s))"
        )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        plot_rswt_fit(
            rwt_df["swt_average_f"].to_numpy(),
            rwt_df["rwt_average_f"].to_numpy(),
            rwt_df["dist_kwh"].to_numpy(),
            bucket_fits=[
                (label, fit, n_hours, mid_load)
                for (_, _, label), fit, n_hours, mid_load in zip(
                    bounds,
                    bucket_fits,
                    bucket_n_hours,
                    bucket_mid_load_kwh,
                    strict=True,
                )
            ],
            water_kg_hour_chc=water_kg_hour_chc,
            water_chc_n_hours=water_chc_n_hours,
            house_alias=self.house_alias,
            n_days=n,
            savepath=self.results_dir / f"{self.house_alias}_rswt_fit_N{n}.png",
        )

        return HouseRswtParams(
            rwt_min=50.0,
            water_kg_hour_chc=water_kg_hour_chc,
            load_0_2=bucket_fits[0],
            load_2_4=bucket_fits[1],
            load_4_6=bucket_fits[2],
            load_6_8=bucket_fits[3],
            load_8_plus=bucket_fits[4],
        )

    def _fit_rwt_vs_swt(self, df: pd.DataFrame) -> tuple[float, float]:
        """OLS fit: rwt_average_f = intercept + slope * swt_average_f."""
        if len(df) < 2:
            raise ValueError(
                f"{self.house_alias}: need at least 2 rows to fit RSWT vs SWT (got {len(df)})"
            )
        swt = df["swt_average_f"].to_numpy(dtype=float)
        rwt = df["rwt_average_f"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(df)), swt])
        coefficients, *_ = np.linalg.lstsq(X, rwt, rcond=None)
        return float(coefficients[0]), float(coefficients[1])

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
            oat_f_colormap_bounds=(
                float(np.min(self.df["oat_f"].to_numpy())),
                float(np.max(self.df["oat_f"].to_numpy())),
            ),
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
        days = np.unique(self.df["day"].to_numpy())
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
                window_df = self.df.loc[self.df["day"].isin(window.tolist())]
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
        self._log_info(f"Scoring recursive horizon forecasts for {max_origin} origins...")
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
                b1_stability_metric = float(np.nanmean(b1_stability.to_numpy()))
            else:
                b1_stability_metric = float(np.nanmean(b1.diff().abs().to_numpy()))
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

    def energy_vs_swt_panel_data(self, params: HouseEnergyParams) -> EnergyVsSwtPanelData:
        """Arrays for SWT vs energy scatter plots (valid cleaned hours only)."""
        df = self.df
        estimated_kwh = self.predict(params, df)
        swt_f = df["swt_average_f"].to_numpy(dtype=float)
        max_zone_heatcall = self._max_zone_heatcall_fraction(df).to_numpy(dtype=float)
        dominant_zone = self._dominant_zone_at_max_heatcall(df).to_numpy(dtype=int)
        valid = np.isfinite(swt_f) & np.isfinite(estimated_kwh)
        if not np.any(valid):
            raise ValueError(
                f"{self.house_alias}: no hours with both swt_average_f and estimated energy"
            )

        oat = df["oat_f"].to_numpy(dtype=float)
        dist_kwh = df["dist_kwh"].to_numpy(dtype=float)
        coldest_oat_estimated_kwh: float | None = None
        coldest_oat_dist_kwh: float | None = None
        coldest_oat_swt_f: float | None = None
        coldest_oat_date_label: str | None = None
        coldest_oat_panel_index: int | None = None
        coldest_i: int | None = None
        if np.any(np.isfinite(oat)):
            coldest_i = int(np.nanargmin(oat))
            if valid[coldest_i]:
                coldest_oat_estimated_kwh = float(estimated_kwh[coldest_i])
                coldest_oat_dist_kwh = float(dist_kwh[coldest_i])
                coldest_oat_swt_f = float(swt_f[coldest_i])
                coldest_oat_date_label = _coldest_hour_legend_label(float(oat[coldest_i]))
                valid_row_indices = np.flatnonzero(valid)
                coldest_pos = np.where(valid_row_indices == coldest_i)[0]
                if len(coldest_pos):
                    coldest_oat_panel_index = int(coldest_pos[0])

        zone_avg_set_f: dict[int, float] = {}
        coldest_hour_zone_set_f: dict[int, float] = {}
        zone_set_f_by_hour: dict[int, np.ndarray] = {}
        for z in self.zones:
            set_col = f"zone{z}_avg_set"
            sets_valid = df.loc[valid, set_col].to_numpy(dtype=float)
            zone_set_f_by_hour[z] = sets_valid
            zone_avg_set_f[z] = float(np.nanmean(sets_valid))
            if coldest_i is not None and valid[coldest_i]:
                coldest_hour_zone_set_f[z] = float(df[set_col].iloc[coldest_i])

        return EnergyVsSwtPanelData(
            house_alias=self.house_alias,
            estimated_kwh=estimated_kwh[valid],
            dist_kwh=dist_kwh[valid],
            swt_f=swt_f[valid],
            max_zone_heatcall=max_zone_heatcall[valid],
            dominant_zone=dominant_zone[valid],
            zone_numbers=tuple(self.zones),
            coldest_oat_estimated_kwh=coldest_oat_estimated_kwh,
            coldest_oat_dist_kwh=coldest_oat_dist_kwh,
            coldest_oat_swt_f=coldest_oat_swt_f,
            coldest_oat_panel_index=coldest_oat_panel_index,
            coldest_oat_date_label=coldest_oat_date_label,
            oat_f=oat[valid],
            hour_start=df["hour_start"].to_numpy()[valid],
            zone_avg_set_f=zone_avg_set_f,
            coldest_hour_zone_set_f=coldest_hour_zone_set_f,
            zone_set_f_by_hour=zone_set_f_by_hour,
            zone_names=self.zone_names,
        )

    def plot_estimated_energy_vs_swt(self, params: HouseEnergyParams) -> None:
        """
        Interactive scatter of model-estimated scaled distribution energy (x) vs measured
        SWT (y) for hours in ``self.df`` (post ``_clean_data`` and feature engineering)
        with SWT data, colored by the zone with the highest heat-call fraction that hour.
        ``params`` should be fit on the same cleaned frame (e.g. ``energy_fit_on_all_cleaned_data``).
        A slider filters to hours with
        max zone heat-call fraction ≥ the chosen value; the OLS regression line updates
        with the filter.
        """
        df = self.df
        max_zone_heatcall = self._max_zone_heatcall_fraction(df).to_numpy(dtype=float)
        dominant_zone = self._dominant_zone_at_max_heatcall(df).to_numpy(dtype=int)
        high_heatcall = max_zone_heatcall > 0.8
        n_high_heatcall = int(high_heatcall.sum())
        self._log_info(
            f"Hours with max zone heat-call > 80%: {n_high_heatcall} "
            f"(of {len(df)} cleaned hours)"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        zone_hc_cols = [f"zone{z}_heatcall_fraction" for z in self.zones]
        high_hc_df = df.loc[high_heatcall, ["hour_start", "oat_f", *zone_hc_cols]].copy()
        high_hc_df["max_zone_heatcall_fraction"] = max_zone_heatcall[high_heatcall]
        high_hc_df["dominant_zone"] = dominant_zone[high_heatcall]
        high_hc_df = high_hc_df.sort_values("hour_start")
        high_hc_csv = self.results_dir / f"{self.house_alias}_high_heatcall_hours.csv"
        high_hc_df.to_csv(high_hc_csv, index=False)
        self._log_info(f"Wrote high heat-call hours to {high_hc_csv}")

        panel = self.energy_vs_swt_panel_data(params)
        savepath = self.results_dir / f"{self.house_alias}_energy_vs_swt.png"
        render_estimated_energy_vs_swt_plot(
            panel.estimated_kwh,
            panel.dist_kwh,
            panel.swt_f,
            panel.max_zone_heatcall,
            panel.dominant_zone,
            zone_numbers=panel.zone_numbers,
            house_alias=panel.house_alias,
            savepath=savepath,
            zone_names=panel.zone_names,
            coldest_oat_estimated_kwh=panel.coldest_oat_estimated_kwh,
            coldest_oat_dist_kwh=panel.coldest_oat_dist_kwh,
            coldest_oat_swt_f=panel.coldest_oat_swt_f,
            coldest_oat_panel_index=panel.coldest_oat_panel_index,
            coldest_oat_date_label=panel.coldest_oat_date_label,
        )
        self._log_info(f"Wrote energy vs SWT plot to {savepath}")
