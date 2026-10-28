"""
Train house energy and RSWT parameters on separate cleaned training windows, then
load and clean test hours (same pipeline as training), apply the fitted models, and
export CSV.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from house_parameters import HouseEnergyParams, HouseEnergyParamsComputer
from plots import plot_test_period_method_comparison, plot_training_hourly_avg_setpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

HOUSE_ALIAS = "beech"
TIMEZONE = ZoneInfo("America/New_York")

ENERGY_TRAINING_DATA_START_TIME = datetime(2025, 10, 17, tzinfo=TIMEZONE)
ENERGY_TRAINING_DATA_END_TIME = datetime(2026, 5, 1, tzinfo=TIMEZONE)

RSWT_TRAINING_DATA_START_TIME = datetime(2025, 10, 17, tzinfo=TIMEZONE)
RSWT_TRAINING_DATA_END_TIME = datetime(2025, 12, 1, tzinfo=TIMEZONE)

TEST_DATA_START_TIME = datetime(2026, 10, 25, 14, tzinfo=TIMEZONE)
TEST_DATA_END_TIME = datetime(2026, 10, 28, 5, tzinfo=TIMEZONE)

OUTPUT_CSV: Path | None = None  # default: results/.../test_period_predictions.csv

# Legacy DerivedGenerator house parameters (SCADA calibration)
SCADA_ALPHA = 8.6
SCADA_BETA = -0.14
SCADA_GAMMA = 0.005
SCADA_INTERMEDIATE_POWER = 3.6
SCADA_INTERMEDIATE_RSWT = 105.0
SCADA_DD_POWER = 8.6
SCADA_DD_RSWT = 155.0


def _local_ts(dt: datetime, tz: ZoneInfo) -> pd.Timestamp:
    return pd.Timestamp(dt.astimezone(tz).replace(tzinfo=None))


def _validate_window(name: str, start: datetime, end: datetime) -> None:
    if end <= start:
        raise ValueError(f"{name}: end time must be after start time")


def _legacy_quadratic_abc() -> tuple[float, float, float]:
    """Coefficients a, b, c for P(RSWT) = a·RSWT² + b·RSWT + c."""
    x0 = -SCADA_ALPHA / SCADA_BETA
    xi = SCADA_INTERMEDIATE_RSWT
    xd = SCADA_DD_RSWT
    yi = SCADA_INTERMEDIATE_POWER
    yd = SCADA_DD_POWER
    c = (xi * xd) / (xi - xd) * (
        (yd * x0) / (xd * (x0 - xd)) - (yi * x0) / (xi * (x0 - xi))
    )
    b = (yi * x0) / (xi * (x0 - xi)) - (x0 + xi) / (x0 * xi) * c
    a = -b / x0 - c / (x0 * x0)
    return a, b, c


def old_rswt_from_power_kw(power_kw: np.ndarray) -> np.ndarray:
    """Legacy quadratic RSWT (°F) for thermal power P (kW)."""
    a, b, c = _legacy_quadratic_abc()
    disc = b * b - 4.0 * a * (c - power_kw)
    rswt = np.full(power_kw.shape, np.nan, dtype=float)
    ok = disc >= 0
    rswt[ok] = (-b + np.sqrt(disc[ok])) / (2.0 * a)
    return rswt


def old_energy_and_rswt(
    oat_f: np.ndarray,
    ws_mph: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy DerivedGenerator-style P (kW) and RSWT (°F) from OAT and wind."""
    power = (
        SCADA_ALPHA
        + SCADA_BETA * oat_f
        + SCADA_GAMMA * ws_mph * (65.0 - oat_f)
    )
    power = np.where(power > 0, power, 0.0)
    return power, old_rswt_from_power_kw(power)


def feature_frame(params: HouseEnergyParams, df: pd.DataFrame) -> pd.DataFrame:
    """Feature values as used in the regression design matrix (excluding intercept)."""
    energy_ratio = params.energy_ratio
    out = pd.DataFrame(index=df.index)
    for name in params.feature_names:
        if name == "previous_dist_kwh_scaled":
            out[name] = df["previous_dist_kwh"].to_numpy(dtype=float) * energy_ratio
        elif name == "dist_kwh_scaled_avg_4h":
            out[name] = df["dist_kwh_avg_4h"].to_numpy(dtype=float) * energy_ratio
        else:
            out[name] = df[name].to_numpy(dtype=float)
    return out


def main() -> None:
    _validate_window(
        "ENERGY_TRAINING_DATA",
        ENERGY_TRAINING_DATA_START_TIME,
        ENERGY_TRAINING_DATA_END_TIME,
    )
    _validate_window(
        "RSWT_TRAINING_DATA",
        RSWT_TRAINING_DATA_START_TIME,
        RSWT_TRAINING_DATA_END_TIME,
    )
    _validate_window("TEST_DATA", TEST_DATA_START_TIME, TEST_DATA_END_TIME)

    run_ms = int(time.time() * 1000)

    energy_computer = HouseEnergyParamsComputer(
        HOUSE_ALIAS,
        ENERGY_TRAINING_DATA_START_TIME,
        ENERGY_TRAINING_DATA_END_TIME,
        timezone=TIMEZONE,
        run_ms=run_ms,
    )
    if energy_computer.df.empty:
        raise ValueError(f"No energy training rows for {HOUSE_ALIAS} after data cleaning")
    energy_params = energy_computer.get_energy_params()
    energy_training_df = energy_computer.df
    baseline_params = energy_computer.fit(energy_training_df, baseline=True)
    energy_training_ratio = float(
        energy_training_df["hp_kwh_th"].sum() / energy_training_df["dist_kwh"].sum()
    )
    training_setpoint_cols = [
        c
        for c in energy_training_df.columns
        if str(c).startswith("zone") and str(c).endswith("_avg_set")
    ]
    energy_training_hourly_avg_setpoint_f = energy_training_df[
        training_setpoint_cols
    ].mean(axis=1)
    logging.info(
        "Energy parameters (trained on %s hour(s)):\n%s",
        len(energy_training_df),
        energy_params,
    )
    logging.info("Baseline energy parameters:\n%s", baseline_params)
    logging.info(
        "Energy training ratio (sum hp / sum dist on energy training window): %.6f",
        energy_training_ratio,
    )
    house_dir = energy_computer.house_dir
    del energy_computer

    rswt_computer = HouseEnergyParamsComputer(
        HOUSE_ALIAS,
        RSWT_TRAINING_DATA_START_TIME,
        RSWT_TRAINING_DATA_END_TIME,
        timezone=TIMEZONE,
        run_ms=run_ms,
    )
    if rswt_computer.df.empty:
        raise ValueError(f"No RSWT training rows for {HOUSE_ALIAS} after data cleaning")
    rswt_params = rswt_computer.get_rswt_params(energy_params=energy_params)
    logging.info(
        "RSWT parameters (trained on %s cleaned hour(s)):\n%s",
        len(rswt_computer.df),
        rswt_params,
    )
    del rswt_computer

    test_start = _local_ts(TEST_DATA_START_TIME, TIMEZONE)
    test_end = _local_ts(TEST_DATA_END_TIME, TIMEZONE)
    feature_buffer = timedelta(hours=HouseEnergyParamsComputer.ROLLING_AVG_HOURS)
    load_test_from = TEST_DATA_START_TIME - feature_buffer

    test_computer = HouseEnergyParamsComputer(
        HOUSE_ALIAS,
        load_test_from,
        TEST_DATA_END_TIME,
        timezone=TIMEZONE,
        run_ms=run_ms,
    )
    test_df = test_computer.df.loc[
        (test_computer.df["hour_start"] >= test_start)
        & (test_computer.df["hour_start"] < test_end)
    ].copy()
    if test_df.empty:
        raise ValueError(
            f"No test rows for {HOUSE_ALIAS} in "
            f"[{test_start.isoformat()}, {test_end.isoformat()}) after data cleaning"
        )
    logging.info(
        "Test period: %s cleaned hour(s) in [%s, %s)",
        len(test_df),
        test_start.isoformat(),
        test_end.isoformat(),
    )
    del test_computer

    estimated = HouseEnergyParamsComputer.predict_with_params(energy_params, test_df)
    estimated = np.where(np.isfinite(estimated), estimated, np.nan)
    baseline_estimated = HouseEnergyParamsComputer.predict_with_params(
        baseline_params, test_df
    )
    baseline_estimated = np.where(np.isfinite(baseline_estimated), baseline_estimated, np.nan)
    features = feature_frame(energy_params, test_df)

    estimated_rswt = HouseEnergyParamsComputer.predict_rswt_from_pred_load(
        estimated, rswt_params
    )

    oat = test_df["oat_f"].to_numpy(dtype=float)
    ws = test_df["ws_mph"].to_numpy(dtype=float)
    old_energy_pred, old_rswt_pred = old_energy_and_rswt(oat, ws)

    export = pd.DataFrame({"timestamp": test_df["hour_start"].values})
    export["actual_energy_use"] = (
        test_df["dist_kwh"].to_numpy(dtype=float) * energy_training_ratio
    )
    export["estimated_energy_use"] = estimated
    export["baseline_energy_use"] = baseline_estimated
    export["estimated_rswt_f"] = estimated_rswt
    export["new_energy_old_rswt_pred"] = old_rswt_from_power_kw(estimated)
    export["old_energy_pred"] = old_energy_pred
    export["old_rswt_pred"] = old_rswt_pred
    for col in features.columns:
        export[col] = features[col].values
    setpoint_cols = [
        c for c in test_df.columns if str(c).startswith("zone") and str(c).endswith("_avg_set")
    ]
    export["avg_setpoint_f"] = test_df[setpoint_cols].mean(axis=1).values
    export["oat_f"] = test_df["oat_f"].values
    export["ws_mph"] = test_df["ws_mph"].values

    numeric_cols = export.select_dtypes(include="number").columns
    export[numeric_cols] = export[numeric_cols].round(2)

    if OUTPUT_CSV is None:
        out_path = house_dir / f"test_predictions_{test_start.date()}_{test_end.date()}.csv"
    else:
        out_path = OUTPUT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path, index=False)
    logging.info("Wrote %s row(s) to %s", len(export), out_path)

    plot_test_period_method_comparison(
        export["timestamp"],
        export["estimated_energy_use"].to_numpy(),
        export["old_energy_pred"].to_numpy(),
        actual=export["actual_energy_use"].to_numpy(),
        # baseline_method=export["baseline_energy_use"].to_numpy(),
        ylabel="Energy use (kWh)",
        title=HOUSE_ALIAS.capitalize(),
    )
    plot_training_hourly_avg_setpoint(
        energy_training_df["hour_start"],
        energy_training_hourly_avg_setpoint_f,
        title=HOUSE_ALIAS.capitalize(),
    )
    plot_test_period_method_comparison(
        export["timestamp"],
        export["estimated_rswt_f"].to_numpy(),
        export["old_rswt_pred"].to_numpy(),
        ylabel="RSWT (°F)",
        title=HOUSE_ALIAS.capitalize(),
    )


if __name__ == "__main__":
    main()
