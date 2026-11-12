"""
Train house energy parameters on a cleaned training window, load test hours with
erroneous-value cleaning only, apply the energy model and design-day-anchored
supply physics (Q_DD, T_supply,DD, n), and export CSV.
"""

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    force=True,
)

import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from house_parameters import (
    HouseEnergyParams,
    HouseEnergyParamsComputer,
)
from plots import plot_test_period_combined_interactive

HOUSE_ALIAS = "oak"
TIMEZONE = ZoneInfo("America/New_York")
USE_SOLAR = False

ENERGY_TRAINING_DATA_START_TIME = datetime(2025, 10, 17, tzinfo=TIMEZONE)
ENERGY_TRAINING_DATA_END_TIME = datetime(2026, 5, 1, tzinfo=TIMEZONE)

RSWT_TRAINING_DATA_START_TIME = datetime(2025, 10, 17, tzinfo=TIMEZONE)
RSWT_TRAINING_DATA_END_TIME = datetime(2026, 5, 1, tzinfo=TIMEZONE)

OUTPUT_CSV: Path | None = None  # default: results/.../test_period_predictions.csv

if HOUSE_ALIAS=='beech':
    TEST_DATA_START_TIME = datetime(2026, 10, 25, 14, tzinfo=TIMEZONE)
    TEST_DATA_END_TIME = datetime(2026, 10, 28, 5, tzinfo=TIMEZONE)
    SCADA_ALPHA = 8.6
    SCADA_BETA = -0.14
    SCADA_GAMMA = 0.005
    SCADA_INTERMEDIATE_POWER = 3.6
    SCADA_INTERMEDIATE_RSWT = 105.0
    SCADA_DD_POWER = 8.6
    SCADA_DD_RSWT = 155.0

else:
    TEST_DATA_START_TIME = datetime(2026, 10, 19, 0, tzinfo=TIMEZONE)
    TEST_DATA_END_TIME = datetime(2026, 10, 27, 18, tzinfo=TIMEZONE)
    SCADA_ALPHA=7.6
    SCADA_BETA=-0.13
    SCADA_GAMMA=0.0024
    SCADA_INTERMEDIATE_POWER = 1.5
    SCADA_INTERMEDIATE_RSWT = 100
    SCADA_DD_POWER = 7.6
    SCADA_DD_RSWT=140


# Q_DD anchor: mean of highest estimated-energy hours on energy training window
PHYSICS_Q_DD_TOP_FRACTION = 0.02

# Design-day supply physics (see docs/rswt_estimation.md). Set PHYSICS_Q_DD_KWH to
# override the data-derived Q_DD anchor.
PHYSICS_Q_DD_KWH: float | None = None
PHYSICS_T_SUPPLY_DD_F = SCADA_DD_RSWT
PHYSICS_N = 1.3

# Measured SWT scatter on RSWT plot: max zone heat-call fraction must exceed this.
CHC_SWT_PLOT_HEATCALL_THRESHOLD = 0.9


def _q_dd_anchor_kwh_from_top_estimated_energy(
    est_kwh: np.ndarray,
    top_fraction: float = PHYSICS_Q_DD_TOP_FRACTION,
) -> float:
    """Mean load (kWh/h) over the top ``top_fraction`` of estimated hourly energy."""
    q = np.asarray(est_kwh, dtype=float)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size == 0:
        raise ValueError("No finite positive estimated energy for Q_DD anchor")
    n_top = max(1, int(np.ceil(q.size * top_fraction)))
    top = np.partition(q, -n_top)[-n_top:]
    return float(np.mean(top))


def _fahrenheit_to_celsius(temp_f: float) -> float:
    return (temp_f - 32.0) * (5.0 / 9.0)


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


def _rswt_supply_physics_scada_copy_paste_lines(
    *,
    t_supply_dd_f: float,
    dd_room_temp_f: float,
    q_dd_kwh: float,
    n: float,
    mass_flow_kg_hour: float,
) -> list[str]:
    """SCADA-style lines for design-day supply physics (cp is a fixed constant, omitted)."""
    return [
        f"dd_rswt_f={int(round(t_supply_dd_f))},",
        f"dd_room_temp_f={int(round(dd_room_temp_f))},",
        f"dd_power_kw={round(float(q_dd_kwh), 1)},",
        f"mass_flow_kg_h={int(round(mass_flow_kg_hour))},",
        f"radiator_exponent={round(float(n), 1)},",
    ]


_SCADA_PRINT_SEPARATOR = "-" * 72


def _print_scada_copy_paste(title: str, lines: list[str]) -> None:
    print(_SCADA_PRINT_SEPARATOR)
    print(title)
    print(_SCADA_PRINT_SEPARATOR)
    print("\n".join(lines))
    print('')


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
        use_solar=USE_SOLAR,
    )
    if energy_computer.df.empty:
        raise ValueError(f"No energy training rows for {HOUSE_ALIAS} after data cleaning")
    if not USE_SOLAR:
        logging.info("Energy model training: solar_w_m2 excluded from regression features")
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
    training_temp_cols = [
        c
        for c in energy_training_df.columns
        if str(c).startswith("zone") and str(c).endswith("_avg_temp")
    ]
    if training_temp_cols:
        physics_t_room_c = float(
            _fahrenheit_to_celsius(
                energy_training_df[training_temp_cols].mean(axis=1).mean()
            )
        )
    elif training_setpoint_cols:
        physics_t_room_c = float(
            _fahrenheit_to_celsius(
                energy_training_df[training_setpoint_cols].mean(axis=1).mean()
            )
        )
    else:
        raise ValueError(
            f"No zone temperature or setpoint columns in energy training data for {HOUSE_ALIAS}"
        )
    _print_scada_copy_paste(
        f"Energy model ({HOUSE_ALIAS}):",
        energy_params.greek_scada_copy_paste_lines()
        + baseline_params.baseline_scada_copy_paste_lines(),
    )
    logging.info(
        "Energy training ratio (sum hp / sum dist on energy training window): %.6f",
        energy_training_ratio,
    )
    logging.info(
        "Physics RSWT T_ROOM: %.2f °C (%.2f °F) from energy training",
        physics_t_room_c,
        physics_t_room_c * (9.0 / 5.0) + 32.0,
    )
    energy_training_estimated = HouseEnergyParamsComputer.predict_with_params(
        energy_params, energy_training_df
    )
    physics_q_dd_kwh = _q_dd_anchor_kwh_from_top_estimated_energy(
        energy_training_estimated
    )
    n_finite_est = int(
        np.sum(np.isfinite(energy_training_estimated) & (energy_training_estimated > 0))
    )
    n_top_q_dd = max(1, int(np.ceil(n_finite_est * PHYSICS_Q_DD_TOP_FRACTION)))
    logging.info(
        "Physics Q_DD anchor: %.3f kWh/h (mean of top %s hour(s), %.0f%% of %s "
        "finite energy-training estimate(s))",
        physics_q_dd_kwh,
        n_top_q_dd,
        PHYSICS_Q_DD_TOP_FRACTION * 100,
        n_finite_est,
    )
    physics_q_dd_kwh = (
        physics_q_dd_kwh
        if PHYSICS_Q_DD_KWH is None
        else float(PHYSICS_Q_DD_KWH)
    )
    cp = HouseEnergyParamsComputer.WATER_CP_KWH_PER_KG_K
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
    water_kg_hour_chc = rswt_params.water_kg_hour_chc
    _print_scada_copy_paste(
        f"RSWT model ({HOUSE_ALIAS}):",
        _rswt_supply_physics_scada_copy_paste_lines(
            t_supply_dd_f=PHYSICS_T_SUPPLY_DD_F,
            dd_room_temp_f=physics_t_room_c * (9.0 / 5.0) + 32.0,
            q_dd_kwh=physics_q_dd_kwh,
            n=PHYSICS_N,
            mass_flow_kg_hour=water_kg_hour_chc,
        ),
    )
    logging.info(
        "RSWT load-bucket fits (trained on %s cleaned hour(s)):\n%s",
        len(rswt_computer.df),
        rswt_params,
    )
    del rswt_computer

    test_start = _local_ts(TEST_DATA_START_TIME, TIMEZONE)
    test_end = _local_ts(TEST_DATA_END_TIME, TIMEZONE)
    feature_buffer = timedelta(hours=HouseEnergyParamsComputer.ROLLING_AVG_HOURS)
    load_test_from = TEST_DATA_START_TIME - feature_buffer

    raw_test_df, test_zones = HouseEnergyParamsComputer.load_hourly_erroneous_only(
        HOUSE_ALIAS,
        load_test_from,
        TEST_DATA_END_TIME,
        timezone=TIMEZONE,
    )
    featured_test_df = HouseEnergyParamsComputer.engineer_features_uncleaned(
        raw_test_df, test_zones
    )
    test_df = featured_test_df.loc[
        (featured_test_df["hour_start"] >= test_start)
        & (featured_test_df["hour_start"] < test_end)
    ].copy()
    if test_df.empty:
        raise ValueError(
            f"No test rows for {HOUSE_ALIAS} in "
            f"[{test_start.isoformat()}, {test_end.isoformat()})"
        )
    logging.info(
        "Test period: %s hour(s) in [%s, %s) (erroneous-value clean only)",
        len(test_df),
        test_start.isoformat(),
        test_end.isoformat(),
    )

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

    chc_swt_timestamp: pd.Series | None = None
    chc_swt_f: np.ndarray | None = None
    clean_test_df, _clean_zones = HouseEnergyParamsComputer.load_hourly_fully_cleaned(
        HOUSE_ALIAS,
        load_test_from,
        TEST_DATA_END_TIME,
        timezone=TIMEZONE,
    )
    clean_test_window = clean_test_df.loc[
        (clean_test_df["hour_start"] >= test_start)
        & (clean_test_df["hour_start"] < test_end)
    ]
    heatcall_cols = [
        c
        for c in clean_test_window.columns
        if str(c).startswith("zone") and str(c).endswith("_heatcall_fraction")
    ]
    if heatcall_cols and "swt_average_f" in clean_test_window.columns:
        max_hc = clean_test_window[heatcall_cols].max(axis=1)
        chc_thr = CHC_SWT_PLOT_HEATCALL_THRESHOLD
        swt = clean_test_window["swt_average_f"].to_numpy(dtype=float)
        chc_mask = (max_hc > chc_thr).to_numpy() & np.isfinite(swt)
        if np.any(chc_mask):
            chc_swt_timestamp = clean_test_window.loc[chc_mask, "hour_start"]
            chc_swt_f = swt[chc_mask]
            logging.info(
                "RSWT plot: %s fully cleaned test hour(s) with measured SWT "
                "and max heat call > %.0f%%",
                int(np.sum(chc_mask)),
                chc_thr * 100,
            )

    plot_test_period_combined_interactive(
        export["timestamp"],
        export["estimated_energy_use"].to_numpy(),
        export["old_energy_pred"].to_numpy(),
        export["old_rswt_pred"].to_numpy(),
        actual_energy=export["actual_energy_use"].to_numpy(),
        t_room_c=physics_t_room_c,
        water_kg_hour=water_kg_hour_chc,
        cp_kwh_per_kg_k=cp,
        initial_q_dd_kwh=SCADA_DD_POWER,
        initial_t_supply_dd_f=PHYSICS_T_SUPPLY_DD_F,
        initial_n=PHYSICS_N,
        title=HOUSE_ALIAS.capitalize(),
        chc_swt_timestamp=chc_swt_timestamp,
        chc_swt_f=chc_swt_f,
    )


if __name__ == "__main__":
    main()
