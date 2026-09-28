from __future__ import annotations

import re
import csv
import os
import dotenv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------
# Configuration
# -----------------

HOUSE_ALIAS = "beech"
LOCAL_TZ = ZoneInfo("America/New_York")
T_START = datetime(2025, 11, 1, tzinfo=LOCAL_TZ)
T_END = datetime(2025, 11, 3, tzinfo=LOCAL_TZ)
OUTPUT_CSV = (
    Path(__file__).resolve().parent / "results"
    / f"{HOUSE_ALIAS}_house_params_data.csv"
)

# -----------------
# Channel names
# -----------------

ZONE_CHANNEL = re.compile(r"^zone(\d+)-.+-(temp|set|heat-call)$")
OIL_BOILER_CHANNEL = "oil-boiler-pwr"
ENERGY_CHANNELS = [
    "hp-ewt",
    "hp-lwt",
    "primary-flow",
    "dist-swt",
    "dist-rwt",
    "dist-flow",
    "charge-discharge-relay3",
    "store-hot-pipe",
    "store-cold-pipe",
    "store-flow",
]

# -----------------
# Querries
# -----------------

# Hourly energy (distribution and heat pump)
ENERGY_SQL = text("""
SELECT
    time_bucket(INTERVAL '1 hour', time_bucket_1s, :local_tz) AS time_bucket,
    avg(hp_kw) AS hp_kwh_th,
    avg(dist_kw) AS dist_kwh
FROM (
    SELECT
        time_bucket_1s,
        500 * primary_flow_gpm * (hp_lwt_c - hp_ewt_c) * 9 / 5 / 3410 AS hp_kw,
        500 * dist_flow_gpm * (dist_swt_c - dist_rwt_c) * 9 / 5 / 3410 AS dist_kw
    FROM (
        SELECT
            time_bucket AS time_bucket_1s,
            (AVG(value) FILTER (WHERE channel_name = 'primary-flow' AND unit = 'GpmTimes100') / 100)
                AS primary_flow_gpm,
            (AVG(value) FILTER (WHERE channel_name = 'hp-lwt' AND unit = 'WaterTempCTimes1000') / 1000)
                AS hp_lwt_c,
            (AVG(value) FILTER (WHERE channel_name = 'hp-ewt' AND unit = 'WaterTempCTimes1000') / 1000)
                AS hp_ewt_c,
            (AVG(value) FILTER (WHERE channel_name = 'dist-flow' AND unit = 'GpmTimes100') / 100)
                AS dist_flow_gpm,
            (AVG(value) FILTER (WHERE channel_name = 'dist-swt' AND unit = 'WaterTempCTimes1000') / 1000)
                AS dist_swt_c,
            (AVG(value) FILTER (WHERE channel_name = 'dist-rwt' AND unit = 'WaterTempCTimes1000') / 1000)
                AS dist_rwt_c
        FROM gridworks.retrieve_readings_1s(
            t_start => :t_start,
            t_end => :t_end_inclusive,
            channels => :energy_channels
        )
        WHERE terminal_asset_alias ILIKE :house_alias
        GROUP BY time_bucket
    ) AS per_second
) AS with_kw
GROUP BY time_bucket
ORDER BY time_bucket
""")

# Average oil boiler power
OIL_PWR_SQL = text("""
SELECT
    time_bucket(INTERVAL '1 hour', r.timestamp, :local_tz) AS time_bucket,
    average(time_weight('LOCF', r.timestamp, r.value)) AS avg_value
FROM gridworks.readings r
INNER JOIN gridworks.reading_channels rc ON rc.id = r.channel_id
WHERE rc.terminal_asset_alias ILIKE :house_alias
  AND rc.name = :oil_boiler_channel
  AND rc.unit = 'PowerW'
  AND r.timestamp >= :t_start
  AND r.timestamp < :t_end
GROUP BY time_bucket
ORDER BY time_bucket
""")

# Heat call fraction, average setpoint and temperature for each zone
ZONE_AND_HEATCALL_SQL = text("""
SELECT
    rc.name AS channel_name,
    rc.unit,
    time_bucket(INTERVAL '1 hour', r.timestamp, :local_tz) AS time_bucket,
    average(time_weight('LOCF', r.timestamp, r.value)) AS avg_value
FROM gridworks.readings r
INNER JOIN gridworks.reading_channels rc ON rc.id = r.channel_id
WHERE rc.terminal_asset_alias ILIKE :house_alias
  AND (
    rc.name ~ '^zone[0-9]+-.+-(temp|set)$'
    OR rc.name ~ '^zone[0-9]+-.+-heat-call$'
  )
  AND rc.name NOT ILIKE '%gw-temp%'
  AND r.timestamp >= :t_start
  AND r.timestamp < :t_end
GROUP BY rc.name, rc.unit, time_bucket
ORDER BY time_bucket, rc.name
""")


def hour_start_key(raw_timestamp: datetime) -> str:
    tb = raw_timestamp
    if tb.tzinfo is None:
        tb = tb.replace(tzinfo=ZoneInfo("UTC"))
    local = tb.astimezone(LOCAL_TZ)
    return local.replace(tzinfo=None).isoformat(timespec="seconds")


def main() -> int:
    dotenv.load_dotenv()
    db_url = os.environ.get("GW_DATA_DB_URL")
    if not db_url:
        raise SystemExit("GW_DATA_DB_URL is not set (e.g. in .env)")
    db_echo = os.environ.get("GW_DATA_DB_ECHO", "").lower() in ("1", "true", "yes")
    engine = create_engine(db_url, echo=db_echo)

    logger.info(f"{HOUSE_ALIAS} - Exporting from {T_START.isoformat()} to {T_END.isoformat()} ({LOCAL_TZ})"),

    # Prepare query parameters
    params = {
        "house_alias": f"%{HOUSE_ALIAS}%",
        "t_start": T_START,
        "t_end": T_END,
        "t_end_inclusive": T_END - timedelta(microseconds=1),
        "local_tz": str(LOCAL_TZ),
        "oil_boiler_channel": OIL_BOILER_CHANNEL,
        "energy_channels": ENERGY_CHANNELS,
    }

    # Execute queries
    with engine.connect() as conn:
        energy_rows = conn.execute(ENERGY_SQL, params).mappings().all()
        oil_rows = conn.execute(OIL_PWR_SQL, params).mappings().all()
        zone_rows = conn.execute(ZONE_AND_HEATCALL_SQL, params).mappings().all()

    # Initialize by-hour dictionary
    by_hour: dict[str, dict[str, float | None]] = {}
    cursor = T_START
    while cursor < T_END:
        hour = cursor.replace(tzinfo=None).isoformat(timespec="seconds")
        by_hour[hour] = {}
        cursor += timedelta(hours=1)

    # Populate by-hour dictionary with energy data
    for row in energy_rows:
        key = hour_start_key(row["time_bucket"])
        bucket = by_hour.setdefault(key, {})
        bucket["dist_kwh"] = 0.0 if row["dist_kwh"] is None else round(float(row["dist_kwh"]), 2)
        bucket["hp_kwh_th"] = 0.0 if row["hp_kwh_th"] is None else round(float(row["hp_kwh_th"]), 2)

    # Populate by-hour dictionary with oil boiler power data
    for row in oil_rows:
        key = hour_start_key(row["time_bucket"])
        bucket = by_hour.setdefault(key, {})
        avg = row["avg_value"]
        bucket["oil_boiler_pwr"] = round(float(avg), 1) if avg is not None else 0.0

    # Populate by-hour dictionary with zone data
    zone_ids: set[int] = set()
    zone_accum: dict[str, list[float]] = {}

    for row in zone_rows:
        name = row["channel_name"]
        parsed = ZONE_CHANNEL.match(name)
        if not parsed:
            continue
        zone = int(parsed.group(1))
        kind = parsed.group(2)
        zone_ids.add(zone)

        unit = row["unit"]
        raw = float(row["avg_value"]) if row["avg_value"] is not None else None
        if raw is None:
            continue
        if kind in ("temp", "set"):
            value = round(raw / 1000.0 if unit == "AirTempFTimes1000" else raw, 1)
        else:
            value = round(raw, 2)

        key = hour_start_key(row["time_bucket"])
        if kind == "heat-call":
            field = f"zone{zone}_heatcall_fraction"
        elif kind == "temp":
            field = f"zone{zone}_avg_temp"
        else:
            field = f"zone{zone}_avg_set"
        zone_accum.setdefault(f"{key}|{field}", []).append(value)

    for compound, values in zone_accum.items():
        hour_key, field = compound.split("|", 1)
        bucket = by_hour.setdefault(hour_key, {})
        bucket[field] = round(sum(values) / len(values), 1 if "avg_" in field else 2)

    zone_id_list = sorted(zone_ids)

    # Build all column_names
    column_names = [
        "hour_start",
        "oat_f",
        "ws_mph",
        "solar_w_m2",
        "dist_kwh",
        "hp_kwh_th",
        "oil_boiler_pwr",
    ]
    for zone in zone_id_list:
        column_names.append(f"zone{zone}_heatcall_fraction")
    for zone in zone_id_list:
        column_names.extend([f"zone{zone}_avg_set", f"zone{zone}_avg_temp"])

    # Write to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=column_names, extrasaction="ignore")
        writer.writeheader()
        for hour in sorted(by_hour.keys()):
            data = by_hour[hour]
            row = {
                "hour_start": hour,
                "oat_f": "",
                "ws_mph": "",
                "solar_w_m2": "",
                "dist_kwh": data.get("dist_kwh", 0.0),
                "hp_kwh_th": data.get("hp_kwh_th", 0.0),
                "oil_boiler_pwr": data.get("oil_boiler_pwr", 0.0),
            }
            for zone in zone_id_list:
                row[f"zone{zone}_heatcall_fraction"] = data.get(f"zone{zone}_heatcall_fraction", 0.0)
            for zone in zone_id_list:
                row[f"zone{zone}_avg_set"] = data.get(f"zone{zone}_avg_set", "")
                row[f"zone{zone}_avg_temp"] = data.get(f"zone{zone}_avg_temp", "")
            writer.writerow(row)
    logger.info(f"Wrote {len(by_hour)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
