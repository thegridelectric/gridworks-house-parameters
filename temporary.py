#!/usr/bin/env python3
"""
House load P (kW thermal) from OAT + wind, then RSWT (°F) from P — same models as DerivedGenerator.
"""

from __future__ import annotations

import math
import sys

# --- House parameters (mirror .env SCADA_*) ---

SCADA_ALPHA = 8.6
SCADA_BETA = -0.14
SCADA_GAMMA = 0.005

SCADA_INTERMEDIATE_POWER = 3.6
SCADA_INTERMEDIATE_RSWT = 105

SCADA_DD_POWER = 8.6
SCADA_DD_RSWT = 155
# SCADA_DD_RSWT = 140  # alternate calibration point

# --- Weather inputs ---

OAT_F = 20.0
WIND_MPH = 0.0


def required_heating_power_kw(oat_f: float, wind_speed_mph: float) -> float:
    """Same as DerivedGenerator.required_heating_power (without load overestimation %)."""
    r = (
        SCADA_ALPHA
        + SCADA_BETA * oat_f
        + SCADA_GAMMA * wind_speed_mph * (65.0 - oat_f)
    )
    return round(r, 2) if r > 0 else 0.0


def quadratic_coefficients(
    *,
    no_power_rswt: float,
    intermediate_rswt: float,
    intermediate_power: float,
    dd_rswt: float,
    dd_power: float,
) -> tuple[float, float, float]:
    """Closed-form a, b, c for P(RSWT) = a·RSWT² + b·RSWT + c."""
    x0 = no_power_rswt
    xi = intermediate_rswt
    xd = dd_rswt
    yi = intermediate_power
    yd = dd_power

    c = (xi * xd) / (xi - xd) * (
        (yd * x0) / (xd * (x0 - xd)) - (yi * x0) / (xi * (x0 - xi))
    )
    b = (yi * x0) / (xi * (x0 - xi)) - (x0 + xi) / (x0 * xi) * c
    a = -b / x0 - c / (x0 * x0)
    return a, b, c


def rswt_for_power(p_kw: float, a: float, b: float, c: float) -> float:
    """Positive root of a·RSWT² + b·RSWT + c = p_kw (same as DerivedGenerator.required_swt)."""
    disc = b * b - 4 * a * (c - p_kw)
    if disc < 0:
        raise ValueError(f"No real RSWT for P={p_kw} kW (discriminant {disc:.6f})")
    return round((-b + math.sqrt(disc)) / (2 * a), 2)


def parse_weather() -> tuple[float, float]:
    if len(sys.argv) >= 3:
        return float(sys.argv[1]), float(sys.argv[2])
    if len(sys.argv) == 2:
        raise SystemExit("Usage: rswt_from_power.py [OAT_F WIND_MPH]")
    return OAT_F, WIND_MPH


def main() -> None:
    oat_f, wind_mph = parse_weather()
    p_kw = required_heating_power_kw(oat_f, wind_mph)

    no_power_rswt = -SCADA_ALPHA / SCADA_BETA
    a, b, c = quadratic_coefficients(
        no_power_rswt=no_power_rswt,
        intermediate_rswt=SCADA_INTERMEDIATE_RSWT,
        intermediate_power=SCADA_INTERMEDIATE_POWER,
        dd_rswt=SCADA_DD_RSWT,
        dd_power=SCADA_DD_POWER,
    )
    rswt_f = rswt_for_power(p_kw, a, b, c)

    print(f"OAT = {oat_f} °F, wind = {wind_mph} mph")
    print(f"P = {p_kw} kW")
    print(f"RSWT = {rswt_f} °F")


if __name__ == "__main__":
    main()
