"""Supply water temperature vs thermal load: K_rad radiator model and nonlinear fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# 4.184 kJ/(kg·K) → kWh/(kg·K) for hourly kWh load with kg/h mass flow
WATER_CP_KWH_PER_KG_K = 4.184 / 3600.0


def hydraulic_rise_c(
    q_kwh: np.ndarray,
    *,
    mdot_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
) -> np.ndarray:
    """Q / (2 · ṁ · cp) in °C."""
    q = np.asarray(q_kwh, dtype=float)
    denom = 2.0 * mdot_kg_hour * cp_kwh_per_kg_k
    if denom <= 0:
        return np.full(q.shape, np.nan, dtype=float)
    return q / denom


def supply_temp_c(
    q_kwh: np.ndarray | float,
    *,
    t_room_c: float,
    k_rad: float,
    n: float,
    mdot_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
) -> np.ndarray:
    """
    T_supply(Q) = T_room + K_rad · Q^(1/n) + Q / (2 · ṁ · cp).

    ``Q`` is hourly thermal energy (kWh/h); ``mdot`` is kg/h.
    """
    scalar_in = np.ndim(q_kwh) == 0
    q = np.atleast_1d(np.maximum(np.asarray(q_kwh, dtype=float), 0.0))
    out = np.full(q.shape, np.nan, dtype=float)
    if mdot_kg_hour <= 0 or n <= 0 or k_rad < 0:
        return float(out[0]) if scalar_in else out
    hyd = hydraulic_rise_c(q, mdot_kg_hour=mdot_kg_hour, cp_kwh_per_kg_k=cp_kwh_per_kg_k)
    rad = k_rad * np.power(q, 1.0 / n)
    ok = np.isfinite(q)
    out[ok] = t_room_c + rad[ok] + hyd[ok]
    if scalar_in:
        return float(out[0])
    return out


def supply_temp_f_from_energy_kwh(
    q_kwh: np.ndarray,
    *,
    t_room_c: float,
    k_rad: float,
    n: float,
    water_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
) -> np.ndarray:
    """Supply temperature (°F) from hourly load (kWh/h)."""
    t_c = supply_temp_c(
        q_kwh,
        t_room_c=t_room_c,
        k_rad=k_rad,
        n=n,
        mdot_kg_hour=water_kg_hour,
        cp_kwh_per_kg_k=cp_kwh_per_kg_k,
    )
    return t_c * (9.0 / 5.0) + 32.0


def supply_temp_dd_anchor_c(
    q_kwh: np.ndarray | float,
    *,
    t_room_c: float,
    q_dd_kwh: float,
    t_supply_dd_c: float,
    n: float,
    mdot_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
) -> np.ndarray:
    """
    Design-day anchored supply temperature (°C).

    T_supply(Q) = T_room + (T_supply,DD - T_room - Q_DD/(2·ṁ·cp))·(Q/Q_DD)^(1/n)
                  + Q/(2·ṁ·cp)
    """
    scalar_in = np.ndim(q_kwh) == 0
    q = np.atleast_1d(np.maximum(np.asarray(q_kwh, dtype=float), 0.0))
    out = np.full(q.shape, np.nan, dtype=float)
    if mdot_kg_hour <= 0 or n <= 0 or q_dd_kwh <= 0:
        return float(out[0]) if scalar_in else out
    hyd = hydraulic_rise_c(q, mdot_kg_hour=mdot_kg_hour, cp_kwh_per_kg_k=cp_kwh_per_kg_k)
    hyd_dd = float(
        hydraulic_rise_c(
            q_dd_kwh,
            mdot_kg_hour=mdot_kg_hour,
            cp_kwh_per_kg_k=cp_kwh_per_kg_k,
        )
    )
    rad_scale = t_supply_dd_c - t_room_c - hyd_dd
    ratio = np.power(q / q_dd_kwh, 1.0 / n, where=q > 0, out=np.zeros_like(q))
    ok = np.isfinite(q)
    out[ok] = t_room_c + rad_scale * ratio[ok] + hyd[ok]
    if scalar_in:
        return float(out[0])
    return out


def supply_temp_f_from_energy_kwh_dd_anchor(
    q_kwh: np.ndarray,
    *,
    t_room_c: float,
    q_dd_kwh: float,
    t_supply_dd_f: float,
    n: float,
    water_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
) -> np.ndarray:
    """Supply temperature (°F) from hourly load using design-day anchors."""
    t_supply_dd_c = (t_supply_dd_f - 32.0) * (5.0 / 9.0)
    t_c = supply_temp_dd_anchor_c(
        q_kwh,
        t_room_c=t_room_c,
        q_dd_kwh=q_dd_kwh,
        t_supply_dd_c=t_supply_dd_c,
        n=n,
        mdot_kg_hour=water_kg_hour,
        cp_kwh_per_kg_k=cp_kwh_per_kg_k,
    )
    return t_c * (9.0 / 5.0) + 32.0


def design_supply_temp_c(
    q_dd_kwh: float,
    *,
    t_room_c: float,
    k_rad: float,
    n: float,
    mdot_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
) -> float:
    """T_supply at fixed design load anchor Q_DD."""
    return float(
        supply_temp_c(
            q_dd_kwh,
            t_room_c=t_room_c,
            k_rad=k_rad,
            n=n,
            mdot_kg_hour=mdot_kg_hour,
            cp_kwh_per_kg_k=cp_kwh_per_kg_k,
        )
    )


@dataclass(frozen=True)
class SupplyPhysicsFitResult:
    k_rad: float
    n: float
    k_rad_std: float | None
    n_std: float | None
    rmse_c: float
    n_rows: int


def _initial_k_rad_guess(
    q_kwh: np.ndarray,
    t_supply_c: np.ndarray,
    *,
    t_room_c: float,
    mdot_kg_hour: float,
    cp_kwh_per_kg_k: float,
    n_initial: float,
) -> float:
    hyd = hydraulic_rise_c(q_kwh, mdot_kg_hour=mdot_kg_hour, cp_kwh_per_kg_k=cp_kwh_per_kg_k)
    residual = t_supply_c - t_room_c - hyd
    pos = q_kwh > 0
    if not np.any(pos):
        return 1.0
    rad_target = residual[pos] / np.power(q_kwh[pos], 1.0 / n_initial)
    rad_target = rad_target[np.isfinite(rad_target) & (rad_target > 0)]
    if rad_target.size == 0:
        return 1.0
    return float(np.median(rad_target))


def fit_k_rad_and_n(
    q_kwh: np.ndarray,
    t_supply_c: np.ndarray,
    *,
    t_room_c: float,
    mdot_kg_hour: float,
    cp_kwh_per_kg_k: float = WATER_CP_KWH_PER_KG_K,
    n_initial: float = 1.3,
    n_bounds: tuple[float, float] = (1.0, 1.6),
    min_rows: int = 20,
) -> SupplyPhysicsFitResult:
    """Nonlinear least squares for K_rad and n with fixed T_room, ṁ, and cp."""
    q = np.asarray(q_kwh, dtype=float)
    t = np.asarray(t_supply_c, dtype=float)
    mask = np.isfinite(q) & np.isfinite(t) & (q >= 0)
    q = q[mask]
    t = t[mask]
    if len(q) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} finite (Q, T_supply) rows for supply physics fit "
            f"(got {len(q)})"
        )
    if mdot_kg_hour <= 0:
        raise ValueError("mdot_kg_hour must be positive for supply physics fit")

    k_rad0 = _initial_k_rad_guess(
        q,
        t,
        t_room_c=t_room_c,
        mdot_kg_hour=mdot_kg_hour,
        cp_kwh_per_kg_k=cp_kwh_per_kg_k,
        n_initial=n_initial,
    )

    def model(q_fit: np.ndarray, k_rad: float, n: float) -> np.ndarray:
        return supply_temp_c(
            q_fit,
            t_room_c=t_room_c,
            k_rad=k_rad,
            n=n,
            mdot_kg_hour=mdot_kg_hour,
            cp_kwh_per_kg_k=cp_kwh_per_kg_k,
        )

    popt, pcov = curve_fit(
        model,
        q,
        t,
        p0=[k_rad0, n_initial],
        bounds=([0.0, n_bounds[0]], [np.inf, n_bounds[1]]),
        maxfev=20_000,
    )
    k_rad, n = float(popt[0]), float(popt[1])
    pred = model(q, k_rad, n)
    rmse = float(np.sqrt(np.mean((pred - t) ** 2)))
    k_std: float | None = None
    n_std: float | None = None
    if pcov.shape == (2, 2) and np.all(np.isfinite(pcov)):
        k_std = float(np.sqrt(max(pcov[0, 0], 0.0)))
        n_std = float(np.sqrt(max(pcov[1, 1], 0.0)))
    return SupplyPhysicsFitResult(
        k_rad=k_rad,
        n=n,
        k_rad_std=k_std,
        n_std=n_std,
        rmse_c=rmse,
        n_rows=len(q),
    )
