"""
The model ladder: a sequence of state-space thermal models of increasing
complexity, each adding exactly one structural claim over the one below.

    rung 1  R1C1     air only                    does any dynamics help?
    rung 2  R2C2     + envelope mass             does envelope storage help?
    rung 3  R4C3     + emitter node              does the emitter node help?
    rung 4  R4C3x2   + second zone               does zoning help?
    rung 5  R4C3x2s  zones share emitter params  is the emitter split real?

Scored against persistence (rung 0), which needs no model at all. Fitting one
model and eyeballing its time constants cannot distinguish "the structure is
wrong" from "the optimiser is stuck"; a ladder can, because each rung isolates
one claim and the rungs below it are small enough to fit reliably.

Everything is expressed in interpretable coordinates rather than raw R and C.
A resistance and a capacitance trade off against each other almost perfectly
(their product, the time constant, is what the data actually identifies), so
fitting R and C separately puts the bounds on quantities nobody has intuition
for and lets the pair drift together to absurd values while the fit looks
fine. Fitting a conductance and a time constant instead puts every bound on
something with a known physical range, which is what makes "pinned at a bound"
a meaningful diagnostic.

Inputs, common to every rung:

    u = [Q1, Q2, T_o, I, 1]

The trailing 1 is a constant carrier for the internal-gains term, so gains
enter as an ordinary input column rather than as a special case.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MINUTE = 60.0
HOUR = 3600.0
N_INPUTS = 5
Q1, Q2, TO, GHI, ONE = range(N_INPUTS)

F1_SOLAR = 0.5          # zone 1 share of solar aperture; collinear with a_sol


@dataclass
class P:
    """One fitted parameter, with bounds in natural units."""
    name: str
    seed: float
    lo: float
    hi: float
    kind: str = "log"          # "log" positive, "logit" 0..1, "linear" sign-free
    unit: str = ""


@dataclass
class Model:
    name: str
    rung: int
    states: list
    params: list
    n_zones: int
    build: Callable            # (p: dict, v: float) -> (A, B)
    obs: np.ndarray = field(default=None)
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.states)

    @property
    def names(self) -> list:
        return [p.name for p in self.params]


# ---------------------------------------------------------------------------
# Shared parameter definitions
#
# Seeds are chosen so that every rung starts at a total conductance near the
# 201 W/K measured by the daily energy balance, and gains near the 1200 W it
# implies. Bounds are wide enough not to prejudge the answer but tight enough
# that hitting one is genuinely informative.
# ---------------------------------------------------------------------------

def p_gains(seed=1200.0, name="G"):
    return P(name, seed, 10.0, 4000.0, unit="W")


def p_solar():
    return P("a_sol", 5.0, 0.1, 60.0, unit="m2")


def p_wind():
    return P("g1", 5.0, 1e-3, 200.0, unit="W/K/(m/s)")


def p_gain_profile():
    """
    Two Fourier harmonics shaping internal gains over the day.

    The measured gain runs 592 W at 09:00 to 1579 W at 19:00 -- people leaving
    and coming home. A constant term forces the optimiser to average that away,
    and the residual daily swing then has to be absorbed by the thermal
    parameters, which is one reason no single parameter set has served both
    the temperature and the energy job.

    Coefficients are sign-free, so they are fitted in linear space. The shape
    is shared by both zones; only the amplitude G_z differs.
    """
    return [P("gp_a1", 0.0, -0.9, 0.9, kind="linear"),
            P("gp_b1", 0.0, -0.9, 0.9, kind="linear"),
            P("gp_a2", 0.0, -0.5, 0.5, kind="linear"),
            P("gp_b2", 0.0, -0.5, 0.5, kind="linear")]


def gain_shape(p: dict, hour):
    """Multiplier on the gains term; mean 1 over a day by construction."""
    h = np.asarray(hour, float) * (2.0 * np.pi / 24.0)
    return (1.0
            + p.get("gp_a1", 0.0) * np.sin(h) + p.get("gp_b1", 0.0) * np.cos(h)
            + p.get("gp_a2", 0.0) * np.sin(2 * h) + p.get("gp_b2", 0.0) * np.cos(2 * h))


def p_basement():
    """
    A basement node between the loop and the house.

    Part of what the meter counts as delivered heat never reaches an emitter:
    it warms pipework and the plant room on the way. That heat is not lost to
    the house -- the basement is inside the envelope and leaks back up -- but it
    arrives late and partly escapes outdoors, which a model that sends every
    watt straight into the emitters cannot represent.
    """
    return [P("f_loss", 0.08, 0.005, 0.6, unit="-"),
            P("K_b1", 60.0, 1.0, 1500.0, unit="W/K"),
            P("K_bo", 25.0, 1.0, 600.0, unit="W/K"),
            P("tau_b", 6 * HOUR, 0.5 * HOUR, 100 * HOUR, unit="s")]


def p_qscale():
    # Process noise. Left free but bounded well below the measurement noise:
    # if the filter is allowed to call the model as unreliable as the sensors
    # it will re-anchor on measurements every step, the innovations go small,
    # and the physics is never forced to be right.
    return P("q_scale", 1e-11, 1e-14, 1e-9, unit="K2/s")


def p_ci(seed, name="C_i"):
    return P(name, seed, 1e5, 5e7, unit="J/K")


def p_uafab(seed, name="UA_fab"):
    return P(name, seed, 5.0, 400.0, unit="W/K")


def p_taum(seed=30 * HOUR, name="tau_m"):
    # envelope time constant; the target range from the instructions is 10-100 h
    return P(name, seed, 2 * HOUR, 400 * HOUR, unit="s")


def p_fim(name="f_im"):
    # fraction of the fabric resistance sitting in the interior film. The film
    # is a small part of the total for any real wall, hence the ceiling.
    return P(name, 0.08, 0.01, 0.6, kind="logit", unit="-")


def p_ginf(seed, name="g0"):
    return P(name, seed, 0.5, 300.0, unit="W/K")


def p_kei(seed=350.0, name="K_ei"):
    return P(name, seed, 20.0, 8000.0, unit="W/K")


def p_taue(seed=40 * MINUTE, name="tau_e"):
    # emitter time constant; observed zone-2 cycling implies tens of minutes
    return P(name, seed, 2 * MINUTE, 8 * HOUR, unit="s")


# ---------------------------------------------------------------------------
# Rung 1: R1C1, single zone, air node only
# ---------------------------------------------------------------------------

def _build_r1c1(p, v):
    A = np.zeros((1, 1))
    B = np.zeros((1, N_INPUTS))
    ci = p["C_i"]
    ua = p["UA0"] + p["g1"] * v          # everything lumped into one conductance

    A[0, 0] = -ua / ci
    B[0, Q1] = 1.0 / ci
    B[0, TO] = ua / ci
    B[0, GHI] = p["a_sol"] / ci
    B[0, ONE] = p["G"] / ci
    return A, B


R1C1 = Model(
    name="R1C1", rung=1, states=["T_i"], n_zones=1, build=_build_r1c1,
    obs=np.array([[1.0]]),
    note="one capacitance, one conductance",
    params=[
        p_ci(3e6),
        P("UA0", 200.0, 5.0, 600.0, unit="W/K"),
        p_wind(), p_solar(), p_gains(), p_qscale(),
    ],
)


# ---------------------------------------------------------------------------
# Rung 2: R2C2, single zone, air + envelope mass
# ---------------------------------------------------------------------------

def _fabric(ua_fab, f_im, tau_m):
    """Split a fabric conductance into its two series stages plus the mass."""
    k_im = ua_fab / f_im                 # interior film
    k_mo = ua_fab / (1.0 - f_im)         # wall + exterior film
    c_m = tau_m * ua_fab
    return k_im, k_mo, c_m


def _build_r2c2(p, v):
    A = np.zeros((2, 2))
    B = np.zeros((2, N_INPUTS))
    ci = p["C_i"]
    ginf = p["g0"] + p["g1"] * v
    k_im, k_mo, cm = _fabric(p["UA_fab"], p["f_im"], p["tau_m"])

    A[0, 0] = -(k_im + ginf) / ci
    A[0, 1] = k_im / ci
    B[0, Q1] = 1.0 / ci
    B[0, TO] = ginf / ci
    B[0, GHI] = p["a_sol"] / ci
    B[0, ONE] = p["G"] / ci

    A[1, 0] = k_im / cm
    A[1, 1] = -(k_im + k_mo) / cm
    B[1, TO] = k_mo / cm
    return A, B


R2C2 = Model(
    name="R2C2", rung=2, states=["T_i", "T_m"], n_zones=1, build=_build_r2c2,
    obs=np.array([[1.0, 0.0]]),
    note="+ envelope mass",
    params=[
        p_ci(3e6), p_uafab(120.0), p_fim(), p_taum(),
        p_ginf(80.0), p_wind(), p_solar(), p_gains(), p_qscale(),
    ],
)


# ---------------------------------------------------------------------------
# Rung 3: R4C3, single zone, emitter + air + envelope mass
# ---------------------------------------------------------------------------

def _build_r4c3(p, v):
    A = np.zeros((3, 3))
    B = np.zeros((3, N_INPUTS))
    ci = p["C_i"]
    ginf = p["g0"] + p["g1"] * v
    k_im, k_mo, cm = _fabric(p["UA_fab"], p["f_im"], p["tau_m"])
    k_ei = p["K_ei"]
    ce = p["tau_e"] * k_ei

    # emitter: heat now enters here, not directly into the air
    A[0, 0] = -k_ei / ce
    A[0, 1] = k_ei / ce
    B[0, Q1] = 1.0 / ce

    A[1, 0] = k_ei / ci
    A[1, 1] = -(k_ei + k_im + ginf) / ci
    A[1, 2] = k_im / ci
    B[1, TO] = ginf / ci
    B[1, GHI] = p["a_sol"] / ci
    B[1, ONE] = p["G"] / ci

    A[2, 1] = k_im / cm
    A[2, 2] = -(k_im + k_mo) / cm
    B[2, TO] = k_mo / cm
    return A, B


R4C3 = Model(
    name="R4C3", rung=3, states=["T_e", "T_i", "T_m"], n_zones=1,
    build=_build_r4c3, obs=np.array([[0.0, 1.0, 0.0]]),
    note="+ emitter node",
    params=[
        p_kei(), p_taue(),
        p_ci(3e6), p_uafab(120.0), p_fim(), p_taum(),
        p_ginf(80.0), p_wind(), p_solar(), p_gains(), p_qscale(),
    ],
)


# ---------------------------------------------------------------------------
# Rungs 4 and 5: two coupled zones, 4R3C each
# ---------------------------------------------------------------------------

E1, I1, M1, E2, I2, M2 = range(6)


def _build_two_zone(p, v, shared_emitter: bool):
    A = np.zeros((6, 6))
    B = np.zeros((6, N_INPUTS))

    kei1 = p["K_ei"] if shared_emitter else p["K_ei1"]
    kei2 = p["K_ei"] if shared_emitter else p["K_ei2"]
    taue1 = p["tau_e"] if shared_emitter else p["tau_e1"]
    taue2 = p["tau_e"] if shared_emitter else p["tau_e2"]
    ce1, ce2 = taue1 * kei1, taue2 * kei2

    ci1, ci2 = p["C_i1"], p["C_i2"]
    k12 = p["K_12"]
    f_im = p["f_im"]
    kim1, kmo1, cm1 = _fabric(p["UA_fab1"], f_im, p["tau_m1"])
    kim2, kmo2, cm2 = _fabric(p["UA_fab2"], f_im, p["tau_m2"])
    ginf1 = p["g0_1"] + p["g1"] * v
    ginf2 = p["g0_2"] + p["g1"] * v
    f1 = F1_SOLAR

    # zone 1
    A[E1, E1] = -kei1 / ce1
    A[E1, I1] = kei1 / ce1
    B[E1, Q1] = 1.0 / ce1

    A[I1, E1] = kei1 / ci1
    A[I1, I1] = -(kei1 + kim1 + k12 + ginf1) / ci1
    A[I1, M1] = kim1 / ci1
    A[I1, I2] = k12 / ci1
    B[I1, TO] = ginf1 / ci1
    B[I1, GHI] = p["a_sol"] * f1 / ci1
    B[I1, ONE] = p["G1"] / ci1

    A[M1, I1] = kim1 / cm1
    A[M1, M1] = -(kim1 + kmo1) / cm1
    B[M1, TO] = kmo1 / cm1

    # zone 2
    A[E2, E2] = -kei2 / ce2
    A[E2, I2] = kei2 / ce2
    B[E2, Q2] = 1.0 / ce2

    A[I2, E2] = kei2 / ci2
    A[I2, I2] = -(kei2 + kim2 + k12 + ginf2) / ci2
    A[I2, M2] = kim2 / ci2
    A[I2, I1] = k12 / ci2
    B[I2, TO] = ginf2 / ci2
    B[I2, GHI] = p["a_sol"] * (1.0 - f1) / ci2
    B[I2, ONE] = p["G2"] / ci2

    A[M2, I2] = kim2 / cm2
    A[M2, M2] = -(kim2 + kmo2) / cm2
    B[M2, TO] = kmo2 / cm2

    return A, B


_OBS2 = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
])
_STATES2 = ["T_e1", "T_i1", "T_m1", "T_e2", "T_i2", "T_m2"]

_SHARED_TWO_ZONE = [
    p_ci(1.5e6, "C_i1"), p_ci(1.5e6, "C_i2"),
    p_uafab(60.0, "UA_fab1"), p_uafab(60.0, "UA_fab2"),
    p_fim(), p_taum(name="tau_m1"), p_taum(name="tau_m2"),
    P("K_12", 200.0, 1.0, 5000.0, unit="W/K"),
    p_ginf(40.0, "g0_1"), p_ginf(40.0, "g0_2"),
    p_wind(), p_solar(),
    p_gains(600.0, "G1"), p_gains(600.0, "G2"),
    P("lam", 0.5, 0.02, 0.98, kind="logit", unit="-"),
    p_qscale(),
]

R4C3x2 = Model(
    name="R4C3x2", rung=4, states=_STATES2, n_zones=2,
    build=lambda p, v: _build_two_zone(p, v, shared_emitter=False),
    obs=_OBS2, note="+ second zone",
    params=[p_kei(name="K_ei1"), p_kei(name="K_ei2"),
            p_taue(name="tau_e1"), p_taue(name="tau_e2"),
            *_SHARED_TWO_ZONE],
)

R4C3x2_SHARED = Model(
    name="R4C3x2s", rung=5, states=_STATES2, n_zones=2,
    build=lambda p, v: _build_two_zone(p, v, shared_emitter=True),
    obs=_OBS2, note="zones share emitter params",
    params=[p_kei(), p_taue(), *_SHARED_TWO_ZONE],
)


# ---------------------------------------------------------------------------
# Extended physics: basement node + shaped internal gains
# ---------------------------------------------------------------------------

B_ = 6                      # index of the basement state


def _build_basement(p, v):
    """Two zones as before, plus a basement the loop passes through."""
    A6, B6 = _build_two_zone(p, v, shared_emitter=False)
    A = np.zeros((7, 7))
    B = np.zeros((7, N_INPUTS))
    A[:6, :6] = A6
    B[:6, :] = B6

    f = p["f_loss"]
    kb1, kbo = p["K_b1"], p["K_bo"]
    cb = p["tau_b"] * (kb1 + kbo)

    # a fraction of the metered heat warms the basement instead of an emitter
    for row, col in ((E1, Q1), (E2, Q2)):
        B[row, col] *= (1.0 - f)
    B[B_, Q1] = f / cb
    B[B_, Q2] = f / cb

    A[B_, B_] = -(kb1 + kbo) / cb
    A[B_, I1] = kb1 / cb
    B[B_, TO] = kbo / cb

    # the basement leaks its heat up into zone 1
    ci1 = p["C_i1"]
    A[I1, B_] += kb1 / ci1
    A[I1, I1] -= kb1 / ci1
    return A, B


_OBS_B = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
])

R4C3x2_GAINS = Model(
    name="R4C3x2+gains", rung=6, states=_STATES2, n_zones=2,
    build=lambda p, v: _build_two_zone(p, v, shared_emitter=False),
    obs=_OBS2, note="+ daily internal-gain profile",
    params=[p_kei(name="K_ei1"), p_kei(name="K_ei2"),
            p_taue(name="tau_e1"), p_taue(name="tau_e2"),
            *_SHARED_TWO_ZONE, *p_gain_profile()],
)

R4C3x2_BASEMENT = Model(
    name="R4C3x2+basement", rung=7, states=[*_STATES2, "T_b"], n_zones=2,
    build=_build_basement, obs=_OBS_B, note="+ basement node",
    params=[p_kei(name="K_ei1"), p_kei(name="K_ei2"),
            p_taue(name="tau_e1"), p_taue(name="tau_e2"),
            *_SHARED_TWO_ZONE, *p_basement()],
)

R4C3x2_FULL = Model(
    name="R4C3x2+both", rung=8, states=[*_STATES2, "T_b"], n_zones=2,
    build=_build_basement, obs=_OBS_B, note="+ basement and gain profile",
    params=[p_kei(name="K_ei1"), p_kei(name="K_ei2"),
            p_taue(name="tau_e1"), p_taue(name="tau_e2"),
            *_SHARED_TWO_ZONE, *p_basement(), *p_gain_profile()],
)

LADDER = [R1C1, R2C2, R4C3, R4C3x2, R4C3x2_SHARED]
EXTENDED = [R4C3x2_GAINS, R4C3x2_BASEMENT, R4C3x2_FULL]


def relax(model: Model, name: str, **overrides) -> Model:
    """
    Copy a model with some parameter bounds or seeds changed.

    Used to re-run a rung after a first pass showed parameters pinned at
    bounds. A bound that is hit is either a wrong prior or a symptom of
    something the model is missing, and the only way to tell them apart is to
    move the bound and see whether the fit improves out of sample.

    overrides: {param_name: (lo, hi)} or {param_name: (lo, hi, seed)}
    """
    params = []
    for q in model.params:
        if q.name in overrides:
            spec = overrides[q.name]
            lo, hi = spec[0], spec[1]
            sd = spec[2] if len(spec) > 2 else min(max(q.seed, lo), hi)
            params.append(P(q.name, sd, lo, hi, q.kind, q.unit))
        else:
            params.append(q)
    return Model(name=name, rung=model.rung, states=model.states,
                 params=params, n_zones=model.n_zones, build=model.build,
                 obs=model.obs, note=model.note)


def derived(model: Model, p: dict) -> dict:
    """Physically interpretable quantities, for sanity-checking a fit."""
    out = {}
    if model.n_zones == 1:
        if "UA_fab" in p:
            out["UA_total [W/K]"] = p["UA_fab"] + p["g0"]
            out["tau_m [h]"] = p["tau_m"] / HOUR
        else:
            out["UA_total [W/K]"] = p["UA0"]
        if "tau_e" in p:
            out["tau_e [min]"] = p["tau_e"] / MINUTE
        out["gains [W]"] = p["G"]
        out["C_i [MJ/K]"] = p["C_i"] / 1e6
    else:
        ua = p["UA_fab1"] + p["UA_fab2"] + p["g0_1"] + p["g0_2"]
        out["UA_total [W/K]"] = ua
        for z in (1, 2):
            out[f"tau_m{z} [h]"] = p[f"tau_m{z}"] / HOUR
            te = p.get(f"tau_e{z}", p.get("tau_e"))
            out[f"tau_e{z} [min]"] = te / MINUTE
        out["gains [W]"] = p["G1"] + p["G2"]
        out["K_12 [W/K]"] = p["K_12"]
        out["lam"] = p["lam"]
    out["glazing [m2]"] = p["a_sol"] / 0.5
    return out
