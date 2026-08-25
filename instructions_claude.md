# Context: grey-box thermal model of a two-zone house for MPC

## Objective

I need to predict the **next hour's heat delivery into the distribution loop (`Q_dist`)** for a house, as an input to an MPC. The MPC has a model of the energy stored in a buffer tank; the next tank state is predicted from the energy the distribution loop draws during the next timestep. So the quantity I need is the **loop-side** heat (what leaves the tank), not room heating demand.

The two differ because of **distribution-side thermal mass**: the water and metal in the emitters plus the water in the loop and pipework all sit between the tank and the room air, so the tank gives up heat before the rooms receive it. That storage is what `C_e` lumps together (it is not only the radiators). Note that envelope and furniture mass do *not* create this gap — they are downstream of the room air and affect how much heat is needed, not the difference between what leaves the tank and what reaches the room. If any pipework runs through unconditioned space, its losses widen the gap further and a single manifold measurement cannot separate them out.

**The MPC forecasts, it does not command.** The existing thermostats decide when heat is called; the MPC has to anticipate what they will demand. So a **thermostat model is required** — the thermal model alone cannot produce `Q_dist`, because nothing in the six state equations decides when heat is called. Building and calibrating that thermostat model is in scope.

## What I started from

A steady-state regression: `Q = α + β·T_o + γ·(T_i−T_o)·v`. With `T_i` assumed constant this is algebraically identical to `Q = [1/R_io + g0 + g1·v]·(T_i−T_o)` — i.e. a single lumped fabric conductance (conduction + convection + radiation bundled) in **parallel** with a wind-dependent infiltration/advection conductance, with zero capacitance and the setpoint hidden inside `α` (since `α = UA·T_i`).

Its failings: (1) assumed `T_i` constant, so it silently hardcoded the setpoint and broke on setback/intermittent operation; (2) assumed a *single* zone when I have two that diverge; (3) no thermal mass anywhere; (4) no solar gains. Crucially it predicted *heat loss*, and only equalled *demand* because steady state forces supplied = lost.

## Physics conventions I've settled on

Four mechanisms, not three: conduction (through material), convection (surface↔fluid boundary layer), radiation (surface↔surface, linearised as `h_rad ≈ 4εσT_ref³`), and **advection** (bulk air crossing the envelope — infiltration; governed by `ṁ·c_p`, not `hA`, which is why it gets its own branch).

RC modelling is valid because in the building's operating range all of these are approximately linear in ΔT, so conductances combine. Conduction and forced convection are exactly linear; radiation and *natural* convection are only approximately so. Capacitance (`m·c_p`) is exact.

Key parameterisation choice: infiltration enters as a **conductance** linear in wind, `1/R_inf = g0 + g1·v`, not as a resistance linear in wind. Physically `ṁ` grows with wind, and conductance = `ṁ·c_p`; writing `R_inf` as linear would be a hyperbola misfit and could go negative.

Wind is placed **entirely** in `R_inf`, not in the exterior convection film inside `R_mo`. Wind does affect the exterior film (`h ≈ 4+4v`), but `R_mo` is conduction-dominated so that's second-order. My `v` p95 is only ~1.25 m/s, so `g1` is expected to be weakly identified regardless.

## Model structure: two zones, 4R3C each, coupled

Six states: `T_e1, T_i1, T_m1, T_e2, T_i2, T_m2`. Two measured outputs: `T_i1, T_i2`. Four hidden states (both emitters, both masses) reconstructed by a Kalman filter.

```
C_e1 dT_e1/dt = Q1 − (T_e1−T_i1)/R_ei1
C_i1 dT_i1/dt = (T_e1−T_i1)/R_ei1 + (T_m1−T_i1)/R_im1 + (T_i2−T_i1)/R_12
                + (T_o−T_i1)·(g0_1+g1·v) + a_sol·f1·I
C_m1 dT_m1/dt = (T_i1−T_m1)/R_im1 + (T_o−T_m1)/R_mo1
```
and mirrored for zone 2 with `(T_i1−T_i2)/R_12`.

Circuit topology: `R_ei`, `R_im`, `R_mo` are in **series** (sequential stages along the fabric path); `R_inf` is in **parallel** with the `R_im+R_mo` series pair (an alternative route bypassing the mass). Each individual resistor internally bundles multiple mechanisms — `R_im` is the interior film (natural convection ∥ radiation); `R_mo` is wall conduction in series with the exterior film (forced convection ∥ radiation).

Node meanings: `T_m` is the **envelope only** (it's the node with an outdoor-facing side). Furniture and interior partitions have no path to outside, so they're implicitly folded into `C_i` — expect fitted `C_i` ≫ `ρ_air·c_p·V`.

`R_12` connects the two **air** nodes and lumps intermediate-floor conduction (symmetric, by reciprocity) with stairwell buoyancy airflow (genuinely asymmetric — zone 1 up into zone 2 more readily than the reverse). Kept symmetric to preserve linearity; diagnostic is to bin residuals by the sign of `(T_i1−T_i2)`.

## Why the emitter node exists

Two separate reasons, both real:

1. **Fitting:** `Q_dist` is measured as `ṁ·c_p·ΔT` at the distribution inlet. It's exactly zero when flow is zero (verified), so there's no phantom heat — but at startup, hot buffer water meets a cold room-temperature slug in the emitters, so `Q_dist` spikes while almost nothing reaches the rooms yet. That heat is real, and it's going into reheating emitter water and metal. Fed into `T_e` rather than `T_i`, it's correctly captured as `C_e` charging, then released to the room through `R_ei`. So `Q_dist` is exactly right as "heat entering the emitter system," and overestimates only if misread as "heat reaching the room air."
2. **Prediction:** I'm predicting the loop-side quantity for the tank model, and over a 1-hour horizon `C_e·ΔT_e` is a material fraction of delivered heat (zone 2's cycles are ~80 min). Whether the loop starts hot or cold changes next-hour `Q_dist` substantially.

Known limitation: `C_e` is genuinely constant (`m·c_p` doesn't change), but `R_ei` is not — radiator output goes roughly as `ΔT^1.3`, so a linear `R_ei` fitted mostly on 40 °C supply will underestimate output at 60 °C supply, which I sometimes use. Options are to accept the bias, or fit `Q_emit = K·(T_e−T_i)^1.3` (needs an EKF/UKF).

## Heat split between zones

I measure only total loop heat, and the return is a blend from all active circuits, so per-zone heat isn't directly measurable. But I have per-zone heat-call status, so I've partitioned the data into three columns. Then:

```
Q1 = Q_dist1_only + λ·Q_dist_together
Q2 = Q_dist2_only + (1−λ)·Q_dist_together
```

`λ` is a single fitted scalar, exact whenever only one zone calls, approximate only in both-active windows. In my data only **7.9%** of delivered energy is both-active, so `λ` is well-constrained by construction but its exact value matters little.

## Why two zones and not one

I initially planned a single lumped zone with an averaged `T_i`. I reversed this. The decisive argument is the **thermostats**: next-hour energy is dominated by whether each zone calls at all, and I have two independent thermostats with independent setpoints. A single node forces a fictitious aggregate thermostat that can't represent "zone 1 calling, zone 2 coasting."

Second argument: my setpoint history shows **regime changes**. Nov: zone 2 setpoint ~15.2 while zone 1 ran ~17.5. Jan–Mar: both at 18.5–19.5. Lumped-model effective parameters are *not* constant across those regimes — not because the building changes (it doesn't; fabric U-values and mass are genuinely constant), but because a single node can't represent where the thermal boundary sits. When zone 2 is cold, the party wall is a loss path and effective UA looks high with low effective `C`; when both are warm it's an internal partition. The two-zone model represents this explicitly, restoring genuine parameter constancy.

## Data

`data.csv`, 5-minute resolution, 2025-11-01 to 2026-03-01 (~52,400 rows, 181 days). Columns:

```
timestamp, T_o, v, GHI, T_i1, T_i2, Q_dist1_only, Q_dist2_only, Q_dist_together
```

- `T_o` in °C (converted from °F — steps of 0.0556 confirm this), `v` in m/s, `GHI` in W/m², `Q_*` in **watts** (average over the interval), zeros when no flow.
- 5 minutes is essential, not optional: zone 2 cycles ~80 min with clear rise / rounded peak / slow decay, giving ~16 samples per cycle. That shape is what identifies `C_e` and `R_ei`. Hourly averaging would alias it away.
- `Q_dist` max ~32.6 kW is a real startup transient (≈0.5 kg/s × 4180 × 15 K), not an artifact — keep it.
- `T_i` sensor resolution ~0.06 K. **Do not pre-smooth** — the Kalman filter has an explicit measurement-noise term and smoothing would bias the capacitances.
- No `T_w` (supply water temperature) yet. I can add it later. It would observe `T_e` almost directly and materially tighten `C_e`/`R_ei`, which are the weakest-identified pair.
- Setpoint history exists (I've plotted it) but isn't currently a CSV column.

## Fitting method

ZOH discretisation at `dt=300 s` via one matrix exponential of the augmented `[[A,B],[0,0]]`. Kalman filter over the record producing one-step-ahead innovations (measured − predicted `T_i`). Sum into a negative log-likelihood. Minimise with L-BFGS-B in **log space** for all positive parameters (guarantees positivity; parameters span ~11 orders of magnitude) and **logit space** for `λ`.

Note this is closed-loop identification — `Q_dist` is thermostat-driven, so input and output are correlated. The prediction-error method with a Kalman filter remains consistent under closed-loop data, which is why it's the right choice here over correlation-based approaches.

**Key simplification in the current version:** the wind term `g1·v·(T_o−T_i)` multiplies a state by an input, which makes A and B time-varying and forces per-wind-bin discretisation plus per-bin Riccati solves. Instead I precompute `w = v·(T_o − T_i_measured)` as an input column. Since `T_i` is measured to ~0.06 K against ΔT of 10–15 K, the error is negligible, and A becomes **constant** — one `expm`, one `solve_discrete_are`, one steady-state Kalman gain for the whole record. The update and predict steps are folded into a single recursion `x_{t+1} = Ad(I−KC)x_t + [Ad K y_t + Bd u_t]` so the loop body is one 6×6 mat-vec. This cut the code from ~690 to ~257 lines with identical results.

Parameters (19): `R_ei1, R_ei2, R_im1, R_im2, R_mo1, R_mo2, R_12, C_e1, C_e2, C_i1, C_i2, C_m1, C_m2, g0_1, g0_2, g1, a_sol, q_scale, λ`. Fixed: `f1, f2` solar split (collinear with `a_sol`, so cannot be fitted alongside it), `σ_meas ≈ 0.06 K`.

Validated end-to-end on synthetic data with known parameters: recovery was tight (`g1` 6.0→6.008, `a_sol` 7.0→7.031, `λ` 0.620→0.621, emitter τ 33.3/80.0 min→33.5/78.7). Note `R_ei` and `C_e` individually came out shaky while their **product** (emitter time constant) was accurate — that trade-off is expected, so read the time constant, not the split.

## Current problem: the fit on real data is degenerate

Latest run converged to cost −3.923 but is physically nonsense:

- Envelope τ of **1324 h** (zone 1) and **277,194 h** (zone 2). Should be 10–100 h.
- Zone 2 fabric UA = **1.0 W/K** (a thermos flask). `R_mo2 = 0.9938` against an upper bound of 1.0, and `C_m2 = 1e9` exactly at its bound — **parameters pinned at bounds means the optimiser ran out of room, not that it found a minimum.**
- Emitter τ of 1.1 min for zone 1, versus ~80 min implied by the observed cycling.
- Implied glazing area 1.1 m².
- `R_ei1`/`C_e1` at 517% relative SE, `R_12` at 849%, Hessian not positive definite.
- Validation: zone 1 RMSE 0.213 K vs **persistence 0.227 K** — only 6% better than assuming the temperature doesn't change. Zone 2 is better (0.184 vs 0.429).

**Diagnosed mechanism:** `q_scale` (process noise) ballooned from a 1e-8 seed to 1.03e-5, i.e. ~0.055 K per step against 0.06 K measurement noise. That tells the filter "the model is about as unreliable as the sensors," so the Kalman gain goes high and the filter re-anchors onto measurements every 5 minutes. One-step innovations become small, the likelihood looks excellent, and **the physics is never forced to be right.** Mass nodes decouple and drift to bounds.

**Fixes already applied:** GHI corrected (it had been reading up to 2368 W/m², physically impossible above ~1100); `q_scale` bounded hard (1e-12, 1e-9) or fixed; physical bounds tightened (`C_m` in (1e6,1e8), `C_e` in (5e4,2e6), `R_mo` so UA lands in 20–500 W/K, `a_sol` in (1,50)).

## Second run, after tightening bounds: worse, and diagnostic

Applying the fixes above (GHI corrected, `q_scale` bounded to ≤1e-9, `C_m` / `C_e` / `R_mo` / `a_sol` tightened) made validation **worse**, not better:

- Zone 1: RMSE 0.618 K vs **persistence 0.227 K**. Zone 2: 0.878 K vs **persistence 0.429 K**. Both zones now lose to assuming the temperature does not change. (The cost went from −3.92 to +27.2, but that is not comparable across runs — changing the `q_scale` bound rescales the likelihood.)
- **Six parameters pinned at bounds:** `C_e1`, `C_e2` at 5e4 (lower); `C_i1`, `C_m1` at 1e8 (upper); `a_sol` at 1 (lower); `q_scale` at 1e-9 (upper). Pinned means the optimiser still wants to go further — it wants *more* process noise, *less* solar, and *enormous* thermal mass.
- Envelope τ still 682 h (zone 1) and 516 h (zone 2). Emitter τ for zone 1 collapsed to 0.1 min.
- `g0_2` = 0.12 W/K, essentially zero infiltration for the upstairs zone, which is not physical.
- Zone 2 fabric UA = 21.3 W/K for a whole floor including the roof.

**The bias is the diagnostic clue.** Holdout bias is **+0.391 K (zone 1) and +0.743 K (zone 2)** over a one-hour horizon, both positive. The model systematically predicts the house getting warmer than it actually does. That is an energy-balance failure: too much heat going in, or too little coming out.

**Arithmetic that shows the gap.** The record is 16,071 kWh over 181 days ≈ **3,700 W average input**. Total fitted conductance is fabric (40.7 + 21.3) plus infiltration (87.3 + 0.1) ≈ **150 W/K**. At a typical winter ΔT of 12–15 K that dissipates only 1,800–2,250 W. So roughly **half the measured input has nowhere to go**. The optimiser responds by maxing out capacitance and process noise to absorb the surplus, which is exactly the pinning pattern above. Tightening bounds further cannot fix this — it is a missing energy sink, not a bad prior.

**Two candidate explanations:** either the real whole-house conductance is roughly double what is being fitted (bounds squeezing it below reality), or a large and roughly constant fraction of `Q_dist` never heats the conditioned space.

**DHW has been ruled out** — this loop does not serve domestic hot water. Remaining candidates for the second case: pipe runs through unconditioned space downstream of the ΔT sensors; a ventilation system not represented in the model; or the two `T_i` sensors not representing the whole conditioned volume (unmeasured colder spaces that are nonetheless being heated).

### FIRST TASK: run this diagnostic before touching bounds or model structure again

Independent of the whole RC/Kalman machinery. Over daily windows the storage terms nearly cancel, so a daily regression gives a direct estimate of total UA, and the intercept catches everything not proportional to ΔT:

```python
d = df.set_index('timestamp').resample('1D').mean()
q  = d.Q_dist1_only + d.Q_dist2_only + d.Q_dist_together
dT = 0.5 * (d.T_i1 + d.T_i2) - d.T_o
slope, intercept = np.polyfit(dT, q, 1)   # slope = whole-house UA [W/K]
```

Interpretation:

- **Slope ≈ 300 W/K** → the real conductance is about double what the fit allows, so the bounds are the problem. Widen `R_mo` and `g0` and re-fit.
- **Slope ≈ 150 W/K with a large positive intercept** → a constant chunk of `Q_dist` is not heating the conditioned space. Chase the sink (pipe losses, ventilation, unmeasured heated volume) and add it to the model as an explicit term rather than letting the optimiser hide it in capacitance.

Also plot `q` against `dT` as a scatter and check for curvature or two distinct branches, which would indicate regime-dependence (the November vs January setpoint regimes) rather than a single constant UA. Report the slope, intercept, R², and the scatter before proposing model changes.

## Validation: what exists now, and what I actually need

**What the code currently measures.** `Q_dist` is an **input** to the thermal model, not an output: the model takes `Q_dist` and predicts `T_i`. So current validation filters through the holdout, then every hour freezes the state and simulates 12 steps open-loop with *measured* `Q_dist` and weather, no measurement updates, comparing predicted `T_i1/T_i2` against measured, with persistence as the baseline. This tests the thermal physics in isolation and needs no setpoints. **Keep it as a unit test of the physics — it is necessary but not sufficient.**

**What I actually need: the real closed-loop `Q_dist` forecast.** This is the deliverable, and it is what must be scored. At each hour boundary:

1. Take the filtered state `[T_e1, T_i1, T_m1, T_e2, T_i2, T_m2]` from the Kalman filter running on live 5-min data.
2. Simulate forward 12 steps of 5 min with forecast `T_o`, `v`, `I`.
3. At each step, **the thermostat model decides the heat**, per zone: `if T_i < setpoint − db/2 → call on`, `if T_i > setpoint + db/2 → call off`, delivering the observed on-power, capped at loop max power.
4. Sum both zones' emitter inputs over the 12 steps → predicted next-hour `Q_dist`.
5. Score against actual hourly loop energy (MAPE and absolute error in kWh) on the holdout.

**A pure diagnostic variant exists but is not what I want.** It inverts the air-node equation to back out the heat implied by the *measured* `T_i` trajectory, including the endpoint. That uses future information, so it cannot run live. It is only worth implementing if we need to isolate whether a bad `Q_dist` score comes from the thermal parameters or from the thermostat model.

**Calibrating the thermostat model from existing data.** No new logging needed. The three columns already encode call status (`Q_dist1_only > 0` = zone 1 alone, `Q_dist_together > 0` = both). For each zone, look at `T_i` at the moments its call switches off (upper edge of the band) versus switches on (lower edge): the midpoint gives the setpoint, the spread gives the deadband. Do this in a **rolling window**, because my setpoints changed substantially over the winter (zone 2 ~15.2 in November, both 18.5–19.5 by January). The on-power is the typical nonzero value of the relevant `Q_dist` column. I also have the setpoint history plotted separately and can add it as a CSV column if that proves more reliable than inferring it.

**Expected weak spot.** Only 7.9% of delivered energy is both-zones-active, so the `Q_dist` score will be dominated by single-zone periods, and `λ` has little leverage on it. Also, hysteresis matters: real control overshoots (zone 2 by more than 1 K) and then loses heat at an elevated ΔT, so an idealised controller will tend to under-predict energy. Worth fitting both an ideal and a hysteresis controller and comparing.

## Live operation

For prediction, `T_e` and `T_m` are never measured — they come from the filter's state estimate, which means **history, not new measurement**. Run the Kalman filter continuously on live 5-min data; at each hour boundary take the current filtered state as the forecast initial condition. Six states summarise the entire past. A gap in the live stream degrades the estimate (covariance grows, `T_e` drifts) and it re-converges over roughly an emitter time constant. Cold start: set `T_e = T_m = T_i`, large covariance, discard the first few hours.

Note the asymmetry: fitting uses measured `Q_dist` and corrects with measured temperatures; forecasting generates `Q_dist` from the thermostat model and never corrects. The filter is the bridge between them.

## What I want from you

Set this up as a proper repo, with clean separation between: data loading and validation, the thermal model as a state-space module, the Kalman filter and parameter fitter, the **thermostat model** (calibrated from the call-status columns), diagnostics, and the live hour-ahead predictor.

Two priorities, in order:

1. **Run the daily energy-balance diagnostic** (see FIRST TASK above) and report the numbers before changing anything. Two successive fits have failed in a way that points to a missing energy sink or wrongly-bounded conductance, and further bound-tuning is not going to resolve it.
2. **Then debug the fit.** I want physically plausible time constants (envelope τ in the 10–100 h range, emitter τ in the tens of minutes), no parameters pinned at bounds, and a thermal model that clearly beats persistence on *both* zones.
3. **Then build and score the closed-loop `Q_dist` forecast**, since that is the actual MPC deliverable. Physics-only temperature validation is a unit test on the way there, not the goal.

Still on the list once the energy balance is resolved: sharing `R_ei`/`C_e` across zones (with only 7.9% both-active energy and very high relative SEs, they cannot be resolved separately); merging `C_m1`/`C_m2`; more multi-start restarts to rule out local minima; and whether the `T_o = −17.8 °C` values (exactly 0.0 °F) are a missing-data sentinel rather than real readings — I have chosen to ignore that for now but it may matter.

Propose a repo structure before writing code.
