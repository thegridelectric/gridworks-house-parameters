Status: LIVING
# Code architecture

Library in `src/house/`, entry points in `scripts/`, outputs in `results/`.
Scripts are numbered in the order they were needed and are reproducible in
that order.

## Modules
| Module | Responsibility |
| --- | --- |
| `data.py` | load/validate CSV, weather-gap policy, contiguous segments |
| `model.py` | state-space ladder R1C1 -> R4C3x2, in (conductance, time constant) coordinates |
| `kalman.py` | ZOH discretisation, per-wind-bin steady-state filter, NLL |
| `fit.py` | log/logit transforms, bounds, L-BFGS-B |
| `thermostat.py` | on/off calibration from heat calls, setpoint-sensor mapping, controller |
| `forecast.py` | closed-loop hour-ahead energy (the deliverable) |
| `required.py` | alternative RC use: heat needed to hold setpoint, no switching |
| `simple.py` | linear energy predictors and their features |
| `predictor.py` | deployable artifact: coefficients + one function |

## Entry points
| Script | Produces |
| --- | --- |
| `01_energy_balance.py` | whole-house UA and gains, model-free |
| `01b_internal_gains.py` | is the intercept a real gain? |
| `02_ladder.py` | does each added state earn its complexity? |
| `03_refine.py` | effect of relaxing pinned bounds |
| `04_thermostat.py` | thermostat calibration |
| `05_qdist.py` | first closed-loop energy score |
| `06_energy_fit.py` | bounded emitter tau, selection on energy |
| `07_direct_energy.py` | 4R3C optimised directly on energy |
| `09_supply.py` | as 07, with a supply-temperature emitter law |
| `08_final.py` | head-to-head of everything (run last) |

## Conventions
Parameters are fitted as conductances and time constants, not R and C: an R
and a C trade off almost exactly, so bounds on the pair are uninterpretable
and "pinned at a bound" stops being a useful diagnostic.
