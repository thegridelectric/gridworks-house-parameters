Status: LIVING
# adr-001: Forecast hour-ahead Q_dist with a linear model, not the 4R3C

## Context and Problem Statement
The MPC needs next-hour loop energy (`Q_dist`). Does the two-zone 4R3C
grey-box model earn its complexity against a regression?

## Considered Options
- Linear regression on ~8 features
- 4R3C + thermostat, fitted on hour-ahead energy

## Decision Outcome
Chosen option: "linear regression", 0.661 kWh against a 4.07 kWh mean. The
4R3C reaches 0.836 on identical windows with honest per-zone behaviour, and
loses all six folds of block cross-validation. Earlier energy fits scored
0.684 by exploiting a degeneracy: only TOTAL Q_dist is measured, so the
optimiser shut zone 1 off (simulated duty 0.000 against an actual 0.732) and
ran zone 2 six times too much, landing the total right through cancelling
errors. Constraining simulated duty to the recorded call status closes it. A
measured occupancy profile and a basement capacitance then buy 2.8% -- real
mechanisms, far too small to close the gap.

### Consequences
- Good, because it deploys as a handful of coefficients with no state to
  maintain (`results/qdist_predictor.json`).
- Good, because its top feature is the gap between room temperature and
  thermostat threshold (0.088 kWh), which alpha/beta/gamma cannot express.
- Bad, because it is fitted for a 60-minute horizon; the 4R3C is not. The
  4R3C deficit is correlation (0.745 vs 0.840), not bias, so no constant
  correction closes it.
- Revisit if the MPC shifts load: storage is ~6% of hourly energy while
  thermostats hold setpoint, but pre-heating would swing temperature.
