Status: LIVING
# gridworks-house-parameters

Grey-box thermal identification for a two-zone house (beech), and hour-ahead
forecasting of heat delivered into the distribution loop (`Q_dist`) as an input
to an MPC over a buffer tank.

## Quickstart

```bash
uv sync
uv run scripts/01_energy_balance.py     # model-free UA and gains
uv run scripts/08_final.py              # head-to-head of every predictor
```

`data.csv` is 5-minute data, 2025-11-01 to 2026-03-01. Outputs land in
`results/`.

## Current answer

Hour-ahead `Q_dist`, 6-fold block cross-validation over the whole winter
(mean actual 4.07 kWh). Folds are week-blocks dealt round-robin, so each fold
spans November to February.

| model | mean MAE kWh | sd |
| --- | --- | --- |
| **linear, 8 features** | **0.674** | 0.054 |
| 4R3C fitted on energy | 0.772 | 0.096 |
| alpha/beta/gamma form, refitted per fold | 0.816 | 0.095 |
| persistence | 1.010 | 0.044 |

The 4R3C lost in all six folds. On a single held-out split with the per-zone
degeneracy closed and the added physics in place it reaches 0.836 against the
linear model's 0.661.

See `docs/adr/adr-001-hour-ahead-qdist-model.md` for why.

`results/qdist_predictor.json` is the deployable linear model: nine
coefficients plus a setpoint-sensor mapping, loadable via
`house.predictor.LinearPredictor.from_json`.

## Three traps in this dataset

- `T_o` blanks were previously exported as `0.0 degF`. Six whole days were
  affected; left in, they flatten the fitted conductance from 205 to 167 W/K.
- The reported thermostat setpoint comes from the thermostat's own sensor,
  which reads ~1.1 K (zone 1) and ~1.7 K (zone 2) below the `T_i` loggers.
  Subtracting it from `T_i` makes sharp on/off switching look modulating.
- Only TOTAL `Q_dist` is measured, so fitting an RC model on energy alone lets
  it put all the heat in one zone and still score well. Constrain simulated
  duty against the per-zone call status (`scripts/15_duty.py`).

## Map

| Where | What |
| --- | --- |
| `docs/architecture/code.md` | modules and entry points |
| `docs/adr/` | decisions (directory listing is the index) |
| `src/house/` | library |
| `scripts/` | numbered entry points |
