# gridworks-house-parameters

Hourly linear regression for house heating load.

The target variable is scaled distribution energy: `dist_kwh_scaled = dist_kwh × (sum(hp_kwh_th) / sum(dist_kwh))`, so total distribution thermal output matches total heat-pump thermal output over the training window:

### Main model

Features:

- `deltaT`: average zone setpoint minus outdoor air temperature (°F), floored at 0
- `windspeed_times_deltaT`: wind speed (mph) * `deltaT`
- `solar_w_m2`: solar irradiance (from open-meteo.org's forecasts, I have a working script for this already)
- `previous_dist_kwh_scaled`: previous hour’s scaled load
- `OAT_avg_4h`: 4-hour rolling mean outdoor temperature
- `dist_kwh_scaled_avg_4h`: 4-hour rolling mean of scaled load
- Optional weekday/weekend hour-of-day indicators for selected hours

$\hat{y}$ is predicted scaled distribution energy (`scaled_dist_kwh`). Coefficients $B_0$ (intercept), $B_1$, … follow feature order in code (same names as the feature list above). We use ordinary least squares (OLS) to fit the model.

$$
\hat{y} = \max\left(0,\; B_0 + B_1 \Delta T + B_2 (w_s \Delta T) + B_3\,\text{solar} + B_4\,\text{prev} + B_5\,\text{OAT4} + B_6\,\text{load4} + \sum_i B_i\, o_i \right)
$$

 With occupancy on, each `wd_hour_*` (weekday) / `we_hour_*` (weekend) column adds another $B_i$ and $o_i$ term.

### Baseline

Same scaled target, but the older two-term weather model (fit in parallel for comparison):

$$
\hat{y} = \alpha + \beta \cdot \text{OAT} + \gamma \cdot w_s \cdot (65 - \text{OAT})
$$

## What the code includes

We load hourly data from the database or from `data/{house}_house_params_data.csv` (depending on the `DATA_SOURCE` variable). We then crop to `[start_time, end_time)`.

After data cleaning, we either fit the model on the last N calendar days ending that day (or all days so far if `GROW_WINDOW_TO_N` is True). This gives the parameters that we will use to make 48-hour predictions.

If we are interested in the model's performance over past data, we can simulate a backtest. Use `trailing_n_day_fits(n)` for one lookback *N*, or `sweep_n` to try several values of *N*.

### Data cleaning

1. **Erroneous data:** clip or NaN bad values and big hour-to-hour jumps.
2. **Missing data:** fill a complete hourly grid, interpolate short gaps, drop long (≥4h) gaps, compute the 4h rolling OAT/load feature where the window is solid. We then drop any hour that still has NaNs.
3. **Bad operating hours**: we remove such hours from the training data:
   - Oil boiler was running
   - Zone was below setpoint
   - Setpoint changed from the previous hour (working on removing also the next hours while we wait for the temperature to reach the new setpoint)
   - Broken thermostat (no heat calls although the temperature is below setpoint, or heat calls but the temperature is above setpoint)

### Feature engineering

After cleaning, we compute the features: `deltaT`, `windspeed_times_deltaT`, `previous_dist_kwh` (first row dropped), optional occupancy dummies, etc.

### Training

For every training day, we fit the model on the last N calendar days ending that day, or all days so far if `GROW_WINDOW_TO_N` (after `MIN_FIT_WINDOW_DAYS`). We then save the coefficients, their standard errors, and R².

Training is done weekly or daily, depending on the `TRAINING_FREQUENCY` variable.

### Out-of-sample backtest

In this part of the code, we are trying to figure out how well our model predicts the next 48 hours (the default `FORECAST_HORIZON_HOURS`) of house heating load. This is the part that will end up in the weekly report.

At each forecast origin (each hour), we use the latest refit whose training window ends before that origin’s calendar day.

With the main model, we recursively predict the next 48 hours. Lead 1 uses observed lagged load (i.e. dist_kwh_scaled from the previous hour), later leads use the prior predicted load. Since the baseline model doesn't have any lag terms, it just predicts the next 48 hours based on the weather forecast.

We then score the predictions vs the actual scaled load (MAE, RMSE overall and by hour in the horizon) and output plots and results. 