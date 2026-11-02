# House load forecast

You received a linear model for this house:

- `alpha` ($\alpha$) — intercept
- `beta` ($\beta$) through `eta` ($\eta$) — one coefficient per feature (same order as the table)
- `rho` ($\rho$) — one number (not a term in the sum; used only when forming the two load-history inputs below)

Use these with weather forecasts and recent load to predict the house’s heat load up to 48 hours ahead.

# Features

For each hour, build the input $x_f$ as follows (the coefficient for that row is the value you were given):

| Coefficient | Feature name | Input $x_f$ |
|-------------|--------------|-------------|
| `beta` ($\beta$) | `deltaT` | $\max(0,\ \overline{T_{\text{zone}}} - T_{\text{oat}})$ |
| `gamma` ($\gamma$) | `windspeed_times_deltaT` | `deltaT` × wind speed |
| `delta` ($\delta$) | `solar_w_m2` | forecast irradiance |
| `epsilon` ($\epsilon$) | `OAT_avg_4h` | mean outdoor temp over the four hours before this hour |
| `zeta` ($\zeta$) | `previous_dist_kwh_scaled` | (previous hour’s $\hat{Q}$) × $\rho$ |
| `eta` ($\eta$) | `dist_kwh_scaled_avg_4h` | (mean $\hat{Q}$ over the four hours before this hour) × $\rho$ |

For `previous_dist_kwh_scaled` and `dist_kwh_scaled_avg_4h`, use heat load in kWh/h for the lag or window, then multiply by $\rho$ to get $x_f$.

$\overline{T_{\text{zone}}}$ = average zone temperature (same zones as at calibration).

# The model

$$
\hat{Q} = \max\!\left(0,\; \alpha + \beta\, x_{\Delta T} + \gamma\, x_{w\Delta T} + \delta\, x_{\text{solar}} + \epsilon\, x_{\text{OAT4}} + \zeta\, x_{\text{prev}} + \eta\, x_{\bar{Q}_4} \right)
$$

($x_{\Delta T}$, … are the inputs from the table for that hour.)

- $\hat{Q}$ = the house’s heat load for that hour (kWh/h)

Units: outdoor and zone temperatures °F, wind mph, solar W/m², heat load kWh/h.

# One hour

1. Build each feature input from the table.
2. $\hat{Q} = \max(0,\ \alpha + \beta x_{\Delta T} + \gamma x_{w\Delta T} + \delta x_{\text{solar}} + \epsilon x_{\text{OAT4}} + \zeta x_{\text{prev}} + \eta x_{\bar{Q}_4})$.

# 48 hours

Index forecast hours by $t = 1, \ldots, 48$. Let $Q^{\mathrm{meas}}(s)$ be measured heat load at hour $s \le 0$ (you need at least $s = 0, -1, -2, -3$ for the first step’s four-hour averages).

For $t = 0, 1, \ldots, 47$, define the load series used in lag and rolling features:

$$
\mathcal{Q}(s) =
\begin{cases}
Q^{\mathrm{meas}}(s) & s \le 0 \\
\hat{Q}(s) & s \ge 1
\end{cases}
\qquad
\bar{Q}_4(t+1) = \frac{1}{4}\sum_{j=0}^{3} \mathcal{Q}(t-j).
$$

Build the feature inputs at hour $t+1$ from your weather forecast and the table, using $\mathcal{Q}(t)$ for $x_{\text{prev}}$ and $\bar{Q}_4(t+1)$ for the four-hour load average.

Anchor the recursion, then advance hour by hour for $t = 0, \ldots, 47$:

$$
\hat{Q}(0) := Q^{\mathrm{meas}}(0)
$$

$$
\hat{Q}(t+1) = \max\!\left(0,\; \alpha + \beta\, x_{\Delta T}(t+1) + \gamma\, x_{w\Delta T}(t+1) + \delta\, x_{\text{solar}}(t+1) + \epsilon\, x_{\text{OAT4}}(t+1) + \zeta\, x_{\text{prev}}(t+1) + \eta\, x_{\bar{Q}_4}(t+1) \right).
$$

Each step uses only past measurements and $\hat{Q}(1), \ldots, \hat{Q}(t)$ already computed.
