# House load forecast

# Baseline model

The baseline is a simple two-term weather model for predicting the heat load of the house $\hat{Q}$ (kWh/h). It uses only outdoor air temperature $T_{\text{out}}$ (°F) and wind speed $w$ (mph):

$$
\hat{Q} = \max\!\left(0,\; \alpha + \beta\, T_{\text{out}} + \gamma\, w\,(65 - T_{\text{out}})\right).
$$

# Main model

The main model is also a linear model, but it has more features:

$$
\hat{Q} = \max\!\left(0,\; \alpha + \beta\,\Delta T + \gamma\, w\,\Delta T + \delta\, I + \epsilon\, \bar{T}_{\text{out},4\mathrm{h}} + \zeta\, \hat{Q}_{\text{prev}} + \eta\, \bar{Q}_{4\mathrm{h}} \right),
$$

with $\Delta T = \max(0,\ \overline{T}_{\text{zone}} - T_{\text{out}})$, $I$ forecast solar irradiance (W/m²), $\bar{T}_{\text{out},4\mathrm{h}}$ the mean $T_{\text{out}}$ over the four hours before this hour, $\hat{Q} = \rho \cdot Q_{\text{dist}}$ when forming $\hat{Q}_{\text{prev}}$ and $\bar{Q}_{4\mathrm{h}}$ from measured loop data.


- `alpha` ($\alpha$) — intercept
- `beta` ($\beta$) through `eta` ($\eta$) — one coefficient per feature (same order as the table)
- `rho` ($\rho$) — converts $Q_{\text{dist}}$ to $\hat{Q}$ for load-history terms

Units: $T_{\text{out}}$ and zone temperatures °F, wind mph, solar W/m², heat load kWh/h.

# Building a 48-hour forecast

Forecast hours $t = 1, \ldots, 48$. Measured distribution load $Q_{\text{dist}}^{\mathrm{meas}}(s)$ for $s \le 0$ (at least $s = 0, -1, -2, -3$), weather forecasts $T_{\text{out}}(t)$, $w(t)$, $I(t)$, $\overline{T_{\text{zone}}}(t)$.

$$
Q^{\mathrm{meas}}(s) = \rho \cdot Q_{\text{dist}}^{\mathrm{meas}}(s),
\qquad
\mathcal{Q}(s) =
\begin{cases}
Q^{\mathrm{meas}}(s) & s \le 0 \\
\hat{Q}(s) & s \ge 1
\end{cases}
$$

For $t = 1, 2, \ldots, 48$:

$$
\hat{Q}(t) = \max\!\left(0,\; \alpha + \beta\,\Delta T(t) + \gamma\, w(t)\,\Delta T(t) + \delta\, I(t) + \epsilon\, \frac{1}{4}\sum_{j=1}^{4} T_{\text{out}}(t-j) + \zeta\, \mathcal{Q}(t-1) + \eta\, \frac{1}{4}\sum_{j=1}^{4} \mathcal{Q}(t-j) \right).
$$

