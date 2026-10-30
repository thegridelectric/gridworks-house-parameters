# RSWT estimation

# Heat transfer rate - the water’s perspective

From the water’s perspective, we know the heat transfer rate ($Q$, in Watts) is proportional to the temperature difference between the supply and return water:

$$
\Delta T_{water} = T_{supply} - T_{return}
$$

In fact, the heat transfer rate is the following:

$$
Q = \dot{m} \cdot c_p\cdot \Delta T_{water}
$$

Where $\dot{m}$ is the flow rate of the water, $c_p$ is its heat capacity. 

# Heat transfer rate - the radiator’s perspective

From the radiator’s perspective, we know $Q$ is a function of the temperature difference between the radiator’s surface and the room ($\Delta T_{rad}$). We approximate the radiator’s surface temperature as the average water temperature inside it ($\bar{T}_{water}$) and write:

$$
\Delta T_{rad} = \bar{T}_{water} - T_{room}
$$

Heat transfer from a radiator to a room occurs via natural convection (the air around the radiator heats and rises, creating a continuous airflow around it) and thermal radiation (infrared is emitted from the radiator's hot surface). Both these processes are not linearly proportional to $\Delta T_{rad}$ (surrounding airflow speeds up as the radiator gets hotter, and thermal radiation scales with temperature to the fourth power). The (empirical) radiator emission law gives the following: 

$$
Q = K \cdot (\Delta T_{\text{rad}})^n
$$

Where $K$ is the physical radiator constant and $n$ is the radiator exponent (they depend on the radiator’s material, geometry, surface area, etc.).

# Bringing both together

It is appropriate to estimate that both heat transfers are equal (the radiator is providing the room with exactly the amount of heat that the water is losing).

In addition to this, since the average water temperature in the radiator is the average between the supply and return temperatures, we can find:

$$
T_{supply}= \bar{T}_{water}+ \frac{\Delta T_{water}}{2}
$$

Adding design day conditions $Q_{DD}$ and $\Delta T_{supply, DD}$, and assuming a constant room temperature, we can bring it all together to get $T_{supply}$ as a function of $Q$:

$$
T_{supply}(Q) = T_{room} + (T_{supply,DD} - T_{room} - \frac{Q_{DD}}{2\dot{m}c_p})\cdot (\frac{Q}{Q_{DD}})^{1/n} +\frac{Q}{2\dot{m}c_p}
$$

If, in the equation above, we use the mass flow rate ($\dot{m}$) that the distribution system provides in constant heat call, we are effectively providing an equation for the required supply water temperature as a function of the required heating rate.

# In practice

From the equation above:
- $\dot{m}_{chc}$ can be estimated using hours in which the house was in constant heat call
- We can find the design day $Q_{DD}$ and $T_{supply,DD}$ from the method used in our paper
- We can find $n$ using the design day conditions in the equation above