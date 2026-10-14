# Takeoff simulation

`SolarSimulator/TakeoffSimulation/` is a standalone, script-style study of how a seaplane
accelerates on the water to liftoff. It is **not imported by the MDP pipeline**. The takeoff
stage power the MDP uses comes from the climb heuristic in
[Aircraft power model](aircraft-power.md). Treat this directory as supporting analysis for
takeoff energy and time.

The directory has no `__init__.py`, and running `gudmunson.py` or `takeoff.py` executes its
simulation and plots at import time. For that reason these modules are not on the API pages.

| File | Contents |
|---|---|
| `gudmunson.py` | `SimulationParams`, `SeaplaneSimulator` (imperial-unit hull and aero force balance), `UAVSimulator` (small-UAV variant driven by the `Motor` map); the module-level script runs a UAV case and plots forces and power against airspeed |
| `motor.py` | `Motor`: static test map for an HQ8040-3 propeller/motor (throttle → thrust, power, g/W) |
| `takeoff.py` | Script: SI-unit takeoff run of a NAX-4-class seaplane with a cubic thrust curve and Gaussian-fit hull resistance; prints liftoff time and distance |

## `gudmunson.py`: force-balance time march

`SeaplaneSimulator.simulate()` integrates $m\,\dot V = T - D - R_\text{water} - R_\text{hydro}$
with explicit Euler steps of size `dt`. It stops when lift exceeds weight **and** the speed
coefficient passes `CV_cutoff`, or when $t \ge 2000$ s.

| Quantity | Code |
|---|---|
| Throttle ramp | $r(t) = 0.25 + 0.75\,\min(t,5)/5$ for $t<5$ s, then 1 |
| Thrust (base class) | $T = r\,(0.01040V^2 - 10.065V + 3225)$ lbf |
| Speed coefficient | $C_V = V/\sqrt{gB}$ |
| Water-resistance coefficient | $C_R = \max(0,\ \text{polyval}([0.0011,-0.0221,0.1062,-0.0149], C_V))$ |
| Trim angle | $\alpha_\text{trim} = 7 + \tanh(3.553C_V - 3.891)$ deg; wing AoA $= \alpha_\text{trim} + i_w$ |
| Lift / drag (base) | $C_L = 0.2 + 4.62\,\alpha\,\pi/180$, $C_D = 0.06 + 0.058C_L^2$ |
| Buoyancy relief | $R_\text{water} = C_R\,(W_0 - L)$ while $C_V \le$ `CV_cutoff`, else 0 |
| Hydroplaning friction | $R_\text{hydro} = f\,B\,V_\text{kt}^2$ once $C_V \ge 2.9$ |
| Electrical power | $P_\text{elec} = P_\text{mech}/\eta_p \times 1.35582$ |

It records a per-step `DataFrame` of speed, forces, distance, power, and cumulative mechanical and
electrical energy.

**`SimulationParams` defaults** (full-size seaplane, imperial units):

| Param | Default | Meaning |
|---|---|---|
| `B` | 5.0 ft | Hull beam |
| `S` | 375 ft² | Wing area |
| `i_w` | 4.5° | Wing incidence |
| `W0` | 8000 lbf | Gross weight |
| `rho_water` / `rho_air` | 63.5 lbf/ft³ / 0.002378 slug/ft³ | Densities |
| `dt` | 0.5 s | Time step |
| `CV_cutoff` | 7.5 | $C_V$ at which displacement resistance is dropped |
| `f` | 0.012 | Hydroplaning friction coefficient |
| `eta_p` | 0.82 | Propeller efficiency |

**`UAVSimulator`** subclasses the seaplane simulator:

- Aerodynamics are a NACA 4414 model: $C_L = 0.25 + 2\pi\alpha$, $C_D = 0.008 + 0.045C_L^2$.
- Thrust and power come from `Motor`, with throttle $= 100\,r$ %.
- The module-level UAV case uses $B = 0.5$ ft, $S = 2.5$ ft², $W_0 = 8$ lbf, `dt` $= 0.01$ s,
  $f = 0.015$, and `CV_cutoff` set from a 20-knot hydroplaning onset.

## `motor.py`: HQ8040-3 static map

`Motor` linearly interpolates static-test data at throttles of 30–100 % in 5 % steps. Thrust runs
from 417 to 2451 g and input power from 78 to 801 W. Throttle is clipped to the tested range.

| Method | Returns |
|---|---|
| `thrust(throttle, units='g')`; units `'g'` or `'N'` | Static thrust |
| `power(throttle, units='W')`; units `'W'` or `'kW'` | Electrical input power |
| `efficiency(throttle)` | Thrust per watt (g/W) |

## `takeoff.py`: SI reference case

A module-level script. It takes aircraft and environment parameters from the "Table 2" of a
NAX-4 seaplane reference, converted to SI. Its comments also cite NACA TN-2481 for the
resistance curve fit.

| Component | Model |
|---|---|
| Hull resistance | $C_r(C_\mu)$ = sum of four Gaussians in $C_\mu = V^2/(gL)$; $R = C_r\,\rho_w g B L$ |
| Friction | $R_f = 0.003\,S_\text{wet}\,V^2$, $S_\text{wet} = 2BL$ |
| Aero drag | $C_D = 0.032 + C_L^2/(\pi\,AR\,e)$, $e = 0.82$, $C_L$ from $L \approx W$ |
| Thrust | Cubic $T(V) = AV^3 + BV^2 + CV + D$ fitted through static thrust and $\eta P/V$ at liftoff, cruise and max speed |
| Throttle ramp | $0.1 + 0.9\,t/10$ for $t < 10$ s |
| Integration | Euler, $\Delta t = 0.1$ s, until $V \ge$ 54 KCAS |

## Running

```bash
# from SolarSimulator/ (gudmunson.py imports TakeoffSimulation.motor)
conda run -n pvlib python -m TakeoffSimulation.gudmunson
conda run -n pvlib python TakeoffSimulation/takeoff.py
```

Both scripts open matplotlib windows.
