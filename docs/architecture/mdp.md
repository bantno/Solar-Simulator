# The MDP

The mission is a finite-horizon MDP. It is implemented by `stochasticMDP` in
`BaseClasses/mdp_base.py`, with the energy and failure dynamics in `StochasticTransitionLogic`
(`BaseClasses/transition_model_base.py`). API pages:
[mdp_base](../api/baseclasses/mdp_base.md),
[transition_model_base](../api/baseclasses/transition_model_base.md).

## Stages

| Quantity | Definition | Source |
|---|---|---|
| Stage length $\Delta t$ | `delta_t` minutes (default 15) | `SimulationFactory.__init__` |
| Horizon $H$ | `horizon` stages; stage $t = 0, \dots, H-1$ | config `horizons` |
| Stage 0 | The stage containing `start_datetime` (month, day, hour, minute must exist in the expected-data table) | `EnvironmentLoader._find_start_index` |
| Discount $\gamma$ | 1.0 (undiscounted) | `build_mdp` passes `gamma=1.0`; the solver uses `_GAMMA = 1.0` |
| Initial state | $(100\%, \text{moored})$ | `create_simulation`: `state0 = [100.0, 0]` |

Each stage $t$ has its own weather distributions. `EnvironmentLoader` slices them out of the
location's expected-data table: a `horizon`-long window starting at `start_datetime`, which
wraps around at the end of the year.

## State

A state is a row `[soc, mode]`:

| Mode | Code | Meaning |
|---|---|---|
| moored | `0` | On the water, charging |
| flying | `1` | Airborne, surveying |
| broken | `2` | Absorbing failure state, stored as `[-1.0, 2]` |

- **SoC grid.** State of charge is in percent, on the grid $\{0, \delta, 2\delta, \dots, 100\}$.
- **Grid step.** The step is set in energy units by `energy_increment_wh`, which is required:
  $$\delta = 100 \cdot \frac{\text{energy\_increment\_wh}}{\text{capacity\_wh}} \quad [\%].$$
  For example, 5 Wh on a 300 Wh battery gives $\delta = 1.667\%$ and 61 SoC levels.
- **Table layout.** `AbstractMDP._get_states` builds the table as all SoC levels for `moored`,
  then all SoC levels for `flying`, then the broken row, so there are $2 n_\text{soc} + 1$ rows.
  The solver relies on this ordering: row $= i_\text{soc} + \text{mode} \cdot n_\text{soc}$
  (`_state_row_index`).
- **Exogenous regime.** When a weather persistence chain is enabled, the decision state also
  includes the current exogenous regime
  $z = b \cdot n_g + g$ (wind bin $b$, solar bin $g$). It is not stored in the state row: the
  value table gets an extra leading dimension instead. See [Solver](solver.md),
  [Wind persistence chain](../models/wind-chain.md), and
  [Solar persistence chain](../models/solar-chain.md).

## Actions

`AbstractMDP.actions = [0, 1]`. What an action means depends on the current mode:

| Mode \ action | `0` | `1` |
|---|---|---|
| moored | stay moored | take off |
| flying | land | continue flying |

If the action succeeds, the next mode is `moored` for `a = 0` and `flying` for `a = 1`.

## Energy dynamics

Energy used in one stage (`StochasticTransitionLogic._calculate_energy_consumption`), with
$\tau = 60\,\Delta t$ seconds:

| Mode \ action | `0` | `1` |
|---|---|---|
| moored | $P_\text{idle}\,\tau$ | $(P_\text{takeoff} + P_\text{idle})\,\tau$ |
| flying | $(P_\text{land} + P_\text{idle})\,\tau$ | $(P_\text{cruise} + P_\text{idle})\,\tau$ |
| broken | 0 | 0 |

The four powers come from the aircraft model, `Seaplane.get_mdp_power_params()`. The
seaplane is built with `capacity = capacity_wh / 22.2`. See
[Aircraft power model](../models/aircraft-power.md).

**Solar energy harvested** in stage $t$ (the environment provider's `_energy_gain_from_solar`):

$$G_t = K_t \cdot G^\text{cs}_t \cdot \tau \cdot A_\text{panel} \cdot \eta_\text{panel}, \qquad K_t \sim \mathrm{Beta}(\alpha_t, \beta_t),$$

where:

- $K_t$ is the clear-sky index;
- $G^\text{cs}_t$ is the stage's clear-sky irradiance (W/m²);
- the panel area and efficiency come from the configured solar panel model (see
  [Solar panel models](../models/solar-panel.md)).

**Next energy.** Given current energy $C$ and consumption $E$:

$$C' = C + G_t - E,$$

- The SoC bin is $\lfloor C' / (\text{capacity}/n) \rfloor \cdot \delta$ (`energy_to_soc`),
  where $n = 100/\delta$ is the number of bins.
- $C' < 0$ is a **battery-depletion failure**: the next state is `broken`.
- In rollouts $C'$ is clipped to $[-1000\,\text{J}, \text{capacity}]$ and carried forward as a
  continuous value (`_update_energy_and_state_continuous`). The solver works on the discrete
  SoC grid.

## Action success (mechanical failure)

Every action succeeds with a probability that depends on the stage's wind speed $w$:

$$p_s(w, a, s) = \texttt{transition\_model.compute\_probability}(w, a, s).$$

- **Model choice.** `transition_model` in the config selects the model (default `moderate`) through
  `ProbabilityModelFactory`. See [Transition models](../models/transition-models.md).
- **Failure.** With probability $1 - p_s$ the next state is `broken` (`_apply_transition_probability`).
- **Failure type.** Rollouts record `failure_type`: `0` for no failure, `1` for a crash
  (mechanical), `2` for battery depletion.

## Reward

`stochasticMDP.reward`:

$$r(s, a, s', t) = g(m, a)\, O_t \;-\; c_f\, \mathbb{1}[s' = \text{broken}]$$

where:

- $O_t$ is the stage's whale-observation probability, from the whale series (see
  [Whale observation](../models/whale-observation.md));
- $c_f$ is the failure penalty (`failure_penalties`, default 15 in `YAMLSimulationRunner`);
- $g(m, a)$ is the **observation gain**, a $3 \times 2$ matrix (`AbstractMDP.observation_gain`):

| Mode \ action | `0` | `1` |
|---|---|---|
| moored | 0 | `takeoff_cruise_fraction` (default 1.0) |
| flying | `landing_cruise_fraction` (default 0.0) | 1 |
| broken | 0 | 0 |

The defaults reproduce the original simplification: a climb is short compared with a stage,
so a takeoff stage earns the full observation reward and a landing stage earns none. Setting
either fraction to `auto` derives $1 - t_\text{maneuver}/\tau$ from the seaplane's
climb and descent times (`SimulationFactory._resolve_cruise_fractions`). Values outside
$[0, 1]$ are rejected when the MDP is constructed.

## Objective

Maximise the expected total reward over the horizon:

$$V^*_0(s_0) = \max_\pi \mathbb{E}\Big[\sum_{t=0}^{H-1} r(s_t, a_t, s_{t+1}, t)\Big].$$

`broken` is absorbing and earns no further reward, so the failure penalty is paid at most once
per episode. [Solver](solver.md) describes how $V^*$ is computed.
