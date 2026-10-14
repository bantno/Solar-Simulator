# Policies

Each experiment compares two families of policy on the same MDP and the same weather:

- **Optimal.** Backward induction solves the finite-horizon MDP, then the policy acts greedily on
  the value table.
- **Threshold.** A hand-tuned rule with an observation threshold and a wind threshold.

Both live in `SolarSimulator/BaseClasses/simulation_base.py`
([API](../api/baseclasses/simulation_base.md)). Both are built by
`SimulationFactory.create_simulation` in `run_sim.py` ([API](../api/baseclasses/run_sim.md)).

## Which simulations a config creates

`YAMLSimulationRunner._build_param_list` expands the config into simulations. For every
`start_datetime`:

| Policy | One simulation per | Condition |
|---|---|---|
| `threshold` | location × horizon × failure penalty × capacity × `threshold_values` × `wind_thresholds` | Both lists non-empty (both default `[]`) |
| `optimal` | location × horizon × failure penalty × capacity | `include_optimal: true` (default) |

Every simulation starts moored at full charge, `state0 = [100.0, 0]`.

## Information and timing at each stage

At stage $t$ the rollout draws the stage's solar, wind and whale-observation samples **first**.
The policy then chooses an action knowing those values (`simulate_episode_batch`). Both policies
therefore act on the realized wind and observation probability of the current stage. The
transition and its success draw use the same wind sample.

## Optimal policy

The optimal policy is `OptimalContinuousAnalyticalPolicySimulation`, built around
`mdpAnalyticalBackwardSolver`:

- `__init__` calls `mdp_solver.solve()`, so every optimal simulation **solves fresh**. The value
  table is written as `.npy` under the run's `solver_tables/`. See [Solver](../architecture/solver.md).
- `choose_action_batch` scores each candidate action $a \in \{0,1\}$ for every episode with a
  two-outcome expectation, then takes the argmax:

$$
Q_t(s,a) = p_\text{succ}(w_t,a,s)\,\Big[r(s,a,s'_\text{succ}) + \mathbb{E}\,V_{t+1}(s'_\text{succ})\Big]
          + \big(1-p_\text{succ}\big)\,r(s,a,s_\text{fail})
$$

| Term | Meaning |
|---|---|
| $p_\text{succ}$ | [Transition model](transition-models.md) at the observed wind |
| $s'_\text{succ}$ | Next state from the continuous energy update, using the observed solar gain |
| $r(\cdot)$ | Observation reward × gain; on failure, minus the penalty ([Whale observation](whale-observation.md)) |
| $\mathbb{E}\,V_{t+1}$ | `value_function_batch` |

`value_function_batch` computes the expectation in one of two ways:

- **i.i.d. weather** (`n_bins == 1`): a direct lookup of the next-stage value table.
- **Weather Markov chain active:** a sum over next exogenous bins, weighted by the stage transition
  row of the current bin, $\sum_b P_t[\text{cur\_bin}, b]\,V_{t+1}(b, s')$. The provider supplies
  `cur_bins`, the joint wind–solar regime index. See [Wind chain](wind-chain.md) and
  [Solar chain](solar-chain.md).

At the final stage the value is the immediate reward alone. The discount is $\gamma = 1$, set by
`build_mdp`.

The optimal policy is always solved on the **distributional** environment provider. If
`historical_weather` is enabled, rollouts run on the historical-bootstrap provider while the
policy still uses the table solved on fitted distributions. See
[Weather modes](weather-modes.md).

## Threshold policy

`UnifiedThresholdContinuousSimulation(mdp, horizon, state0, observation_threshold, wind_threshold, ...)`.

**Low-battery reserve.** `calculate_low_battery_threshold` sets a SoC reserve equal to one
15-minute stage of cruise plus one of landing:

$$
\text{SoC}_\text{reserve} = 100\cdot\frac{(P_\text{cruise} + P_\text{landing})\cdot 900\ \text{s}}{E_\text{cap}}
$$

**Decision rule** (`choose_action_batch`), with $o_t$ the whale observation probability and $w_t$
the wind speed:

| Mode | Action 1 when | Otherwise |
|---|---|---|
| Moored (0) | $w_t < w_\text{th}$ **and** $o_t > o_\text{th}$ **and** SoC $> \text{SoC}_\text{reserve}$ **and** SoC $> 95$ → take off | stay moored |
| Flying (1) | none of the land conditions hold → continue | land if SoC $< \text{SoC}_\text{reserve}$ **or** $o_t < o_\text{th}$ **or** $w_t \le w_\text{th} - 3$ |

The threshold policy never solves anything, so it produces no `solver_tables/` entry.

Its thresholds come from the YAML lists:

```yaml
threshold_values: [0.25, 0.5]   # observation-probability thresholds o_th
wind_thresholds:  [8.0, 10.0]   # wind-speed thresholds w_th, m/s
```

When benchmarking against the optimal policies, `Scripts/compare_chain_sweep.py`
(`best_threshold_for_cell`) picks the threshold pair with the highest average reward in each
cell. It also runs a split-half check of that selection (select on even episodes, evaluate on
odd). See [Threshold benchmark](../design/wind-threshold-benchmark.md).

## Other policy classes

These classes exist in `simulation_base.py`, but `SimulationFactory` does not instantiate them:

| Class | Rule |
|---|---|
| `AlwaysFlySimulation` / `AlwaysFloatSimulation` | Constant action 1 / 0 (discrete-SoC `AbstractSimulation` base) |
| `ObservationThresholdContinuousSimulation` | Take off if $w < w_\text{th}$, $o > o_\text{th}$ and SoC $> 15$ |

## Episode outcomes

Each episode ends at the horizon or at the first transition into mode 2. `failure_type` records
how it ended:

| Value | Meaning |
|---|---|
| 0 | No failure |
| 1 | Crash: the action failed the success draw |
| 2 | Battery depletion: next energy $< 0$ |

`flight_hrs` is the number of stages with action 1, divided by 4. See
[Episode simulation](../architecture/simulation.md) and [Result storage](../architecture/storage.md).
