# Weather modes

The simulator has three ways of generating weather. They differ in **what the optimal policy
is solved against** and **what the episodes are rolled out on**. Each mode is a combination
of config blocks, not a single switch.

| Mode | Config | Solve on | Roll out on |
|---|---|---|---|
| **iid** (published model) | every chain and `historical_weather` off | Per-stage fitted distributions, independent in time | The same fitted distributions |
| **chain** | `wind_chain.enabled` and/or `solar_chain.enabled` | Fitted distributions plus a Markov chain over weather bins | The same chain-modulated distributions |
| **historical** | `historical_weather.enabled` (combines with either of the above) | Whatever the chain settings say (iid or chain) | Real historical weather, block-bootstrapped |

!!! note "Solve and evaluate are separate"
    `historical_weather` never changes the value function. `SimulationFactory` always builds
    the distributional provider (`self.env_provider`) and solves the MDP against it. With
    `historical_weather` enabled it builds a second provider, `self.sim_env_provider`, and
    `create_simulation` uses that one for rollouts
    ([`run_sim`](../api/baseclasses/run_sim.md)). This is how a policy solved under a
    model's assumptions is tested on weather that does not satisfy them.

## iid: fitted distributions

Provider: `StochasticWindSolarEnvironmentProvider` in
[`environment_provider_base`](../api/baseclasses/environment_provider_base.md), with no
transition arrays.

`EnvironmentLoader` finds the row of the expected-data file that matches `start_datetime`
(month, day, hour, minute). It then slices `horizon` consecutive rows, wrapping at the end
of the year. For each stage $t$:

- wind $W_t \sim \text{Weibull}(k_t, \lambda_t)$ from `weibull_k` / `weibull_scale`;
- irradiance $G_t = K_t \cdot G_{cs,t}$ with $K_t \sim \text{Beta}(\alpha_t, \beta_t)$;
- harvested energy $G_t \cdot \Delta t \cdot A_{panel} \cdot \eta_{panel}$ from the
  [solar panel model](solar-panel.md);
- the whale observation value from the [whale series](whale-observation.md), which is the
  same deterministic value on every lane.

Draws are independent across stages. The solver integrates over these distributions
analytically; see [Solver](../architecture/solver.md).

## chain: Markov-modulated weather

The same provider, given per-stage transition stacks. The exogenous state gains a discrete
**regime** $z$:

- wind only: $z = b$, the wind bin ([Wind persistence chain](wind-chain.md));
- solar only: $z = g$, the clear-sky-index bin ([Solar persistence chain](solar-chain.md));
- both: $z = b \cdot n_g + g$, with transition $P_t = P^{wind}_t \otimes P^{solar}_t$
  (`get_exo_transition`, which assumes the two chains are independent).

On each step the provider samples the continuous value inside the lane's current bin,
exposes the bin (`last_exo_bins`), and then advances the bin with the stage's transition
matrix. The marginal distribution at each stage is unchanged; only the correlation over
time is added. With one bin per chain the model reduces exactly to iid.

## historical: block bootstrap on real weather

Provider: `HistoricalBootstrapEnvironmentProvider`, reading the calendar cube
(see [Weather data pipeline](weather-data.md)).

- The episode keeps the mission's calendar dates. The horizon is cut into consecutive blocks
  of `block_length_days` (default 7).
- Each (episode lane, block) pair draws a random year from the record. Wind and solar
  use the **same** year for that block, so their real cross-correlation is kept.
- Values are read straight from the cube at the stage's calendar slot. Memory is
  $O(n \cdot n_{blocks})$ (a year-assignment matrix), not $O(H \cdot n)$
  (commit `b40ae34`).
- The start date cannot be Feb 29, which is not in the cube.

When the policy was solved with a chain, the factory passes the chain's bin edges, transition
stacks, and stage Beta parameters to the historical provider. Realized weather is then
**digitized into bins** so the chain-solved value table can be looked up:

- wind: `np.digitize` against the chain's interior edges;
- solar: $\lfloor n_g \cdot F_{\text{Beta}}(K) \rfloor$ using the same stage Beta as the solve,
  at solar-valid stages. At night the previous bin is held. A mission that starts at night
  draws a uniform initial bin.

Because the rollout exposes bins the same way in both modes, `simulate_episode_batch` and
`choose_action_batch` run unchanged on distributional and historical weather.

```yaml
historical_weather:
  enabled: true
  block_length_days: 7      # default
  # path: optional; default <data_path>_histcube.pkl
```

## Reproducibility and paired comparisons

`simulate_multiple_episodes` calls `env_provider.reset(0)` before each simulation, so every
simulation in a run starts from the same seed. Two simulations that differ only in policy,
such as optimal and threshold, or iid-solved and chain-solved on historical weather, therefore
see the same weather sequence episode for episode. This is common random numbers.

The chain-vs-iid sweep relies on this pairing, and `Scripts/compare_chain_sweep.py --verify`
checks it (see [Sweeps](../experiments/sweeps.md)).

## Choosing a mode

| Goal | Mode |
|---|---|
| Reproduce the published (journal) results | iid |
| Let the policy anticipate weather persistence | chain (wind first; solar is experimental) |
| Judge any policy on realistic, correlated weather | historical, with the solve mode of your choice |
| Headline comparison of iid-solved vs chain-solved | historical with and without `wind_chain`, same seed |

Example configs: `iid_small.yaml`, `chain_small.yaml`, `solarchain_small.yaml`,
`historical_smoke.yaml`, `journal_threshold_vs_optimal_historical.yaml`,
`journal_threshold_vs_optimal_chain_historical.yaml`
([Example configs](../experiments/examples.md)).
