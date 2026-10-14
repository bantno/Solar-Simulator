# Episode simulation

Policies are evaluated by Monte Carlo rollouts.

- **Rollout code:** `BaseClasses/simulation_base.py`
  (API: [simulation_base](../api/baseclasses/simulation_base.md)).
- **Orchestration:** `BaseClasses/simulation_run_manager.py`
  (API: [simulation_run_manager](../api/baseclasses/simulation_run_manager.md)).

## Simulation classes

| Class | Policy | Built by |
|---|---|---|
| `OptimalContinuousAnalyticalPolicySimulation` | Optimal: one-step Bellman lookahead on the solved value table | `SimulationFactory.create_simulation("optimal", ...)` |
| `UnifiedThresholdContinuousSimulation` | Threshold rules on observation probability, wind and battery | `SimulationFactory.create_simulation("threshold", ...)` |

Both derive from `AbstractContinuousEnergySimulation`, which owns the rollout loop.

- **Where the solve happens.** The optimal class calls `mdp_solver.solve()` in its
  constructor, so a simulation object only exists once its value table is ready.
- **Other classes.** `ObservationThresholdContinuousSimulation`, `AlwaysFlySimulation`
  and `AlwaysFloatSimulation` are also defined in the module, but the factory doesn't build
  them.

The decision rules are covered in [Policies](../models/policies.md). In brief:

- **Optimal.** For each candidate action it computes the success probability $p_s$ at the
  realized wind, and the next state if the action succeeds at the realized solar gain. It
  scores the action as
  $$p_s\,\big(r_\text{success} + V_{t+1}(s')\big) + (1 - p_s)\, r_\text{failure}$$
  and takes the argmax. With a weather chain, $V_{t+1}$ is weighted by the transition row of
  the lane's current regime (`cur_bins`).
- **Threshold.**
    - Takeoff from moored requires all three of:
        - wind below `wind_threshold`;
        - observation probability above `observation_threshold`;
        - SoC above both 95 % and a low-battery reserve.
    - A flying aircraft lands if the battery is below the reserve, the observation
      probability is below the threshold, or wind ≤ `wind_threshold − 3`.

## Per-stage order of operations

For each stage $t$, `simulate_episode_batch` does the following for every still-active lane
(episode):

1. **Sample weather.** Draw solar energy, wind speed and the whale-observation value from the
   environment provider for **all** $n$ lanes, then keep the active ones. Sampling all lanes
   means lane $i$ always receives draw $i$, whichever other lanes have already failed.
2. **Read the regime.** If a weather chain is active, read each lane's current regime from
   `env_provider.last_exo_bins`.
3. **Choose.** Call `choose_action_batch(state, solar, wind, whale, t, cur_bins)`. The policy
   sees the current stage's realized weather.
4. **Step.** `step_batch` calls
   `transition_continuous_energy_with_wind_and_energy` to update continuous energy and SoC and
   to apply the wind-dependent success draw, then computes the reward with `mdp.reward`.
5. **Record.** Accumulate reward and flight time. Lanes that reach `broken` get
   `failure_step = t + 1` and a `failure_type` (`2` if energy went negative, otherwise `1`),
   and are marked done so later stages leave them alone.

A lane that never fails has `failure_step = horizon` and `failure_type = 0`.

## Vectorized batches

- **One batch per simulation.** All episodes run together as `(n, ...)` arrays, so the Python
  loop runs over stages rather than over episodes × stages. The original per-episode loop
  (`simulate_episode`) is kept for reference and verification.
- **Measured speed-up.** `Tests/verify_runtime_opt.py` measured about 810× for threshold
  policies and about 1347× for the optimal policy at 2000 episodes.
- **Agreement with the original loop.** Aggregate statistics match the original loop within
  Monte Carlo error, not bit for bit, because the order of random draws changed.
- **Full history.** Per-step history is kept only for the first $K$ lanes: $K = n$ when
  `save_states: true`, otherwise $K = $ `full_history_episodes`. This bounds memory use.

`simulate_multiple_episodes(n)` runs the batch and yields one dict per episode:

| Key | Always | Full-history lanes only |
|---|---|---|
| `metadata.episode_index`, `total_reward`, `flight_hrs`, `failure`, `failure_step`, `failure_type` | ✓ | |
| `trajectory`, `actions`, `rewards`, `solar_series`, `wind_series`, `whale_series`, `energy_series` | | ✓ |

`flight_hrs` is the sum of `1` actions divided by 4, i.e. 15-minute stages.

## Randomness and pairing

- **Weather is seeded.** `simulate_multiple_episodes` calls `env_provider.reset(0)` before each
  batch, which re-seeds the provider's `np.random.default_rng(0)` and clears any chain state.
- **Paired runs share weather.** Every simulation in a run therefore draws the same weather
  sequence for the same location, start and horizon. For example, in historical mode every
  simulation sees the same bootstrap years episode for episode, which is what makes paired
  policy comparisons valid (`Scripts/compare_chain_sweep.py --verify` checks this).
- **Success draws are not seeded.** The action-success draw in `_apply_transition_probability`
  uses NumPy's global `np.random.rand`, not the provider's generator.
- **No bit-exact reproducibility.** The harness doesn't seed runs for bit-exact
  reproducibility. The copied `config.yaml` and the git SHA in `run_metadata.json` record what
  ran.

## SimulationRunManager

`SimulationRunManager(episodes_per_simulation, storage_dir, sim_name_prefix)` opens one
HDF5 file, `<prefix>_<YYYYmmdd_HHMMSS>.h5`, in `storage_dir`.

**`run_simulations`:**

- **Serial or parallel.**
    - `use_multiprocessing=False` runs simulations one after another in-process.
    - `True` uses a `multiprocessing.Pool` (`num_workers`, default CPU − 1) with `imap_unordered`.
    - Either way, each simulation's episodes run in a single worker (`_run_one_sim`).
- **Summary.** `_run_one_sim` computes the simulation-level summary: `average_reward`,
  `failure_percentage`, `average_failure_step`, `average_flight_hrs`, plus identifying metadata
  (type, capacity, horizon, penalty, thresholds, `location_id`, `start_time`).
- **Storage.** As each result arrives, the manager stores it as one group, named by
  `_make_group_name`, and flushes. See [Result storage](storage.md).

In the harness, simulations are first **built** in parallel. For optimal simulations that
includes the solve (`create_simulation_wrapper` over an `mp.Pool`). The list of built
simulations is then passed to `run_simulations`.
