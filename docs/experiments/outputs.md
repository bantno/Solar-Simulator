# Outputs

What a harness run writes and how to read it. The layout inside the HDF5 file (groups, attributes, the columnar `episode_scalars/` datasets, full-history episode groups) is covered on [Result storage](../architecture/storage.md).

## Run folder

Each call to `run_experiment` creates one timestamped folder (`harness/run_experiment.py::run_experiment`):

```text
<storage_dir>/<config_basename>/<YYYYmmdd_HHMMSS>/
    config.yaml                      resolved spec (absolute paths, "_" keys removed)
    <config_basename>_<ts>.h5        all simulations of the run, one group each
    summary.csv                      one row per simulation
    run_metadata.json                reproducibility record
    solver_tables/*.npy              optimal-policy value tables
    figures/                         sweep summary (or paper figure) + trajectory replay
    _episode_scalars.csv             (sweep runs only) per-episode analysis cache
```

- `<storage_dir>` is the CLI `--out` if given, otherwise the config's `storage_dir`, otherwise `<repo>/results`.
- Re-running a config creates a new timestamped folder next to the old one. Earlier runs are never overwritten.
- The HDF5 timestamp `<ts>` comes from `SimulationRunManager` and is a few seconds later than the folder timestamp.
- `results/` is ignored by git.

### `config.yaml`

The config as it was executed. Two differences from the file you wrote:

- Every `data_path` and artifact `path` is made absolute.
- The harness-internal keys (`_run_output_dir`, `_config_basename`) are removed.

The file can be re-run directly with the harness.

### `solver_tables/`

One value table per optimal-policy simulation, written by `mdpAnalyticalBackwardSolver._vf_filename`:

```text
<config_basename>_<capacity>Wh_<horizon>h_<penalty>p_<start[0:12]>.npy
e.g. cvi_capgrid_lat30.0_lon-90.0_iid_hist_300.0Wh_2880h_5.0p_2020-07-01 0.npy
```

The name has no location in it. That is why the chain-vs-iid sweep writes one config per location.

- **i.i.d. solves** produce a table indexed by (state, stage).
- **Chain solves** add a leading exogenous-bin axis (see [Backward-induction solver](../architecture/solver.md)).

The tables are written for inspection and analysis only. Every run solves afresh and never reloads them. `Scripts/plot_value_function.py` plots one quickly.

## `summary.csv`

One row per HDF5 group, written by `harness/summarize.py::summarize_hdf5` / `write_summary_csv`. Rows are sorted by `simulation_type`, `battery_capacity`, `wind_threshold`, `observation_threshold`. Every row carries the experiment-level columns, so the file stands alone without a join. Treat the schema as stable: it is the seam a future regression mode will diff.

**Identity**

| Column | Meaning |
|---|---|
| `group` | HDF5 group name, e.g. `optimalcontinuousanalyticalpolicysimulation_c300_f5_h2880_lat30.0_lon-90.0_d183` |

**Per-simulation columns**

These are the HDF5 group attributes written by `SimulationRunManager._run_one_sim`.

| Column | Meaning |
|---|---|
| `simulation_type` | Class name, e.g. `OptimalContinuousAnalyticalPolicySimulation`, `UnifiedThresholdContinuousSimulation` |
| `battery_capacity` | Wh |
| `horizon` | Stages |
| `failure_penalty` | Penalty value |
| `observation_threshold`, `wind_threshold` | Threshold-policy knobs; empty for the optimal policy |
| `location_id` | `lat<lat>_lon<lon>` |
| `start_time` | Mission start |
| `episodes_count` | Episodes simulated |
| `failure_percentage` | Fraction of episodes that ended in failure, in [0, 1] |
| `average_failure_step` | Mean stage of termination. Episodes that did not fail count as the horizon (their `failure_step` equals the horizon). |
| `average_reward` | Mean total episode reward |
| `average_flight_hrs` | Mean flight hours per episode |

**Experiment-level columns**

These come from the config (`_experiment_columns`). Defaults match the ones in `run_sim.py`.

| Column | Source |
|---|---|
| `config_basename` | Config file name without extension |
| `weather_mode` | `historical` if `historical_weather.enabled`, else `chain` if `wind_chain.enabled`, else `iid` |
| `wind_chain_enabled` | `wind_chain.enabled` |
| `transition_model`, `solar_panel_model`, `clearsky_model` | Model keys |
| `whale_series`, `whale_block_hours`, `whale_block_values` | Whale series and custom blocks (lists joined with `;`) |
| `whale_species`, `whale_g0` | Species card name and its `g0` (config override, else the built-in card value) |
| `whale_strip_width_km`, `whale_ground_speed_ms` | From `whale_platform` |
| `whale_density_per_km2` | Each location's `density_per_km2`, joined with `;` |
| `whale_light_mode` | `whale_light.mode` |
| `takeoff_cruise_fraction`, `landing_cruise_fraction` | Cruise-fraction keys, as written in the config (`auto` stays `auto`) |
| `energy_increment_wh` | SoC grid step |
| `solver_backend`, `rollout_backend` | Reserved; `vectorized` / `batched` |

!!! note
    CSV files written before a column existed don't have it. For example, older sweep outputs have no whale columns. They may also have `whale_series` written as `real`, the old name of `JoA2026`.

## `run_metadata.json`

Written last by `run_experiment`:

| Field | Meaning |
|---|---|
| `harness_version` | `harness.HARNESS_VERSION` |
| `git_sha` | `git rev-parse HEAD` at run time (`unknown` if git is unavailable). Uncommitted changes are **not** recorded. |
| `timestamp` | Run-folder timestamp |
| `command` | The command line (`sys.argv`) |
| `config_basename`, `experiment_name`, `description` | Identity |
| `n_simulations` | Simulations built and run |
| `episodes_per_simulation` | `episodes` |
| `hdf5` | HDF5 file name in the run folder |

There is no seeding mechanism for bit-exact reproduction beyond this record. Rollouts do, however, reset the environment provider with a fixed seed per batch, which is what makes paired runs share weather (see [Episode simulation](../architecture/simulation.md)).

## `figures/`

Written by `harness/figures.py` (see [Journal figures](figures.md) for the paper path):

| File | When | Content |
|---|---|---|
| `sweep_<metric>.png` | `paper_figure` not set | `average_reward`, `failure_percentage` and `average_flight_hrs` against the swept parameter with the most distinct values. There is one line per value of the second swept parameter, with no averaging. The optimal policy is overlaid, as a horizontal reference when the x-axis is a threshold knob. |
| Paper figure file(s) | `paper_figure` set | The named journal figure, redrawn from this run's HDF5 |
| `trajectory_<label>_<cap>Wh.png` | The run has full-history episodes | Wind, SoC and actions over mission time for the median-reward saved episode of a representative simulation (the first optimal one). `<label>` is `chain` or `i.i.d.`. |

The trajectory figure needs `full_history_episodes > 0` or `save_states: true`. Without either it is skipped with a warning.

## Sweep analysis outputs

The chain-vs-iid tooling ([Sweeps](sweeps.md)) writes extra files under its results base (`results/chain_vs_iid_sweep[_smoke]/`):

| Path | Written by | Content |
|---|---|---|
| `sweep_run_log.json` | `run_chain_sweep.py` | List of `{config, run_dir, seconds}` for each config run |
| `<config>/<ts>/_episode_scalars.csv` | `run_chain_sweep.py` via `compare_chain_sweep.read_all_episode_scalars` | One row per episode: `group, episode_index, total_reward, failure, failure_step, flight_hrs`. It is a cache of the HDF5's per-episode scalars and is read by the comparison. |
| `_analysis/comparison_cells.csv` | `compare_chain_sweep.py` | One row per (sweep, location, capacity, start, horizon, world) cell. It holds each arm's metrics (`failure_pct_*`, `avg_reward_*`, `avg_flight_hrs_*`, `cvar10_*`), the paired deltas with CI bounds (`d_<metric>`, `_lo`, `_hi`), and the threshold-benchmark columns (`thresh_best_*`, `*_vs_thresh`). |
| `_analysis/comparison_report.md` | `compare_chain_sweep.py` | The same comparisons as readable Markdown tables, one section per sweep, plus the optimal-vs-best-threshold table |
| `_analysis/figures/fig<N>_*.png` | `plot_chain_sweep.py` | Figures 1–8 (see [Sweeps § Plot](sweeps.md#4-plot)) |
| `_analysis/figures/persistence_stats.json` | `plot_chain_sweep.py` | Per-location wind-persistence statistics used by figure 4 |

## Reading results in Python

- **Per-simulation table:** read `summary.csv` with pandas, or build the same table from one or more HDF5 files with `Figures/Scripts/results_io.load_summary` (see [Journal figures](figures.md#results_io)).
- **Per-episode data:** use `Scripts/compare_chain_sweep.read_all_episode_scalars(run_dir)`. It reads the columnar `episode_scalars/` layout, falls back to the legacy group-per-episode layout, and caches the result as `_episode_scalars.csv`.
- **Interactive browsing:** the results viewer in `gui/combined_gui.py` opens the same HDF5 files (see [GUI and tools](gui-and-tools.md)).
