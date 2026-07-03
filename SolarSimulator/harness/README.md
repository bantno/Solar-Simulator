# Simulation validation harness

A declarative driver for the Solar-Simulator. One **self-describing YAML** fully describes an
experiment (the sweep matrix + all behavior settings); the CLI expands and executes it, reusing
the existing `SimulationFactory` / `SimulationRunManager` / HDF5 stack, and collects everything
for that experiment into a single timestamped run directory.

It replaces the click-driven `gui.py` flow for validation runs: configs are committed artifacts,
runs are scriptable, and each run's config + data + summary + figures live together.

## Usage

Use the `pvlib` conda env (base Anaconda has a numpy/h5py ABI mismatch). Run from the
`SolarSimulator` dir (or anywhere — the script puts itself on `sys.path`):

```bash
conda run -n pvlib python harness/run_experiment.py experiment harness/examples/iid_small.yaml
conda run -n pvlib python harness/run_experiment.py experiment harness/examples/chain_small.yaml
conda run -n pvlib python harness/run_experiment.py experiment ./my_configs --workers 12
```

Options: `--out <dir>` (storage base), `--workers N`, `--no-multiproc`, `--pattern` (when the
path is a directory of configs).

`regression` and `perf` are reserved subcommands (not implemented yet — see *Extending* below).

## Output layout

```
<storage_dir>/<config_basename>/<YYYYmmdd_HHMMSS>/
    config.yaml            # copy of the resolved spec (paths absolutized)
    <config_basename>_*.h5 # raw episode data, one HDF5 group per simulation
    summary.csv            # tidy metrics, one row per simulation
    run_metadata.json      # git SHA, timestamp, command, harness version
    solver_tables/         # value-function .npy files (redirected here, not cwd)
    figures/
        sweep_<metric>.png      # metric vs each swept parameter
        trajectory_<arm>_<cap>.png  # real-weather replay of a representative sim
```

Re-running the same config creates a **new timestamped sub-run**; prior runs are never clobbered.
`<storage_dir>` defaults to `config['storage_dir']` or `<repo>/results`.

## Config schema

The harness keeps the existing YAML schema. Any value given as a **list is a swept dimension**;
the full cross-product of the matrix below is run (× optimal and/or threshold policies).

| Key | Meaning |
|---|---|
| `experiment_name`, `description` | Free text, copied into `run_metadata.json`. |
| `start_datetime` *(or `start_datetimes`)* | Mission start; must exist in the location data. |
| `battery_capacities` | List, Wh. |
| `threshold_values` | List, observation thresholds (threshold policy). |
| `wind_thresholds` | List, m/s (threshold policy). |
| `horizons` | List, time steps. |
| `failure_penalties` | List. |
| `episodes` | Monte-Carlo episodes per simulation. |
| `transition_model` | e.g. `moderate`. |
| `solar_panel_model` | e.g. `constant`. |
| `whale_series` | e.g. `real`. |
| `energy_increment_wh` | SoC grid step (Wh) for the solver. |
| `delta_t` | Minutes per step (default 15). |
| `include_optimal` | Whether to also run the optimal-policy sims (default true). |
| `save_states`, `full_history_episodes` | Full per-step history retention. |
| `locations` | List of `{latitude, longitude, data_path}`. |
| `wind_chain` | `{enabled: bool, path: <optional>}`. `enabled: false` → i.i.d. wind. `path` defaults to the data_path with `_windchain` inserted. |
| `paper_figure` | *(optional)* Name of a function in `Figures/Scripts/generate_journal_paper_figures.py` (e.g. `fig_threshold_sweep_combined`). When set, the harness reproduces that exact paper figure from this run's HDF5 instead of the generic summary plots. |

**Self-describing:** unlike the GUI/`run_batch.py` path, `include_optimal`, `save_states`, and
`full_history_episodes` are read from the YAML — no behavior comes from CLI flags. i.i.d. and
chain experiments are kept as **separate configs** (see `examples/`).

**Chain-vs-IID evaluation sweep:** `Scripts/generate_chain_sweep_configs.py` emits paired
configs (identical except `wind_chain.enabled` / `historical_weather`) into
`configs/chain_vs_iid_sweep[_smoke]/` plus a manifest; `Scripts/run_chain_sweep.py` provisions
location data and runs them all (`--resume` to continue an interrupted sweep);
`Scripts/compare_chain_sweep.py` joins the pairs into per-cell deltas with paired bootstrap CIs
(hist-world pairs see identical bootstrap weather — verified with `--verify`), CVaR tail metrics,
and native-vs-historical calibration gaps; `Scripts/plot_chain_sweep.py` draws the evaluation
figures (capacity–reliability frontier, delta panels, reward CDFs, benefit-vs-persistence,
calibration, storm-onset composites, value-table diagnostics, three-way policy comparison).

**Threshold-policy benchmark:** `generate_chain_sweep_configs.py --thresholds` additionally
emits one arm-agnostic hist-world config per sweep scenario running the full
observation-threshold × wind-threshold grid (no value-function solve; the bootstrap weather is
identical to both optimal arms episode-for-episode, so three-way paired comparisons hold).
These configs store scalars only (`full_history_episodes: 0`); after each one the runner
compacts the HDF5 into the `_episode_scalars.csv` analysis cache (a 210-sim × 3000-episode
config is ~1.7 GB raw vs ~40 MB compacted; disable with `--keep-h5`). The comparison script
selects the per-cell best combo by mean reward (with a split-half selection-bias check and a
min-failure envelope) and reports optimal-vs-threshold paired deltas; figures 1, 3, and 8 show
the threshold benchmark in green.

**Wind-bin selection:** sweep configs specify `wind_chain: {n_bins: N}` with no `bin_edges`
key, which builds equal-occupancy (quantile) bins per location from its historical record at
artifact-build time; explicit `bin_edges` (interior cutpoints, m/s) are still honored.
Provisioning rebuilds a stale chain artifact whenever the configured binning differs from the
one on disk (edge array mismatch, bin-count mismatch, or quantile-vs-explicit mode mismatch —
artifacts record a `quantile_derived` flag). Because quantile edges differ per location, the
analysis scripts read edges from each location's `*_windchain.pkl`
(`compare_chain_sweep.chain_edges_for`), never from the sweep manifest.

## `summary.csv`

One row per simulation, produced by `summarize.summarize_hdf5` from the HDF5 group attrs plus
experiment-level columns: `simulation_type, battery_capacity, horizon, failure_penalty,
observation_threshold, wind_threshold, location_id, start_time, episodes_count,
failure_percentage, average_failure_step, average_reward, average_flight_hrs,
wind_chain_enabled, transition_model, solar_panel_model, whale_series, energy_increment_wh,
config_basename, solver_backend, rollout_backend`.

This is the stable seam a future regression mode snapshots/diffs.

## Figures

- **`paper_figure` set** → the harness reuses the journal figure script's own plotting functions
  on this run's HDF5 (its `load_summary` reads the same group attrs the harness writes), so the
  output **matches `generate_journal_paper_figures.py` by construction**. Currently wired:
  `fig_threshold_sweep_combined`, `fig_capacity_mean_reward`, `fig_optimal_capacity_by_location`
  (extend `_PAPER_FIGURE_H5_KEY` in `figures.py` for more).
- **otherwise** → a generic family-of-curves summary: each metric vs the primary swept parameter,
  one line per secondary-parameter value (no averaging), with the optimal policy overlaid.
- Plus a real-weather **trajectory replay** for a representative sim, when historical data exists.

## What this harness does NOT do

- Run the old scalar solver or batched-rollout's scalar baseline — **vectorized path only**.
- Reuse a pre-solved value table — every run **solves fresh** (value-table `.npy` is written to
  `solver_tables/`).
- Seed for bit-exact reproducibility — deferred (the config copy + git SHA capture the run).

## Extending (reserved seams)

- **`regression` mode:** collect `Tests/verify_*.py` under pytest + golden-file diff of
  `summary.csv`.
- **`perf` mode:** pytest-benchmark around `solve()` / rollouts.
- **Old-vs-new solver comparison:** the `solver_backend` / `rollout_backend` columns
  (`vectorized` / `batched` today) are reserved so a future `scalar` backend can be run and its
  `summary.csv` diffed against the vectorized one — additive, no schema change.
