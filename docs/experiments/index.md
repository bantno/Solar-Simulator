# Simulation validation harness

!!! info "Where this page came from"
    This page was `SolarSimulator/harness/README.md`. The pages next to it go deeper: every YAML key
    is in the [Config reference](config-reference.md), every output file in [Outputs](outputs.md),
    and the multi-config pipelines in [Sweeps](sweeps.md).

A declarative driver for the Solar-Simulator. One **self-describing YAML** fully describes an
experiment (the sweep matrix + all behavior settings); the CLI expands and executes it, reusing
the existing `SimulationFactory` / `SimulationRunManager` / HDF5 stack, and collects everything
for that experiment into a single timestamped run directory.

It replaces the click-driven `gui.py` flow ([GUI and tools](gui-and-tools.md)) for validation runs: configs are committed artifacts,
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

`regression` and `perf` are reserved subcommands (not implemented yet — see [Extending](#extending-reserved-seams) below).

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
`<storage_dir>` defaults to `config['storage_dir']` or `<repo>/results`. Each file is described in
[Outputs](outputs.md).

**HDF5 layout:** each simulation is one top-level group. Per-episode scalars
(`total_reward`, `failure`, `failure_step`, `flight_hrs`, `failure_type`, `episode_index`)
are stored as one 1-D `(episodes,)` dataset per field under `<group>/episode_scalars/`;
`<group>/episodes/episode <i>` groups exist only for full-history episodes and hold their
array series (`wind_series`, `trajectory`, ...). Older files stored every episode as its own
group of scalar datasets, which cost ~0.5 KB of HDF5 metadata per object (a scalars-only
210-sim x 3000-episode run was ~1.6 GB for ~40 MB of content); all readers
(`compare_chain_sweep.read_all_episode_scalars`, the GUI, and the `Figures/Scripts` loaders)
still fall back to that legacy layout. Full details: [Result storage](../architecture/storage.md).

## Config schema

The harness keeps the existing YAML schema. Any value given as a **list is a swept dimension**;
the full cross-product of the matrix below is run (× optimal and/or threshold policies). This table
is a summary; the [Config reference](config-reference.md) lists every key with its type, default and
the code that reads it (including `solar_chain` and `historical_weather`).

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
| `solar_panel_model` | `constant` (default), `pvwatts` (pvlib PVWatts DC at fixed cell temperature; defaults reproduce `constant` exactly), or `variable`. |
| `solar_panel_kwargs` | *(optional)* Mapping passed to the panel constructor, e.g. `{pdc0: 65.0, gamma_pdc: -0.003, cell_temperature: 45.0}` for `pvwatts`. |
| `clearsky_model` | Clear-sky envelope for the Beta/chain normalization: `fatemi` (default, A·cos z — the original pipeline), `haurwitz`, or `ineichen` (pvlib). Non-default models resolve `data_path` and its derived chain/cube artifacts to `_<model>`-suffixed files, built on demand, so fatemi artifacts are never overwritten. |
| `whale_series` | `JoA2026` (default; the journal-results profile: Scott et al. 2001 Table 2 surface proportions with 20:00-06:00 zeroed — formerly `real`, still accepted with a deprecation warning), `encounter`, `custom`, `constant`, or `sinusoidal`. Block-based series are expanded with `delta_t` and rolled so stage 0 holds the value for `start_datetime`'s time of day. |
| `whale_block_values`, `whale_block_hours` | For `whale_series: custom`: per-block-of-day observation values starting at 00:00 (each in [0, 1]) and the block length in hours (default 2); `len(values) * hours` must be 24. Experiment-wide, not a swept dimension. |
| `whale_species` | `encounter` only. A species card: a built-in name (`kogia_spp`, `humpback`; see `BaseClasses/whale_base.py::SPECIES_CARDS` for the cited numbers) or a mapping naming a card with overrides, e.g. `{name: kogia_spp, g0: 1.0}`, or a bare mapping with `availability_profile` (`scott2001`, a scalar, or a 12-block list) and `g0`. Default `kogia_spp`. |
| `whale_platform` | `encounter` only, required when a location has a density: `{strip_width_km, ground_speed_ms}` — swept area per stage `A = strip_width_km · ground_speed_ms · delta_t`. |
| `whale_light` | `encounter` only. The camera's daylight gate: `{mode: fixed_hours, day_start: 6, day_end: 20}` (default; reproduces the JoA2026 night zeros), `{mode: clearsky_gate, threshold_wm2: 0}` (sun above the horizon from the location's clear-sky series), or `{mode: none}`. |
| `locations[].density_per_km2` | `encounter` only, optional. Animal density `D` for the location: a scalar, 12 monthly values, or `{month: value}` (picked by `start_datetime`'s month). Given: `λ = D·A·a(t)·g0·light(t)`, `p = 1 − exp(−λ)`. Omitted: the focal form `p = a(t)·g0·light(t)` (a whale assumed under the aircraft; with `g0: 1.0` this is `JoA2026`). |
| `takeoff_cruise_fraction`, `landing_cruise_fraction` | Fraction of the takeoff (moored, fly) and landing (flying, land) stages spent surveying at cruise; multiplies the observation reward on those stages. Defaults `1.0` / `0.0` reproduce the original model (climb short relative to the stage). `auto` derives `1 - t_maneuver / delta_t` from the seaplane climb / descent model. |
| `energy_increment_wh` | SoC grid step (Wh) for the solver. |
| `delta_t` | Minutes per step (default 15). |
| `include_optimal` | Whether to also run the optimal-policy sims (default true). |
| `save_states`, `full_history_episodes` | Full per-step history retention. |
| `locations` | List of `{latitude, longitude, data_path}`. |
| `wind_chain` | `{enabled: bool, path: <optional>}`. `enabled: false` → i.i.d. wind. `path` defaults to the data_path with `_windchain` inserted. |
| `paper_figure` | *(optional)* Name of a function in `Figures/Scripts/generate_journal_paper_figures.py` (e.g. `fig_threshold_sweep_combined`). When set, the harness reproduces that exact paper figure from this run's HDF5 instead of the generic summary plots. |

**Self-describing:** unlike the GUI/`run_batch.py` path, `include_optimal`, `save_states`, and
`full_history_episodes` are read from the YAML — no behavior comes from CLI flags. i.i.d. and
chain experiments are kept as **separate configs** (see [Example configs](examples.md)).

**Chain-vs-IID evaluation sweep** (step-by-step in [Sweeps](sweeps.md))**:** `Scripts/generate_chain_sweep_configs.py` emits paired
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
writes the `_episode_scalars.csv` analysis cache (`--keep-h5` skips this). With the columnar
`episode_scalars` layout the HDF5 stays small and is kept; legacy group-per-episode files
(~1.7 GB for a 210-sim × 3000-episode config) are deleted once cached. The comparison script
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
wind_chain_enabled, transition_model, solar_panel_model, clearsky_model, whale_series,
whale_block_hours, whale_block_values, whale_species, whale_g0, whale_strip_width_km,
whale_ground_speed_ms, whale_density_per_km2, whale_light_mode, takeoff_cruise_fraction,
landing_cruise_fraction,
energy_increment_wh, config_basename, solver_backend, rollout_backend`.

The file also starts with a `group` column and includes `weather_mode`; see
[Outputs § summary.csv](outputs.md#summarycsv) for what each column means.

This is the stable seam a future regression mode snapshots/diffs.

## Figures

See also [Journal figures](figures.md).

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
