# Config reference

An experiment is described entirely by one YAML file. This page lists every key the code reads, its type and default, what it controls, and where it is read. There is no schema validator. Each key is read with `config.get(key, default)` at the point of use, so an unknown key is silently ignored and a wrong value fails only when the code that uses it runs.

Keys are read in three places:

| File | What it reads |
|---|---|
| `SolarSimulator/BaseClasses/run_sim.py` | The sweep matrix (`YAMLSimulationRunner`), model settings (`SimulationFactory.__init__`), whale options (`_whale_kwargs_from_config`), and the optional artifact blocks (`_load_wind_chain`, `_load_solar_chain`, `_load_historical_cube`). |
| `SolarSimulator/harness/run_experiment.py` | Output location, behavior flags, path resolution (`_resolve_paths_in_place`), artifact provisioning (`_ensure_location_data`), and `paper_figure`. |
| `SolarSimulator/harness/summarize.py` | Experiment-level columns for `summary.csv` (`_experiment_columns`). It repeats several defaults from `run_sim.py`. |

API: [`BaseClasses.run_sim`](../api/baseclasses/run_sim.md), [`harness`](../api/harness.md).

## Sweep semantics

The keys marked **swept** below take lists. `YAMLSimulationRunner._build_param_list` runs the full cross-product of them:

```text
for start in start_datetimes:
    threshold sims:  locations × horizons × failure_penalties × battery_capacities
                     × threshold_values × wind_thresholds
    optimal sims:    locations × horizons × failure_penalties × battery_capacities
                     (only if include_optimal)
```

- Both threshold lists default to `[]`, so a config that omits either one runs **no** threshold-policy simulations.
- The optimal policy ignores the thresholds, so it is built once per (location, horizon, penalty, capacity) cell.
- Every simulation in the product is one HDF5 group and one row of `summary.csv` (see [Outputs](outputs.md)).
- The other keys are experiment-wide scalars, including all the whale options. To compare two model settings, write two configs; the repo keeps i.i.d. and chain experiments as separate files (see [Example configs](examples.md)).

## Experiment identity and output

| Key | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `experiment_name` | str | — | Free text, copied into `run_metadata.json`. | `run_experiment.py::run_experiment` |
| `description` | str | — | Free text, copied into `run_metadata.json`. | `run_experiment.py::run_experiment` |
| `storage_dir` | path | `<repo>/results` | Output root. The run is written to `<storage_dir>/<config_basename>/<YYYYmmdd_HHMMSS>/`. The CLI `--out` flag takes precedence. | `run_experiment.py::run_experiment` |
| `paper_figure` | str | unset | Name of a function in `Figures/Scripts/generate_journal_paper_figures.py`. When set, the harness reproduces that journal figure from this run's HDF5 instead of the generic sweep plots. See [Journal figures](figures.md). | `run_experiment.py::run_experiment` |

The config's own file name (without extension) is the **config basename**. It names the run folder, the HDF5 file, the value-table files, and the `config_basename` column.

## Sweep matrix

| Key | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `start_datetime` | ISO datetime str | required unless `start_datetimes` is given | Mission start. It must match a (month, day, hour, minute) row in the location's expected-data file (`EnvironmentLoader._find_start_index`); the year is not matched. Its time of day also aligns block-based whale series, and its month selects monthly whale densities. | `SimulationFactory.__init__` |
| `start_datetimes` | list of str | `[start_datetime]` | **Swept.** Several mission starts. Each value is written into `start_datetime` in turn. | `YAMLSimulationRunner.__init__` |
| `battery_capacities` | list of float (Wh) | required | **Swept.** Battery capacity. | `YAMLSimulationRunner._build_param_list` |
| `horizons` | list of int (steps) | `[horizon]` | **Swept.** Mission length in stages of `delta_t` minutes. | `YAMLSimulationRunner.__init__` |
| `horizon` | int | — | Single-horizon form, used when `horizons` is absent. | `YAMLSimulationRunner.__init__` |
| `failure_penalties` | list of float | `[failure_penalty]` | **Swept.** Penalty applied on the absorbing failure. | `YAMLSimulationRunner.__init__` |
| `failure_penalty` | float | `15` | Single-penalty form, used when `failure_penalties` is absent. | `YAMLSimulationRunner.__init__` |
| `threshold_values` | list of float | `[]` | **Swept.** Observation-probability thresholds for the threshold policy. | `YAMLSimulationRunner._build_param_list` |
| `wind_thresholds` | list of float (m/s) | `[]` | **Swept.** Wind-speed thresholds for the threshold policy. | `YAMLSimulationRunner._build_param_list` |
| `locations` | list of mappings | required | **Swept.** One entry per location (see [Locations](#locations)). | `YAMLSimulationRunner.__init__` |

## Run behavior

| Key | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `episodes` | int | `3000` | Monte-Carlo episodes per simulation. | `run_experiment.py::run_experiment` |
| `include_optimal` | bool | `true` | Also build and solve the optimal-policy simulations. | `YAMLSimulationRunner._build_param_list` |
| `save_states` | bool | `false` | Keep full per-step history for **every** episode. | `run_experiment.py::run_experiment` |
| `full_history_episodes` | int | `0` (harness) | Keep full per-step history for the first *K* episodes only; the rest store scalars. The trajectory-replay figure needs *K* > 0. | `run_experiment.py::run_experiment` |
| `delta_t` | int (minutes) | `15` | Stage length. Used by the MDP, the whale series, the historical block length, and artifact provisioning (`interval_minutes`). | `SimulationFactory.__init__`, `_ensure_location_data` |
| `energy_increment_wh` | float (Wh) | required | Step of the state-of-charge grid used by the solver and the rollouts. It is converted to a percentage of capacity per simulation: `soc_inc = energy_increment_wh / capacity × 100`. | `SimulationFactory.build_mdp` |

See [Result storage](../architecture/storage.md) for how history retention changes the HDF5 layout.

## Aircraft and solar models

| Key | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `transition_model` | str | `moderate` | Action success-probability model. Case-insensitive. One of `moderate`, `linear`, `nofail`, `realistic`, `optimistic`, `test`, `some`, `nowind`, or `wind based` (`transition_model_base.py::ProbabilityModelFactory.models`). See [Transition models](../models/transition-models.md). | `SimulationFactory.__init__` |
| `solar_panel_model` | str | `constant` | `constant`, `pvwatts`, or `variable` (`solar_panel_base.py`, registered by `model_name`). With default settings `pvwatts` reproduces `constant` exactly. See [Solar panel](../models/solar-panel.md). | `SimulationFactory.__init__` |
| `solar_panel_kwargs` | mapping | none | Passed to the panel constructor. For `pvwatts`: `area` (0.65), `pdc0` (65.0), `gamma_pdc` (-0.003), `cell_temperature` (25.0). For `variable`: `area`, `base_efficiency`, `degradation`. | `SimulationFactory.__init__` |
| `clearsky_model` | str | `fatemi` | Clear-sky envelope used to normalize irradiance for the Beta fits and the solar chain: `fatemi`, `haurwitz`, or `ineichen`. Any value other than `fatemi` resolves to suffixed artifact files (see [Artifact paths](#artifact-paths)). See [Clear-sky models](../models/clearsky.md). | `SimulationFactory.__init__`, `_ensure_location_data` |
| `takeoff_cruise_fraction` | float in [0, 1] or `auto` | `1.0` | Fraction of the takeoff stage spent surveying at cruise; it multiplies the observation reward on that stage. `auto` computes `max(0, 1 − climb_time_s / (delta_t·60))` from `Seaplane.get_maneuver_times_s`. | `SimulationFactory._resolve_cruise_fractions` |
| `landing_cruise_fraction` | float in [0, 1] or `auto` | `0.0` | The same for the landing stage, using `descent_time_s`. | `SimulationFactory._resolve_cruise_fractions` |

The cruise-fraction defaults (`1.0` / `0.0`) reproduce the original model: the takeoff stage earns the full observation reward and the landing stage earns none. `AbstractMDP.__init__` rejects values outside [0, 1].

## Whale observation

All of these keys are experiment-wide. `_whale_kwargs_from_config` maps them onto the arguments of `WhaleRewardSeriesFactory.create_series` through the `_WHALE_OPTION_KEYS` table. See [Whale observation](../models/whale-observation.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `whale_series` | str | `JoA2026` | `JoA2026`, `custom`, `encounter`, `constant`, or `sinusoidal`. `real` is still accepted as the old name of `JoA2026`, with a deprecation warning. |
| `whale_block_values` | list of float in [0, 1] | built-in JoA2026 profile | `custom` only. Observation value for each time-of-day block, starting at 00:00. `len(values) × whale_block_hours` must equal 24. |
| `whale_block_hours` | float | `2.0` (`whale_base.DEFAULT_BLOCK_HOURS`) | Block length for the block-based series. |
| `whale_species` | str or mapping | `kogia_spp` | `encounter` only. A built-in species card name (`kogia_spp`, `humpback`; `whale_base.SPECIES_CARDS`), a mapping that names a card and overrides fields (`{name: kogia_spp, g0: 1.0}`), or a bare mapping with `availability_profile` and `g0`. |
| `whale_platform` | mapping | — | `encounter` only. `{strip_width_km, ground_speed_ms}`; the area swept per stage is `A = strip_width_km · ground_speed_ms · delta_t`. Required when any location sets `density_per_km2`. |
| `whale_light` | mapping | `{mode: fixed_hours, day_start: 6, day_end: 20}` | `encounter` only. Daylight gate: `fixed_hours`, `clearsky_gate` (with `threshold_wm2`), or `none`. |

Block-based series are expanded to `delta_t` resolution and rotated so that stage 0 holds the value for `start_datetime`'s time of day. The same series is used for solving and for historical rollouts.

## Locations

Each entry in `locations` is a mapping:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `data_path` | path | required | Expected-data artifact (fitted Weibull/Beta parameters per 15-min slot), normally `Data/EXPECTED_DATA/data_expected_lat<lat>_lon<lon>_15min.pkl`. A relative path is resolved against the repo root. |
| `latitude`, `longitude` | float | `0.0` in the factory | Used by the seaplane model and by artifact provisioning. Provisioning refuses to build missing artifacts if either is unset. |
| `timezone` | str | `UTC` | Passed to `Seaplane`. |
| `density_per_km2` | float, list of 12, or `{month: value}` | unset | `encounter` only. Animal density *D*. The month is taken from `start_datetime`. When unset, the encounter model uses its focal form. |

A top-level `data_path` / `latitude` / `longitude` / `timezone` is accepted as the single-location form when `locations` is absent.

## Weather persistence and historical rollouts

These three optional blocks select the weather mode (see [Weather modes](../models/weather-modes.md)). All default to off, which gives i.i.d. weather sampled from the fitted distributions.

### `wind_chain`

| Field | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `enabled` | bool | `false` | Solve with the Markov-modulated wind chain. | `run_sim.py::_load_wind_chain` |
| `path` | path | derived (`_windchain`) | Explicit artifact path. With an explicit path the artifact is never rebuilt automatically. | `_load_wind_chain`, `_ensure_location_data` |
| `bin_edges` | list of float (m/s) | unset | Interior cut points. The artifact stores `[0, *bin_edges, ∞]`. Leave it out to use equal-occupancy (quantile) bins. | `run_experiment.py::_configured_bin_edges` |
| `n_bins` | int | `3` | Number of quantile bins when `bin_edges` is absent. | `_ensure_location_data` |

See [Wind persistence chain](../models/wind-chain.md).

### `solar_chain`

| Field | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `enabled` | bool | `false` | Solve with the Markov-modulated clear-sky index. | `run_sim.py::_load_solar_chain` |
| `path` | path | derived (`_solarchain`) | Explicit artifact path; never rebuilt automatically. | `_load_solar_chain`, `_ensure_location_data` |
| `n_bins` | int | `3` | Number of stage-relative quantile bins. There is no `bin_edges` option. | `_ensure_location_data` |

Enabling both chains solves over the joint exogenous index (`n_wind × n_solar` bins). See [Solar persistence chain](../models/solar-chain.md).

### `historical_weather`

| Field | Type | Default | Meaning | Read in |
|---|---|---|---|---|
| `enabled` | bool | `false` | Roll episodes out on block-bootstrapped historical weather. The policy is still solved on the fitted distributions or chains. | `SimulationFactory.__init__` |
| `path` | path | derived (`_histcube`) | Explicit historical-cube artifact path. | `_load_historical_cube`, `_ensure_location_data` |
| `block_length_days` | float | `7` | Bootstrap block length. It is converted to `block_days · 24 · 60 / delta_t` steps. | `SimulationFactory.__init__` |

## Artifact paths

Every path is derived from the location's `data_path`:

1. **Clear-sky suffix** (`run_sim.py::_resolve_data_path`). For any `clearsky_model` other than `fatemi`, `_<model>` is inserted before the extension. For example, `data_expected_lat30.0_lon-90.0_15min.pkl` becomes `…_15min_ineichen.pkl`. Artifacts normalized with different envelopes are not interchangeable, so they never overwrite each other. `fatemi` keeps the configured path unchanged.
2. **Derived artifacts** take the (possibly suffixed) data path and insert a tag before the extension:

    | Artifact | Tag | Function |
    |---|---|---|
    | Wind chain | `_windchain` | `_derive_chain_path` |
    | Solar chain | `_solarchain` | `_derive_solar_chain_path` |
    | Historical cube | `_histcube` | `_derive_histcube_path` |

    For example, `data_expected_lat30.0_lon-90.0_15min_ineichen_solarchain.pkl`.

**Provisioning.** Before any simulation is built, `run_experiment.py::_ensure_location_data` checks each location. It (re)builds an artifact when:

| Artifact | Rebuilt when |
|---|---|
| Expected data | The file is missing. |
| Wind chain | Missing, or its stored binning differs from the config: a different edge array, a different `n_bins`, or quantile vs explicit mode (the artifact's `quantile_derived` flag). |
| Solar chain | Missing, or `n_bins` or `clearsky_model` differs from the artifact. |
| Historical cube | Missing (only when `historical_weather.enabled`). |

Builds read the location's historical record from the `HISTORICAL_DATA` directory next to `EXPECTED_DATA`. If that record is absent, it is fetched from Open-Meteo (1950–2022) first. See [Weather data pipeline](../models/weather-data.md).

## Internal keys

The harness adds these keys to the in-memory config. They are removed from the `config.yaml` copied into the run folder (every key starting with `_` is dropped):

| Key | Set by | Purpose |
|---|---|---|
| `_run_output_dir` | `run_experiment.py::run_experiment` | Sends the solver's value-table `.npy` files to `<run>/solver_tables/`. |
| `_config_basename` | `run_experiment.py::run_experiment` | Carried into the `config_basename` column of `summary.csv`. |

The harness also makes every `data_path` and artifact `path` absolute (`_resolve_paths_in_place`), so spawned workers do not depend on the working directory. The copied `config.yaml` therefore contains absolute paths.

## Reserved CSV keys

| Key | Default | Meaning |
|---|---|---|
| `solver_backend` | `vectorized` | Written to `summary.csv` only. It does not select a solver. |
| `rollout_backend` | `batched` | Written to `summary.csv` only. It does not select a rollout path. |
