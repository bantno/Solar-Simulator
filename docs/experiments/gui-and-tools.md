# GUI and tools

The code under `SolarSimulator/` that is outside the core library and the harness: the PyQt5 GUIs, the batch runner, `Tools/`, `SolarArray/`, and the utility scripts in `Scripts/` that the [Scripts API page](../api/scripts.md) does not cover. For validation experiments, prefer the [harness](index.md). It reads every behavior setting from the YAML, whereas the GUI flow takes some from checkboxes and CLI flags.

## GUIs (`gui/`)

`gui/` is not a package. Run its scripts directly from `SolarSimulator/`. They need PyQt5 in the environment.

### `gui.py`: Simulation Runner

```bash
python gui/gui.py
```

It has two tabs:

- **Run Simulation.** Choose a YAML config and tick *Use Multiprocessing*, *Save Full State Info* and *Include Optimal Algorithm*, and set the number of full-history episodes. The run uses the same `YAMLSimulationRunner` → `SimulationFactory` → `SimulationRunManager` stack as the harness and writes one timestamped HDF5 into `simulation_results/` under the working directory. The checkboxes, not the YAML, decide history retention and whether the optimal policy runs.
- **Create Config.** A form for the sweep lists (start dates, capacities, observation and wind thresholds, latitudes, longitudes, horizons, failure penalties, episodes). *Export Config File* writes it out as YAML. Keys the form does not cover must be added by hand.

### `combined_gui.py`: All-In-One Simulation Toolkit

```bash
python gui/combined_gui.py
```

A results viewer for the HDF5 files written by the harness, the GUI, or `run_batch.py`. It has two tabs:

| Tab | Class | What it does |
|---|---|---|
| Episode Inspector | `InspectorTab` | Open an HDF5 file, pick simulations and an episode, and plot the stored series (`solar_series`, `wind_series`, `whale_series`, `energy_series`, `actions`, `rewards`) against stage. Has a moving window, a cursor line, and an option to open an episode in a new figure. Needs full-history episodes. |
| Reward Plotter | `RewardPlotterTab` over `HDF5RewardPlotter` | Summary plots from the group attributes: reward, failure step and failure % by threshold pair; reward against capacity, horizon or penalty; optimal-policy reward and failure-step distributions by penalty; a metric by location, duration, start date or penalty. It also exports the summary as CSV or a LaTeX table. |

The Reward Plotter reads per-episode scalars from the columnar `episode_scalars/` layout and falls back to the legacy per-episode groups.

### `run_batch.py`: CLI mirror of the GUI flow

```bash
python gui/run_batch.py ./configs                         # every *.y*ml in a directory
python gui/run_batch.py ./configs --pattern "case*.yaml"
python gui/run_batch.py ./configs/case1.yaml
python gui/run_batch.py ./configs --workers 20 --chunksize 1 --maxtasksperchild 1 --verbose-workers
```

It builds simulations exactly as the GUI does (`YAMLSimulationRunner._build_param_list` → `create_simulation_wrapper`), runs simulations in parallel processes with each simulation's episodes run serially, and writes one timestamped HDF5 per config.

| Flag | Meaning |
|---|---|
| `--out` | Output directory (default: the config's `storage_dir`, else `./simulation_results`) |
| `--workers`, `--build-workers` | Process counts for running and for building simulations |
| `--no-multiproc` | Run serially |
| `--chunksize`, `--maxtasksperchild` | Pool scheduling |
| `--include-optimal {auto,yes,no}` | Override the config's `include_optimal` |
| `--save-history`, `--full-history-eps` (default 20) | History retention |
| `--verbose-workers` | Per-worker START/DONE messages |

Unlike the harness, it does not provision data, write `summary.csv` or figures, or create a run folder.

## `Tools/`

Stand-alone helpers from the original development of the model. None of them is used by the harness or the core library.

| Module | Purpose |
|---|---|
| `weather_api.py` | Minimal Open-Meteo archive request (wind speed/direction and shortwave radiation for a hard-coded point and date range) printed as a DataFrame. The production fetch is `Scripts/create_weather_distributions.WeatherDataProcessor` (see [Weather data pipeline](../models/weather-data.md)). |
| `fake_weather.py` | `FakeDataGenerator`: synthetic 15-min wind/solar test cases (constant low/high, low→high, high→low, short low spell) for exercising policies on controlled weather. |
| `weather_analysis.py` | `EnvironmentalDataProcessor`: seaborn exploration of a weather DataFrame. |
| `weather_plot.py` | Plot a weather DataFrame, or its first year. |
| `plotting.py` | Early plotting helpers (solar, endurance, battery sweep, simulation results, yearly DC energy). |
| `plot_ev_table.py` | Plot an expected-value table for the wind and solar simulations. |
| `stl_slice.py` | Hull geometry from an STL mesh (trimesh + shapely): cross-section slicing, second moment of area, draft and horizontal-stabilizer calculations. |
| `wind_solar_test.py` | `simulate_floating_tracker`: prototype of solar tracking on a floating base. |
| `analyze_results.py`, `code_profile.py` | Small scratch snippets: load a pickle, print `pstats` from `output.prof`. |

## `SolarArray/`

`SolarArray/SolarArray.py` is an exploratory pvlib model of a single-axis solar tracker on a base that rolls, pitches and yaws, including a "no tracking" case with the panel fixed to the base. It covers yaw/pitch/roll rotation matrices, conversion of a panel normal to surface tilt and azimuth, and plane-of-array irradiance (`poa_for`). The simulator's panel models do not use it; they treat irradiance as horizontal (see [Solar panel models](../models/solar-panel.md)).

## `TakeoffSimulation/`

Water-takeoff and motor models. See [Takeoff simulation](../models/takeoff.md).

## Other `Scripts/` utilities

The pipeline, precheck and inspection scripts are documented on the [Scripts API page](../api/scripts.md). The rest are helpers and scratch scripts. Several read inputs hard-coded at the top of the file; edit those before running.

| Script | Purpose |
|---|---|
| `create_weather_distributions.py` | The weather-data library: `WeatherDataProcessor` (Open-Meteo fetch, resampling, Beta/Weibull fitting), synthetic-year generation, and the artifact builders `build_wind_chain_artifact`, `build_solar_chain_artifact` and `build_historical_cube_artifact`. The harness's provisioning step and the `build_*.py` scripts call these. See [Weather data pipeline](../models/weather-data.md). |
| `integrate.py` | Numerical check of the marginal action-success probability when success depends on wind through a sigmoid and the wind is Weibull (`quad` vs Simpson). |
| `plot_surfaces.py`, `plot_surfaces_plotly.py` | `EVTablePlotterPlotly`: interactive Plotly surfaces of saved value tables. `python Scripts/plot_surfaces.py <file.npy or dir> --outdir DIR`. |
| `validate_weather.py` | Sanity-check every weather pickle in a directory. |
| `histogram.py` | Plotly histograms of rewards from a directory of result pickles (hard-coded input). |
| `plot_states.py` | Plot selected series of one episode from an HDF5 file (hard-coded input). |
| `ts_localize.py`, `read_pkl.py` | Inspect or time-localize a weather pickle (hard-coded input). |
| `vis.py` | Bar chart of average reward per policy from hard-coded values. |
