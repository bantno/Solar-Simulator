# AHABS: Autonomous Hydrofauna Aerial Biome Surveyor — A Solar-Powered Seaplane Simulation

## Overview
This repository contains the simulation framework for modeling the operation of a
solar-powered seaplane designed for long-term oceanic monitoring (e.g. whale
observation). The plane spends most of its time moored on the water charging from
its solar array, and periodically takes off to fly and observe whales. The
simulation models this as a finite-horizon **Markov Decision Process (MDP)** whose
environment is driven by location- and season-specific wind, solar, and whale
distributions, and it compares energy-management policies for maximizing
observations over a mission.

## Model

### State and actions
- **State** = `(battery state-of-charge %, mode)` plus an absorbing **broken** state.
  - Modes: `moored` (on the water, charging) and `flying`.
  - The broken state is terminal and is entered stochastically on a failure.
- **Actions**: `idle`/stay vs. `take off`/fly (a binary action set, `[0, 1]`).
- **Reward**: accrued from whale observations while flying; entering the broken
  state incurs a configurable failure penalty.

### Environment
Each mission stage draws from fitted, time-varying distributions:
- **Wind speed** — Weibull (shape `k`, scale).
- **Solar irradiance** — Beta distribution over a clearsky-normalized index,
  following the Fatemi–Kuh–Fripp (2018) normalization `x = r / (A·cos z)`.
- **Whale observations** — a per-stage observation-probability series.

Battery charge/discharge dynamics are derived from the solar array model and the
seaplane's idle / cruise / takeoff / landing power draw.

### Policies
- **Optimal** — computed by analytical backward induction over the MDP
  (`mdpAnalyticalBackwardSolver`).
- **Threshold** — fly when the whale-observation probability exceeds an
  `observation_threshold` and the wind is below a `wind_threshold`. A **greedy**
  policy is the special case `observation_threshold = 0`.

The MDP solver and episode rollouts are fully vectorized for runtime performance.

## Repository layout
```
SolarSimulator/
  BaseClasses/      MDP, solver, environment provider, seaplane/solar models,
                    simulation drivers, storage, and the run_sim.py entry point
  Scripts/          data prep (create_weather_distributions.py), plotting, utilities
  SolarArray/       solar array model
  TakeoffSimulation/ takeoff / motor power modeling
  Tools/            analysis, profiling, weather utilities
  Tests/            unit tests and runtime-optimization verification scripts
  gui/              batch-run GUI front end
configs/            YAML experiment configs (journal sweeps, tests)
Figures/            figure-generation scripts and saved journal figures
environment.yml     conda environment specification
```

## Installation
This project uses Conda. The environment is named `pvlib` (the repo relies on
`pvlib`/`h5py` builds from conda-forge; the base Anaconda environment has an
incompatible numpy/h5py ABI).

```bash
git clone https://github.com/bantno/Solar-Simulator.git
cd Solar-Simulator
conda env create -f environment.yml
conda activate pvlib
```

Optionally install the package itself (Python ≥ 3.11):
```bash
cd SolarSimulator
pip install -e .
```

## Usage

### 1. Generate environment data
Weather distributions are fetched from the Open-Meteo archive and fit per
location, then saved as pickled DataFrames (e.g.
`Data/EXPECTED_DATA/data_expected_lat30.0_lon-75.0_15min.pkl`):
```bash
python Scripts/create_weather_distributions.py
```

### 2. Run simulations from a config
Experiments are defined by YAML configs under `configs/`. Run them with the
`run_sim.py` driver (from inside the `SolarSimulator/` directory):
```bash
python BaseClasses/run_sim.py -c ../configs/final_journal_configs/battery_capacity_sweep/battery_sweep_config.yaml
```
Add `-p` / `--parallel` to enable multiprocessing.

A config sweeps over any combination of battery capacities, horizons, failure
penalties, start dates, locations, and threshold values, and runs both the
optimal and threshold policies (`include_optimal: true` by default). Key fields:

| Field | Meaning |
| --- | --- |
| `battery_capacities` | list of battery capacities (Wh) |
| `horizons` | mission length(s) in stages |
| `failure_penalties` | penalty for entering the broken state |
| `episodes` | Monte-Carlo episodes per simulation |
| `locations` | list of `{data_path, latitude, longitude}` |
| `start_datetime` / `start_datetimes` | mission start time(s) |
| `delta_t` | stage length in minutes (default 15) |
| `energy_increment_wh` | SoC discretization step (Wh) |
| `threshold_values`, `wind_thresholds` | threshold-policy parameters |
| `transition_model`, `solar_panel_model` | model variants |
| `storage_dir`, `save_states` | output location and history logging |

### 3. Outputs
Each run writes a single batch **HDF5** file (`<config_name>_<timestamp>.h5`) to
`storage_dir`, with results grouped per simulation. Plotting and analysis scripts
live in `Figures/Scripts/` and `SolarSimulator/Tools/`.

## Development environment
- **Package management**: Conda (`environment.yml`)
- **Python**: 3.11
- **Platform**: developed on Windows

## License
TBD
