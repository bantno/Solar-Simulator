# Installation

## Requirements

| Item | Value |
|---|---|
| Package manager | Conda |
| Environment name | `pvlib` |
| Environment file | `environment.yml` (repository root) |
| Python | 3.11 (`environment.yml` pins 3.11.9; `SolarSimulator/setup.py` requires `>=3.11`) |
| Platform | Developed on Windows |

!!! warning "Always use the `pvlib` environment"
    The repository needs `pvlib` and `h5py` builds from conda-forge. The base Anaconda
    environment has an incompatible numpy/h5py ABI, so imports that touch HDF5 fail there.
    Run everything with `conda run -n pvlib ...` or from an activated `pvlib` env.

## Create the environment

```bash
git clone https://github.com/bantno/Solar-Simulator.git
cd Solar-Simulator
conda env create -f environment.yml
conda activate pvlib
```

`environment.yml` is a full export of the working environment. Key packages:

- **Numerics:** numpy 1.26, scipy 1.13, pandas 2.2.
- **Solar and data:** pvlib 0.10.5, h5py 3.11.
- **Plotting and config:** matplotlib, plotly, pyyaml.
- **Weather fetching:** the Open-Meteo client libraries.
- **This wiki:** the packages in the last block of the `pip:` section.

### Optional: install the package

```bash
cd SolarSimulator
pip install -e .
```

This isn't needed to run anything. The code is run from inside `SolarSimulator/`, which
is the import root: modules import each other as `BaseClasses.*`, `harness.*`, `Scripts.*`.

## Repository layout

```
Solar-Simulator/
  SolarSimulator/          import root (run code from here)
    BaseClasses/           MDP, solver, environment providers, aircraft/solar/whale models,
                           simulation drivers, HDF5 storage, run_sim.py
    harness/               declarative YAML experiment runner (+ examples/)
    Scripts/               artifact builders, sweep pipeline, prechecks, plotting utilities
    Tests/                 verify_*.py verification scripts and unittest tests
    Tools/                 weather/analysis/plotting helpers
    gui/                   legacy batch-run GUI and results viewer
    SolarArray/            solar-array model
    TakeoffSimulation/     takeoff / motor power modelling
  configs/                 experiment YAMLs (journal sweeps, chain-vs-iid sweep)
  Figures/Scripts/         journal-paper figure scripts
  Data/                    weather data and fitted artifacts (git-ignored)
  results/                 experiment outputs (git-ignored)
  docs/                    this wiki's source
  mkdocs.yml               this wiki's configuration
  environment.yml          conda environment
```

## Weather data

The simulator reads fitted weather distributions from `Data/EXPECTED_DATA/`, for example
`Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min.pkl`. It builds those
distributions from raw historical data in `Data/HISTORICAL_DATA/`. Both directories are
git-ignored.

You don't have to build them by hand. When the harness runs an experiment, it provisions any
missing artifact for each configured location (`_ensure_location_data` in
`SolarSimulator/harness/run_experiment.py`):

1. Fetch the historical record from the Open-Meteo archive (1950-2022) if it isn't already on disk.
2. Build the expected-data pickle.
3. Build the wind-chain, solar-chain and historical-cube artifacts if the config enables them.

See [Weather data pipeline](../models/weather-data.md).

## Build this wiki

The wiki is built locally with MkDocs Material and is never hosted. From the repository root:

```bash
conda run -n pvlib mkdocs build
```

This writes a static site to `site/`, which is git-ignored. Open `site/index.html` directly
from disk. The site is built with the `offline` plugin and `use_directory_urls: false`, so
navigation, search, diagrams and maths all work from `file://` with no network access.

For a live preview that reloads as you edit:

```bash
conda run -n pvlib mkdocs serve
```

Then open `http://127.0.0.1:8000`.

The build runs in `strict` mode, so any broken link, broken anchor, or page missing from
`nav` fails the build. API pages under **Reference > API** are generated from docstrings
by mkdocstrings. The source is read statically, so heavy dependencies are never imported.
See [API reference](../api/index.md).

Next: [First run](first-run.md).
