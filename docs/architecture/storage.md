# Result storage

Results are written to HDF5 by `SimulationStorageHDF5` in `BaseClasses/simulation_storage.py`
(API: [simulation_storage](../api/baseclasses/simulation_storage.md)). One experiment run
produces one HDF5 file, and each simulation is a top-level group in it.

## Run directory

The harness gathers each run into its own directory:

```
results/<config_basename>/<YYYYmmdd_HHMMSS>/
    config.yaml                  resolved config (paths absolutized, internal "_" keys dropped)
    <config_basename>_<ts>.h5    raw results (this page)
    summary.csv                  one row per simulation group
    run_metadata.json            harness_version, git_sha, timestamp, command, n_simulations, ...
    solver_tables/*.npy          value tables for the optimal simulations
    figures/                     sweep plots / paper figure, trajectory replay
```

`summary.csv`, `run_metadata.json` and the figures are described in
[Outputs](../experiments/outputs.md).

## HDF5 layout

```
<file>.h5
└── <group name>                      one per simulation
    ├── attrs                         simulation summary + identifying metadata
    ├── episode_scalars/
    │   ├── total_reward   (episodes,)
    │   ├── failure        (episodes,)
    │   ├── failure_step   (episodes,)
    │   ├── flight_hrs     (episodes,)
    │   ├── failure_type   (episodes,)
    │   └── episode_index  (episodes,)
    └── episodes/
        └── episode <i>                only for full-history episodes (i is 1-based)
            ├── attrs                  episode metadata (episode_index)
            ├── trajectory             (steps+1, 2)  [soc, mode]
            ├── actions                (steps,)
            ├── rewards                (steps,)
            ├── solar_series           (steps,)   harvested energy per stage [J]
            ├── wind_series            (steps,)   wind speed [m/s]
            ├── whale_series           (steps,)
            └── energy_series          (steps+1,) continuous battery energy [J]
```

- **Group attributes**, written in `_run_one_sim`:
    - identity: `simulation_type`, `battery_capacity`, `horizon`, `failure_penalty`,
      `initial_state`, `start_time`, `location_id`;
    - threshold settings: `observation_threshold` and `wind_threshold` (threshold simulations only);
    - run size: `episodes_count`;
    - averages: `average_reward`, `failure_percentage`, `average_failure_step`,
      `average_flight_hrs`.
- **Episode scalars** are columns: one 1-D dataset per field, indexed by episode. A field
  missing for some episodes is stored as a float column with `NaN` in the gaps.
- **Full-history groups** exist only for episodes that carry array fields, i.e. the first
  `full_history_episodes` episodes, or all of them with `save_states: true`. Arrays are
  written with `lzf` compression, byte shuffle, and chunks of at most about 1 MB. Episodes are
  truncated at failure, so `steps` can be smaller than the horizon.
- **The `episodes/` group** is always created, even when it stays empty.

### Group names

`SimulationRunManager._make_group_name` builds names from the lower-cased class name and the
metadata. Examples:

```
optimalcontinuousanalyticalpolicysimulation_c300_f5_h150_lat30.0_lon-90.0_d92
unifiedthresholdcontinuoussimulation_c300_f5_h150_lat30.0_lon-90.0_t0.25_w8.0_d92
```

| Part | Meaning |
|---|---|
| `c<cap>` | battery capacity (Wh) |
| `f<penalty>` | failure penalty |
| `h<horizon>` | horizon (stages) |
| `lat<..>_lon<..>` | location |
| `t<obs>`, `w<wind>` | threshold policy parameters |
| `d<doy>` | day of year of `start_time` |

A name has to be unique within a file; `store_simulation` raises an error if the group
already exists.

## Why columnar scalars

Every HDF5 object carries about 0.5 KB of file metadata. The earlier layout stored every
episode as its own group of scalar datasets. For a scalars-only run of 210 simulations ×
3000 episodes, that made the file about 1.6–1.7 GB for about 40 MB of actual data. Storing the
scalars as columns made such files about 88× smaller, around 19 MB (commit `32e0d4a`,
*Store per-episode scalars as columnar HDF5 datasets*).

- **Legacy files in sweeps:** `run_chain_sweep` still writes its `_episode_scalars.csv`
  cache, but it deletes only legacy-sized HDF5 files (≥ 100 MB), because columnar files don't
  need compacting.
- **Readers:** every in-repo reader was updated with a fast columnar path and a fallback for
  the legacy layout.

## Reading results

- **Current layout:** read `<group>/episode_scalars/<field>[...]` directly, or use `pandas`
  on `summary.csv` for the per-simulation averages.
- **Legacy files**, where every episode has an `episodes/episode <i>` group of scalar datasets and
  there is no `episode_scalars/`: all in-repo readers fall back to that layout. The readers are
  `Scripts/compare_chain_sweep.py` (`read_all_episode_scalars`), the GUI results viewer, and
  the `Figures/Scripts` loaders (`results_io`).
- **Sweep cache:** `Scripts/run_chain_sweep.py` also writes a per-config
  `_episode_scalars.csv` analysis cache. See [Sweeps](../experiments/sweeps.md).

```python
import h5py
with h5py.File(path, "r") as f:
    for name, grp in f.items():
        reward = grp["episode_scalars/total_reward"][...]
        print(name, grp.attrs["average_reward"], reward.std())
```

## Storage budget

Full per-step histories are what make files large. At 64 episodes × 2880 steps that is
about 16 MB per simulation. Per-episode scalars are enough for comparisons, reward CDFs and
CVaR.

- **Default:** keep `save_states: false` and `full_history_episodes: 0`.
- **When a figure needs trajectories:** keep a small full-history budget only on the specific
  cell that figure uses.

The older `SimulationStorage` class, which writes one `.npz` per simulation, is still in the
same module, but the run managers no longer use it.
