# First run

This page runs the smallest example experiment end to end: `iid_small.yaml`. It then shows
what lands on disk and how to read the results.

## 1. The config

`SolarSimulator/harness/examples/iid_small.yaml`:

```yaml
--8<-- "SolarSimulator/harness/examples/iid_small.yaml"
```

In this config, any value given as a list is a swept dimension. The harness runs the full cross-product:

| Policy | Combinations | Simulations |
|---|---|---|
| Threshold | 2 capacities × 1 observation threshold × 2 wind thresholds × 1 horizon × 1 penalty | 4 |
| Optimal (`include_optimal: true`) | 2 capacities × 1 horizon × 1 penalty | 2 |

Each simulation runs 500 Monte Carlo episodes. `horizons: [150]` at `delta_t: 15` minutes
is a 37.5-hour mission starting at `2020-04-01T00:00`.

## 2. Run it

From the `SolarSimulator/` directory:

```bash
conda run -n pvlib python harness/run_experiment.py experiment harness/examples/iid_small.yaml
```

Useful options:

| Flag | Effect |
|---|---|
| `--no-multiproc` | Run serially (simplest to debug) |
| `--workers N` | Number of worker processes for building and running simulations (default: CPU count − 1) |
| `--out DIR` | Output base directory (default: the config's `storage_dir`, else `<repo>/results`) |
| `--pattern GLOB` | When the path is a directory of configs, which files to run (default `*.y*ml`) |

What happens, in order (`run_experiment` in `SolarSimulator/harness/run_experiment.py`):

1. **Provision data.** Missing weather artifacts for each location are fetched or built (see
   [Installation](index.md)).
2. **Build.** One `SimulationFactory` is created per parameter combination. Building an
   *optimal* simulation runs the backward-induction solve immediately. Building is parallel
   across workers.
3. **Run.** `SimulationRunManager` rolls out every simulation's episodes and writes each
   simulation as one group in a single HDF5 file.
4. **Summarize.** `summary.csv` gets one row per simulation, and figures are drawn.
5. **Record.** `run_metadata.json` captures the git SHA, timestamp and command.

## 3. What lands on disk

```
results/iid_small/<YYYYmmdd_HHMMSS>/
    config.yaml                 resolved copy of the config (paths made absolute)
    iid_small_<timestamp>.h5    raw results, one HDF5 group per simulation
    summary.csv                 one row per simulation
    run_metadata.json           harness version, git SHA, timestamp, command
    solver_tables/              value-function tables (.npy), one per optimal simulation
    figures/
        sweep_<metric>.png      each metric against the swept parameters
        trajectory_*.png        real-weather replay of a representative simulation
```

Re-running the same config creates a new timestamped directory and never overwrites an
earlier run. The trajectory replay is drawn only when historical weather data exists for the
location.

## 4. Read the results

`summary.csv` is the stable, tidy view of a run. The columns you'll use most:

| Column | Meaning |
|---|---|
| `simulation_type` | `OptimalContinuousAnalyticalPolicySimulation` or `UnifiedThresholdContinuousSimulation` |
| `battery_capacity`, `horizon`, `failure_penalty` | The swept model parameters |
| `observation_threshold`, `wind_threshold` | Threshold-policy parameters (empty for optimal) |
| `average_reward` | Mean total reward per episode |
| `failure_percentage` | Fraction of episodes that ended in `broken` (0-1) |
| `average_failure_step` | Mean stage at which episodes ended (equals the horizon for survivors) |
| `average_flight_hrs` | Mean hours flown per episode |

The full column list, including the config-wide columns, is in
[Outputs](../experiments/outputs.md). To go beyond the averages, read the per-episode columns
in the HDF5 file. [Result storage](../architecture/storage.md) describes the layout.

```python
import pandas as pd
df = pd.read_csv("results/iid_small/<timestamp>/summary.csv")
df.sort_values("average_reward", ascending=False)[
    ["simulation_type", "battery_capacity", "wind_threshold",
     "average_reward", "failure_percentage"]]
```

!!! tip "Check existing results first"
    `results/` may already hold runs of the experiment you have in mind. Look at their
    `summary.csv` files before starting a new simulation.

## Next steps

- Try the other examples in [Example configs](../experiments/examples.md). `chain_small.yaml`
  is the same sweep with the wind persistence chain enabled.
- Learn every config key: [Config reference](../experiments/config-reference.md).
- Understand what ran: [Architecture](../architecture/index.md).
