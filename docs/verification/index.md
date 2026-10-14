# Verification

This page explains how model changes in this repository are checked, and how to run the
checks. The [test catalogue](tests.md) lists what each script asserts.

## Philosophy: reduce to the baseline, then check direction

Every model extension added since the journal paper was built to leave the published
i.i.d. model reachable as a special case. Examples are the
[wind chain](../models/wind-chain.md), the [solar chain](../models/solar-chain.md),
[pluggable clear-sky models and the PVWatts panel](../models/clearsky.md),
[configurable whale series and cruise fractions](../models/whale-observation.md), and the
[vectorized solver](../architecture/solver.md). Each change is verified in two steps, in
this order.

1. **Exact or statistical reduction to the baseline.** With the new feature off, or set to
   its degenerate configuration, the new code must reproduce the old model:
    - **Bit-identical** (or equal to ~1e-8) wherever only the loop structure or
      bookkeeping changed. Examples: vectorized vs scalar `solve()`; PVWatts defaults vs
      the `constant` panel; `custom` whale blocks carrying the built-in values; value
      tables at default cruise fractions.
    - **Within Monte Carlo error** wherever the random-number draw order changed.
      Examples: batched episode rollout vs the per-episode loop; a rank-1 (memoryless)
      chain vs the i.i.d. run.

    This step shows the extension is a strict superset of the published model and did
    not quietly change it.
2. **Directional (behavioral) checks.** With the feature on, its effect must point the
   way the physics says. The value function falls as the wind bin rises and climbs as
   the solar bin rises. A hot PVWatts cell derates by the expected factor. A 06:00
   start rolls the whale profile to the 06:00–08:00 block.

The original scalar code paths were deliberately kept in the source alongside the
vectorized ones: `_value`, `simulate_episode`, and scalar `choose_action`. That makes
"new vs. old" an in-process diff rather than a comparison against stored numbers. See
the decision log in [Design records](../design/index.md).

!!! note "Where each check is argued in full"
    The mathematical side of each reduction is written up in the design records:
    [wind-persistence validity](../design/wind-persistence-validity.md) §6–7 and
    [solar-persistence math](../design/solar-persistence-math.md) §6.

## Running the verification scripts

The `verify_*.py` scripts are standalone programs, not pytest tests. Each prints one line
per check (`[PASS]`, `ok -`, or a measured value followed by `PASS`) and stops with a
non-zero exit at the first failure, through `assert` or `sys.exit(1)`. Each
script resolves its own paths from `__file__`, so it runs from **any working directory**.
They must run in the `pvlib` conda env (see
[Installation](../getting-started/index.md)):

```bash
conda run -n pvlib python SolarSimulator/Tests/verify_wind_chain.py
conda run -n pvlib python SolarSimulator/Tests/verify_solar_chain.py
conda run -n pvlib python SolarSimulator/Tests/verify_pvlib_solar.py
conda run -n pvlib python SolarSimulator/Tests/verify_whale_model.py
conda run -n pvlib python SolarSimulator/Tests/verify_runtime_opt.py   # --horizon 2000 --episodes 3000
conda run -n pvlib python SolarSimulator/Tests/verify_solver_speed.py  # --horizon 1000
```

**Input data.** The model checks build temporary configs against the Gulf baseline site
(lat 30 / lon −90):

- `Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min.pkl` is required.
- Chain artifacts are built on first use if they are missing. For example,
  `verify_wind_chain.py` builds `..._15min_windchain.pkl` from
  `Data/HISTORICAL_DATA/data_30_-90.pkl`.

`Data/` is git-ignored. On a fresh clone, generate the data first (see
[Weather data pipeline](../models/weather-data.md)), or run any harness example once,
since the harness provisions missing artifacts automatically.

!!! tip "When to run which"
    | You changed… | Run at least |
    |---|---|
    | `backward_induction_base.py` (solver) | `verify_runtime_opt.py`, `verify_solver_speed.py`, `verify_wind_chain.py`, `verify_solar_chain.py` |
    | `simulation_base.py` (rollout / policies) | `verify_runtime_opt.py`, `verify_wind_chain.py` |
    | `environment_provider_base.py`, `run_sim.py` wiring | `verify_wind_chain.py`, `verify_solar_chain.py`, `verify_pvlib_solar.py` |
    | `clearsky_models.py`, `solar_panel_base.py` | `verify_pvlib_solar.py`, `verify_solar_chain.py` |
    | `whale_base.py`, `mdp_base.py` reward / gain | `verify_whale_model.py` |

    `verify_wind_chain.py` is the regression gate for any change that touches the joint
    exogenous index. The solar-chain refactor was accepted partly because it passed
    unchanged ([solar-persistence math](../design/solar-persistence-math.md) §6).

## Unit tests

The older `Tests/test_*.py` files are `unittest` test cases that import the package
modules as top-level `BaseClasses.*`, so they must run from `SolarSimulator/` (the same
root `.vscode/settings.json` uses). Run the maintained cases by module:

```bash
cd SolarSimulator
conda run -n pvlib python -m unittest -v Tests.test_environment_provider_base Tests.test_climb_power Tests.test_reward
```

To run a single file:
`conda run -n pvlib python -m unittest Tests.test_climb_power -v`.

## Sweep-level check: common random numbers

Paired comparisons between policies on historical weather rely on each arm seeing
**identical** bootstrap weather, episode for episode. The provider RNG is re-seeded
per batch; see [Episode simulation](../architecture/simulation.md). The sweep comparison
script checks this directly:

```bash
# from SolarSimulator/
conda run -n pvlib python Scripts/compare_chain_sweep.py --verify
```

The script compares `wind_series` across up to 8 full-history episodes that the i.i.d.
and chain historical runs have in common (`verify_crn` in
`Scripts/compare_chain_sweep.py`), and prints one of three results:

- **identical:** the weather matches across arms.
- **diverge:** the pairing is broken, and paired deltas cannot be trusted.
- **inconclusive:** neither run kept full histories.

Full histories are kept only on the baseline-location capgrid pair, so that the check
has data; see [Sweeps](../experiments/sweeps.md).
