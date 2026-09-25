# Agent orientation

This page is for AI coding agents and new maintainers who work in this repository. It
covers the things that aren't obvious from reading any single file: how imports resolve, the
quirks of the environment, and the conventions the maintainers expect.

## Import root and working directory

- The **import root is `SolarSimulator/`**, not the repository root. Modules import each
  other as top-level packages: `from BaseClasses.run_sim import ...`, `from harness import ...`,
  `from Scripts.create_weather_distributions import ...`.
- **Run code from `SolarSimulator/`.** Several entry points also insert that directory into
  `sys.path` themselves, so they work from any directory. These include
  `harness/run_experiment.py`, the `Scripts/build_*` and chain-sweep scripts, and every
  `Tests/verify_*.py`.
- **Config paths are relative to the repository root.** For example,
  `data_path: Data/EXPECTED_DATA/...`. The harness converts them to absolute paths
  (`_resolve_paths_in_place`) so spawned worker processes don't depend on the working directory.
- **`Figures/Scripts/` is a second, separate import root.** Its scripts import each other as bare
  sibling modules (`from results_io import ...`).
  `harness/figures.py` adds that directory to `sys.path` at runtime.
- The API reference mirrors this layout: identifiers are `BaseClasses.whale_base`,
  `harness.run_experiment`, `results_io`, and so on. See [API reference](../api/index.md).

## Environment gotchas

| Issue | What to do |
|---|---|
| Base Anaconda has a numpy/h5py ABI mismatch | Always use the `pvlib` env: `conda run -n pvlib python <script>.py`, or its interpreter directly (`<anaconda>/envs/pvlib/python.exe`). |
| The Windows console is cp1252 | Keep stdout ASCII (write `->`, not `→`), or set `PYTHONIOENCODING=utf-8` before running. Non-ASCII output can raise `UnicodeEncodeError`. |
| `conda run ... python -c "<code>"` | `conda run` rejects arguments that contain newlines, and `-c` code can't be re-imported by multiprocessing `spawn` workers. Put the code in a script file and run the file. |
| Multiprocessing uses `spawn` | `run_experiment.main()` forces the `spawn` start method. Worker functions must be defined at module top level so they can be pickled (for example `create_simulation_wrapper`, `_run_one_sim`). |
| Disk space is tight | Full per-episode histories dominate disk use. See *Result storage conventions* below. |

## Conventions

### Code

- **Docstrings are Google style** (`Args:`, `Returns:`, `Raises:`). The API reference is
  rendered from them with `docstring_style: google`.
- **New model features default to off**, and with the default settings they reproduce the
  original model exactly. Examples: `wind_chain.enabled: false`, `solar_chain.enabled: false`,
  `clearsky_model: fatemi`, `solar_panel_model: constant`, and cruise fractions `1.0` / `0.0`.
  When a feature is off, the original code path runs unchanged. See
  [Models overview](../models/index.md).
- **Keep the original scalar code paths next to the vectorized ones.** Examples are `_value`
  next to `_value_batch`, `simulate_episode` next to `simulate_episode_batch`, and
  `lookup_future_values` next to `_lookup_future_values_fast`. The verification scripts diff
  the new path against the original in the same process.

### Verification

Every feature comes with a `Tests/verify_<feature>.py` script. Each one follows the same
pattern:

1. **Reduce to the baseline.** With the feature off, or set to its degenerate
   configuration, results match the original model exactly: byte-identical value tables, or
   agreement within Monte Carlo error for rollouts.
2. **Check the direction.** With the feature on, the effect goes the way the theory predicts.

The scripts are standalone (not pytest), print PASS/FAIL, generate their own temporary configs
against `Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min*.pkl`, and clean up after
themselves. See [Verification](../verification/index.md).

### Performance work

- **Do performance work on its own branch**, not on `main`.
- **Never automatically reuse pre-solved value tables.** Every run solves the MDP fresh. The
  `.npy` tables in `solver_tables/` are outputs for inspection, not a cache. Loading them back
  risks silently stale results. The filename scheme may stay as an open hook, but solving must
  always recompute.

### Result storage

- **Don't store full episode histories by default.** Set `save_states: false` and
  `full_history_episodes: 0` in sweep configs. Per-episode scalars (`total_reward`,
  `failure`, `failure_step`, `flight_hrs`, `failure_type`) are always stored and are
  enough for comparisons, CDFs and CVaR.
- **Keep a small full-history budget only where a figure needs it.** For example, 64
  episodes on the one baseline cell used for trajectory replays and event-aligned composites.
  `Scripts/generate_chain_sweep_configs.py` encodes this rule.
- See [Result storage](../architecture/storage.md) for why this matters for file size.

### Running experiments

- **Check existing results before running new simulations.** Many headline experiments have
  already been run. Each run directory under `results/` holds:
    - `summary.csv`, one row per policy/config;
    - `config.yaml`, for matching runs by diffing configs;
    - `run_metadata.json`.

  Threshold and optimal rows sit in the same `summary.csv`, so comparing them needs no new
  run. Allow for Monte Carlo spread between repeated runs.
- **Keep experiments self-describing.** Behavior comes from the YAML, not from CLI flags. Keep
  i.i.d. and chain experiments as separate configs.

### Git

- **Never mention Claude or other AI assistants in commit messages.**
- Features are developed on feature branches: `wind-persistence`, `solar-chain`,
  `pvlib-solar`, `observation-model-fidelity`, and others.

## Building this wiki

```bash
conda run -n pvlib mkdocs build      # -> site/, open site/index.html from disk
conda run -n pvlib mkdocs serve      # live preview at http://127.0.0.1:8000
```

- **Build rules:**
    - The build is `strict`: broken links or anchors, and pages missing from `nav` in
      `mkdocs.yml`, fail it.
    - Link to other pages by relative `.md` path.
    - Show source files as inline code paths, not as links.
- **Adding an API page:** add a `::: module.path` directive to a page under `docs/api/` and
  list the page in `nav`.
- **Updating hand-written pages:** do it in the same change as the code they describe.
