# Journal figures

How the journal-paper figures are produced and how a harness run can reproduce one. The code lives in `Figures/Scripts/`. Those scripts import each other as bare sibling modules (`from results_io import ...`), so that directory is its own import root. API: [Figure scripts](../api/figures.md), [`harness.figures`](../api/harness.md).

## `generate_journal_paper_figures.py`

The single entry point that re-renders every journal-paper figure. All figures share one style: sans-serif, fonts of at least 8 pt, 1.0 pt data lines, 0.75 pt axes, 600 dpi, drawn at final print width so LaTeX does not rescale the fonts.

```bash
python Figures/Scripts/generate_journal_paper_figures.py            # render all
python Figures/Scripts/generate_journal_paper_figures.py --only fig_threshold_sweep_combined
python Figures/Scripts/generate_journal_paper_figures.py --list     # print figure names
```

`--continue-on-error` skips a failing figure and renders the rest.

**Output.** Figures are written to `Figures/Journal_Paper_Figures/` (the module-level `OUT_DIR`).

**Inputs.** Each figure reads fixed artifacts listed in the module-level `RESULTS_PATHS` dict:

| Key | Artifact |
|---|---|
| `threshold_h5`, `threshold_value_npy` | Threshold sweep HDF5 and its value table (`Figures/Results/ThresholdSweep/`) |
| `battery_h5` | Battery-capacity sweep (`Figures/Results/BatterySweep/`) |
| `duration_h5`, `duration_value_npy_short`, `duration_value_npy_long` | Duration sweep and two value tables (`Figures/Results/DurationSweep/`) |
| `failure_h5` | Failure-penalty sweep (`Figures/Results/FailurePenaltySweep/`) |
| `loc_capacity_h5` | Location × capacity sweep (`Figures/Results/LocationCapacitySweep/`) |
| `wind_solar_data_dir` | `Data/EXPECTED_DATA/`, for the weather-climatology figures |

These results come from the configs in `configs/final_journal_configs/` (see [Sweeps § Journal-paper configs](sweeps.md#journal-paper-configs)). `Figures/Results/` holds local result files and is not part of the repository.

**Figures.** The registered figures (the `FIGURES` dict):

| Name | Content |
|---|---|
| `fig_total_monthly_solar_energy` | Monthly solar energy by location |
| `fig_mean_wind_speed` | Mean wind speed by location |
| `fig_wind_months_mean_plus_3sigma` | Wind mean ± 3σ, selected months |
| `fig_solar_months_mean_plus_3sigma` | Solar mean ± 3σ, selected months |
| `fig_episode_4_combos` | Example episodes for four threshold combinations |
| `fig_threshold_example_value_combined` | Value-function slice alongside threshold lines |
| `fig_threshold_sweep_combined` | Observation × wind threshold sweep against the optimal policy |
| `fig_capacity_mean_reward` | Mean reward vs capacity |
| `fig_world_map` | Study locations |
| `fig_optimal_capacity_by_location` | Optimal-policy performance vs capacity per location |
| `fig_failure_penalty_combined` | Failure-penalty sweep |
| `fig_horizon_sweep_combined` | Mission-duration sweep |
| `fig_duration_value_soc_slices_short`, `..._long` | Value-function SoC slices for short and long horizons |
| `fig_whale_observation_probability` | Whale observation-probability profile |
| `fig_failure_probability` | Transition-model failure probability |

The module docstring notes that some paper figures, such as the iso and locations figures, are not produced by this script and are regenerated separately.

## Reproducing a figure from a harness run (`paper_figure`)

When a config sets `paper_figure: <function name>`, `harness/figures.py::render_paper_figure` redraws that journal figure from **this run's** HDF5 instead of the generic sweep plots:

1. Import `generate_journal_paper_figures`, adding `Figures/Scripts` and `SolarSimulator` to `sys.path`.
2. Look up which `RESULTS_PATHS` key the figure reads, from `_PAPER_FIGURE_H5_KEY`.
3. Point that key at the run's HDF5 and `OUT_DIR` at the run's `figures/` folder.
4. Call `apply_paper_style()` and the figure function.

The plotting code is the paper's own, and its `load_summary` reads the same group attributes the harness writes. The output therefore matches the paper figure by construction. Figures wired up today:

| `paper_figure` | Reads |
|---|---|
| `fig_threshold_sweep_combined` | `threshold_h5` |
| `fig_capacity_mean_reward` | `loc_capacity_h5` |
| `fig_optimal_capacity_by_location` | `loc_capacity_h5` |

To support another figure, add its function name and `RESULTS_PATHS` key to `_PAPER_FIGURE_H5_KEY` in `harness/figures.py`. If a name is unknown, unmapped, or fails to render, the harness issues a warning and skips the figure; it does not fail the run.

`harness/examples/journal_threshold_vs_optimal.yaml` is the worked example (see [Example configs](examples.md#journal_threshold_vs_optimal)).

## `results_io`

`Figures/Scripts/results_io.py` holds the shared loader and style helpers. It does not depend on the GUI.

| Function | Purpose |
|---|---|
| `load_summary(h5_paths)` | One row per simulation group across one or more HDF5 files. Columns: `sim_type`, thresholds, `battery_capacity`, `horizon`, `failure_penalty`, `mean_reward`, `failure_percentage`, `mean_failure_step`, `average_flight_hrs`, `latitude`/`longitude` (parsed from `location_id`), `start_time`, `source_file`. Summary attributes are preferred; if they are missing, the values are aggregated from legacy per-episode groups. |
| `steps_to_days(steps, minutes_per_step=15)` | Convert a stage count to days |
| `apply_style(style, rc)` | Apply a matplotlib style and rc overrides |
| `savefig_all_formats(fig, outdir, basename, formats=("png","pdf"), dpi=300)` | Save a figure in several formats |

`generate_journal_paper_figures.load_summary` wraps `results_io.load_summary` and suppresses its debug output.

## Other scripts in `Figures/Scripts/`

These standalone scripts came before the unified generator. Each targets one sweep or one plot type; see its module docstring on the [Figure scripts](../api/figures.md) API page.

| Script | Purpose |
|---|---|
| `batterySweep.py`, `battery_sweep_variance.py` | Battery-capacity sweep figures and variance views |
| `durationSweep.py`, `duration_sweep_variance.py` | Horizon sweep subfigures and violin plots |
| `failurePenaltySweep.py`, `failure_penalty_reward_histogram.py` | Failure-penalty sweep and reward histograms |
| `locationCapacitySweep.py`, `loc_capacity_subfigs.py` | Location × capacity sweep |
| `startDateSweep.py` | Start-date × metric subfigures |
| `threshold_plots.py` | Threshold sweep plotter |
| `plot_episode.py`, `plot_episode_split.py` | A single episode from a results HDF5 |
| `plot_reward_histogram.py` | Total-reward histograms by failure penalty |
| `plot_value_function.py`, `surface_plot_alternatives.py` | Value-function heatmaps, surfaces and SoC-slice plots |
| `wind_mean_var.py` | Monthly wind mean ± 3σ panels |

## Harness sweep figures

Runs without `paper_figure` get the generic plots from `harness/figures.py::plot_sweep_summary` and a trajectory replay from `plot_trajectory_replay`. Both are described in [Outputs § figures](outputs.md#figures). The replay helpers are copied from `Scripts/compare_policies_episode.py` so that the harness does not depend on that script's module-level setup. Matplotlib runs headless (Agg).
