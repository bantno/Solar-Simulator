# Example configs

The example experiments in `SolarSimulator/harness/examples/`. Each is embedded below straight from the source file, so this page always matches the repository. Run any of them from `SolarSimulator/`:

```bash
conda run -n pvlib python harness/run_experiment.py experiment harness/examples/<name>.yaml
```

The small examples all share one baseline: lat 30 / lon −90, start 2020-04-01, capacities 300 and 400 Wh, horizon 150, 500 episodes. That makes their run folders directly comparable.

| Config | Weather mode | What it exercises |
|---|---|---|
| [`iid_small`](#iid_small) | i.i.d. | Baseline small sweep |
| [`chain_small`](#chain_small) | wind chain | Markov-modulated wind with explicit bin edges |
| [`solarchain_small`](#solarchain_small) | solar chain | Markov-modulated clear-sky index |
| [`pvlib_small`](#pvlib_small) | solar chain | Ineichen clear-sky model + PVWatts panel |
| [`whale_custom_small`](#whale_custom_small) | i.i.d. | Whale profile supplied from the config |
| [`whale_encounter_small`](#whale_encounter_small) | i.i.d. | Distance-sampling encounter model |
| [`historical_smoke`](#historical_smoke) | historical | Minimal smoke test of the bootstrap rollout |
| [`journal_threshold_vs_optimal`](#journal_threshold_vs_optimal) | i.i.d. | Reproduces the journal threshold-vs-optimal figure |
| [`journal_threshold_vs_optimal_historical`](#journal_threshold_vs_optimal_historical) | historical | Journal matrix, rolled out on real weather |
| [`journal_threshold_vs_optimal_chain_historical`](#journal_threshold_vs_optimal_chain_historical) | wind chain + historical | Chain-solved policy on real weather |
| [`journal_threshold_vs_optimal_solarchain_historical`](#journal_threshold_vs_optimal_solarchain_historical) | solar chain + historical | Solar-chain-solved policy on real weather |

For the meaning of each key, see the [Config reference](config-reference.md).

## iid_small

The baseline: 2 capacities × 2 wind thresholds with memoryless wind, running both the optimal and threshold policies.

```yaml
--8<-- "SolarSimulator/harness/examples/iid_small.yaml"
```

## chain_small

Identical to `iid_small` except that the wind chain is on, with explicit aircraft-based bin edges `[5, 10]` m/s. Compare its run folder against `iid_small`'s to see the effect of wind persistence.

```yaml
--8<-- "SolarSimulator/harness/examples/chain_small.yaml"
```

## solarchain_small

Solar persistence only: the clear-sky-index chain with 3 stage-relative quantile bins, and wind still i.i.d. Turn `wind_chain` on as well for the joint run.

```yaml
--8<-- "SolarSimulator/harness/examples/solarchain_small.yaml"
```

## pvlib_small

Ineichen clear-sky normalization with the PVWatts panel. The first run builds `_ineichen`-suffixed artifacts and leaves the fatemi artifacts untouched.

```yaml
--8<-- "SolarSimulator/harness/examples/pvlib_small.yaml"
```

## whale_custom_small

`iid_small` with the whale profile given as `whale_block_values`. The values shipped here are the built-in `JoA2026` profile and the cruise fractions are the defaults, so the results are identical to `iid_small`. Edit either to depart from it.

```yaml
--8<-- "SolarSimulator/harness/examples/whale_custom_small.yaml"
```

## whale_encounter_small

`iid_small` using the encounter model `p = 1 − exp(−D·A·a(t)·g0·light(t))`: species card, per-location density, and survey platform. Setting `density_per_km2: null` and `g0: 1.0` reduces it to `JoA2026`.

!!! note "In progress"
    The encounter model and this config are part of the in-progress observation-model work on the current branch. See [Whale observation](../models/whale-observation.md).

```yaml
--8<-- "SolarSimulator/harness/examples/whale_encounter_small.yaml"
```

## historical_smoke

The smallest end-to-end check of the historical block-bootstrap rollout path: one capacity, one threshold pair, 100 episodes.

```yaml
--8<-- "SolarSimulator/harness/examples/historical_smoke.yaml"
```

## journal_threshold_vs_optimal

Reproduces the journal threshold-vs-optimal comparison (the matrix of `configs/final_journal_configs/threshold_sweep/observation_and_windspeed_threshold_config.yaml`). It sets `paper_figure: fig_threshold_sweep_combined`, so the run's figure is drawn by the journal figure code. See [Journal figures](figures.md).

```yaml
--8<-- "SolarSimulator/harness/examples/journal_threshold_vs_optimal.yaml"
```

## journal_threshold_vs_optimal_historical

The journal matrix with the policy solved on the fitted i.i.d. distributions, but episodes driven by 7-day blocks of the lat 30 / lon −90 historical record.

```yaml
--8<-- "SolarSimulator/harness/examples/journal_threshold_vs_optimal_historical.yaml"
```

## journal_threshold_vs_optimal_chain_historical

The chain-solved optimal policy against the threshold policy, both evaluated on historical weather.

```yaml
--8<-- "SolarSimulator/harness/examples/journal_threshold_vs_optimal_chain_historical.yaml"
```

## journal_threshold_vs_optimal_solarchain_historical

The solar-chain-solved optimal policy against the threshold policy on historical weather. Enable `wind_chain` as well for the joint wind + solar run.

```yaml
--8<-- "SolarSimulator/harness/examples/journal_threshold_vs_optimal_solarchain_historical.yaml"
```
