# Test catalogue

This page lists what each verification script and unit test checks. It is drawn from the
scripts' own docstrings and assertions, and the rendered docstrings are also on the
[verification scripts API page](../api/verify.md). How to run them:
[Verification](index.md).

## Verification scripts (`SolarSimulator/Tests/verify_*.py`)

### `verify_wind_chain.py` — Markov-modulated wind

Model: [Wind persistence chain](../models/wind-chain.md). Site lat 30 / lon −90, horizon 150,
4000 episodes. Builds the wind-chain artifact from `HISTORICAL_DATA` if it is missing.

| # | Check | Kind |
|---|---|---|
| 1 | Chain **off** (`n_bins == 1`) takes the unchanged i.i.d. path: solver table is 2-D and the sim runs. | reduction (exact) |
| 2 | With a rank-1 transition whose rows equal the per-stage Weibull bin masses, bin draw + within-bin truncated-Weibull sampling reproduces the i.i.d. Weibull marginal (mean and quantiles within tolerance). | reduction (mixture identity) |
| 3 | A chain-on optimal sim with that rank-1 transition matches the i.i.d. optimal sim's rollout statistics within Monte Carlo error. | reduction (statistical) |
| 4 | With the real fitted transition, the value table varies across wind bins (high-wind bin worth less) and failure statistics shift sensibly. | directional |

### `verify_solar_chain.py` — Markov-modulated clear-sky index

Model: [Solar persistence chain](../models/solar-chain.md). Same site and data as above.

| # | Check | Kind |
|---|---|---|
| 1 | Chain **off** (`n_solar_bins == 1`) is the unchanged i.i.d. path (2-D table, runs). | reduction (exact) |
| 2 | Solar bins are stage-Beta quantile bands (mass exactly 1/n), so the memoryless chain is the **uniform** transition; with it, within-bin sampling reproduces the full stage-Beta marginal and rollout statistics match the i.i.d. run within MC error. | reduction |
| 3 | With the fitted chain, the value table **rises** with the solar bin at a solar-valid mid-day stage, and failure/reward statistics shift. | directional |
| 4 | Joint wind + solar: value table is `(n_wind·n_solar, |S|, T)`; with the solar factor forced uniform, the joint rollout matches the wind-only chain within MC error (validates the Kronecker / joint-index plumbing). | reduction (plumbing) |
| 5 | Night/dawn structure: per-stage solar transition stack is identity at held (night) stages except exactly one dawn matrix per night gap; degenerate (saturated, κ≈1e6) stage Betas give finite conditional CDFs and samples. | structure / edge cases |

### `verify_pvlib_solar.py` — clear-sky models and PVWatts panel

Models: [Clear-sky models](../models/clearsky.md), [Solar panel](../models/solar-panel.md).

| # | Check | Kind |
|---|---|---|
| 1 | The `fatemi` path of `clearsky_ghi_series` reproduces the original `A·cos z` formula exactly, and `_fit_beta` with the default model reproduces the `clearsky_irradiance` stored in the existing expected-data pickle. | reduction (exact) |
| 2 | `haurwitz` and `ineichen` are zero at night, positive at midday, and below the `fatemi` envelope at noon. | sanity |
| 3 | PVWatts defaults reproduce the `constant` panel (area 0.65, efficiency 0.1); a hotter cell derates by `1 + γ_pdc·(T − 25)`; factory kwargs reach the panel. | reduction + directional |
| 4 | `StochasticWindSolarEnvironmentProvider` built with `solar_panel_model="pvwatts"` (defaults) gives bit-identical `sample_sunlight` draws to the `constant` build at the same seed; a hot cell scales energy by the derate. | reduction (exact) |
| 5 | `_resolve_data_path` is the identity for `fatemi` and suffixes other models before the extension. | wiring |

### `verify_whale_model.py` — whale series and cruise fractions

Model: [Whale observation](../models/whale-observation.md).

| # | Check | Kind |
|---|---|---|
| 1 | `block_series` with the built-in profile, `delta_t = 15` and a midnight start reproduces the original `np.repeat(block_values, 8)` / `np.tile` construction for horizons 1–1000; the legacy `create_series("JoA2026", horizon)` call, the deprecated `"real"` alias and `custom` with the same values agree. | reduction (exact) |
| 2 | `delta_t = 30` gives 4 steps per block; a 06:00 / 06:15 start rolls the series so stage 0 holds the 06:00–08:00 value; invalid inputs are rejected. | directional / validation |
| 3 | Default MDP observation gain is `[[0,1],[0,1],[0,0]]` and `reward` equals the original `actions * samples`; custom fractions scale only the takeoff and landing stages. | reduction + semantics |
| 4 | The analytical solver on `harness/examples/iid_small.yaml` gives **bit-identical** value tables with defaults, explicit `(1.0, 0.0)` fractions, `whale_series: custom` carrying the built-in values, and `whale_series: encounter` in focal form (`kogia_spp` card, `g0 = 1`, no density); `(0.5, 0.0)` changes the table. | reduction (exact) |
| 5 | `"auto"` fractions come from the seaplane climb/descent times and lie in (0, 1). | derived-parameter sanity |
| 6 | `summarize._fmt_list` formats list-valued config entries. | output |
| 7 | `start_datetime` alignment works through `SimulationFactory`. | wiring |
| 8 | Encounter model: focal form with `g0 = 1` equals `JoA2026`; species cards (`kogia_spp`, `humpback`, overrides, bare); swept area; density forms (scalar / monthly list / month map); `λ → 1 − exp(−λ)` against hand-computed values; fixed-hours and clear-sky light gates; end-to-end through `SimulationFactory` with a location density and the clear-sky gate. | reduction + unit values |

### `verify_runtime_opt.py` — vectorized solver and batched rollouts

Page: [Backward-induction solver](../architecture/solver.md),
[Episode simulation](../architecture/simulation.md). Options: `--horizon`, `--episodes`.

| Check | Criterion |
|---|---|
| Solver batch path vs the scalar `_value` formula, over sampled stages and actions | `max |batch − scalar| < 1e-8` |
| Fast index-arithmetic value lookup vs the original equality-scan lookup | `max |fast − scan| < 1e-12` |
| Batched episodes (threshold and optimal) vs the per-episode scalar baseline | statistical agreement of failure %, mean reward, mean flight hours; reports the speedup |
| Full-history episode shapes | `trajectory` is `(failure_step + 1, 2)`; `actions`/`rewards` have length `failure_step` |

Batched rollouts are **statistically**, not bit-for-bit, equivalent: episodes are no longer
re-seeded per index, so RNG order differs (commit `13fb2fb`).

### `verify_solver_speed.py` — solver speed-up

Runs an inlined copy of the original per-state scalar `solve()` loop on a fresh solver built
from the same MDP, times it against the vectorized `solve()`, and asserts the two value
tables agree to `< 1e-8`. The speed-up grows with horizon (the scalar loop is
O(horizon × states) Python calls, the vectorized one O(horizon)). Option: `--horizon`.

## Unit tests (`SolarSimulator/Tests/test_*.py`)

| File | What it tests |
|---|---|
| `test_transition_model_base.py` | Success-probability model classes (`LinearSuccessProbability`, `OnlySuccessProbability`, `TestSuccessProbability`, `RealisticSuccessProbability`, `OptimisticSuccessProbability`): vectorized output length, broadcasting of every model in `ProbabilityModelFactory.list_models()` over scalar and array inputs, and spot values. See [Transition models](../models/transition-models.md). |
| `test_environment_provider_base.py` | `AbstractEnvironmentProvider` contract (abstract methods enforced); `StochasticWindSolarEnvironmentProvider` parameter getters (Weibull shape/scale, Beta α/β, clear-sky irradiance), seeding and `reset`, sample shapes and positivity, that samples follow the fitted Weibull and Beta distributions; the regime-switching and sinusoidal providers' initialization, diurnal solar pattern, and wind-scale periodicity. |
| `test_climb_power.py` | `Seaplane.climb_energy`: energy positive for a climb, monotone in altitude gained, a zero climb angle rejected, and agreement with the previous implementation. See [Aircraft power model](../models/aircraft-power.md). |
| `test_reward.py` | Standalone script (`python Tests/test_reward.py`): the vectorized reward (`actions * samples − penalty on broken`) matches the original `np.where` formulation on random batches. |

Fixtures for the older tests live in `Tests/test_data/` (sample pickles and
`simulation_params.yaml`).
