# Glossary

Terms and symbols used in the code, the configs, and the design records. Each entry links
to the page that defines it. The design records carry their own full symbol tables:
[wind-persistence math](../design/wind-persistence-math.md) (end of page) and
[wind-persistence theory](../design/wind-persistence-theory.md) (appendix).

## Model and MDP

AHABS
:   Autonomous Hydrofauna Aerial Biome Surveyor: the solar-powered seaplane this
    simulator models. See [Home](../index.md) and [Project context](project-context.md).

Stage, `t`, `delta_t`
:   One control step of the finite-horizon MDP. Its length is `delta_t` minutes, 15 by
    default. Stages are indexed `0 … T−1`. See [The MDP](../architecture/mdp.md).

Horizon, `T`
:   The number of stages in a mission (`horizons` / `horizon` in the config). At 15-minute
    stages, 2880 stages is 30 days. See [Config reference](../experiments/config-reference.md).

State, `s = (σ, m)`
:   The endogenous state: battery state of charge `σ` and operating mode `m`. With the
    persistence chains on, it is augmented by the exogenous regime index. See
    [The MDP](../architecture/mdp.md).

SoC, `σ`
:   Battery state of charge. The solver discretizes it in steps of `energy_increment_wh`.

Mode, `m`
:   `0` = moored (on the water, charging), `1` = flying, `2` = broken (absorbing). The
    rows of `AbstractMDP.observation_gain` follow the same order.

Action, `a`
:   Binary. `0` stays moored (when moored) or lands (when flying). `1` takes off (when
    moored) or keeps flying (when flying).

Broken state, failure
:   The absorbing state entered on a failed takeoff or landing (wind-driven, from the
    [transition model](../models/transition-models.md)) or when the battery is depleted.
    Entering it costs the failure penalty.

Failure penalty, `c_f`
:   The negative reward on entering the broken state (`failure_penalties`). Reward is
    therefore risk-inclusive; see [Findings](../findings.md) for the operating-point caveat.

Reward
:   The per-stage whale-observation probability earned while surveying, multiplied by the
    observation gain, minus the failure penalty on failure. See
    [Whale observation](../models/whale-observation.md).

Value function, value table, `V`
:   The backward-induction solution. It is 2-D `(|S|, T)` for i.i.d. weather and 3-D
    `(n_regimes, |S|, T)` with a chain. Saved as `.npy` under `solver_tables/`. See
    [Solver](../architecture/solver.md).

Transition model
:   A success-probability curve for takeoff and landing as a function of wind speed
    (`transition_model`: `moderate`, `linear`, `realistic`, …). See
    [Transition models](../models/transition-models.md).

## Policies

Optimal policy
:   The greedy policy with respect to the solved value function
    (`OptimalContinuousAnalyticalPolicySimulation`). See [Policies](../models/policies.md).

Threshold policy
:   A heuristic (`UnifiedThresholdContinuousSimulation`). It takes off when the observation
    probability is above an observation threshold, the wind is below a wind threshold, and
    the battery is sufficiently charged. The *greedy* policy is the special case of
    observation threshold 0. See [Policies](../models/policies.md).

i.i.d.-optimal / chain-optimal
:   The optimal policy solved with memoryless weather, or with the wind (and optionally
    solar) chain. Both can be evaluated on the same historical weather.

Best tuned threshold
:   The threshold combination with the highest mean reward in a cell of the threshold
    grid. It is the benchmark in the [threshold benchmark](../design/wind-threshold-benchmark.md).

## Weather

Weather mode (iid / chain / historical)
:   How weather is generated. **iid**: independent per-stage draws from fitted
    distributions. **chain**: Markov-modulated draws. **historical**: block-bootstrap
    replay of the real record, used for evaluation only. See
    [Weather modes](../models/weather-modes.md).

Wind bin, `β`, `n_bins`
:   A discrete wind-speed interval. Its edges are quantiles of the site's record or
    configured `bin_edges`. The current bin is the wind chain's state. See
    [Wind persistence chain](../models/wind-chain.md).

Within-bin (truncated) Weibull
:   The continuous wind distribution inside the current bin: the stage Weibull truncated
    to the bin. It keeps the failure integral resolved.

Clear-sky index, `K`
:   `K = GHI / G_cs`: irradiance normalized by a clear-sky envelope, modeled as a Beta
    variable per stage. See [Clear-sky models](../models/clearsky.md).

Clear-sky model
:   The envelope `G_cs`: `fatemi` (`1150·cos z`, the default), `haurwitz`, or `ineichen`
    (`clearsky_model`).

Solar bin, `g`
:   A stage-relative Beta-quantile band of `K`. It is the state of the solar chain. See
    [Solar persistence chain](../models/solar-chain.md).

Dusk→dawn matrix
:   A per-month transition applied once before each dawn. It carries solar regime
    persistence across the night. At night stages the solar transition is the identity.

Exogenous (regime) index, `z`
:   The flattened joint weather state `z = b·n_g + g` over wind bin `b` and solar bin `g`,
    with joint transition `P = kron(P_wind, P_solar)`.

Rank-1 / memoryless chain
:   A transition matrix whose rows are all equal. The next bin does not depend on the
    current one, so the chain collapses to i.i.d. This is used in verification.

Time-conditioned chain
:   Transition matrices indexed by `(month, hour)`. The wind artifact has shape
    `(13, 24, n, n)`.

Historical block bootstrap
:   The evaluation weather: consecutive blocks (default 7 days) drawn from random years of
    the real record, with wind and solar from the same year within a block.

Histcube
:   The `_histcube` artifact: historical wind and solar packed into a
    `(slots_per_year, n_years)` calendar cube for fast bootstrapping. See
    [Weather data pipeline](../models/weather-data.md).

Artifact
:   A derived data file next to the expected-data pickle: `_windchain`, `_solarchain`, or
    `_histcube`, plus a `_<model>` suffix for non-default clear-sky models. The harness
    builds missing artifacts on demand.

## Whale observation

Whale series
:   The per-stage observation-probability profile (`whale_series`: `JoA2026`, `custom`,
    `encounter`, `constant`, `sinusoidal`). See [Whale observation](../models/whale-observation.md).

Block series
:   A daily profile given per block of hours (default 2 h), expanded to stages and rolled
    to the mission start time.

`p = ψ·a·g`
:   The factorised detection model from the literature: presence `ψ`, availability `a`
    (fraction of time at the surface), and perception `g` (P(seen | surfaced)). See the
    [literature survey](../design/whale-observation-literature.md).

Encounter model, `λ`
:   `λ(t) = D·A·a(t)·g0·light(t)` expected sightings per stage, and `p = 1 − exp(−λ)`.
    `D` is density, `A` is swept area, and `light` is the daylight gate. With no density
    it is the *focal* form `p = a·g0·light`.

`g0`
:   Trackline perception probability.

Species card
:   A named parameter set in `SPECIES_CARDS` (for example `kogia_spp`, `humpback`) with its
    citations.

Cruise fraction
:   The share of a takeoff or landing stage spent surveying at cruise
    (`takeoff_cruise_fraction`, `landing_cruise_fraction`). It scales that stage's reward
    through the observation gain. `auto` derives `1 − t_maneuver/delta_t`.

Observation gain
:   A `(mode, action)` table multiplying the stage observation reward: 1 while cruising,
    the cruise fraction on maneuver stages, and 0 while moored.

## Experiments and statistics

Harness
:   The declarative YAML experiment runner (`harness/run_experiment.py`). See
    [Running experiments](../experiments/index.md).

Cell
:   One scenario of a sweep: location × capacity × start date × horizon.

Native world / historical world
:   A policy evaluated on the weather model it was solved with (native), or on the
    historical bootstrap (historical). Comparing the two is the calibration gap.

Common random numbers (CRN)
:   Paired runs draw identical weather, episode by episode, because the provider RNG is
    reset with a fixed seed. This makes per-episode deltas valid. It is checked with
    `compare_chain_sweep.py --verify`.

CVaR₁₀
:   Conditional value at risk: the mean episode reward over the worst 10% of episodes.

CMI
:   Conditional mutual information, for example `I(Next ; Prev | Curr, month, hour)`. It
    is the extra information a higher-order chain would carry. It is used in the
    persistence prechecks and bias-corrected with a permutation null.

Precheck
:   A diagnostic run before building a model extension, to test whether the persistence is
    worth modeling (`Scripts/wind_persistence_precheck.py`, `solar_persistence_precheck.py`).

`V0`
:   The solver-predicted value at stage 0 for the initial state.
