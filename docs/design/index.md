# Design records

Design records are the long-form documents that explain **why** the model is built the
way it is: derivations, validity arguments, prechecks, benchmarks, and literature. They
were written alongside the work and moved here from `SolarSimulator/docs/` unchanged,
except for link fixes. Each one states its own status and date in its header.

!!! note "Paths in the documents"
    Source paths quoted inside the records (for example `BaseClasses/run_sim.py` or
    `harness/summarize.py:52`) are relative to `SolarSimulator/` and give line numbers
    as of when each record was written. Cross-references between records link to the
    pages listed below.

## Records

### Wind persistence

| Record | Purpose |
|---|---|
| [Briefing](wind-persistence-briefing.md) | Change summary and advisor briefing: what changed, how it works, sanity checks, the single-site empirical result, limitations, talking points and objections. Start here. |
| [Mathematics](wind-persistence-math.md) | Formula reference: wind binning, within-bin truncated Weibull, the time-conditioned chain, binned failure probability, augmented Bellman recursion, i.i.d. reduction, rollout, the CMI precheck, cost, and a symbol table. |
| [Theory](wind-persistence-theory.md) | Paper-style argument: correlated wind breaks the Markov property of the original state, augmentation with the bin restores it, the marginal is preserved, the model reduces to the baseline, and first order is the right order. A typeset PDF is also available: [wind-persistence-theory.pdf](wind-persistence-theory.pdf). |
| [Validity and implementation](wind-persistence-validity.md) | Correctness argument plus a file-by-file before/after implementation diff (§9). This is the best code map for the wind chain. |
| [Threshold benchmark](wind-threshold-benchmark.md) | Three-way comparison (i.i.d.-optimal vs chain-optimal vs best tuned threshold) over the 34-cell historical sweep, the Hawaii within-bin diagnosis, and the 4-quantile-bin rerun. |
| [Held-out validation](apl-holdout-validation.md) | Gulf-site check with the model fitted on 1950–2004 and evaluated on 2005–2022, against an in-sample twin, at failure penalties 5, 20 and 50. Chain gains survive the year split; i.i.d.-optimal falls behind the tuned threshold only at penalty 20 and above. |

### Solar persistence

| Record | Purpose |
|---|---|
| [Mathematics and design record](solar-persistence-math.md) | The clear-sky-index Beta chain: stage-relative quantile bins, the dusk→dawn night channel, Kronecker composition with the wind chain, precheck results, verification, and cost. |

### Whale observation

| Record | Purpose |
|---|---|
| [Literature survey](whale-observation-literature.md) | Aerial visual and passive-acoustic detection literature, species-agnostic parameter ranges, where the implemented `encounter` model comes from, the Tier 0–3 integration design, and open questions. |
| [Model extensions](whale-observation-extensions.md) | Decision-theoretic options for the observation model: occupancy vs detection, POMDP/MOMDP, Bayes-adaptive, robust MDP, path planning, MOMDP sizing, and recommended sequencing. Its "nothing implemented" status line predates Tier 0/1; see [Whale observation](../models/whale-observation.md) for what exists now. |

Results that cut across these records are collected in [Findings](../findings.md).

## Decision log

The table lists decisions that shape the current code, with the reason recorded at the
time. Sources are the git commit bodies (from June 2026 on), the design records above,
and the runtime-optimization handoff notes. Earlier history (2025, the journal-paper
period) has one-line commit messages and no recorded rationale.

| Date | Decision | Reason (as recorded) | Source |
|---|---|---|---|
| 2026-06-03 | Vectorize the backward-induction solve and run all episodes of a simulation as one batch. | The two pure-Python hot loops dominated runtime; ~28× faster solve at horizon 200 and ~800–1300× faster rollout. | `13fb2fb` |
| 2026-06-03 | Accept **statistical**, not bit-for-bit, equivalence for batched rollouts. | Episodes are no longer re-seeded per index, so RNG order changes; aggregate statistics match within MC error. | `13fb2fb` |
| 2026-06-03 | Keep the original scalar code paths (`_value`, `simulate_episode`, scalar `choose_action`) alongside the vectorized ones. | Lets verification be an in-process new-vs-old diff. | `13fb2fb`, handoff notes |
| 2026-06-03 | Do **not** automatically reuse pre-solved value-function tables; leave only the filename scheme as a hook. | Risk of silently using stale results. | handoff notes |
| 2026-06-03 | Add wind persistence as a first-order, (month, hour)-conditioned Markov chain over wind bins, with the bin as an exogenous state variable; **default off**; `n_bins == 1` is a separate, unchanged path. | Real wind is autocorrelated; augmenting the state keeps the problem Markov and exactly preserves the published i.i.d. model as a special case. | `d73b543` |
| 2026-06-03 | Bins govern persistence, but the failure integral keeps a **continuous within-bin** (truncated Weibull) wind distribution. | Keeps the steep failure-probability curve resolved. | `d73b543` |
| 2026-06-03 | Frame the extension as *adding* persistence while preserving the Markov property, not as "fixing a broken Markov property". | The i.i.d. model was already self-consistent; weather was simply memoryless. | `d73b543`, [Validity](wind-persistence-validity.md) |
| 2026-06-03 | Evaluate policies on real weather with a historical **block bootstrap** (7-day blocks, same year for wind and solar within a block), while still **solving** on fitted distributions. | Preserves real persistence and wind/solar cross-correlation in evaluation; lets i.i.d.- and chain-solved policies be evaluated unchanged. | `3ee1da7` |
| 2026-06-03 | Bootstrap provider stores a lazy `(n, n_blocks)` year-assignment matrix instead of pre-allocated weather arrays. | Memory fell from O(H·n) (~1.4 GB per sim) to O(n·n_blocks) (~1.2 MB). | `b40ae34` |
| 2026-06-04 | Support user-specified `wind_chain.bin_edges`, with quantile derivation as the fallback; rebuild stale artifacts automatically. | Allow aircraft-threshold bins; keep artifacts consistent with the config. | `81e3d4d` |
| 2026-06-04 | Match historical windows on calendar position (month/day/hour/minute), ignoring the year. | Journal configs use future years, but historical data ends in 2022. | `4d1ac02` |
| 2026-06-04 | Keep printed output ASCII-only. | Windows cp1252 consoles raise `UnicodeEncodeError` on em dashes and arrows. | `0460bf4` |
| 2026-06-04 | Draw the trajectory figure from a saved full-history episode in the HDF5 rather than a synthetic replay. | Shows what the simulator actually did; needs `full_history_episodes > 0` or `save_states`. | `a104606` |
| 2026-06-28 | Google-style docstrings across the codebase. | Not recorded (one-line commit). The API pages parse docstrings in this style. | `6b43c88` |
| 2026-07-03 | Sweep configs use **4 equal-occupancy quantile wind bins** per location instead of fixed `[5, 10]` m/s edges. | Fixed edges pooled the decision-relevant ~6 m/s band at Hawaii into one bin; with quantile bins, chain-optimal beats the best threshold in 34/34 cells. | `3598426`, [Threshold benchmark](wind-threshold-benchmark.md) §9 |
| 2026-07-03 | Keep full episode histories only on the baseline-location capgrid pair; scalars everywhere else. | Disk: a scalars-only 210-sim config is ~40 MB instead of ~1.7 GB; the histories are needed only for the composite and trajectory figures and the CRN check. | `3598426` |
| 2026-07-03 | Solar persistence chain on the clear-sky index uses **stage-relative Beta-quantile bins**, not global edges. | The hour-averaged GHI record biases K low near the terminator, so global bins degenerate at dusk and dawn; rank bins recover the day-to-day channel and make the conditional CDF edge-free. | `59939eb`, [Solar persistence](solar-persistence-math.md) §2 |
| 2026-07-03 | Handle night entirely in the per-stage solar transition stack: identity at night stages and one fitted per-month dusk→dawn matrix before each dawn. | Carries multi-day cloud-spell persistence (day-scale first-order MI ≈ 9.4% of daily entropy; higher order 0.5%). | `59939eb`, [Solar persistence](solar-persistence-math.md) §3, §5 |
| 2026-07-03 | Compose wind and solar through a flattened exogenous index with `P = kron(P_wind, P_solar)`. | Keeps the value table 3-D; a fitted joint chain can later replace only `get_exo_transition`. | `59939eb` |
| 2026-07-03 | Store per-episode scalars as **columnar** HDF5 datasets (`episode_scalars/<field>`), with episode groups only for full histories; all readers accept both layouts. | Each HDF5 object costs ~0.5 KB of metadata; the columnar layout is ~88× smaller. | `32e0d4a` |
| 2026-07-03 | Debias the stratified CMI with a permutation-null floor rather than Miller–Madow. | The plug-in estimate over up to 288 strata is biased upward; the permutation null preserves the Curr→Next structure that an analytic correction would miscount. | `8356b5a` |
| 2026-08-20 | `clearsky_model` switch (fatemi default, haurwitz, ineichen); non-default models write `_<model>`-suffixed artifacts. | Allows other envelopes without ever overwriting the fatemi artifacts. | `76812fa` |
| 2026-08-20 | PVWatts panel folded into the linear `area × efficiency` interface at a fixed cell temperature. | Defaults reproduce the constant panel bit for bit, and the solver's linear-in-K integration is unchanged. | `76812fa` |
| 2026-08-29 | Whale observation parameters are **species-agnostic ranges** meant to be swept. | Literature values vary widely by species and site. | [Literature survey](whale-observation-literature.md) §5 |
| 2026-08-29 | The first acoustic (hydrophone-while-moored) channel is **reward-only**, with no hidden presence state; the POMDP/MOMDP option is deferred. | Only worth the state-space cost once the channel is shown to change decisions. | `6855bd3`, [Model extensions](whale-observation-extensions.md) |
| 2026-08-29 | Whale block series derives steps-per-block from `delta_t` and rolls to `start_datetime`'s time of day. | The old code hard-coded 8 steps per block and indexed from midnight, misaligning the whale profile with the weather for non-midnight starts. | `f281dca` |
| 2026-08-29 | Takeoff and landing stages earn the observation reward scaled by `takeoff_cruise_fraction` / `landing_cruise_fraction`; defaults 1.0 / 0.0; `auto` derives `1 − t_maneuver/delta_t`. | Credits the part of a maneuver stage spent surveying at cruise; the defaults reproduce the original model exactly. | `f281dca` |

!!! tip "Adding to the log"
    Record new decisions in the commit body, following the convention since June 2026:
    what changed, **why**, and how it was verified. Then add a row here.
