# Markov Chain Effectiveness Sweep (markov_ablation)

**Status: Phases 0–5 COMPLETE (main sweep done); Phase 6 (solar-resolution follow-up) IN PROGRESS.** Update the Decision Log at the bottom as checkpoints resolve. This document plus the config manifests is the complete, resumable state of the experiment: if work is interrupted at any point, read this file top to bottom and continue from the first unchecked phase.

> **Phase 6 addendum (2026-07-07):** the main sweep found solar-only ≈ 0 benefit at n_bins=2. Before concluding solar persistence is worthless, re-test it at higher bin resolution across ALL locations. See **§6** at the bottom for design, commands, and decision rules.

## 1. Question

How much does modeling weather *persistence* (Markov chains) improve the optimal dispatch policy over an IID-solved policy and a threshold benchmark — separately for solar (clear-sky-index chain) and wind (speed-bin chain), and jointly?

Five arms, all evaluated on **identical historical block-bootstrap weather episodes** (common random numbers):

| arm | wind_chain | solar_chain | policy |
|---|---|---|---|
| `iid` | off | off | optimal solve, IID weather model |
| `wind` | on, 5 bins, explicit path | off | optimal solve, wind persistence |
| `solar` | off | on, n_bins from Phase 2 checkpoint | optimal solve, solar persistence |
| `joint` | on (5 bins) | on | optimal solve, joint table `(5·n_g, |S|, T)` |
| `thresh` | off | off | threshold policy grid, no solve |

### Metrics (all four requested)
1. **Paired per-episode Δreward vs `iid`** with 95% bootstrap CIs (episodes joined on `episode_index`; valid because `simulate_multiple_episodes` calls `env_provider.reset(0)` before every batch and the bootstrap weather is the first RNG draw → identical weather across arms).
2. **Fraction of gap closed**: `(R̄_arm − R̄_thresh*) / (R̄_best − R̄_thresh*)`, threshold\* chosen by full-sample mean with split-half winner's-curse check.
3. **Failure rate + cost decomposition**: failure %, penalty cost `fp·failure_rate`, non-penalty reward `mean(total_reward + fp·failure)`.
4. **Interaction trends**: Δ vs capacity, penalty, season, and site persistence score.

## 2. Grid (full sweep, Phase 4)

| Dimension | Values |
|---|---|
| Locations | Florida (27.0, −79.5), Hawaii (21.0, −158.0), N. Atlantic (45.0, −45.0), Bering (65.0, −169.0) |
| Battery capacity (Wh) | 150, 300, 600 |
| Failure penalty | 5, 20, 80 |
| Start dates | 2025-06-10T00:00:00, 2025-12-10T00:00:00 |
| Horizon | 5760 steps (60 days @ 15 min) |
| Episodes | 3000 |
| Eval world | `historical_weather: {enabled: true, block_length_days: 7}` (hist only, no native runs) |
| Threshold grid (`thresh` arm) | `threshold_values: [0.0, 0.1, 0.2, 0.3]` × `wind_thresholds: [0.0, 4.0, 8.0, 12.0]`, **`failure_penalties: [5.0]` only** |

20 configs = 4 locations × 5 arms (`configs/markov_ablation/mkv_{loc}_{arm}.yaml`). One location per config (value-table filenames omit location); caps × penalties × dates cross inside each config. 288 optimal solves + 384 threshold rollouts; est. **10–12 h wall @ 16 workers**.

### Key design facts (verified in code, 2026-07-06)
- **Threshold policies are penalty-invariant in behavior** (`choose_action_batch` uses thresholds+SoC only; penalty enters reward once at failure). The `thresh` arm therefore runs at fp=5 only; other penalties are recomputed per episode: `reward(fp) = total_reward + 5·failure − fp·failure`. **Never use threshold `summary.csv` rows for fp ∈ {20, 80}.**
- Episode i's bootstrap weather is invariant to the total episode count (`_year_choice` drawn row-major) → a later, larger rerun stays paired with this one.
- `full_history_episodes: 8` on the five **Florida** configs only (feeds `verify_crn`); 0 elsewhere.

### ⚠ Artifact-path safety rules
- `Data/EXPECTED_DATA/*_windchain.pkl` (default derived path) are **12-bin** artifacts from the failbin dim sweeps. Configuring a different `n_bins` at the default path **destructively rebuilds them**. All wind arms use an explicit `wind_chain.path`.
- The wind arms reuse `Data/EXPECTED_DATA/data_expected_{loc}_15min_windchain_wind5.pkl` — verified 2026-07-06 for all four sites: `n_bins=5, quantile_derived=True`, transition `(13, 24, 5, 5)`. Do not rebuild; Phase 3 gate checks their mtimes are unchanged.
- Solar artifacts (`*_solarchain_g{n}.pkl`, explicit per bin count) do not exist yet; they are built during Phase 2/provisioning (needs pvlib env).

## 3. Execution

Python: `C:\Users\bepstein8\AppData\Local\anaconda3\envs\pvlib\python.exe` (conda not on PATH; invoke directly). Working dir: `SolarSimulator/`. All long runs in the background; `--resume` skips configs that already have a `summary.csv` under the out dir. Per-config wall times land in `<out>/sweep_run_log.json`. Consider pausing OneDrive sync on `results/` during long runs.

### Phase checklist
- [ ] **Phase 0** — this document written. ✔ once committed alongside Phase 1 code.
- [ ] **Phase 1** — code:
  1. `SolarSimulator/Scripts/generate_markov_ablation_configs.py` (new; pattern-copy of `generate_chain_sweep_configs.py`; flags: `--solar-bins N` default 3, `--smoke-solar-bins`, `--smoke`, `--out`).
  2. `SolarSimulator/Scripts/run_chain_sweep.py::provision_all` patched to provision per-config (solar chains + explicit wind paths), not the wind-only superset.
  3. `SolarSimulator/Scripts/compare_markov_ablation.py` (new; multi-arm analysis reusing `compare_chain_sweep.py` helpers).
- [ ] **Pre-check**: `python Tests/verify_solar_chain.py` passes (~2 min; validates joint Kronecker plumbing).
- [ ] **Phase 2 — solar bin-resolution smoke** (~15 min):
  - Generate: `python Scripts/generate_markov_ablation_configs.py --smoke-solar-bins`
  - Run: `python Scripts/run_chain_sweep.py --configs ../configs/markov_ablation_smoke_solar --out ../results/markov_ablation_smoke_solar --workers 16`
  - Florida only; `solar` arm at n_bins ∈ {2, 3, 4, 5} + `iid` reference; 300 Wh, fp 20, June start, horizon 2880, 2000 episodes, `full_history_episodes: 8`.
  - Analyze: `python Scripts/compare_markov_ablation.py --solar-bins-study ../results/markov_ablation_smoke_solar`
  - **CHECKPOINT / decision rule**: choose the smallest n_bins whose paired Δreward-vs-IID is within the 95% CI of the best n_bins (knee rule); tie → 3. Record below; pass as `--solar-bins` to all later generation.
- [ ] **Phase 3 — joint all-arm smoke gate** (~10 min):
  - Generate: `python Scripts/generate_markov_ablation_configs.py --smoke --solar-bins <chosen>`
  - Run: `python Scripts/run_chain_sweep.py --configs ../configs/markov_ablation_smoke --out ../results/markov_ablation_smoke --workers 16`
  - Florida, all 5 arms, horizon 480, 300 episodes, fp 5, 300 Wh, June start + one 1-cap winter (2025-12-10) config.
  - **PASS GATE (all required)**: (a) `verify_crn` PASS for iid↔wind, iid↔solar, iid↔joint, iid↔thresh; (b) joint value table shape `(5·n_g, |S|, 480)` in `solver_tables/`; (c) sane `summary.csv` rows for all arms; (d) mtimes of `*_windchain.pkl` and `*_windchain_wind5.pkl` unchanged; (e) winter config runs without error.
- [ ] **Phase 4 — full sweep** (overnight, ~10–12 h):
  - Generate: `python Scripts/generate_markov_ablation_configs.py --solar-bins <chosen>`
  - Run: `python Scripts/run_chain_sweep.py --configs ../configs/markov_ablation --out ../results/markov_ablation --workers 16 --resume`
  - Config order: thresh/iid first (fast feedback), joint last per location. If interrupted: re-run the same command with `--resume`.
  - Memory note: worst joint table ≈ `(5·n_g)·483·5760·8 B` (~550 MB at n_g=5); drop to `--workers 8` if RAM pressure.
- [ ] **Phase 5 — analysis**:
  - `python Scripts/compare_markov_ablation.py --results ../results/markov_ablation --verify`
  - Outputs: `results/markov_ablation/_analysis/markov_ablation_cells.csv`, `markov_ablation_report.md`, figures (forest plots per location, gap-closed bars, decomposition stacks, interaction trends).
  - Sanity: joint ≥ max(solar, wind) within CI in most cells; threshold penalty re-weighting cross-checked at fp=5; flag any cell where IID materially beats a chain arm.

## 4. Known risks
- Joint path unit-tested (`Tests/verify_solar_chain.py` §4) but never run at scale — that is what the Phase 3 gate exists for.
- Bering summer cells may have degenerate failure rates at some penalties; gap-closed is guarded (falls back to absolute deltas when |gap| < bootstrap noise).
- Winter starts never run before (histcube keys month/day/hour, so 2025-12-10 should work) — Phase 3 includes a winter config to confirm.

### Incident note (2026-07-06, benign): CWD-relative artifact paths
The configs' relative `Data/...` chain paths resolve against the run CWD (`SolarSimulator/`), so provisioning wrote chain artifacts to a shadow dir `SolarSimulator/Data/EXPECTED_DATA/` and the full sweep's provisioning "rebuilt" the wind5 artifacts there (repo-root failbin originals verified untouched, and rebuilt bin edges verified identical — deterministic quantile builds from the same historical record). Runs load via the same CWD-relative resolution, so all solves/rollouts are self-consistent. Solar artifacts were copied to repo-root `Data/` (no overwrites) so analysis (which resolves from repo root) and future runs are CWD-independent. If touching this later: make `_ensure_location_data`/`_load_wind_chain`/`_load_solar_chain` resolve chain paths with `_abspath` (repo-root-relative) like `data_path`.

## 6. Phase 6 — Solar bin-resolution follow-up (2026-07-07)

**Why:** the main sweep ran solar at n_bins=2 (chosen on an underpowered Florida-only smoke) and found ≈0 benefit, even hurting Hawaii. Before declaring solar persistence worthless, re-test it at higher resolution across all sites under full conditions.

**Design:** solar-*only* arm (wind IID) at `n_bins ∈ {2, 3, 4, 6, 8}`, plus an iid reference, at every location — full sweep conditions (3 caps × 3 penalties × 2 seasons × 5760-step horizon × 3000 episodes). 24 configs (4 loc × (iid + 5 solar)), 432 optimal solves. Each solar-g sim pairs against the same-condition iid sim on identical bootstrap weather. Configs `configs/markov_solar_res/`, results `results/markov_solar_res/`.

**Commands** (pvlib env, from `SolarSimulator/`):
```
python Scripts/generate_markov_ablation_configs.py --solar-res            # --solar-bins-list "2,3,4,6,8"
python Scripts/run_chain_sweep.py --configs ../configs/markov_solar_res --out ../results/markov_solar_res --workers 16 --resume
python Scripts/compare_markov_ablation.py --solar-res-study ../results/markov_solar_res
```
Provisions solar artifacts at g6/g8 (and any missing g3/g4) for all sites first (fast). Est. runtime ≈ 5–6 h at 16 workers (solar solve ∝ bins; g8 ≈ 4× g2). Analysis emits `solar_res_summary.csv` (mean paired Δreward vs IID per location + POOLED, with cell-bootstrap CIs), `solar_res_cells.csv`, `solar_res_verdict.json`, `solar_res_curve.png`.

**Decision rules** (applied to the POOLED-across-locations mean paired Δreward vs IID, `best_g` = argmax):
1. **Significant-benefit gate** — if the POOLED CI at `best_g` is *not* strictly above 0, conclude solar persistence gives no benefit at any tested resolution: **do not** run a joint rerun, report and stop.
2. **Clear winner** — if significant AND no other bin count has a pooled mean within `best_g`'s CI (i.e. `tie_set` empty): pick `best_g`, then **rerun the joint (solar+wind) arm at that resolution** across all conditions and compare against the existing wind-only results (does solar add on top of wind?).
3. **No clear winner** — if significant but several resolutions are statistically tied at the top: **ask the user** which n_bins to use (offer the tied set), then run the joint rerun at their choice.

The verdict JSON records `best_g`, pooled Δ + CI, `significant_benefit`, `clear_winner`, `tie_set` — read it to route the decision.

> **Scope caveat on the solar null (added 2026-07-18):** the clear-sky-index chain was
> fitted and evaluated on HOURLY reanalysis irradiance (linearly interpolated to the
> model step). Sub-hourly cloud-field variability — where short-horizon cloud
> persistence actually lives — is absent from the data by construction, so the correct
> statement of this phase's conclusion is *"solar persistence carries no actionable
> dispatch signal at hourly data resolution"*, not that solar persistence is worthless
> per se. Testing the stronger claim needs native sub-hourly GHI (e.g., NSRDB 5–30 min)
> — see the TODO list (§10). The wind conclusion is not affected in the same way: the
> Phase 7 Δt=60 study reran wind on the native (uninterpolated) hourly record and the
> benefit survived.

## 7. Phase 7 — Timestep-resolution check, delta_t=60 (2026-07-11)

**Why:** everything so far ran at dt=15 on hourly weather linearly interpolated to 15 min, which inflates short-lag persistence. At dt=60 the model runs on the NATIVE hourly record (60-min resample of hourly data is the identity), eliminating the interpolation artifact entirely. If the wind-chain benefit survives at dt=60, the +13.8 finding is robust to both timestep and interpolation-kernel choices.

**Design:** iid + wind arms only (solar settled by Phase 6), 4 locations × 3 caps × 3 penalties × 2 seasons, `delta_t: 60`, `horizons: [1440]` (= 60 days), 3000 episodes, 60-min data artifacts (`_60min.pkl`) and a wind chain **fitted at the 60-min step** (per-step transition matrices don't transfer across dt). 8 configs, 144 solves. Configs `configs/markov_dt60/`, results `results/markov_dt60/`.

**dt-hardcoding fixes applied first** (audit 2026-07-11; all identity at dt=15, verified numerically + full `verify_solar_chain.py` regression):
1. `whale_base.py` — "real" whale series was laid out per 15-min step (`repeat(...,8)`, 96/day); now parameterized on delta_t (was stretching the diurnal pattern 4× at dt=60). Threaded through `run_sim.py:138,275`.
2. `create_weather_distributions.py:395` — wind-chain fit gated consecutive pairs on `== 15` min; now infers the step (median spacing) like the solar fit. At dt=60 the old gate silently degenerated the chain to uniform.
3. `seaplane_base.py` — takeoff/landing average power spread fixed maneuver energy over a hardcoded 900 s; now over the real step (was 4× maneuver energy at dt=60 → artificial endurance collapse). Threaded via `run_sim.py:_compute_power_params`.
4. `simulation_base.py:601` — threshold-policy battery reserve hardcoded 15 min → `mdp.delta_t`.
5. `simulation_base.py:414,137,146` — `flight_hrs` divided by 4 (15-min steps) → `* delta_t/60`.
6. `mdp_base.py` — constant floating/flying per-step hazards rescaled `p_step = 1-(1-p_15)^(dt/15)` so cumulative wall-clock risk is dt-invariant (sigmoids are per-maneuver, unscaled).

**Commands:**
```
python Scripts/generate_markov_ablation_configs.py --dt60
python Scripts/run_chain_sweep.py --configs ../configs/markov_dt60 --out ../results/markov_dt60 --workers 16 --resume
```
Provisioning builds per-site 60-min expected-data, histcube, and `_60min_windchain_wind5.pkl` artifacts (new names — nothing 15-min is touched). Smoke gate before the full run: florida iid+wind at horizon 120 (5 days), 300 episodes; check the fitted 60-min chain is NOT uniform (fix #2), whale series alignment, and sane failure/flight-hrs magnitudes.

**Readout:** wind−iid paired Δ at dt=60 vs the dt=15 main sweep on comparable units — Δreward per mission-day (raw totals differ: 1440 vs 5760 steps; per-step whale rewards are wall-clock aligned so per-day is comparable) and Δfailure/Δflight-hrs. Expect the same qualitative story; quantitative shrinkage would indicate part of the dt=15 chain benefit rode on interpolation-inflated persistence.

## 8. Phase 8 — Thesis sweep: battery resolution × penalty resolution × mission duration (2026-07-11)

**Purpose:** thesis-chapter dataset. Fine-grained battery and penalty response surfaces plus a mission-duration curve to one year, for arms **iid**, **wind chain (5 bins)**, and **threshold**, at 5 sites.

**Design (star around 300 Wh / 60 d / June+Dec):**
| Family | Grid | Solves |
|---|---|---|
| `batgrid` | caps **100:50:600 Wh** (11) × penalties **{2.5, 5, 10, 20, 40, 80}** (6) × 2 start dates (2025-06-10, 2025-12-10) × 5 loc × {iid, wind} @ 5760 steps (60 d) | 1320 |
| `duration` | horizons **{30, 60, 90, 180, 270, 365} d** = {2880, 5760, 8640, 17280, 25920, 35040} steps × penalties {5, 20, 80} × 300 Wh × 5 loc × {iid, wind}, June start only (a 365-d mission spans all seasons; the single-start caveat applies to the short end) | 180 |
| `thresh` | same grids, `include_optimal: false`, 4×4 threshold combos, **penalty 5 only** (penalty-invariant; analysis re-weights) | ~2160 rollout sims |

- **Locations (5):** the 4 provisioned sites + **Gulf of Mexico (30.0, −90.0)** — historical record already on disk from the Phase 7 unit-test provisioning; histcube + 5-bin wind chain build locally, no network.
- Fixed: episodes 3000 (CRN-paired, seed-0 batch reset), block bootstrap 7 d, `energy_increment_wh 5`, `delta_t 15`, `full_history_episodes 8` on florida configs only.
- Configs `configs/thesis_sweep/` (basenames `ths_{loc}_{family}_{arm}`), results `results/thesis_sweep/`.
- **Memory note:** 365-d cells vectorize 3000×35040 arrays (~0.4 GB/series per sim). The duration family runs at `--workers 8`; batgrid at 16.
- **Runtime budget:** batgrid ≈ 5–6 h wall @16 workers (132 solves per wind config ≈ 45 min each); duration ≈ 2–3 h @8; thresholds ≈ 30 min. Total ≈ 8–10 h, resumable via `--resume`.

**Commands:**
```
python Scripts/generate_markov_ablation_configs.py --thesis            # + --thesis-smoke for the gate
python Scripts/run_chain_sweep.py --configs ../configs/thesis_sweep_batgrid --out ../results/thesis_sweep --workers 16 --resume
python Scripts/run_chain_sweep.py --configs ../configs/thesis_sweep_duration --out ../results/thesis_sweep --workers 8 --resume
python Scripts/compare_markov_ablation.py --results ../results/thesis_sweep --manifest ../configs/thesis_sweep_batgrid/markov_ablation_manifest.json --verify
```

**Smoke gate (before the full run):** (a) Gulf provisioning produces a NON-uniform 5-bin chain and a histcube; (b) florida mini batgrid (3 caps × 2 pens, 300 eps) runs all 3 arms with CRN PASS; (c) one **full-length 365-d** config (150 Wh, iid+wind, 300 eps) completes — validates calendar wrap-around across the year boundary, memory headroom, and gives the long-solve timing anchor. Record timings before committing to the full run.

## 9. Phase 9 — Penalty-extension study: where does the threshold policy collapse? (2026-07-12)

**Question:** how sensitive is the policy ranking to failure penalty? Thesis data (2.5–80) shows the threshold's gap to the wind chain NARROWS with penalty (+6.2 → +1.5; the threshold's fixed ~26% failure conservatism approximates the optimal policy's behavior as penalty rises), but three signals point to an eventual collapse beyond 80: grid-edge saturation (18% of best combos on the 4×4 edge at pen 80), a floored failure rate (~23%) implying linearly negative reward in penalty, and a U-turn in the relative gap at 80. This study resolves the >80 regime and de-confounds the coarse tuning grid.

**Design** (fixed 300 Wh / 60 d / 2 seasons / 5 sites / 3000 episodes — cells pool exactly with the thesis batgrid 300 Wh rows, same seed-0 weather):
- **New DP solves:** iid + wind at penalties **{160, 320, 640}** → 5 loc × 2 arms × 3 pens × 2 dates = 60 solves.
- **Fine threshold arm** (removes the grid handicap): `threshold_values` 0→0.5 step 0.05 (11) × `wind_thresholds` {0, 2, …, 14} (8) = **88 combos**, run ONCE at fp=5 (penalty-invariant), re-weighted in analysis across the FULL ladder 2.5→640 → 5 loc × 88 × 2 dates = 880 rollout sims.
- Penalty curve assembled in analysis from: thesis 300 Wh cells (2.5–80) + new cells (160–640) + fine-grid threshold at every penalty; also compare fine-grid vs 4×4 threshold at 2.5–80 to quantify how much of the old benchmark was tuning-limited.
- Configs `configs/penalty_ext/` (`pex_{loc}_{arm}`), results `results/penalty_ext/`. Est. ≈ 2.5–3 h @16 workers.

**Readouts:** reward-vs-penalty per arm (semilog-x, 2.5→640); wind−threshold gap vs penalty (does it re-widen — U-shape confirmed?); failure-vs-penalty per arm (does the fine grid let the threshold buy below ~23%?); best-combo migration + edge saturation vs penalty; crossover penalty (if any) where threshold reward goes negative while wind stays positive.

**Smoke gate:** florida iid+wind at pen 640 (full 60-d horizon, 300 eps — checks reward scale/no numeric issues at extreme penalty) + florida fine-grid threshold config (300 eps) verifying 176 sims build and the conservative corner (obs 0.5 / wind 2) is exercised.

## 10. Future investigations (TODO)

Prioritized open items as of 2026-07-18. Each is scoped so a future session can execute
without replanning; run via the `run-study` / `analyze-study` skills.

**Direct loose ends (cheap, run-ready):**
- [ ] **ZOH confound split** — Δt=15 on zero-order-held hourly weather; separates the two
  causes of Phase 7's 55% retention (interpolation-inflated persistence vs coarser
  control). ~1 h.
- [ ] **Wind bin-resolution study** — Phase-6-style paired study for the wind chain
  (bins {3, 5, 8, 12}) under thesis conditions; 5 bins was a directive, never a finding.
  ~3–4 h.
- [ ] **Native-world calibration runs** — solve AND evaluate in the synthetic world for
  iid/wind across the penalty ladder; per-arm calibration gap (native − historical)
  quantifies how much of the Phase 9 corner loss is model mismatch. ~2 h.
- [ ] **Start-date sweep** — monthly starts (12 dates) × iid/wind at 300 Wh / 60 d;
  never run (thesis used only Jun 10 / Dec 10; duration family June-only; the legacy
  cvi startdate configs were never executed). ~2–3 h for all 5 sites.
- [ ] **Perfect-foresight oracle arm** — per-episode deterministic DP on the realized
  weather trace; upper-bounds the value of weather information so the chain's benefit
  can be reported as "X% of realizable information value". ~1 day incl. code.

**Robustness checklist (defense prep):**
- [ ] **Bootstrap block length** — only 7 days ever tested; rerun a slice at 3/14/28-day
  blocks.
- [ ] **CRN seed sensitivity** — every result rests on the single seed-0 bootstrap draw
  set (pairing is valid, but the episode collection is one realization); rerun a slice
  at 2–3 other seeds to show conclusions are draw-set-invariant. Requires threading a
  seed through `simulate_multiple_episodes`' `reset(0)`.
- [ ] **SoC grid sweep** — `energy_increment_wh: 5` was never varied; convergence check
  at {2.5, 5, 10} Wh on a slice (solve cost scales ∝ 1/increment).
- [ ] **Mass–power coupling sensitivity** — the 250–300 Wh optimum inherits the linear
  capacity→mass assumption in `Seaplane`; vary the coupling to band the sizing
  recommendation.

**Deeper questions:**
- [ ] **Solar at native sub-hourly resolution** — see the caveat appended to §6: the
  solar null is currently a claim about hourly data. Acquire sub-hourly GHI (NSRDB
  5–30 min for US-adjacent sites), refit the clear-sky-index chain, rerun the solar
  arm.
- [ ] **Empirical-distribution solve** — replace parametric Weibull/Beta stage fits
  with empirical histograms from the historical record; isolates the parametric-fit
  component of model mismatch.
- [ ] **Robust-DP variant** — inflate maneuver failure probabilities by a swept safety
  factor λ; does a conservatism-tuned DP recover the low-failure corner from the
  threshold?
- [ ] **Richer model-free heuristic** — add SoC-cutoff and time-of-day dials to the
  threshold family; separates policy-class richness from information value (residual
  DP gap ≈ pure value of the weather model).
- [ ] **NWP forecast-in-the-loop arm** — archived GFS forecasts driving the policy;
  brackets the chain between IID and the oracle on an information ladder. Requires a
  rolling-origin evaluation design (real timeline, not bootstrap) and forecast data
  plumbing — post-thesis paper scale.
- [ ] **Hawaii solar diagnosis** — compare fitted vs realized clear-sky-index
  autocorrelation at Hawaii; explain why the solar chain actively degrades there.

## 5. Decision Log
<!-- Append entries as checkpoints resolve. Format: date — decision — evidence. -->
- 2026-07-12 — **Frontier representation adopted as the summary form**: (harvest, P(fail)) is a sufficient statistic for the penalty axis (reward = harvest − p·P(fail)); harvest reported as capture ratio (÷ Σ whale series, 9.34/day). Repo tooling `Scripts/frontier_analysis.py` (frontier points, threshold Pareto hull from the fp=5 fine grid, matched-risk table, frontier.png); reframed report at `results/penalty_ext/_analysis/frontier_report.html`. Frontier readings: wind ≈2× iid capture at matched risk everywhere; iid frontier strictly inside the threshold hull; threshold ceiling 5.2% capture @ 27% fail (wind extends to 6.3%); wind↔threshold crossover ≈13% fail. Skills `analyze-study` and `findings-report` updated with the construction rules.
- 2026-07-12 — **Phase 9 complete: the penalty-sensitivity hypothesis INVERTS — the threshold policy does not collapse at high penalty; it OVERTAKES the DP arms beyond pen ≈ 100–150, while the chain's value evaporates.** Full curve (300 Wh/60 d, 90 cells, fine 11×8 grid): wind−thresh +4.6 @2.5 → +1.9 @20 → ~0 @40–80 → **−4.6 @160, −4.9 @320, −9.3 @640 (sig−)**. Wind−iid peaks +18.6 @10, +4.3 @160, then exactly **0.0 @320/640** — at extreme penalty both DP arms converge to near-never-fly (6.1% failure ≈ the unavoidable floating-hazard floor over 60 d) and persistence information is worth nothing. The threshold at 640 fails 4.7% vs 6.1% — its edge (+9.3) ≈ 640×1.4pp: a pure failure-avoidance gap, most plausibly the DP arms' synthetic-solve/historical-eval model mismatch pricing maneuver risk slightly wrong, which the model-free heuristic doesn't inherit. Fine-vs-coarse grid gain is small (+0.4–1.3) except at pen 160 (+6.9) — the coarse grid's saturation regime, as predicted. Run: 15 configs in 26 min; analysis `results/penalty_ext/_analysis/penalty_curve_cells.csv`. Thesis framing: chain value is a low-to-mid-penalty phenomenon (peak ≈ pen 10); at extreme risk aversion prefer the simple conservative heuristic.: pen-640 solves finite and sane (iid −29.9 @ 4.7% failure = penalty accounting checks out); fine 11×8 threshold grid runs 88 combos, interior optimum at fp=5 (obs 0.15/wind 4) consistent with the coarse grid. Smoke hint: wind < iid at pen 640 (300 eps, noisy) — the full run resolves the extreme-penalty ranking. Full run launched.
- 2026-07-12 — **Phase 8 complete (thesis dataset)**: batgrid 15 configs / 1320 solves in 12.6 h; duration 15 configs / 180 solves in 2.6 h; zero failures/resumes. Analysis (n_boot 3000, CRN PASS iid↔wind and iid↔thresh): **batgrid (660 cells)** — wind vs iid +15.06 mean (sig+ 631/660, never sig−); wind vs best-threshold +3.68 (sig+ 499, sig− 48); iid vs threshold −11.39 (sig− 637); best arm = wind in 649/660. Capacity response is an inverted U: chain benefit peaks at 200–300 Wh; ABSOLUTE wind-arm reward peaks at 250–300 Wh and declines beyond (battery mass → cruise power cost), 90% of the 600 Wh reward already at 150 Wh. Penalty response peaks at 10, still +9.9 at 80. Sites: florida 19.7 > hawaii 17.0 > gulf 15.7 > bering 11.9 > natlantic 11.1; summer 17.9 > winter 12.2. Failure-rate crossover: wind fails MORE than iid below ~250 Wh, LESS above. **Duration (90 cells)** — wind−iid grows monotonically +10.3 @30 d → +56.9 @365 d (sig+ 87/90); per-day rate declines 0.34→0.16; wind−threshold WIDENS with duration +2.3 → +8.3; year-long failure rates: thresh 81.7% > wind 78.3% > iid 66.7% (wind/thresh trade failures for harvest). 114 degenerate gaps flagged. Outputs: `results/thesis_sweep/_analysis_batgrid/`, `_analysis_duration/` (cells CSVs, reports, figures).
- 2026-07-11 — **Phase 8 smoke gate PASS**: Gulf provisioned (histcube + 5-bin chain, non-uniform, mean diag 0.93 — most persistent site); 365-day configs run clean through the year boundary (150 Wh year mission: iid reward 25.8 vs wind 93.4, failure ≈ 89% as expected for a tiny battery over a year); mini batgrid all 3 arms OK, wind > iid in 6/6 cells. Timing anchors: florida wind mini-config (6 solves incl 600 Wh) 646 s wall @16 workers; 365-d wind solve+rollout @150 Wh = 521 s. Batgrid launched.
- 2026-07-11 — **Phase 7 complete: wind-chain benefit survives at native hourly resolution (dt=60), retaining ~55% of the dt=15 per-step benefit — the finding is real, not an interpolation artifact, but roughly half the dt=15 magnitude rode on interpolation-inflated persistence and/or finer decision granularity (confounded; separating needs a dt=15-on-ZOH run).** Pooled paired dReward/step: dt=15 +0.00239 [0.00199, 0.00278] vs dt=60 +0.00130 [0.00094, 0.00169], SIG+ in all 4 locations at both dts. 60-min chain diagonal 0.79 vs 0.95–0.97 interpolated. At dt=60 the chain buys gains more aggressively (dFail +8.9pp vs +2.7pp). Absolute performance collapses at hourly decisions (iid per-step reward goes negative) — sub-hourly control matters independent of the chain. Six dt-hardcoding fixes applied and regression-gated first (see §7); dt=15 results bit-identical after fixes. Outputs: `results/markov_dt60/_analysis/` (dt_resolution_summary.md, dt_resolution_cells.csv). Study cost: ~35 min (provision + 8 configs).
- 2026-07-07 — **Phase 6 complete: solar persistence gives NO significant benefit at any resolution → joint rerun NOT run** (decision rule #1). Solar-only vs IID, pooled across all 4 locations, full conditions (432 solves, 24 configs, ~5 h): every bin count is non-significant — g2 +0.053 [−0.16, +0.25], g3 −0.009, g4 −0.017, g6 +0.030, g8 −0.014 (all CIs span 0). Higher resolution does not help and does not monotonically improve. Per-location: Florida mildly positive (~+0.5, still ns), Bering/N.Atlantic ≈ 0, **Hawaii consistently negative and getting *worse* with more bins** (−0.62 at g2 → −0.81 at g8) — the main-sweep Hawaii degradation is real, not a too-coarse-bins artifact. Confirms the main-sweep finding that solar-clear-sky persistence carries no actionable dispatch signal here (wind persistence does all the work). Bug fixed during analysis: `import json` + ASCII console prints (cp1252 can't encode Δ). Outputs: `results/markov_solar_res/_analysis/` (solar_res_summary.csv, solar_res_cells.csv, solar_res_verdict.json, solar_res_curve.png).
- 2026-07-06 — Wind bins fixed at 5 (user directive); reuse verified `_windchain_wind5.pkl` artifacts (n_bins=5, quantile) rather than rebuilding.
- 2026-07-06 — Threshold arm runs at fp=5 only; per-penalty rewards recomputed analytically (threshold behavior verified penalty-invariant).
- 2026-07-06 — Pre-check `Tests/verify_solar_chain.py`: all 5 sections PASS (incl. §4 joint Kronecker plumbing). lat30/lon-90 provisioned for the test (hist pkl copied to the test's legacy name `data_30_-90.pkl`).
- 2026-07-06 — **Solar chain n_bins = 2** (Phase 2 checkpoint). Study: florida, 30 d, fp 20, 2000 eps, paired vs IID: Δreward g2 +0.32 [−0.87, +1.48], g3 −0.23, g4 +0.74 (best), g5 +0.57 — no bin count statistically distinguishable from the best (paired g-vs-g4 CIs all span 0), so the knee rule selects the smallest, g=2. Caveat: study underpowered to rank bins; it only had to pick a resolution. Full sweep resolves the solar effect itself. Data: `results/markov_ablation_smoke_solar/_analysis/solar_bins_study.csv`. Joint table is 5×2 = 10 bins.
- 2026-07-06 — **Phase 4 complete**: all 20 configs, ~4.4 h wall at 16 workers (launched 06:15, finished 10:38). Per-location ≈ 64 min (iid 4, joint 34, solar 7, thresh 1, wind 18 min). No failures, no resume needed. `results/markov_ablation/sweep_run_log.json`.
- 2026-07-06 — **Phase 5 complete**: 72 cells analyzed, CRN PASS all four iid↔arm pairs. Headline: wind chain +13.8 mean paired Δreward vs IID (sig+ in 68/72 cells, never sig−); joint +14.0 (65/72); solar-only ≈ 0 (−0.05 mean; sig− in 6 Hawaii cells where it raises failure 5.7%→22%). Only persistence-aware arms beat the best threshold policy (wind +2.1, joint +2.4 vs thresh; IID and solar-only LOSE to threshold by ~11.6, sig− in 71/72). Gap-closed: wind ≈ 0.94, joint ≈ 0.97 (IID defines 0 well below threshold). Benefit peaks at 300 Wh, shrinks (stays positive) at penalty 80, summer > winter. Solar increment on top of wind (joint−wind): +0.23 mean, indistinguishable in 63/72 cells. Caveats: solar n_bins=2 was chosen on a Florida-only underpowered study; Hawaii solar degradation may be a resolution/site artifact worth a follow-up. 17 cells have degenerate threshold gaps (flagged). Outputs: `results/markov_ablation/_analysis/`.
- 2026-07-06 — **Phase 3 gate PASS** (all 5 criteria): CRN identical-weather PASS for iid↔wind, iid↔solar, iid↔joint, iid↔thresh (8 episodes each); joint table shape (10, 123, 480) = (5·2 bins, |S|, T); sane summary rows all arms; `_windchain.pkl`/`_windchain_wind5.pkl` mtimes unchanged; winter-start config ran clean. Also fixed during gate: report writer now writes UTF-8 (cp1252 crash), CRN verify extended to the thresh arm. Full sweep cleared for launch with `--solar-bins 2` configs.
