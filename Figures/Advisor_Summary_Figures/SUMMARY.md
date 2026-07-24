# IID vs Chain vs Threshold — summary of studies run 1–12 July 2026

**Question.** Does a dynamic-programming (DP) dispatch policy earn its complexity over a
simple tuned threshold heuristic — and does that depend on modeling weather persistence
(Markov chains for wind / solar) instead of an i.i.d. weather model?

**Method (common to all studies).** Three arms evaluated on identical paired historical
block-bootstrap weather episodes (common random numbers, 3 000 episodes/cell):
*Best threshold* = tuned SoC-observation/wind threshold heuristic, re-selected as the
best-scoring grid combination per cell (16-combo grid in the thesis sweeps with a
split-half stability check; 88-combo fine grid on the penalty curve, full-sample best
per penalty); *Optimal (i.i.d.)* = DP solved against an i.i.d. weather model;
*Optimal (wind chain)* = the same DP solved against a 5-bin Markov wind model.
Figures show absolute mean reward per arm, broken out per location, conditioned on a
single mission start date (June primary; December where the sweep ran one). Nothing is
pooled across sites or start dates — different sites/starts sample different weather
and are not comparable.

## What was run, and why (chronological)

1. **Bin validation (Jul 3).** How should the wind chain discretize wind, and how many
   bins? → Equal-occupancy quantile bins win; gains saturate by ~8–12 bins.
   Failure-space bin placement (our starting hypothesis) was refuted — it collapses
   resolution into the calm regime and the resulting policy never flies.
2. **Markov ablation (Jul 6, 4 sites, 72 cells).** Which persistence channel matters —
   wind, solar, or both? → Only the wind-informed arms (wind, joint) beat the tuned
   threshold; the i.i.d. DP loses to it in 71/72 cells. Joint (wind+solar) is
   indistinguishable from wind alone; solar alone tracks i.i.d.
3. **Solar bin-resolution follow-up (Jul 7, 4 sites).** Is the solar null a
   too-coarse-2-bin artifact? → No: pooled over all conditions the solar chain is
   indistinguishable from i.i.d. at every bin count 2–8 (all 95 % CIs span zero).
4. **Timestep robustness (Jul 11, 4 sites).** Is the wind benefit an artifact of
   interpolating hourly weather to 15-min steps? → No: on native hourly data (dt = 60)
   the wind arm still beats i.i.d. significantly at every site, retaining ~55 % of the
   per-step delta (aggregate across the study's 72 cells).
5. **Thesis sweep + penalty extension (Jul 11–12, 5 sites incl. Gulf of Mexico).** Full
   response surfaces (11 capacities × 6 penalties; durations to 1 year; penalties
   extended to 640). → Wind chain is the best arm in 649/660 cells. Mean reward peaks
   at an intermediate capacity — between 150 and 400 Wh depending on site and start —
   and always declines toward 600 Wh (battery mass feeds cruise power). Surprise: at
   high penalty the tuned threshold draws level with and passes the DP arms, at a
   site- and start-dependent penalty (see figure guide).

## Figure guide (all in this folder; one panel per location)

- **capacity_by_location_june.png / _december.png** — Mean reward vs battery capacity
  (60 d, penalty 5). The wind chain beats the threshold in all 110 cells (11 capacities
  × 5 sites × 2 starts), and the threshold beats i.i.d. in 107/110 — the exceptions are
  Gulf June at 500–600 Wh, where the i.i.d. DP significantly beats the threshold.
  The i.i.d. handicap is largest at the windy sites (N. Atlantic, Bering), where the
  i.i.d. policy is too conservative to exploit calm spells.
- **penalty_by_location_june.png / _december.png** — Mean reward vs failure penalty
  (300 Wh, 60 d, penalties 2.5–80). The wind chain leads at low penalty at every site
  and start. Where the threshold first draws level with or passes the wind chain is
  site- and start-dependent: June — penalty ≈ 40 at Hawaii and N. Atlantic, ≈ 80 at
  Bering, beyond 80 at Florida (160) and Gulf (320); December — by penalty 20–80 at
  all five sites. (The extension run to penalty 640 — not shown — has the threshold
  ahead of both DP arms at extreme penalties everywhere, as the DP arms converge to
  near-never-fly.)
- **penalty_fixed_threshold_by_location_june.png / _december.png** — Same axes, adding
  a fourth series: the single best combination at penalty 5, held fixed across the
  penalty axis (its reward at other penalties re-weighted exactly from its φ=5 run).
  Separates "the threshold family, re-tuned per penalty" (orange envelope) from "a
  threshold you'd actually deploy" (brown). Which sites suffer depends on the start:
  in June the fixed tuning collapses at Hawaii and N. Atlantic (≈ −27 by penalty 80)
  while tracking the envelope at Florida/Gulf/Bering; in December it collapses at
  Florida and N. Atlantic instead. The chosen fixed combo is printed in each panel.
- **duration_by_location_june.png** — Mean reward per stage vs mission duration
  (300 Wh, penalty 5). The wind chain is on top at every site and duration except
  N. Atlantic at 365 d, where wind vs threshold is a statistical tie
  (Δ = −0.6, 95 % CI [−2.6, +1.4]). The i.i.d. arm trails everywhere and its
  per-stage reward collapses toward zero at N. Atlantic/Bering/Hawaii.
- **solar_bins_mean_reward_june.png** — Solar-chain DP vs its i.i.d. reference across
  bin counts (300 Wh, penalty 5, 4 sites). At this reference condition the solar arm
  sits slightly above i.i.d. (+0.4 to +2.2 by site), but pooled across all conditions
  the effect is null (every CI spans zero) and Hawaii degrades in winter/high-penalty
  cells — hence "no actionable solar signal at hourly data resolution."
- **dt_mean_reward_per_step_by_location_june.png** — Per-step mean reward at dt = 15
  (interpolated) vs dt = 60 (native hourly), 300 Wh, penalty 5. The wind-over-i.i.d.
  gap survives on native hourly data at all four sites; at hourly decisions the i.i.d.
  arm drops to ≈ 0 or slightly negative at Hawaii/N. Atlantic/Bering, so sub-hourly
  control matters independently of the chain.

## Per-site numbers at the reference condition (June start, 300 Wh, penalty 5, 60 d)

| Site          | Threshold | i.i.d. DP | Wind-chain DP | wind − thresh | iid − thresh |
|---------------|----------:|----------:|--------------:|--------------:|-------------:|
| Florida       |      54.9 |      35.5 |          59.7 |          +4.8 |        −19.3 |
| Hawaii        |      12.5 |       1.9 |          18.8 |          +6.3 |        −10.6 |
| Gulf of Mexico|      67.6 |      58.9 |          71.8 |          +4.2 |         −8.7 |
| N. Atlantic   |      23.3 |       5.2 |          31.2 |          +7.8 |        −18.1 |
| Bering Sea    |      29.9 |      10.1 |          36.7 |          +6.8 |        −19.8 |

Deltas are paired within-cell (same site, same start date, same episodes); every
wind − thresh and iid − thresh entry above is significant at the per-cell 95 % level.

## Caveats

- Solar claim is scoped to **hourly source data** — sub-hourly cloud variability is
  absent from the reanalysis record by construction.
- dt = 60 retention (~55 %) is **confounded** between interpolation-inflated
  persistence and decision granularity.
- The penalty-figure crossover points are point-estimate comparisons (no per-point CI
  on that curve); several crossings are by small margins.
- All results use one common-random-number seed set and 7-day bootstrap blocks;
  seed/block-length robustness and the 5-bin wind-chain setting are on the follow-up
  list (5 bins was a chosen default; bin validation suggests ~8–12 saturates).

Reproduction: `docs/markov_ablation_experiment.md` (design + decision log);
figures from `Figures/Scripts/generate_advisor_summary_figures.py`; cells CSVs under
`results/{thesis_sweep, penalty_ext, markov_solar_res, markov_dt60}/_analysis/`.
