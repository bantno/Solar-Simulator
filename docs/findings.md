# Findings

This page collects the empirical results recorded in the repository so far. Every claim
here is quoted from a design record or from a generated analysis report, with its source
beside it. For methods, caveats, and the full tables, follow the links. Nothing on this
page was recomputed for the wiki.

!!! warning "Read the operating-point caveat before quoting failure rates"
    With the failure penalty used in these runs (5.0), the optimal policies deliberately
    accept high failure rates in exchange for reward. Reward already includes the penalty
    and is the objective. Failure % on its own compares policies at different operating
    points. — [Threshold benchmark](design/wind-threshold-benchmark.md) §4

## 1. Wind persistence improves the optimal policy on real weather

### Single site (lat 30 / lon −90)

Source: [Wind-persistence briefing](design/wind-persistence-briefing.md) §4. Setup:
historical weather from 1950–2022 in 7-day blocks, 300 Wh, failure penalty 5.

- **Chain-optimal has the highest reward of the three policies at both horizons
  studied (3000 and 4000 stages).** It also beats i.i.d.-optimal on all three metrics at
  once: roughly 5–9 pp lower failure, +10–15% reward, and +5–8 flight hours.
- **i.i.d.-optimal roughly ties a tuned threshold heuristic** at the long horizon: about
  48 vs about 48.5 reward. Adding wind persistence turns "optimal ≈ heuristic" into
  "optimal clearly wins" (about 54).
- **Bin definition is the lever.** With equal-occupancy quantile bins the chain gives
  the gain above. With coarse aircraft-threshold bins of `[5, 10]` m/s, the policy falls
  back to staying on the water almost all the time (reward about 3).
- **Mechanism.** On one real two-week window, the two policies follow similar SoC paths
  but diverge around weather transitions. The chain policy abstains or commits a step
  earlier around high-wind spells (figure in the briefing, produced by
  `Scripts/compare_policies_episode.py`).

### Multi-site sweep: 34 historical cells

Source: [Threshold benchmark](design/wind-threshold-benchmark.md) §§3, 9. The cells are
4 locations × 6 capacities, plus 6 start months and 4 durations, at 3,000 episodes each.
Deltas are episode-paired with 95% bootstrap CIs.

| Comparison | 3 fixed bins `[5, 10]` m/s | 4 quantile bins (current default) |
|---|---|---|
| chain-optimal beats best tuned threshold | 30 / 34 (2 losses, both Hawaii) | **34 / 34** |
| i.i.d.-optimal loses to best tuned threshold | 19 / 34 | 19 / 34 |

- **Solving the MDP under i.i.d. wind is often worse than tuning a two-parameter
  threshold rule** on real weather. Adding wind persistence restores the optimal policy's
  advantage.
- **The Hawaii losses came from bin resolution, not from how the chain was fitted.** July
  trade winds sit above 6 m/s 48% of the time, almost all of it inside the 5–10 m/s
  bin, so the value function could not see the decision-relevant level. Quantile edges
  (4.1 / 5.7 / 7.3 m/s) bracket that band, and both Hawaii cells flip to wins
  (§5, §9).
- **Choosing the best threshold combo on the same episodes it is scored on
  (winner's-curse bias) is small.** Split-half holdout rewards are within 0.5 of the
  full-sample values in 29 of 34 cells and differ by up to 1.58 (May start cell), below every
  chain-minus-threshold margin (*corrected 2026-10-20 against `comparison_cells.csv`; the
  benchmark text said "within ~0.5 everywhere"*).

The generated per-cell report for the 4-quantile-bin sweep,
`results/chain_vs_iid_sweep/_analysis/comparison_report.md`, is git-ignored and exists
only where the sweep was run. It additionally shows:

- **Chain vs i.i.d. on historical weather:** in every one of the 34 cells (capgrid,
  duration, startdate), the paired Δreward (chain − i.i.d.) is positive.
- **Calibration:** the i.i.d. model's native-world failure prediction is **optimistic in
  every cell**: its failure gap is negative, meaning it under-predicts its real
  failure rate. The chain model's gaps are much smaller and of mixed sign.

Regenerate the report with `Scripts/compare_chain_sweep.py`; see [Sweeps](experiments/sweeps.md).

## 2. First order is enough; resolution is the remaining lever

**Wind.** Sources: [Wind-persistence math](design/wind-persistence-math.md) §9 and the
[briefing](design/wind-persistence-briefing.md) §4.

- The conditional mutual information `I(Next ; Prev | Curr, month, hour)` is the extra
  information a second-order chain would add. At the coarse bins the policy can use, it
  is about **1.4–2%** of the remaining entropy. It rises with the number of bins, not
  with history depth.
- Conclusion: the exploitable structure lives in bin **resolution**, not Markov
  **order**.
- The high-resolution values were later bias-corrected with a permutation null (commit
  `8356b5a`; see the [decision log](design/index.md)).

**Solar.** Source: [Solar-persistence design record](design/solar-persistence-math.md) §5.
Data: lat 30 / lon −90, 1950–2022, hourly.

- **Day scale, first order:** the information shared between one day's regime and the
  next is 9.4% of daily regime entropy. The daily-mean-K ACF(1) is 0.40, an e-folding
  time of about 2 days. The dusk→dawn channel is worth modeling.
- **Day scale, higher order:** 0.5% after bias correction, so first order is right.
- **Intra-day, higher order:** 3.2% after bias correction, somewhat above wind's 1.4–2%.
  The record treats this as a limitation, not as grounds for a deeper chain.

## 3. Engineering results

| Result | Source |
|---|---|
| Vectorized `solve()` is ~28× faster at horizon 200 (grows with horizon), with an identical value table; batched rollouts are ~800–1300× faster and statistically equivalent. | commit `13fb2fb`; [Verification](verification/index.md) |
| Columnar episode-scalar storage is ~88× smaller than group-per-episode (~19 MB vs ~1.7 GB for a 210-sim × 3000-episode scalars-only run). | commit `32e0d4a`; [Result storage](architecture/storage.md) |
| Historical-bootstrap memory dropped from ~1.4 GB to ~1.2 MB per simulation (30k episodes, horizon 3000). | commit `b40ae34` |

## Stated limitations and open items

These are quoted from the records rather than added for this page.

- **Wind, single-site result:** it is one site and effectively one season window.
  Optimal and threshold policies sit at different failure operating points. Solar was
  still i.i.d. at the time. ([Briefing](design/wind-persistence-briefing.md) §6)
- **Threshold benchmark:** the 3-bin run's per-episode data was deleted, so the §§3–5
  numbers cannot be regenerated from disk. A later refactor changed the RNG ordering, so
  the 3-bin vs 4-bin comparison is approximate (about ±0.2 reward). Principled bin
  selection is open. ([Threshold benchmark](design/wind-threshold-benchmark.md) §9)
- **Solar:** one site so far. The hour-averaged GHI vs clear-sky mismatch is absorbed by
  rank bins. There is a ~3% intra-day residual. The wind and solar chains are assumed
  independent; a fitted joint chain is the designed next step.
  ([Solar persistence](design/solar-persistence-math.md) §7)
- **Whale observation:** the open questions are in the
  [literature survey](design/whale-observation-literature.md) §6.

<!-- additions-2026-10-20 -->
## Additions and corrections, 2026-10-20

Recorded from runs made for the APL interview backups. Scripts and numbers:
`APL Interview/analysis/` and `APL Interview/out/wind_backups/slide_numbers.json`.

- **Held-out check (Gulf site, July 1, 30 days, 150/300/500 Wh, corrected solver after `779dc44`).** Weibull/Beta table and 4-quantile-bin chain fitted on 1950 to 2004, bootstrap cube from 2005 to 2022 (`results/apl_holdout_2026-10-20`). Chain minus i.i.d. reward +7.06 [+6.17, +7.90] at 150 Wh, +2.22 [+1.53, +2.88] at 300 Wh, +2.38 [+1.81, +2.96] at 500 Wh. The in-sample twin on the full record with the same solver (`results/apl_insample_2026-10-20`) gives +6.82, +3.16, +1.70, so the July in-sample sweep was not inflated by shared years.
- **The single-site claim that i.i.d.-optimal "barely ties" the tuned threshold is not reproduced at the Gulf 30-day cells on disk at penalty 5.** Held-out: i.i.d. minus threshold +1.9, +3.2, +4.5. July sweep Gulf capacity grid: +2.5 to +7.1. In the July sweep the i.i.d. policy loses to the threshold at Hawaii, lat 45 (most capacities) and Bering, and in the 60-day Gulf cell, not in the 30-day Gulf cells. The section 1 single-site numbers come from CSVs that are not in the repo.
- **Penalty sensitivity, held out.** At penalty 50 the i.i.d. policy loses to the threshold rule at all three capacities (-10.4, -4.4, -3.2) while the chain beats it (+4.4, +3.5, +3.3), and the chain's failure at 300 Wh falls from 31.4 % to 23.5 % (i.i.d. 34.0 to 31.2, threshold 21.5 unchanged). So the "persistence is what makes optimal win" claim holds at the Gulf at penalty 20 or more, not at 5.
- **Triplicate baseline cell in the July sweep** (+2.914 / +2.821 / +4.185): crash draws were unseeded before `779dc44`, so the three runs differ only by crash randomness. `V0` was identical.
- **Chain fitting detail.** Transitions are counted for every consecutive pair one model step apart, including across midnight. The Weibull table has one fit per (month, day, hour, minute) slot. Earlier wording in the math record and the briefing said otherwise and was corrected in place.
- Caveat: `compare_chain_sweep.py` derives the V0 and calibration columns from the full-record expected-data path, so those columns in the held-out `comparison_cells.csv` files are not valid.
<!-- end additions-2026-10-20 -->
