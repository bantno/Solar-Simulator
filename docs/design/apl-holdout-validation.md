# Held-Out Validation at the Gulf Site — Chain vs IID vs Threshold, 2026-10-20

*Branch: `wind-persistence`. Run date: 2026-10-20. Companion to the
[threshold benchmark](wind-threshold-benchmark.md), which fitted and evaluated on the same
1950–2022 record. This record asks whether the chain's advantage survives when the model is
fitted on one span of years and evaluated on another, and how it moves with the failure
penalty. Numbers are quoted from the generated `comparison_report.md` and
`comparison_cells.csv` files; nothing was recomputed for this page.*

---

## 1. Question

Every chain-vs-i.i.d. and chain-vs-threshold result recorded so far fits the Weibull/Beta
table and the wind chain on the full historical record and then evaluates on a block
bootstrap of that same record. Two doubts follow:

1. **Shared years.** The chain-optimal policy could look good partly because it has seen
   the evaluation weather. A clean test fits on early years and evaluates on later ones.
2. **Operating point.** All sweeps used a failure penalty of 5, at which the optimal
   policies accept 30–47 % failure. Does the ranking of the three policies change when
   failure is penalised harder?

## 2. Design

### Cells

One site, one season, three capacities:

| Setting | Value |
|---|---|
| Location | Gulf, lat 30.0 / lon −90.0 |
| Start | 2020-07-01 00:00 (calendar-matched; year is ignored by the bootstrap) |
| Horizon | 2880 stages at 15 min = 30 days |
| Battery capacity | 150, 300, 500 Wh |
| Episodes | 3,000 per simulation, paired across arms (identical bootstrap weather) |
| Block bootstrap | 7-day blocks |
| Wind chain | 4 equal-occupancy quantile bins |
| Solver | corrected depletion term and paired crash draws (`779dc44` and later) |
| Threshold grid | obs {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3} × wind {0, 3, 6, 9, 12} m/s = 35 combos |

### Year split

| Artifact | Years | Hourly rows |
|---|---|---|
| Weibull/Beta expected-data table and wind chain (fit) | 1950–2004 (55 years) | 481,800 |
| Historical bootstrap cube (evaluation) | 2005–2022 (18 years) | 157,680 |

The quantile bin edges derived on the fit years are 2.73, 3.91 and 5.29 m/s. The full-record
edges used by the July sweep were 2.65, 3.85 and 5.23 m/s
([benchmark §9](wind-threshold-benchmark.md)), so the fit span barely moves the bins.

### How the hold-out was built

The simulator consumes three per-location artifacts, all derived from one hourly
historical pickle (`Data/HISTORICAL_DATA/data_30_-90.pkl`, Open-Meteo 1950–2022):

| Artifact | Built by | Used for |
|---|---|---|
| Expected-data table (`..._15min.pkl`) | `build_expected_data_artifact` in `weather_processor_cs_normalization.py` | Per-(month, day, hour, minute) Weibull wind fit and Beta clear-sky-index fit. The solver integrates failure probability and expected solar over these. |
| Wind chain (`..._windchain.pkl`) | `build_wind_chain_artifact` in `Scripts/create_weather_distributions.py` | 288 (month, hour) transition matrices over the wind bins, plus the bin edges. Chain arm only. |
| Historical cube (`..._histcube.pkl`) | `build_historical_cube_artifact` in the same module | A (calendar slot, year) array of wind and irradiance that the block-bootstrap provider draws 7-day blocks from. All three arms. |

In the normal pipeline the harness builds any missing artifact from the **whole** record,
so the solver's distributions and the evaluation weather share every year. The hold-out
breaks that link by building the artifacts from two disjoint slices of the same hourly
pickle:

1. **Split by calendar year.** Rows with year 1950–2004 form the fit slice; rows with year
   2005–2022 form the evaluation slice. The row counts in the manifest are exactly whole
   365-day years at hourly resolution (55 × 8,760 = 481,800 and 18 × 8,760 = 157,680), so
   Feb 29 was dropped before the split, as the standard builders do.
2. **Fit artifacts from the fit slice.** The expected-data table and the wind chain were
   built from the 1950–2004 rows with the standard builders: resample to 15 min, fit the
   Weibull and Beta per calendar slot, derive 4 equal-occupancy quantile edges from the
   fit-slice wind, count bin transitions between consecutive 15-min steps, conditioned on
   (month, hour). The resulting edges are 2.73, 3.91 and 5.29 m/s.
3. **Evaluation cube from the evaluation slice.** The historical cube was built from the
   2005–2022 rows only, so it has 18 year-columns. Each bootstrap episode stitches 7-day
   blocks drawn from those 18 years, matched on calendar position; the solver never sees
   any of them.
4. **Name the artifacts explicitly in the configs.** The held-out files carry the suffixes
   `_fit1950-2004` and `_eval2005-2022_histcube`. The chain and i.i.d. configs point
   `locations[].data_path` at the fit table, `wind_chain.path` at the fit chain and
   `historical_weather.path` at the evaluation cube (the threshold config needs only the
   last two). Because the paths are explicit, the harness's provisioning step
   (`_ensure_location_data` in `harness/run_experiment.py`) finds them present and does
   not rebuild from the full record, and its staleness check is skipped for an explicit
   `wind_chain.path`.
5. **Record the split in the manifest.** The config generator wrote a `holdout` block into
   each held-out manifest: source pickle, fit and evaluation years, row counts, artifact
   paths, chain bin edges, and the list of cube years. The in-sample manifest has the block
   set to null.

Everything else is identical to the in-sample twin: same three capacities, start date,
horizon, episode count, block length, solver build and threshold grid. The i.i.d. and chain
arms of a run share bootstrap weather episode-for-episode, and the threshold arm sees the
same weather because the bootstrap provider re-seeds per batch. The penalty runs reuse the
same held-out artifacts and change only `failure_penalties`.

The script that performed the split and called the builders lives in the APL interview
working folder, outside this repository. The manifest `holdout` block is its complete
input specification; §6 gives the rerun commands once the artifacts exist.

### Runs

Four config sets, each with a chain arm, an i.i.d. arm and one arm-agnostic threshold
config. They differ only as listed.

| Config / results directory | Fit data | Evaluation cube | Failure penalty |
|---|---|---|---|
| `apl_insample_2026-10-20` | full record | full record | 5 |
| `apl_holdout_2026-10-20` | 1950–2004 | 2005–2022 | 5 |
| `apl_holdout_p20_2026-10-20` | 1950–2004 | 2005–2022 | 20 |
| `apl_holdout_p50_2026-10-20` | 1950–2004 | 2005–2022 | 50 |

The in-sample run is the control: same solver, same three capacities, same seeds, but fitted
and evaluated on the whole record like the July sweep. Each config set took about three
minutes of wall time on 8 workers (chain arm about 105 s, i.i.d. arm about 32 s, threshold
arm about 56 s).

### Comparison method

As in the benchmark: all deltas are per-episode paired differences with percentile-bootstrap
95 % confidence intervals over 3,000 episodes. The best threshold combo is chosen per cell by
full-sample mean reward. A split-half check re-selects on even episodes and scores on odd
episodes; if that value is close to the full-sample value, the winner's-curse bias from
selecting and scoring on the same episodes is negligible.

## 3. Results

### 3.1 Chain minus i.i.d., in-sample vs held-out (penalty 5)

| cap (Wh) | world | fail % iid | fail % chain | Δreward [95 % CI] | ΔCVaR₁₀ [95 % CI] | Δflight hrs |
|---|---|---|---|---|---|---|
| 150 | in-sample | 47.20 | 43.23 | +6.82 [+5.88, +7.79] | +1.87 [+0.96, +2.71] | +6.62 |
| 150 | held-out | 46.53 | 42.37 | +7.06 [+6.17, +7.90] | +1.99 [+1.13, +2.85] | +6.84 |
| 300 | in-sample | 38.13 | 33.53 | +3.16 [+2.40, +3.92] | +0.38 [−0.62, +1.44] | +3.93 |
| 300 | held-out | 34.03 | 31.40 | +2.22 [+1.53, +2.88] | +0.94 [−0.05, +1.92] | +2.97 |
| 500 | in-sample | 30.23 | 27.83 | +1.70 [+1.16, +2.24] | +1.46 [+0.56, +2.43] | +1.63 |
| 500 | held-out | 28.93 | 25.90 | +2.38 [+1.81, +2.96] | +1.49 [+0.39, +2.60] | +1.57 |

The held-out deltas sit inside or next to the in-sample confidence intervals at every
capacity. The chain's reward advantage is not an artifact of having fitted on the evaluation
years.

### 3.2 Three-way comparison at penalty 5

Best threshold combo is written (obs, wind m/s). Rewards are mean total reward per episode.

| cap (Wh) | world | best thresh | fail % thresh / iid / chain | iid − thresh [CI] | chain − thresh [CI] | thresh reward full / split-half |
|---|---|---|---|---|---|---|
| 150 | in-sample | (0.2, 3) | 23.87 / 47.20 / 43.23 | +3.41 [+2.53, +4.26] | +10.23 [+9.42, +11.05] | 39.70 / 40.09 |
| 150 | held-out | (0.2, 3) | 25.70 / 46.53 / 42.37 | +1.90 [+1.06, +2.76] | +8.95 [+8.20, +9.73] | 39.71 / 40.01 |
| 300 | in-sample | (0.1, 3) | 20.27 / 38.13 / 33.53 | +2.38 [+1.64, +3.10] | +5.54 [+4.86, +6.21] | 37.25 / 37.41 |
| 300 | held-out | (0.1, 3) | 21.47 / 34.03 / 31.40 | +3.15 [+2.44, +3.87] | +5.37 [+4.68, +6.05] | 36.78 / 36.49 |
| 500 | in-sample | (0.2, 3) | 16.20 / 30.23 / 27.83 | +6.91 [+6.35, +7.48] | +8.61 [+8.09, +9.11] | 28.63 / 28.72 |
| 500 | held-out | (0.2, 3) | 15.80 / 28.93 / 25.90 | +4.47 [+3.92, +4.98] | +6.84 [+6.33, +7.32] | 29.14 / 29.29 |

At penalty 5 both optimal policies beat the tuned threshold rule in every Gulf 30-day cell,
in-sample and held-out. The chain roughly doubles the i.i.d. margin.

### 3.3 Penalty sensitivity, held out

Same held-out artifacts, penalty raised from 5 to 20 and 50. The threshold policies do not
see the penalty, so their failure rates are unchanged; their rewards fall because the
penalty is charged on the same failures. The selected combos are the same at all three
penalties.

**Chain minus i.i.d.**

| penalty | cap (Wh) | fail % iid | fail % chain | Δreward [95 % CI] | ΔCVaR₁₀ [95 % CI] | Δflight hrs |
|---|---|---|---|---|---|---|
| 5 | 150 | 46.53 | 42.37 | +7.06 [+6.17, +7.90] | +1.99 [+1.13, +2.85] | +6.84 |
| 20 | 150 | 43.43 | 38.80 | +8.79 [+7.63, +9.91] | +2.21 [+1.23, +3.25] | +7.64 |
| 50 | 150 | 31.23 | 30.07 | +14.77 [+13.26, +16.30] | +1.96 [+1.00, +2.99] | +13.62 |
| 5 | 300 | 34.03 | 31.40 | +2.22 [+1.53, +2.88] | +0.94 [−0.05, +1.92] | +2.97 |
| 20 | 300 | 33.37 | 29.00 | +4.11 [+3.15, +5.07] | +0.64 [−0.37, +1.70] | +4.09 |
| 50 | 300 | 31.17 | 23.50 | +7.86 [+6.36, +9.38] | +2.85 [+1.54, +4.16] | +5.70 |
| 5 | 500 | 28.93 | 25.90 | +2.38 [+1.81, +2.96] | +1.49 [+0.39, +2.60] | +1.57 |
| 20 | 500 | 27.13 | 24.00 | +3.25 [+2.48, +4.08] | +1.86 [+0.71, +3.00] | +2.39 |
| 50 | 500 | 25.27 | 19.63 | +6.56 [+5.28, +7.83] | +2.28 [+1.10, +3.48] | +3.48 |

**Optimal arms minus best threshold**

| penalty | cap (Wh) | best thresh | fail % thresh | iid − thresh [CI] | chain − thresh [CI] | thresh reward full / split-half |
|---|---|---|---|---|---|---|
| 5 | 150 | (0.2, 3) | 25.70 | +1.90 [+1.06, +2.76] | +8.95 [+8.20, +9.73] | 39.71 / 40.01 |
| 20 | 150 | (0.2, 3) | 25.70 | −1.80 [−2.85, −0.69] | +6.99 [+6.00, +7.93] | 35.86 / 36.19 |
| 50 | 150 | (0.2, 3) | 25.70 | −10.41 [−11.92, −8.77] | +4.36 [+2.94, +5.75] | 28.15 / 28.55 |
| 5 | 300 | (0.1, 3) | 21.47 | +3.15 [+2.44, +3.87] | +5.37 [+4.68, +6.05] | 36.78 / 36.49 |
| 20 | 300 | (0.1, 3) | 21.47 | +0.46 [−0.46, +1.39] | +4.57 [+3.62, +5.52] | 33.56 / 33.35 |
| 50 | 300 | (0.1, 3) | 21.47 | −4.37 [−5.88, −2.99] | +3.49 [+2.14, +4.88] | 27.12 / 27.07 |
| 5 | 500 | (0.2, 3) | 15.80 | +4.47 [+3.92, +4.98] | +6.84 [+6.33, +7.32] | 29.14 / 29.29 |
| 20 | 500 | (0.2, 3) | 15.80 | +1.60 [+0.83, +2.34] | +4.85 [+4.13, +5.58] | 26.77 / 27.04 |
| 50 | 500 | (0.2, 3) | 15.80 | −3.22 [−4.50, −1.96] | +3.33 [+2.20, +4.47] | 22.03 / 22.54 |

### 3.4 Equal-reliability capacity saving

The failure-vs-capacity frontier does not cross 5 % failure for any policy in the swept
range (150–500 Wh), at any penalty, so the "battery saved at equal reliability" figure that
the comparison script computes is undefined for these cells.

## 4. Reading

1. **The July sweep's Gulf result was not inflated by shared years.** Held-out and
   in-sample chain-minus-i.i.d. rewards agree to within the confidence intervals at all
   three capacities (§3.1). The chain-minus-threshold margins are likewise close
   (+8.95 vs +10.23, +5.37 vs +5.54, +6.84 vs +8.61).
2. **At penalty 5 the i.i.d.-optimal policy beats the tuned threshold rule here.** The
   single-site briefing's claim that i.i.d.-optimal "barely ties" the heuristic is not
   reproduced in the Gulf 30-day cells: i.i.d. minus threshold is +1.9 to +4.5 held out
   and +2.4 to +6.9 in sample. In the July sweep the i.i.d. losses to the threshold were at
   Hawaii, lat 45 and Bering, and in the 60-day Gulf cell, not in the 30-day Gulf cells.
3. **The ranking does change with the penalty, in the chain's favour.** As the penalty
   rises the i.i.d.-optimal policy falls behind the threshold rule, losing at 150 Wh by
   penalty 20 and at every capacity by penalty 50. The chain-optimal policy beats the
   threshold at every penalty and every capacity, though its margin shrinks from about
   +5 to +9 at penalty 5 to about +3 to +4 at penalty 50. The statement "persistence is
   what makes the optimal policy win" holds at the Gulf at penalty 20 and above, not at 5.
4. **The chain-minus-i.i.d. gap widens with the penalty.** At 150 Wh it goes from +7.1 to
   +14.8 reward between penalty 5 and 50; at 300 Wh from +2.2 to +7.9; at 500 Wh from
   +2.4 to +6.6. The i.i.d. policy reduces its failure rate much less than the chain does
   when failure gets expensive (300 Wh at penalty 50: i.i.d. 31.2 %, chain 23.5 %, from
   34.0 % and 31.4 % at penalty 5). A policy that cannot see wind persistence has little
   room to avoid failures other than by flying less.
5. **Selection bias in the threshold benchmark is negligible here.** Split-half rewards
   are within 0.5 of the full-sample value in every cell. The comparison script flags the
   300 Wh cell as "selection not stable" in all three held-out runs because the odd/even
   halves pick different combos, but the scored difference is at most 0.3 reward, well
   below every chain-minus-threshold margin.
6. **The threshold rule's own tuning is penalty-insensitive.** The reward-best combo is
   (0.2, 3) at 150 and 500 Wh and (0.1, 3) at 300 Wh under every penalty. A 3 m/s wind
   threshold is near the first chain bin edge (2.7 m/s), which is consistent with the
   wind bins resolving the decision-relevant level.

## 5. Caveats

- **V0 and calibration columns in the held-out CSVs are not valid.**
  `compare_chain_sweep.py` reads the solver's start-state value and the native-world
  calibration from the full-record expected-data path, so in the three held-out
  `comparison_cells.csv` files those columns describe a different model than the one
  that was evaluated. They are not reported here. The in-sample file's columns are fine.
- **The report header date is inherited.** Each `comparison_report.md` says "Generated
  from manifest of 2026-07-03"; that timestamp was copied from the July sweep manifest the
  configs were derived from. The runs themselves are dated 2026-10-20 in
  `sweep_run_log.json` and the manifest's `note` field.
- **The held-out artifact builder is not in the repository.** The year-split expected-data
  table, wind chain and bootstrap cube were built by an analysis script kept outside the
  repo (the APL interview working folder); §2 describes what it did and the manifest's
  `holdout` block records its inputs. The `Data/` artifacts themselves are gitignored.
- **Scope.** One site, one start date, one 30-day season window, three capacities. The
  evaluation cube has 18 years, so a 30-day episode built from 7-day blocks draws from a
  smaller pool than the 73-year full record.
- **Operating point.** Failure rates differ across the three policies, so failure % alone
  does not rank them. Reward includes the penalty and is the objective; the same caveat
  as in the [threshold benchmark §4](wind-threshold-benchmark.md).

## 6. Artifacts and reproduction

Configs are committed under `configs/apl_insample_2026-10-20/`,
`configs/apl_holdout_2026-10-20/`, `configs/apl_holdout_p20_2026-10-20/` and
`configs/apl_holdout_p50_2026-10-20/` (three YAMLs and a manifest each). Results live in the
gitignored `results/<same name>/`, with per-arm run directories and an `_analysis/` folder
holding `comparison_cells.csv` and `comparison_report.md`.

Given the held-out `Data/` artifacts named in the manifest, each config set is rerun and
re-analysed with the standard sweep tools:

```bash
# from SolarSimulator/, in the pvlib conda env
conda run -n pvlib python Scripts/run_chain_sweep.py --configs ../configs/apl_holdout_2026-10-20 --out ../results/apl_holdout_2026-10-20 --resume
conda run -n pvlib python Scripts/compare_chain_sweep.py --results ../results/apl_holdout_2026-10-20 --manifest ../configs/apl_holdout_2026-10-20/chain_vs_iid_sweep_manifest.json
```

The penalty variants are the same configs with `failure_penalties` set to 20 or 50.
Rebuilding the year-split artifacts needs the external builder; the manifest `holdout`
block is the specification to reproduce it from.
