# Whale-Observation Model — Literature Survey & Sensor-Model Design

*Status (2026-08-29): design/literature document, nothing implemented. Companion to
`whale_observation_extensions.md`, which covers the decision-theoretic side (occupancy vs.
detection, MOMDP belief, Bayes-adaptive). This document covers the **sensor-physics and
survey-ecology** side: what the literature says about (a) aerial visual detection from a UAV
and (b) passive acoustic detection from a single hydrophone on a surface platform, and how
both wire into the existing `(SoC, mode[, wind bin, solar bin])` MDP without adding state.*

Decisions already taken (see §5): parameters are **species-agnostic ranges** meant to be swept;
an acoustic detection while moored is a **reward-only** channel (no hidden state / belief);
this pass is documentation only.

---

## 0. One-paragraph summary

Today the whale reward is one deterministic time-of-day series `p(t)` (twelve hand-set
two-hour block values) and the stage reward is `action · p(t)`. Survey ecology decomposes that
single number as `p = ψ · a · g` — presence × availability × perception — and the literature
gives explicit, published functional forms for `a` and `g` in terms of quantities the simulator
**already tracks**: wind speed (sea state / ambient noise), time of day (light, diurnal calling),
and mode (flying camera vs. moored hydrophone). The proposal is therefore a factorised
observation model `p_vis(t, wind)` earned while flying and a new `p_ac(t, wind)` earned while
moored with a hydrophone, both derived from cited formulas with config-overridable parameters,
and both reducing bit-for-bit to the current behaviour at default settings.

---

## 1. Where we are today, in survey-ecology terms

| Symbol | Meaning | Where it lives today |
|---|---|---|
| `ψ(t)` | P(whale present in the vehicle's sensing footprint during stage `t`) | folded into `p(t)` |
| `a` | **availability**: P(whale is at the surface — or vocalising — during the stage) | folded into `p(t)` |
| `g` | **perception**: P(detect \| present & available); depends on sea state, glare, light, noise | folded into `p(t)` |
| `p(t) = ψ·a·g` | what `block_values` hard-codes | `BaseClasses/whale_base.py:24-37` |

Relevant code (line numbers as of commit 76812fa on `pvlib-solar`):

- `BaseClasses/whale_base.py` — `WhaleRewardSeriesFactory` (`_series_mapping`: `sinusoidal`,
  `constant`, `real`). `RealWhaleRewardSeries` does `np.repeat(block_values, 8)` (hard-codes
  `delta_t = 15` min) and `np.tile` to the horizon; **index 0 is assumed to be 00:00**, which
  only holds because every committed config starts at midnight.
- `BaseClasses/mdp_base.py:118-129` — `stochasticMDP.reward`: `rewards = actions * samples`,
  minus the failure penalty. **No mode gate**: `action = 1` from mode 0 (takeoff) and from
  mode 1 (continue flying) earn the same `p(t)`. `docs/wind_persistence_theory.md` describes
  the reward as earned "only when flying"; the code does not enforce that. Treat as a known
  discrepancy to resolve in Tier 0 (§4).
- `BaseClasses/backward_induction_base.py:595` — `expected_reward(a_k, O_k, penalty, p_failure)
  = a_k·O_k − penalty·p_failure`, called from `_value` (300-313), `_value_batch` (357-380) and
  `_value_batch_bin` (513-534). Each fetches a **scalar** `observation_k = self.mdp.get_obs(t)`.
- `BaseClasses/environment_provider_base.py` — every provider's `sample_whale_observation(t, n)`
  broadcasts `whale_reward_series[t]`; there is no Bernoulli draw, the *expected* value is banked.
- `BaseClasses/transition_model_base.py:802-813` — the 3×2 energy table
  `[[moored_float, takeoff], [land, continue_flight], [0, 0]]`; a hydrophone's draw would be
  added to the `moored_float` entry.
- `BaseClasses/simulation_base.py` — per-step whale draw at 62 / 214 / 341; the threshold
  policy (`UnifiedThresholdContinuousSimulation`, ~580-663) is the only *decision rule* that
  reads the whale value.
- `BaseClasses/run_sim.py:268` (`whale_series` key), `:155-158` and `:299` (two separate places
  that build the series — the historical-bootstrap branch must be updated in parallel);
  `harness/summarize.py:52`; `harness/README.md:73`.

Nothing acoustic exists anywhere in the repo (grep for `hydrophone|acoustic|sonar|PAM` → 0).

---

## 2. Literature map

Citations verified against publisher / Europe-PMC / Crossref metadata. Full texts marked
**[paywalled]** could only be read as abstracts; the numbers quoted for them come from the
abstract or from secondary sources and should be re-checked against the full paper before use.

### 2a. Aerial visual detection — availability + perception (flying, mode 1)

| Paper | What it gives the model |
|---|---|
| **Hodgson, A., Peel, D. & Kelly, N. (2017).** Unmanned aerial vehicles for surveying marine fauna: assessing detection probability. *Ecological Applications* 27(4):1253–1267. doi:10.1002/eap.1519 | Fixed-wing UAV (ScanEagle, 732 m altitude, 490 m strip) over humpbacks. **Availability formula** `ā = (s̄ + t/2)/(s̄ + d̄ + t/2)` (after Laake et al. 1997), `s̄` = mean surfacing bout, `d̄` = mean dive, `t` = observation window. Numbers: `ā = 0.63` (0.86 mother/calf, 0.59 adults), overall detection **0.33**, ~35 % of pods missed through movement bias. Beaufort 1–5, cloud and glare had **no significant effect** on image-based detection. |
| **Aniceto, A.S., Biuw, M., Lindström, U., Solbø, S.A., Broms, F. & Carroll, J. (2018).** Monitoring marine mammals using unmanned aerial vehicles: quantifying detection certainty. *Ecosphere* 9(3):e02122. doi:10.1002/ecs2.2122 | UAV surveys of humpback and killer whales in Norwegian fjords: **sea state ↓, luminance ↑** detection certainty. Maps directly onto the simulator's wind bin and clear-sky/time-of-day state. |
| **Barlow, J. (2015).** Inferring trackline detection probabilities, g(0), for cetaceans from apparent densities in different survey conditions. *Marine Mammal Science* 31(3):923–943. doi:10.1111/mms.12205 **[paywalled]** | Species-specific **g(0) vs. Beaufort** curves — the cleanest published "perception vs. wind" function. Candidate for the `g_vis(wind)` multiplier. |
| **Borchers, D.L., Zucchini, W., Heide-Jørgensen, M.P., Cañadas, A. & Langrock, R. (2013).** Using hidden Markov models to deal with availability bias on line transect surveys. *Biometrics* 69(3):703–713. doi:10.1111/biom.12049 **[paywalled]** | Two-state (surface/dive) Markov availability model; shows that assuming temporally independent availability is badly biased when dive cycles are long relative to the observation window. Justifies a window-length-aware `ā(t)` and is the same machinery as the wind/solar chains. |
| **Brown, A.M. et al. (2023).** Using unoccupied aerial vehicles to estimate availability and group size error for aerial surveys of coastal dolphins. *Remote Sensing in Ecology and Conservation*. doi:10.1002/rse2.313 | Surface/dive durations from UAV focal follows (e.g. 7.9 s surface / 39 s dive for dolphins); ≥ 9 dive cycles needed for `ā` to stabilise — informs how `ā` scales with stage length. |
| **Sigourney, D.B., DeAngelis, A., Cholewiak, D. & Palka, D. (2023).** Combining passive acoustic data from a towed hydrophone array with visual line transect data to estimate abundance and availability bias of sperm whales. *PeerJ* 11:e15850. doi:10.7717/peerj.15850 | The bridge paper: joint visual + acoustic detection with a whale state-space (surface vs. foraging/vocal); acoustic range ≫ visual (clicks 5–16 km vs. ~3.8 km visual effective strip width); visual availability 0.72. Template for a **two-channel detection model with a shared whale state**. |
| **Hodgson, A., Kelly, N. & Peel, D. (2013).** Unmanned aerial vehicles (UAVs) for surveying marine fauna: a dugong case study. *PLoS ONE* 8(11):e79556. doi:10.1371/journal.pone.0079556 | Earlier fixed-wing UAV detection-rate baseline; turbidity and glare covariates. |
| **Lambert, C. et al. (2023).** Matching visual and acoustic events to estimate detection probability for small cetaceans in the ACCOBAMS Survey Initiative. *Frontiers in Marine Science* 10:1244474. doi:10.3389/fmars.2023.1244474 | Formal definitions used throughout this doc: availability bias = submerged **or acoustically silent**; perception bias = available but missed. The same taxonomy applies to both channels. |

### 2b. Passive acoustic detection from a single surface sensor (moored, mode 0)

| Paper | What it gives the model |
|---|---|
| **Thomas, L. & Marques, T.A. (2012).** Passive acoustic monitoring for estimating animal density. *Acoustics Today* 8(3):35–44. (open PDF) | The canonical **cue-counting** framework: `D̂ = n(1−ĉ) / (a · p̂ · r̂ · T)` (Eq. 2); the passive sonar equation `SNR = SL − DL − TL − NL` (Eq. 5); `p̂` via Monte-Carlo over source level, noise and range. Inverted, it gives the simulator what it needs: **expected cue detections in a stage = D · πw² · r · p̄ · Δt**. Also notes wind-noise bias in `p̂` (Ward 2011). |
| **Marques, T.A., Thomas, L., Martin, S.W., Mellinger, D.K., Ward, J.A., Moretti, D.J., Harris, D. & Tyack, P.L. (2013).** Estimating animal population density using passive acoustics. *Biological Reviews* 88(2):287–309. doi:10.1111/brv.12001 | Full review of the above; cue-rate variability with time of day, group size, season — i.e. `r(t)` should be diurnal, the same shape the sim's `p(t)` already has. |
| **Küsel, E.T., Mellinger, D.K., Thomas, L., Marques, T.A., Moretti, D. & Ward, J. (2011).** Cetacean population density estimation from single fixed sensors using passive acoustics. *JASA* 129(6):3610–3622. doi:10.1121/1.3583504 **[paywalled]** | **Single-hydrophone** detection probability from propagation modelling + an SNR→detection regression — exactly the sensor configuration of a moored seaplane with one hydrophone. Estimated `p̂ = 0.014` (CV 18 %) for beaked whale clicks vs. 0.032 from tag data. |
| **Helble, T.A., D'Spain, G.L., Hildebrand, J.A., Campbell, G.S., Campbell, R.L. & Heaney, K.D. (2013).** Site specific probability of passive acoustic detection of humpback whale calls from single fixed hydrophones. *JASA* 134(3):2556–2570. doi:10.1121/1.4816581 **[paywalled]** | Detection probability of humpback calls on a single hydrophone varies by **> 10×** across sites and over time at one site, from bathymetry, sediment, sound speed and noise. Argues for a **site-specific effective radius** config parameter rather than a universal constant. |
| **Harris, D.V., Mellinger, D.K., Heaney, K.D., Clarke, T., Miles, D. & Thomas, L. (2025).** Estimating the detection probability of long-ranging baleen whale song using a single sensor: towards density estimation. *JASA* 158(6):4582. doi:10.1121/10.0036892 **[paywalled]** | Most recent single-sensor method (Monte-Carlo over TL, noise, source level 179–189 dB, detector efficiency). Gives the density → detection-rate pipeline to copy. |
| **Johnson, H.D., Taggart, C.T., Newhall, A.E., Lin, Y.-T. & Baumgartner, M.F. (2022).** Acoustic detection range of right whale upcalls identified in near-real time from a moored buoy and a Slocum glider. *JASA* 151(4):2558. doi:10.1121/10.0010124 **[paywalled]** | **Hard numbers for a surface-ish platform**: P(det) = 0.33 at 6.2 km (buoy) / 8.6–13.4 km (glider); 0.10 at 14.4 km / 22.6–27.5 km. Detection depends on range, **ambient noise, platform depth, calling rate**; P(det) rises substantially if more than one call is available per window. Ready-made shape for `g_ac(range, NL)` and a range for `w`. |
| **Baumgartner, M.F. et al. (2020).** Slocum gliders provide accurate near real-time estimates of baleen whale presence from human-reviewed passive acoustic detection information. *Frontiers in Marine Science* 7:100. doi:10.3389/fmars.2020.00100 | DMON/LFDCS real-time detector on an autonomous platform, reviewed in **15-min tally periods** (= the sim's `delta_t`): **< 0.3 % false positives per 15 min, 0 % daily; 34–64 % missed occurrences per 15 min, 17–24 % daily**. Use as per-stage detector recall bounds. Also the operational model: low-power onboard detector + Iridium uplink every 2 h. |
| **Lammers, M.O. et al. (2023).** The occurrence of humpback whales across the Hawaiian archipelago revealed by fixed and mobile acoustic monitoring. *Frontiers in Marine Science* 10:1083583. doi:10.3389/fmars.2023.1083583 | Wave Glider (surface vehicle) hydrophone vs. moorings: **listening radius 4–28 km for the surface platform, 5–55 km for moorings**; song detection ↓ with noise (GAM). Relevant because the Hawaii trade-wind site (lat 20 / lon −159) is already one of the sim's locations. |
| **Fregosi, S., Harris, D.V., Matsumoto, H., Mellinger, D.K., Barlow, J., Baumann-Pickering, S. & Klinck, H. (2020).** Detections of whale vocalizations by simultaneously deployed bottom-moored and deep-water mobile autonomous hydrophones. *Frontiers in Marine Science* 7:721. doi:10.3389/fmars.2020.00721 | Detection ranges by call type (minke boing P ≈ 0.8–0.9 at 10 km, ≥ 0.5 to ~25 km; beaked whale clicks < 3.5 km on-axis, < 1 km off-axis); platform self-noise floor matters (10 dB lower floor → fainter signals detectable in low sea state). |
| **Harris, D., Fregosi, S., Klinck, H., Mellinger, D.K., Barlow, J. & Thomas, L. (2017).** Evaluating autonomous underwater vehicles as platforms for animal population density estimation. *JASA* 141(5 Suppl.):3606. doi:10.1121/1.4987720 (AFFOGATO project) | Framework for density/detection from slow or drifting autonomous platforms — the closest analogue to a moored floating seaplane; track deviations up to 20 km biased density ≤ 9 %. |
| **Hildebrand, J.A., Frasier, K.E., Baumann-Pickering, S. & Wiggins, S.M. (2021).** An empirical model for wind-generated ocean noise. *JASA* 149(6):4516–4533. doi:10.1121/10.0005430 (open PDF at cetus.ucsd.edu, password-protected for text extraction) | **Ambient noise level as a function of wind speed and frequency** (Wenz-curve replacement; wind dominates 400 Hz–50 kHz; saturates / reverses above ~15 m/s at high frequency). This is the link from the simulator's **existing wind state** to acoustic detection radius: `NL(wind) → SNR → w(wind)`. |
| **Ward, J., Jarvis, S., Moretti, D., Morrissey, R., DiMarzio, N., Thomas, L. & Marques, T.A. (2011).** Beaked whale (*Mesoplodon densirostris*) passive acoustic detection with increasing ambient noise. *JASA* 129(2):662–669. doi:10.1121/1.3531844 **[paywalled]** | Empirical detection-probability drop with added wind noise — **smaller than theory predicts**. Sanity bound on how steep `g_ac(wind)` should be. |

### 2c. Landed / water-surface UAV hydrophone deployments (the "observe while landed" precedent)

| Paper | What it gives the model |
|---|---|
| **Babatunde, D., Pomeroy, S., Lepper, P., Clark, B. & Walker, R. (2020).** Autonomous deployment of underwater acoustic monitoring devices using an unmanned aerial vehicle: the Flying Hydrophone. *Sensors* 20(21):6064. doi:10.3390/s20216064 | UAV that **lands, listens for a fixed dwell (10 s), takes off, repeats** at waypoints; motors off while listening; custom < 250 g recorder streaming FFTs. Explicit energy argument: persistent landing cuts consumption vs. hovering. Direct precedent for a listen-while-moored action. |
| **Frouin-Mouy, H., Tenorio-Hallé, L., Thode, A., Swartz, S. & Urbán, J. (2020).** Using two drones to simultaneously monitor visual and acoustic behaviour of gray whales in Baja California, Mexico. *J. Exp. Mar. Biol. Ecol.* 525:151321. doi:10.1016/j.jembe.2020.151321 **[paywalled]** | Waterproof quadcopter (Swellpro) landing on water with a SoundTrap on a 2 m line while a second drone films — the **two-channel (visual + acoustic) observation** configuration. |
| **Erbe, C., Parsons, M., Duncan, A.J., Osterrieder, S. & Allen, K. (2017).** Aerial and underwater sound of unmanned aerial vehicles (UAV, drones). *Journal of Unmanned Vehicle Systems* 5(3):92–101. doi:10.1139/juvs-2016-0018 **[paywalled]** | UAV motor noise couples poorly into water; with motors off, the floating hull's self-noise is wave-slap / flow noise — i.e. treat the moored platform like a drifting buoy (Johnson 2022 numbers apply). |
| **Peel, D., Miller, B.S., Kelly, N., Dawson, S., Slooten, E. & Double, M.C. (2014).** A simulation study of acoustic-assisted tracking of whales for mark-recapture surveys. *PLoS ONE* 9(5):e95602. doi:10.1371/journal.pone.0095602 | **Acoustic cue → visual follow-up** simulation: whale groups as correlated random walks; sperm whale 30–50 min dives, clicking ~80 % of the dive, 7–14 min at surface; blue whale acoustic range 50 km (sensitivity 10–100 km); visual detection via effective strip width, daylight only, Beaufort ≤ 5. The "listen while moored, take off on detection" loop, already simulated. |
| **Keen, E.M., Hendricks, B., Shine, C., Wray, J., Picard, C.R. & Alidina, H.M. (2022).** A simulation-based tool for predicting whale-vessel encounter rates. *Ocean & Coastal Management* 223:106183. doi:10.1016/j.ocecoaman.2022.106183 **[paywalled]** | Encounter-rate simulator (Poisson encounters per grid cell from density + movement) — reusable if presence `ψ` is ever made spatial. |

### 2d. Decision-theoretic references

Already collected in `whale_observation_extensions.md` §3/§6: MacKenzie et al. 2002 and Royle
2004 (occupancy / N-mixture), Chadès et al. 2008 & 2012 (POMDP surveying, MOMDP), Ong et al.
2010, Kaelbling et al. 1998, Hollinger & Sukhatme 2014. Not repeated here.

---

## 3. Parameters and formulas (species-agnostic ranges)

All values are **ranges to sweep**, not fixed defaults; species-specific numbers are worked
examples only.

| Parameter | Range / form | Sources |
|---|---|---|
| Visual availability `a_vis` per stage | `ā = (s̄ + t/2)/(s̄ + d̄ + t/2)`. Short windows: 0.6–0.65 (humpback), 0.72 (sperm whale). For a 15-min stage `t ≫ d̄`, so `ā → 1` for most baleen whales and the binding terms become presence and perception. | Hodgson 2017; Sigourney 2023; Borchers 2013 |
| Visual perception `g_vis(wind)` | Monotone decreasing in Beaufort (from wind m/s via the standard Beaufort table); image-based UAV detection roughly flat for Beaufort 1–5 (Hodgson 2017) but real-time / low-altitude detection falls with sea state (Aniceto 2018); hard cutoff at Beaufort 5–6 (Peel 2014). Sweep a linear ramp `g_vis = clip(1 − k·(B − B0), 0, 1)` with `B0 ∈ {2, 3}`, `k ∈ {0.1, 0.25}`. | Barlow 2015; Aniceto 2018; Hodgson 2017; Peel 2014 |
| Visual light gate `light(t)` | 0 at night, 1 by day (already implicit in the zero block values); optional luminance scaling by clear-sky index. | Aniceto 2018 |
| Acoustic effective radius `w` (surface platform, baleen whale) | **4–28 km** (Wave Glider); 0.33 detection at ~6 km, 0.10 at ~14 km (buoy). Sweep `w_km ∈ {4, 8, 15, 28}`. Small odontocete clicks: < 1–3.5 km. | Lammers 2023; Johnson 2022; Fregosi 2020 |
| Acoustic detection vs. noise | Sonar equation `SNR = SL − TL(r) − NL(wind)`, `TL ≈ 15–20·log10 r`, `NL(wind)` from Hildebrand 2021 (Wenz-like, saturating above ~15 m/s). Simplest usable form: a **detection-radius table by wind bin**. | Thomas & Marques 2012; Hildebrand 2021; Ward 2011 |
| Per-stage detector recall `p̄_det` | **0.36–0.66** at 15-min scale, 0.76–0.83 daily; false-positive rate ≈ 0. | Baumgartner 2020 |
| Cue rate `r(t)` | Species- and time-of-day-dependent; humpback song often higher at night; sperm whale clicks ~80 % of each dive. Treat as a diurnal series with the same shape as today's `block_values`. | Marques 2013; Peel 2014 |
| Stage acoustic detection probability | `p_ac = 1 − exp(−ψ · D · π w² · r · p̄_det · Δt)` — Poisson cue counting inverted. | Thomas & Marques 2012 |
| Hydrophone + detector power | O(1 W) (DMON-class) added to the mode-0 idle draw. | Baumgartner 2020 (qualitative) |

### Two formulas to keep

**Availability (Laake 1997 / Hodgson 2017):**

```
ā(t_window) = (s̄ + t_window/2) / (s̄ + d̄ + t_window/2)
```

**Acoustic detection in one stage (inverted cue counting, Thomas & Marques 2012 Eq. 2):**

```
λ(t, wind) = D · π · w(wind)² · r(t) · p̄_det          # expected detected cues per unit time
p_ac(t, wind) = 1 − exp(−ψ(t) · λ(t, wind) · Δt)
w(wind): from SNR = SL − TL(w) − NL(wind) ≥ threshold, or a per-wind-bin table
```

With `D·πw²` interpreted as the expected number of whales in the listening disc, this is
"P(at least one cue detected in Δt)". At small `λΔt` it reduces to `ψ · λ · Δt`, linear in
presence — the same structural form as the visual channel.

---

## 4. Integration design — three tiers, each independently useful

Guiding invariant (from `clearsky_model` / `pvwatts`, commit 76812fa): **every new config key
defaults to bit-identical current behaviour**, verified by a `Tests/verify_*.py` script that
keeps the old code path in-source and diffs new vs. old in-process.

### Tier 0 — Mode gate + configurable `p(t)` (no model change)

- **Mode gate.** Add config key `reward_takeoff_stage` (default `true` = current behaviour).
  When `false`, `reward = p(t) · 1[action = 1 ∧ mode = 1]` so the takeoff stage earns nothing.
  Touch `mdp_base.py:118-129` (`reward` gets `states` already), `backward_induction_base.py:595`
  `expected_reward` and its callers `_value` / `_value_batch` / `_value_batch_bin` (each has
  `states` in scope), and `simulation_base.py` step paths. The threshold policy already gates
  on mode and needs no change.
- **Configurable series.** `whale_series: custom` + `whale_block_values: [12 floats]`
  (+ `whale_block_hours: 2`) as a new class in `whale_base._series_mapping`; derive steps-per-
  block from `delta_t` instead of the hard-coded 8; roll the daily series by the start hour from
  `start_datetime`. Register the key in `run_sim.py:268` and thread it into **both** series
  construction sites (`:155-158` and the historical-bootstrap branch `:299`), add a column in
  `harness/summarize.py:52`, and a row in `harness/README.md:73`.
- **Verify**: `Tests/verify_whale_model.py` asserts `custom` with the default block values equals
  `real` exactly, and that the value table from `harness/examples/iid_small.yaml` is unchanged.

### Tier 1 — Factorised visual model `p_vis(t, wind) = ψ(t) · a_vis · g_vis(wind) · light(t)`

- New class `FactoredWhaleObservationModel` in `whale_base.py` returning a **(T, n_wind_bins)**
  array (or (T,) when there is no chain, by taking the Weibull-weighted expectation over bins).
  Inputs: `ψ(t)` = the block series; `a_vis` from `surface_s`, `dive_d`, `delta_t` via the
  availability formula; `g_vis` = a Beaufort ramp evaluated at each wind bin's representative
  speed (`environment_provider_base` already exposes bin edges); `light(t)` from the solar
  pipeline's sunrise/sunset or a clear-sky-index scaling.
- Solver: `_value_batch_bin` (`backward_induction_base.py:513-534`) currently calls
  `self.mdp.get_obs(t)` for a scalar; make `get_obs(t)` return a per-bin vector and index it with
  the wind bin `b` computed on line 521. `_value` / `_value_batch` (i.i.d. solver) use the
  bin-averaged scalar, so the value-table shape is unchanged.
- Simulation: `sample_whale_observation(t, n)` must return the value for each lane's *current*
  wind bin (`last_wind_bins` is already tracked in the chain provider at
  `environment_provider_base.py:271`).
- **Result**: wind now affects the reward, not just the transition risk. Calm windows become
  doubly valuable (safe *and* good visibility) — a trade-off the wind-chain policy can exploit.

### Tier 2 — Acoustic channel while moored: `p_ac(t, wind)` earned in mode 0 (the target)

- **Model** (§3): `p_ac(t, wind) = 1 − exp(−ψ(t) · λ(t, wind) · Δt)`, with `w(wind)` either from
  the sonar equation with `NL(wind)` (Hildebrand 2021) or — recommended first — a
  **detection-radius table by wind bin** seeded from Johnson 2022 / Lammers 2023.
- **Config** (new block, all defaults off/zero so existing runs are untouched):

  ```yaml
  acoustic_model:
    enabled: false
    reward_weight: 0.0          # alpha_ac; 0 => current behaviour
    radius_km_by_wind_bin: [15, 10, 6, 3]   # or noise_model: wenz + source_level_db, tl_coeff
    cue_rate_per_hr: 30         # r(t) scalar, or a 12-block diurnal list
    detector_recall: 0.5        # p_bar_det, 15-min scale (0.36-0.66)
    density_per_km2: 0.01       # D
    hydrophone_power_w: 1.0
  ```

- **Reward**: `reward = 1[mode=1, act=1]·p_vis(t, wind) + α_ac · 1[mode=0, act=0]·p_ac(t, wind)`.
  Landing (`mode=1, act=0`) and takeoff (`mode=0, act=1`) stages earn nothing under the Tier 0
  gate. Implementation: `get_obs(t)` returns a small (n_modes × n_actions[, n_bins]) table
  instead of a scalar; `expected_reward` indexes it by `(mode, action)`.
- **Energy**: add `hydrophone_power_w · Δt` to `moored_float_energy` in
  `transition_model_base.py:804`. Listening is **always-on while moored** in this design;
  making it a switchable third action (2 → 3 actions) is deferred (open question §6).
- **Simulation / threshold policy**: `UnifiedThresholdContinuousSimulation` compares `whale >
  observation_threshold` to decide takeoff; with two channels it should compare `p_vis` (the
  gain from flying) — expose both channels to `choose_action` and keep the threshold on `p_vis`.
- **Why it matters**: the acoustic channel works at night and in high wind — exactly when
  `p_vis = 0`. The optimal policy should move from "sleep at night" to "listen at night, fly at
  dawn", changing the day/night structure of every existing result. This is the headline
  experiment (§5).

### Tier 3 (if later) — acoustic detection as information (MOMDP)

Not designed here. If Tier 2 shows the acoustic channel changes decisions, the natural next
step is `whale_observation_extensions.md` §4: hidden presence `z_t` with a belief coordinate,
where `P(o_ac | z, moored)` comes from Tier 2's `p_ac` and Baumgartner 2020's FP/miss rates,
and `P(o_vis | z, flying)` from Tier 1. The policy becomes "listen while moored → belief rises →
take off when `b` clears a threshold" (Peel 2014; Sigourney 2023).

---

## 5. Proposed headline experiment (after Tier 2)

- **Grid**: `acoustic_model.reward_weight ∈ {0, 0.25, 0.5, 1.0}` × `radius_km_by_wind_bin`
  scaled by `{0.5, 1, 2}` × the existing capacity sweep (300–400 Wh) × the three sites already in
  the configs (Gulf lat 30/−90, SE-US shelf lat 30/−75, Hawaii lat 20/−159). Keep the chain
  configs (`wind_chain.enabled`) so `p_ac(wind)` interacts with the wind-persistence policy.
- **Readout** from the existing `summary.csv` columns: mean reward split by channel (add
  `reward_visual` / `reward_acoustic` columns), fraction of night stages spent moored vs. the
  baseline (unchanged at `reward_weight = 0`), number of takeoffs per day (expect fewer, later
  takeoffs as listening substitutes for marginal daytime flights), and failure rate (expect a
  small rise if the hydrophone draw is non-trivial at low SoC).
- **Hypotheses**: (i) at `reward_weight ≥ 0.25` night reward is non-zero and total reward rises
  monotonically; (ii) the threshold-policy vs. chain-optimal gap widens because the threshold
  rule cannot trade acoustic reward against flight; (iii) high-wind sites (Hawaii) benefit least
  because `w(wind)` shrinks exactly when flying is also grounded.
- Follow `check-existing-results-first`: the `reward_weight = 0` cells are the existing runs —
  reuse rather than rerun.

---

## 6. Open questions

1. **Always-on vs. switchable listening.** Always-on (this doc) keeps the action space at two
   and folds the hydrophone draw into idle power. A third action ("moored + listen") would let
   the policy save ~1 W at critically low SoC; only worth it if the failure-rate readout in §5
   moves.
2. **Luminance proxy.** Should `p_vis` scale with the clear-sky index (Aniceto 2018 found
   luminance improves detection certainty)? Cheap to add in Tier 1 since the solar bin is
   already in the state when `solar_chain.enabled`.
3. **Site-specific `w`.** Helble 2013 shows > 10× site variation. Per-location
   `radius_km_by_wind_bin` in the `locations` block, or a single experiment-wide value swept?
4. **Relative worth of an acoustic vs. visual detection (`reward_weight`).** A mission-design
   choice, not a literature question; the sweep in §5 makes it explicit rather than hidden.
5. **Cue rate `r(t)` diurnality.** For song-producing species the acoustic diurnal curve is
   roughly the *complement* of the visual one; whether to model it as such or hold it flat
   determines how strong the "listen at night" effect is.
