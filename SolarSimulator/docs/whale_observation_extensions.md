# Whale-Observation Model — Extension Ideas & Design Notes

*Status: design discussion, nothing implemented. Captures a conversation about extending
the whale-observation reward model and whether to move to a POMDP/MOMDP formulation.*

---

## 0. One-paragraph summary

The current whale reward is a **deterministic, known, time-of-day series** `p(t)` (the
`block_values` in `whale_base.py`, e.g. `0.278` for 18:00–20:00), and the reward for
looking is `reward = action · p(t)` — you bank the *expected* sighting value every step you
choose `action = 1`. Whale presence is not a state variable, and looking neither changes
dynamics nor yields information. The concern that motivated this discussion: that
formulation requires a probability of observing a whale that **we don't actually know**.
The conclusion is a sequenced plan: (1) build a proper **occupancy + detection model** to
estimate that probability honestly, separating "a whale is present" from "we detect it
given present"; then, if warranted, (2) move to a **Mixed-Observability MDP (MOMDP)** that
tracks a belief over hidden whale presence — which adds only one state coordinate and is
still solvable by backward induction. Step 1 is a prerequisite for step 2, because the
MOMDP's observation model *is* the detection probability that step 1 estimates.

---

## 1. Where we are today

Relevant code:

- `BaseClasses/whale_base.py` — `WhaleRewardSeriesFactory` builds the series
  (`sinusoidal`, `constant`, `real`). The `real` series is 12 two-hour block values tiled
  across the horizon. The plotting label already calls this "Whale Observation Probability".
- `BaseClasses/mdp_base.py` — `stochasticMDP.reward()` computes
  `rewards = actions * samples`, where `samples = env_provider.sample_whale_observation(t, n)`.
- `BaseClasses/environment_provider_base.py` — every provider's
  `sample_whale_observation(t, n)` just returns `whale_reward_series[t]` broadcast to `n`.

Key properties of the current formulation:

- `p(t)` is **deterministic, known, exogenous, and periodic** (time-of-day).
- Reward is the **expected** sighting value, `action · p(t)` — collected independently each
  step.
- Whale presence is **not** in the state. Looking does not change dynamics and does not
  produce information. The whale side of the policy is therefore essentially myopic given
  the energy constraints.
- **The single number conflates two things**: `p(t) = ψ(t) · d`, where `ψ` = probability a
  whale is present and `d` = probability we detect it given present. The current model
  cannot separate them.

---

## 2. The key conceptual clarification

"We don't know the observation probability" and "use a POMDP" are **two different
problems**, and a naive POMDP does **not** make the unknown probability go away:

- A standard POMDP is about acting under a **hidden state** inferred from noisy
  observations — but specifying it *requires* an observation model `P(o | hidden state, look)`,
  which **contains exactly the detection probability we said we don't know**. A POMDP
  relocates the problem, it doesn't remove it.
- "We don't know `p`" as an **unknown fixed parameter** is properly a **Bayes-Adaptive MDP**
  (learn it online) or a **robust MDP** (hedge against it) — not a classical hidden-state
  POMDP. (A BAMDP *is* formally a POMDP whose hidden state is the parameter, which is where
  the "POMDP" intuition is right — but the practical machinery is Bayesian learning.)

**A POMDP earns its keep only if whale *presence* is a genuinely latent, temporally
correlated state worth tracking** — because then "looking to find out whether whales are
around" has information value the current `action · p(t)` reward cannot represent.

---

## 3. Extension options (grouped by which problem they solve)

### A. Estimate `p(t)` honestly — separate presence from detectability *(do this first)*

- **Occupancy models** — MacKenzie et al. (2002), *"Estimating site occupancy rates when
  detection probabilities are less than one,"* Ecology 83(8). Repeat-visit data identifies
  occupancy `ψ` and detection `d` **separately** — precisely the missing parameter.
- **N-mixture models** — Royle (2004), Biometrics — same idea for abundance/counts.

Does not change the MDP structure; yields a defensible, uncertainty-quantified `p(t)`
instead of hand-set block values. Highest-value, lowest-risk move.

### B. If presence is a hidden, correlated state → genuine POMDP / MOMDP

Model latent presence `z_t` evolving as a Markov chain (reuse the wind-persistence chain
machinery), emitting observations when you look, driving reward; the policy maps
*belief → look/don't-look*, so looking becomes valuable partly for information.

- **Chadès et al. (2008), *"When to stop managing or surveying cryptic threatened
  species,"* PNAS 105(37)** — almost exactly this problem: optimal surveying under
  imperfect detection, as a POMDP. Read first.
- **Chadès et al. (2012), *"MOMDPs,"* AAAI** — mixed-observability MDP: SOC/mode fully
  observed, presence hidden. The efficient formulation for our case.
- **Ong, Png, Hsu & Lee (2010), *"Planning under uncertainty for robotic tasks with mixed
  observability,"* IJRR** — the MOMDP factorization + exact alpha-vector solution.
- **Kaelbling, Littman & Cassandra (1998), Artificial Intelligence 101** — foundational
  belief-MDP reference.

### C. If `p` is just unknown, and you want to *learn* it → Bayes-adaptive / bandit

- **Beta–Bernoulli learning**: `Beta(α, β)` prior per time-of-day block, updated from
  actual sightings — a **Bayes-Adaptive MDP** (Ghavamzadeh et al. 2015, *"Bayesian RL: A
  Survey"*). Formally a POMDP over the parameter belief.
- **Restless multi-armed bandits** — Whittle (1988): each time-block/region is an arm with
  unknown, autocorrelated payoff; Whittle-index scheduling of observations.
- **Thompson sampling** — Russo et al. (2018) tutorial — cheap way to act while learning.

### D. If you want robustness without learning → Robust MDP

- **Nilim & El Ghaoui (2005), Operations Research** and **Iyengar (2005), Math of OR** —
  plan against the worst-case `p(t)` in an uncertainty set. Good paper baseline / hedge.

### E. If the drone's *path* is a decision → informative path planning

- **Hollinger & Sukhatme (2014), *"Sampling-based robotic information gathering,"* IJRR** —
  "where to fly to see whales" as active perception. Pairs with B/C.

---

## 4. MOMDP sizing & solvability

**State-vector growth.** MOMDP carries a belief only over the *hidden* part; `(SOC, mode)`
stay fully observed.

- **Binary presence** `z ∈ {present, absent}`: belief is one scalar `b = P(z = present)`.
  State dimension 2 → 3: `(SOC, mode)` → `(SOC, mode, b)`. State *count* multiplies by `B`
  (belief-grid resolution). Value table shape `(T, N_soc, N_mode)` →
  `(T, N_soc, N_mode, B)` — one extra trailing axis; vectorization preserved.
- **K-level hidden state**: belief on the `(K−1)`-simplex; grid size `C(m+K−1, K−1)` for
  resolution `m` (`m+1` for K=2, `~m²/2` for K=3, combinatorial after). **Keep K at 2–3.**

**Why it's cheap here specifically.** The hidden whale state affects **only reward**, not
SOC/mode dynamics (flying/energy/wind don't care about whales). Consequences:

1. `x`-dynamics (existing `transition` logic) are unchanged.
2. Belief evolves as its own tiny HMM, coupled to control *only* through the look decision.
3. Expected reward `= action · b · d` is **linear in belief**, preserving PWLC structure.

**Backward induction still works — naturally.** The problem is already finite-horizon and
time-indexed (nonstationary whale series, solver sweeps `t = T…0`), which is exactly
backward induction's home turf. The belief axis changes only what happens *inside* each
backup:

- **Belief-grid route (recommended, drop-in):** discretize `b` into `B` points. Per backup,
  per action:
  - `action = look`: for each observation `o ∈ {sighting, no-sighting}`, Bayes-update
    `b → b'`, interpolate onto the grid, weight by `P(o | b)`;
  - `action = don't look`: no observation — HMM *predict* step only (belief drifts, no
    correction).
  That **predict-only-when-not-looking asymmetry** is what gives looking an information
  value and makes the policy non-myopic — something `action · p(t)` cannot express.
- **Exact alpha-vector route (PWLC):** keep `b` continuous, one alpha-vector set per
  fully-observed `x`, each vector of length `|Z|`. Finite-horizon backward induction over
  alpha-vectors is exact POMDP value iteration (Ong et al. 2010).

**New modeling inputs the MOMDP needs (and can't invent):**

- `P(z' | z)` — presence autocorrelation (small transition matrix; reuse chain machinery);
- `P(o | z, look)` — the **detection model** (this *is* the detection probability `d`);
- the prior / stationary belief.

---

## 5. Recommended sequencing

**Build the occupancy + detection model first.** The dependency is causal:

> occupancy/detection estimates → MOMDP transition + observation model → belief-grid
> backward induction.

Building the planner first would force you to make up `P(o | z, look)` — the very number
this whole thread is about not knowing. Only the occupancy model separates `ψ` from `d`,
and the MOMDP's hidden state *is* presence while its observation model *is* detection, so it
cannot be populated faithfully from a collapsed `p(t) = ψ · d`.

**Two things to decide before writing estimation code:**

1. **Is there repeat-visit / repeat-detection structure in the data?** Occupancy
   identifiability (MacKenzie 2002) rests on repeat observations at the same site/time with
   presence assumed constant. If the data is single-pass sightings, `ψ` and `d` are **not**
   separable without an extra assumption (a prior on `d`, an auxiliary sensor, or a
   literature value for `d`). This determines whether occupancy modeling is even estimable
   here.
2. **Hidden-state granularity.** Binary present/absent → one extra belief coordinate. If the
   occupancy model naturally yields abundance (N-mixture), decide whether to bin to 2–3
   levels or stay binary — that choice flows from what you estimate now.

**Clean intermediate milestone (no state-space change):** even before the full MOMDP, an
occupancy model upgrades the *existing* MDP — feed `E[sighting | look, t] = ψ(t) · d` as a
principled, uncertainty-quantified replacement for the hand-set block values. That is a
publishable improvement and a validated observation model, after which the MOMDP is a
well-posed next step rather than a leap.

---

## 6. References

- MacKenzie et al. (2002). Estimating site occupancy rates when detection probabilities are
  less than one. *Ecology* 83(8):2248–2255.
- Royle (2004). N-mixture models for estimating population size from spatially replicated
  counts. *Biometrics* 60(1):108–115.
- Chadès et al. (2008). When to stop managing or surveying cryptic threatened species.
  *PNAS* 105(37):13936–13940.
- Chadès et al. (2012). MOMDPs: a solution for modelling adaptive management problems.
  *AAAI*.
- Ong, Png, Hsu & Lee (2010). Planning under uncertainty for robotic tasks with mixed
  observability. *IJRR* 29(8):1053–1068.
- Kaelbling, Littman & Cassandra (1998). Planning and acting in partially observable
  stochastic domains. *Artificial Intelligence* 101(1–2):99–134.
- Ghavamzadeh et al. (2015). Bayesian reinforcement learning: a survey. *Foundations and
  Trends in ML* 8(5–6).
- Whittle (1988). Restless bandits: activity allocation in a changing world. *Journal of
  Applied Probability* 25A:287–298.
- Russo et al. (2018). A tutorial on Thompson sampling. *Foundations and Trends in ML*
  11(1).
- Nilim & El Ghaoui (2005). Robust control of Markov decision processes with uncertain
  transition matrices. *Operations Research* 53(5):780–798.
- Iyengar (2005). Robust dynamic programming. *Mathematics of Operations Research*
  30(2):257–280.
- Hollinger & Sukhatme (2014). Sampling-based robotic information gathering algorithms.
  *IJRR* 33(9):1271–1287.
