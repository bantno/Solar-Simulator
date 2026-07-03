# Wind-Persistence Extension — Validity, Theory, and Implementation Differences

*Branch: `wind-persistence`. Status: experimental, **default off**. This document explains
**why** the wind-persistence extension is a legitimate modification of the published i.i.d.
MDP — the theory and the math that make it valid — and then catalogues exactly **what
changed in the implementation** relative to the pre-persistence code.*

*Companion documents: [`wind_persistence_math.md`](wind_persistence_math.md) gives the full
formula reference and symbol table; [`wind_persistence_briefing.md`](wind_persistence_briefing.md)
gives the empirical results and the advisor briefing. This document is the correctness
argument that sits between them, and is self-contained on the validity question. Bracketed
tags like `[backward_induction_base.py: _solve_chain]` point at the implementing code.*

---

## 0. The claim, in one paragraph

The published model draws wind independently at every 15-minute control step. The extension
replaces that i.i.d. draw with a **first-order, time-conditioned Markov chain over
wind-speed bins** and adds the current bin as a state variable. The core validity claim is:

> **Adding the wind bin to the state is not a heuristic patch — it is the state
> augmentation that makes the controlled process Markov again once wind is temporally
> correlated. The augmented model is a bona-fide finite-horizon MDP, backward induction on
> it is exactly optimal, the modeled wind *climatology* (marginal distribution) is preserved
> identically, and the published i.i.d. model is recovered exactly as the one-bin special
> case.**

Everything below substantiates that sentence: §1–§2 (why the un-augmented state stops being
Markov and how augmentation fixes it), §3–§6 (the math: transition kernel, Bellman
recursion, marginal-preservation identity, sampling correctness), §7 (exact reduction to
i.i.d.), §8 (why *first* order is the correct model order), and §9 (implementation
differences).

---

## 1. Why we need the extension at all: correlated wind breaks the Markov property of `s`

### 1.1 The published model

The endurance problem is a finite-horizon MDP over stages $t = 0,\dots,T-1$
[`mdp_base.py`, `backward_induction_base.py`]:

- **Endogenous state** $s = (\sigma, m)$: battery state-of-charge $\sigma$ on a discrete
  energy grid, and operating mode $m\in\{\text{moored}=0,\ \text{flying}=1,\ \text{broken}=2\}$.
- **Action** $a\in\{0,1\}$.
- **Exogenous weather** at stage $t$: wind $W_t$ and solar $G_t$.
- **Reward** $r(s,a,s',t)$ with failure penalty $c_f$; **discount** $\gamma$ (code sets
  $\gamma=1$, `self._GAMMA = 1.0`).

In the published model the weather is **independent across stages**,
$W_t \sim \text{Weibull}(k_t,\lambda_t)$ i.i.d. (with month/hour-dependent parameters), and
the value function depends only on $(s,t)$:

$$
V_t(s) = \max_{a} \; \mathbb{E}_{W_t, G_t}\!\big[\, r + \gamma V_{t+1}(s') \,\big],
\qquad V_T(s) = 0. \tag{1}
$$

This is valid **because $W_t$ is independent of everything before $t$.** The distribution of
$s_{t+1}$ given $(s_t, a_t)$ does not depend on the past, so $s_t$ is a Markov state and (1)
is the correct Bellman equation.

### 1.2 What real wind does to that argument

Real wind is autocorrelated: calm follows calm, gusty follows gusty (the ACF/PACF and
Markov-order diagnostics quantify this — §8). Suppose we keep the *same* endogenous state
$s = (\sigma, m)$ but let the true wind process be temporally correlated. Now the
distribution of $W_t$ depends on $W_{t-1}, W_{t-2},\dots$, and since $W_t$ drives the failure
and energy transition of $s$, the transition kernel of $s$ becomes

$$
\Pr(s_{t+1}\mid s_t, a_t, \underbrace{s_{t-1}, a_{t-1}, \dots}_{\text{history}})
\;\ne\; \Pr(s_{t+1}\mid s_t, a_t).
$$

**The endogenous state $s$ is no longer a Markov state.** Backward induction on $s$ alone —
equation (1) — is therefore *not* the right Bellman equation for correlated weather: it
implicitly assumes the wind you face next step is independent of the wind you face now,
which is exactly false. A policy solved from (1) is optimal for the i.i.d. world, not for
the correlated one.

This is the precise sense in which the extension is needed. It is **not** that the i.i.d.
model was internally inconsistent — it was correct for its own assumptions. It is that on
correlated wind, the state must be enlarged to stay Markov.

---

## 2. The fix: state augmentation with a sufficient statistic

The standard remedy for a non-Markov controlled process is to **augment the state with a
sufficient statistic of the exogenous process's memory**. For a first-order wind model the
sufficient statistic is the *current wind regime*. We discretize wind into bins and track the
current bin $\beta_t$.

Define the **augmented state** $x_t = (\beta_t, s_t)$, where $\beta_t \in \{0,\dots,n_b-1\}$
is the current wind-bin index. The claim of validity rests on two facts, proved in §3:

1. **$\beta_t$ is a sufficient statistic for the wind process's future** (under a first-order
   bin chain). Given $\beta_t$, the future wind — hence the future of $s$ — is conditionally
   independent of all history. So $x_t = (\beta_t, s_t)$ *is* Markov.
2. **$\beta_t$ is observable at decision time.** The controller sees the current wind regime
   before it acts (the simulator exposes the bin, then advances the chain — §6, and
   `sample_wind_speed` in `environment_provider_base.py`). An observable component of the
   state may be conditioned on by an optimal policy without any loss of optimality; that is
   just the definition of a Markov policy on the augmented state.

Together these make $x_t = (\beta_t, s_t)$ a legitimate MDP state, so the Bellman equation on
$x$ is exactly optimal. That is the whole validity argument in miniature; the rest is making
each step precise.

> **Framing note.** The extension *preserves* the Markov property by enlarging the state; it
> does not "correct a bug." The wind bin is an **exogenous** state variable: the controller
> cannot influence it, only observe it and react. This is the same move as adding a
> weather/regime variable to any stochastic-control problem.

---

## 3. The math that makes it valid

### 3.1 Wind discretization and the within-bin distribution

Partition wind speed into $n_b$ contiguous bins with edges
$0 = e_0 < e_1 < \dots < e_{n_b} = \infty$, bin $b = [e_b, e_{b+1})$, and bin map
$\beta(w) = \#\{i: e_i \le w\} - 1$. Interior edges are chosen either as equal-occupancy
quantiles of the historical record (default) or as user-supplied physical thresholds
[`create_weather_distributions.py: fit_wind_transition_chain`].

Crucially, **persistence is carried by the bin, but the wind used inside the failure
integral stays continuous.** Conditioned on bin $b$, stage-$t$ wind is the stage-$t$ Weibull
*truncated and renormalized* to $[e_b, e_{b+1})$:

$$
f_t(w\mid b) = \frac{f_t^{\mathrm W}(w)}{F_t^{\mathrm W}(e_{b+1}) - F_t^{\mathrm W}(e_b)}\,
\mathbb 1\{e_b \le w < e_{b+1}\},
\qquad
\pi_t(b) = F_t^{\mathrm W}(e_{b+1}) - F_t^{\mathrm W}(e_b), \tag{2}
$$

with $F_t^{\mathrm W}(e) = 1 - \exp[-(e/\lambda_t)^{k_t}]$
[`backward_induction_base.py: _get_wind_grid_bin`; `environment_provider_base.py:
_sample_within_bin`].

### 3.2 Marginal-preservation identity (why the climatology is unchanged)

The single most important consistency property: **binning + truncation does not change the
modeled wind distribution at any stage.** By the law of total probability, the bin-mixture of
the truncated densities is exactly the original Weibull:

$$
\boxed{\;\sum_{b=0}^{n_b-1} \pi_t(b)\, f_t(w\mid b) \;=\; f_t^{\mathrm W}(w)\;} \tag{3}
$$

*Proof.* Each term is $\pi_t(b)\cdot f_t^{\mathrm W}(w)/\pi_t(b)\cdot\mathbb 1\{w\in[e_b,e_{b+1})\}
= f_t^{\mathrm W}(w)\,\mathbb 1\{w\in[e_b,e_{b+1})\}$; summing the indicator over the disjoint
bins that tile $[0,\infty)$ gives $f_t^{\mathrm W}(w)$. $\square$

Equation (3) is what guarantees the extension changes only the **temporal correlation** of
wind, never its **marginal climatology**. It is checked numerically in
`Tests/verify_wind_chain.py` (the within-bin mixture reproduces the full Weibull marginal).

### 3.3 The time-conditioned first-order chain

The bin evolves as a first-order Markov chain whose transition matrix is conditioned on the
calendar (month, hour) of the stage:

$$
\Pr(\beta_{t+1} = b' \mid \beta_t = b) = P^{(m_t, h_t)}_{b, b'} \;\equiv\; P^{(t)}_{b,b'}. \tag{4}
$$

The full object is a tensor $P\in\mathbb R^{13\times 24\times n_b\times n_b}$ (one stochastic
matrix per month/hour, $288$ matrices), fit by stratified transition counting and
row-normalization with a uniform fallback for unobserved rows
[`fit_wind_transition_chain`; `run_sim.py: EnvironmentLoader._build_per_stage_transition`]:

$$
P^{(m,h)}_{b,b'} =
\begin{cases}
N^{(m,h)}_{b,b'}\big/\sum_{b''} N^{(m,h)}_{b,b''}, & \text{row sum} > 0,\\
1/n_b, & \text{otherwise.}
\end{cases}
$$

Because (month, hour) is a **deterministic function of the stage index $t$**, conditioning
the transition on it does not break the Markov property of $x_t=(\beta_t,s_t)$: it is
time-inhomogeneous, not history-dependent. Putting the diurnal/seasonal drift *into* the
chain also matters for the diagnostic in §8 (it isolates genuine higher-order memory).

### 3.4 The augmented transition kernel factorizes correctly

Write the one-step dynamics of $x_t=(\beta_t, s_t)$ under action $a$. Two exogenous draws
occur: the within-step wind $W_t\sim f_t(\cdot\mid\beta_t)$ (used only to decide
success/failure and the energy update of $s$), and the next bin
$\beta_{t+1}\sim P^{(t)}_{\beta_t,\cdot}$. Solar $G_t$ is drawn i.i.d. independently. The kernel is

$$
\Pr\big(\beta_{t+1}=b',\, s_{t+1}=s' \,\big|\, \beta_t=b,\, s_t=s,\, a\big)
\;=\;
\underbrace{P^{(t)}_{b,b'}}_{\text{next bin}}
\;\cdot\;
\underbrace{\Pr\big(s_{t+1}=s' \mid s, a, \beta_t=b, t\big)}_{\text{next endogenous state}}. \tag{5}
$$

The factorization in (5) encodes one explicit, deliberate **modeling assumption**:

> **Conditional independence of the next bin from the within-step outcome.** Given the
> current bin $\beta_t=b$, the next bin $\beta_{t+1}$ is drawn from $P^{(t)}_{b,\cdot}$
> independently of the realized within-step wind $W_t$ and of whether the step succeeded or
> failed.

This is exactly how the code advances the chain — `_advance_bins` samples $\beta_{t+1}$ from
$P[\beta_t]$, using only the bin, not the continuous $W_t$ [`environment_provider_base.py`].
It is the natural first-order assumption: the continuous within-bin position is a *nuisance*
variable used to keep the failure integral sharp (§3.5), and it is not allowed to feed back
into the regime dynamics. The factorization (5) is what lets the future-value term split into
a bin-average and an energy-average in the Bellman recursion (§3.6). It is consistent (it
never contradicts (3)) and it is the assumption a first-order chain is *defined* by.

### 3.5 Failure probability (binned) — well-defined and reduces correctly

A step fails if the battery depletes (a solar/energy event $p_B$) **or** a mechanical wind
event occurs ($p_M$), combined as independent causes
[`_compute_failure_probability_bin`]:

$$
p_{\text{fail}}(s,a,t,b) = p_B(s,a,t) + \big(1 - p_B(s,a,t)\big)\,p_M(s,a,t,b). \tag{6}
$$

- **Solar/energy term $p_B$** — *unchanged and not binned* (solar stays i.i.d.). With energy
  deficit $\Delta = E_{\text{req}}(s,a) - E(\sigma)$ and normalizer
  $G_{\max}=\max(G^{\text{cs}}_t, 10)$,
  $p_B = I_u(\alpha_t,\beta_t)$, $u=\operatorname{clip}(\Delta/G_{\max},0,1)$, with $I_u$ the
  regularized incomplete beta function.
- **Mechanical/wind term $p_M$** — the expected one-step mechanical failure *given wind is in
  bin $b$*, integrated against the truncated density (2):

$$
p_M(s,a,t,b) = \int_{e_b}^{e_{b+1}} \big(1 - q(w,a,s)\big)\, f_t(w\mid b)\, \mathrm dw, \tag{7}
$$

  where $q(w,a,s)$ is the mechanical success curve (a logistic/sigmoid in wind speed;
  `transition_model_base.py`). The integral is evaluated by trapezoidal quadrature on a
  per-(stage, bin) grid [`_mechanical_failure_probability_bin`].

**Reduction check.** Combining (7) with the marginal identity (3): the bin-mass-weighted
average of the binned mechanical failure is the original full-Weibull failure integral,

$$
\sum_b \pi_t(b)\, p_M(s,a,t,b)
= \int_0^\infty (1 - q(w,a,s))\, f_t^{\mathrm W}(w)\, \mathrm dw
= p_M^{\text{iid}}(s,a,t).
$$

So the binned failure model is not a different physics — it is the *same* failure integral,
sliced by regime. With $n_b=1$ it is identical to the published $p_M$.

### 3.6 The Bellman recursion on the augmented state

The value function gains a bin dimension, $V_t(b,s)$, stored as an
$n_b\times|\mathcal S|\times T$ table [`_initialize_future_value_table`]. Using the kernel
factorization (5), the state-action value is

$$
Q_t(b,s,a) =
\Big[\, a\,O_t - c_f\, p_{\text{fail}}(s,a,t,b)\,\Big]
\;+\;
\gamma\,\big(1 - p_{\text{fail}}\big)\!\!
\sum_{\text{energy bins } e}\!\! \Delta P^{(t)}_{e}(s,a)
\underbrace{\sum_{b'=0}^{n_b-1} P^{(t)}_{b,b'}\, V_{t+1}\!\big(b', s'_e\big)}_{\text{effective next-stage value } \widetilde V_{t+1}(b,\,s'_e)}, \tag{8}
$$

with $O_t$ the whale-observation reward, $\Delta P^{(t)}_e$ the solar-driven probability of
landing in successor SoC bin $s'_e$ (the regularized-beta bin masses of the survival
contribution), and the Bellman optimality / terminal conditions

$$
V_t(b,s) = \max_a Q_t(b,s,a),\qquad V_T(b,s)=0,\qquad
\pi^\star_t(b,s) = \arg\max_a Q_t(b,s,a). \tag{9}
$$

*(This matches the code exactly:
`expected_reward = a·O − c_f·p_fail`; the survival contribution multiplies
$(1-p_{\text{fail}})$ by the solar energy-bin masses and dots into the effective next value;
and $\widetilde V$ is the bin contraction `P_row @ V_next` in `_vnext_eff`, equivalently the
`einsum('nb,bn->n')` in `value_function_batch`.)*

**Two structural points that make (8) correct:**

1. **The double sum is legitimate precisely because of the factorization (5).** The next SoC
   $s'_e$ (driven by i.i.d. solar) and the next bin $b'$ (driven by $P^{(t)}$) are
   conditionally independent given $(b,s,a)$, so the joint expectation of $V_{t+1}$ separates
   into an inner bin-average $\widetilde V_{t+1}$ and an outer energy-average. Survivors
   inherit the *unconditional* bin transition $P^{(t)}_{b,\cdot}$ because failure is
   conditionally independent of the next bin (§3.4).
2. **The policy conditions on the current regime.** $\pi^\star_t(b,s)$ depends on $b$, which
   is optimal *and* valid because $b$ is an observed state component. This is the mechanism
   behind the empirical gains: the aircraft can pre-empt a *persisting* high-wind spell
   rather than treating each step as a fresh coin flip.

Because $x_t=(\beta_t,s_t)$ is Markov (§2–§3.4), rewards depend only on the current
transition, and the horizon is finite, backward induction on (8)–(9) is exactly optimal by
the standard finite-horizon dynamic-programming theorem. No approximation is introduced by
the augmentation itself; the only approximations are the pre-existing discretizations (SoC
grid, wind-bin edges, quadrature grid) that the i.i.d. model already had.

---

## 4. Observability and rollout consistency

For the policy to legitimately condition on $\beta_t$, the controller must *observe* it at
decision time. The rollout enforces this ordering [`environment_provider_base.py:
sample_wind_speed`; `simulation_base.py: simulate_episode_batch`]:

1. **Expose the current bin** $\beta_t$ (stored as `last_wind_bins`) to the policy, which
   picks $a = \pi^\star_t(\beta_t, s)$.
2. **Sample within the current bin** by inverse-CDF on the truncated support for the failure
   draw.
3. **Advance the bin** $\beta_{t+1}\sim\operatorname{Categorical}(P^{(t)}_{\beta_t,\cdot})$.

This "observe-then-advance" order is what makes the simulated information pattern match the
one the solver assumed. On **historical** rollouts the wind is the *real* record (block
bootstrap), not sampled — but the policy still conditions on the realized bin
$\beta_t = \beta(W_t)$ via `np.digitize`, so the same solved $V_t(b,s)$ is applied
consistently to real weather [`HistoricalBootstrapEnvironmentProvider.sample_wind_speed`].

### Within-bin sampling is exact (inverse-CDF on the truncated Weibull)

To draw $W_t\sim f_t(\cdot\mid b)$, the code draws
$U\sim\operatorname{Uniform}\big(F_t^{\mathrm W}(e_b),\,F_t^{\mathrm W}(e_{b+1})\big)$ and
inverts the Weibull CDF:

$$
W_t = \lambda_t\,[-\ln(1-U)]^{1/k_t}. \tag{10}
$$

Since the Weibull CDF is continuous and strictly increasing on $[e_b, e_{b+1})$, restricting
$U$ to $[F(e_b), F(e_{b+1}))$ and inverting produces a draw with exactly density (2) — the
standard inverse-transform result for a truncated distribution. So the rollout's within-bin
wind is an exact sample from the same $f_t(\cdot\mid b)$ used in the solver's failure integral
(7). Solver and simulator are consistent by construction.

---

## 5. Why the reward and endogenous dynamics are untouched

Validity also requires that the augmentation not silently change the *rest* of the model. It
does not:

- **Reward** depends only on $(s, a, s', t)$ — observation reward $a\,O_t$ minus penalty on
  the failure transition — and never on the bin directly. Adding $b$ to the state does not
  add or remove reward; it only changes which action is chosen and the failure probability
  through the wind it implies (`mdp_base.py: reward`, unchanged).
- **Endogenous transition** of $(\sigma, m)$ — energy accounting, mode switching, the
  broken-state absorption — is byte-for-byte the same logic (`transition_model_base.py:
  StochasticTransitionLogic`). The chain only changes *which wind speed* is fed to the
  success probability and *what the next bin is*.
- **Solar** remains i.i.d.; $p_B$ is unbinned. The extension is deliberately wind-only in
  this version.

---

## 6. Exact reduction to the published i.i.d. model

The extension is a **strict superset** of the published model, and the reduction is exact in
two independent ways:

1. **One bin ($n_b = 1$).** The single bin is $[0,\infty)$, so $\pi_t(0)=1$,
   $f_t(\cdot\mid 0)=f_t^{\mathrm W}$ (no truncation), and $P^{(t)}=[1]$. The bin sum in (8)
   collapses, $V_t(0,s)=V_t(s)$, and (8)–(9) become (1) verbatim. The implementation takes a
   **separate, unchanged code path** for $n_b=1$ — a 2-D value table and the original solver
   loop — so the published results are reproduced *bit-for-bit*, not merely in the limit
   [`solve → _solve_iid`]. The provider likewise falls back to ordinary i.i.d. Weibull
   sampling when no chain is supplied (`use_wind_chain = False`).

2. **Rank-1 chain (statistical analogue).** If every row of $P^{(t)}$ is identical, the next
   bin is independent of the current bin — the bins become i.i.d. draws from a fixed
   categorical — and the multi-bin machinery reproduces the i.i.d. outcome within Monte-Carlo
   error. `Tests/verify_wind_chain.py` asserts this ($|\Delta\text{failure}\%| < 3$ pp,
   $|\Delta\text{reward}| < 0.5$).

These two facts are the operational proof that turning the feature on cannot corrupt the
published pipeline: off by default, and exactly i.i.d. when on with one bin.

---

## 7. Behavioral correctness checks (`verify_wind_chain.py`)

The validity argument above is backed by executable checks:

- **Superset:** a rank-1 chain collapses to i.i.d. within MC error (§6.2).
- **Marginal preservation:** the within-bin truncated mixture reproduces the full Weibull
  marginal — the numerical form of identity (3).
- **The bin state actually informs the policy:** a *persistent* chain makes $V_t(b,s)$ fall
  monotonically in the wind bin (a high-wind regime is worth less), and shifts the
  failure/reward statistics — confirming the exogenous state variable is doing real work, not
  sitting inert.

---

## 8. Why *first* order is the correct model order (not a shortcut)

A fair objection to a first-order chain is "why not second order?" The answer is measured,
not assumed [`Scripts/wind_persistence_precheck.py`; memory `wind-persistence-precheck`].
Because the chain already conditions on (month, hour), the *only* thing a deeper chain could
add is genuine higher-order memory beyond that. We quantify it as the (month, hour)-stratified
**conditional mutual information** between the next and previous bin given the current bin:

$$
I\big(\beta_{t+1};\,\beta_{t-1}\,\big|\,\beta_t,\,m,\,h\big). \tag{11}
$$

This is exactly the information a second-order chain would add over the first-order,
(month,hour)-conditioned chain in use. Conditioning on $(m,h)$ removes the diurnal/seasonal
memory the chain already models, isolating true higher-order structure. On the lat30/lon−90
record:

- At the **coarse bins the policy can actually exploit** (3 quantile bins, or aircraft
  $[5,10]$ m/s edges), the previous bin adds only **~1.4–2%** of the remaining conditional
  entropy $H(\beta_{t+1}\mid\beta_t)$ beyond the current bin. First order captures
  essentially all the exploitable persistence.
- The residual CMI **grows monotonically with bin resolution**, not with deeper history — so
  the missing structure lives in wind *resolution* (representation richness), which a finer
  or continuous wind state would capture, not in Markov *order*.
- **Caveat (validity of the diagnostic itself):** the stratified plug-in CMI is
  positively biased (a separate 2nd-order table is fit inside up to 288 strata). The code
  subtracts an empirical **permutation-null bias floor** — shuffling `Prev` within each
  $(\text{curr}, m, h)$ stratum destroys real $\text{Prev}\to\text{Next}$ information while
  preserving all marginal counts and the $\text{Curr}\to\text{Next}$ structure — and reports
  the bias-corrected value, because a Miller–Madow correction would over-correct on the
  near-tridiagonal wind matrix (many structurally empty cells). Judge by bits/%-entropy, not
  by the p-value (which is ~0 at this sample size regardless).

**Conclusion:** first order is not a convenient simplification — it is the *correct* model
order for the exploitable signal. Deeper history is provably (to within the diagnostic's
bias correction) not worth the added state.

---

## 9. Implementation differences (before vs. after wind persistence)

Everything below is gated behind `wind_chain: {enabled, path}` in the run config and is
**inert when disabled** (`n_bins == 1`). The table maps each concept to what existed before
and what the branch added.

### 9.1 Data / artifacts

| Aspect | Before (i.i.d.) | After (wind persistence) |
|---|---|---|
| Wind model input | Per-stage Weibull params $(k_t,\lambda_t)$ only | Same, **plus** a fitted chain artifact: `bin_edges` + `transition_by_month_hour` tensor $(13,24,n_b,n_b)$ |
| Fitting code | — | `create_weather_distributions.py: fit_wind_transition_chain` / `build_wind_chain_artifact`; CLI `Scripts/build_windchain.py` |
| Artifact storage | — | Companion pickle (gitignored; regenerated from `HISTORICAL_DATA`) |

### 9.2 `environment_provider_base.py` — `StochasticWindSolarEnvironmentProvider`

- **Constructor** gains optional `wind_bin_edges`, `wind_transition`. When
  `wind_transition is None` → `use_wind_chain = False`, `n_wind_bins = 1`, and behavior is the
  **original i.i.d. Weibull sampling**.
- **New per-lane chain state:** `_wind_bins` (current bin per Monte-Carlo lane),
  `last_wind_bins` (the bins exposed to the policy this step), initialized lazily from the
  stage-0 bin masses (`_init_wind_bins`).
- **New methods:** `_sample_within_bin` (inverse-CDF truncated Weibull, eq. 10),
  `_advance_bins` (sample next bin from `P[bins]`), `get_wind_transition(t)`,
  `_weibull_cdf`.
- **`sample_wind_speed`** changed from a one-line i.i.d. draw to observe-then-advance: sample
  within current bin → expose `last_wind_bins` → advance the chain. The i.i.d. branch is
  untouched.
- **`reset`** now also clears `_wind_bins`/`last_wind_bins`.
- **New provider** `HistoricalBootstrapEnvironmentProvider` accepts the same
  `wind_bin_edges`/`wind_transition` and, on real weather, digitizes realized wind into bins
  so chain-solved policies can be evaluated on the historical record (§4).

### 9.3 `backward_induction_base.py` — `mdpAnalyticalBackwardSolver`

| Element | Before | After |
|---|---|---|
| Value table shape | `(num_states, horizon)` | `(n_bins, num_states, horizon)` when chain on; **2-D preserved** when `n_bins==1` (`_initialize_future_value_table`) |
| Solve entry point | single loop | `solve()` dispatches to `_solve_iid()` (unchanged) or `_solve_chain()` |
| Per-stage solve | value for both actions over all states | `_solve_chain` adds an **outer loop over bins**; per bin uses the stage matrix row `P[b]` |
| Failure integral | full-Weibull `_mechanical_failure_probability` | per-bin `_mechanical_failure_probability_bin` on the truncated grid (`_get_wind_grid_bin`) |
| Future value | direct SoC lookup | **bin contraction** $\widetilde V = P_{b,\cdot}\!\cdot V_{t+1}$ (`_vnext_eff`), then energy-bin survival mass (`_value_batch_bin`) |
| Saved table | 2-D `.npy` | 3-D when chain on; 2-D saved for the i.i.d. path (back-compatible) |

Per project preference, the solver **always recomputes** the value table and never
auto-loads a pre-solved `.npy` (memory `runtime-optimization-prefs`).

### 9.4 `simulation_base.py` — optimal policy

- `choose_action_batch` gains a `cur_bins` argument. When the chain is active it fetches the
  stage matrix `P = get_wind_transition(t)` and evaluates the future value **through the
  current bin** via `mdp_solver.value_function_batch(..., cur_bins=cur_bins, P=P)`, which does
  the `einsum('nb,bn->n')` bin-weighted lookup. When `cur_bins is None` it is the original
  i.i.d. SoC lookup.
- `simulate_episode_batch` reads `env_provider.last_wind_bins` for the active lanes and passes
  it as `cur_bins` **only** when `use_wind_chain` is set.
- **Threshold policies are unchanged** — they never receive or use the bin (`cur_bins` is
  accepted and ignored). This keeps the baseline heuristics identical for the comparison.

### 9.5 `run_sim.py` — config and wiring

- `_load_wind_chain(config, location)` reads `wind_chain: {enabled, path}` (default off) and
  loads the artifact; path defaults to `<data>_windchain.pkl`.
- `EnvironmentLoader._build_per_stage_transition` maps each mission stage's `(month, hour)` to
  its fitted matrix, producing the per-stage `(horizon, n_b, n_b)` transition stack used by
  both solver and provider.
- For historical evaluation, the bootstrap provider is handed the *same* `wind_bin_edges` and
  `wind_transition` as the distributional solve-side provider, so the 3-D value table is
  indexed consistently on real weather.

### 9.6 Tests / diagnostics added

- `Tests/verify_wind_chain.py` — superset (rank-1 → i.i.d.), marginal preservation, and
  monotonic-value-in-bin checks (§7).
- `Scripts/wind_persistence_precheck.py` (+ `wind_persistence_plots.py`) — the higher-order
  CMI diagnostic with permutation-null bias correction (§8).

---

## 10. Summary of the validity argument

| Question | Answer | Where |
|---|---|---|
| Why is the extension *needed*? | Correlated wind makes the endogenous state $s$ non-Markov; (1) is no longer the right Bellman equation. | §1 |
| Why is augmenting with $\beta_t$ *valid*? | $\beta_t$ is a sufficient statistic for a first-order wind process and is observed at decision time, so $x=(\beta,s)$ is Markov and the policy may condition on $\beta$ with no loss. | §2, §3.4 |
| Does it change the modeled weather? | No — the bin-mixture identity (3) preserves the marginal climatology exactly; only temporal correlation is added. | §3.2 |
| Is backward induction still exactly optimal? | Yes — $x$ is a finite-horizon MDP state; (8)–(9) is the standard DP recursion; the double-sum factorization is exact under the conditional-independence assumption (5). | §3.4, §3.6 |
| Is the simulator consistent with the solver? | Yes — observe-then-advance ordering + exact inverse-CDF within-bin sampling (10). | §4 |
| Can it corrupt the published results? | No — off by default; $n_b=1$ is a separate code path reproducing i.i.d. bit-for-bit; rank-1 chain reproduces i.i.d. within MC error. | §6, §7 |
| Is first order the right complexity? | Yes — measured higher-order memory is ~1–2% of remaining entropy at exploitable bins; the residual is in resolution, not order. | §8 |

*Implementation: `BaseClasses/{backward_induction_base, environment_provider_base, run_sim,
simulation_base, transition_model_base}.py`, `Scripts/create_weather_distributions.py`.
Verification: `Tests/verify_wind_chain.py`. Diagnostic:
`Scripts/wind_persistence_precheck.py`. Companion math reference:
[`wind_persistence_math.md`](wind_persistence_math.md); empirical briefing:
[`wind_persistence_briefing.md`](wind_persistence_briefing.md).*
