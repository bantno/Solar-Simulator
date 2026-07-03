# State Augmentation for Temporally Correlated Wind in a Finite-Horizon Endurance MDP

*Formulation and validity of the Markov-modulated wind extension.*

---

## 1. Introduction

The published endurance model draws wind independently at every control step. The extension considered here replaces that i.i.d. draw with a first-order, time-conditioned Markov chain over wind-speed bins and adds the current bin to the state. This document establishes the following claims:

1. Under temporally correlated wind, the original endogenous state ceases to be a Markov state, so the original Bellman recursion is no longer the correct optimality equation (§3).
2. Augmenting the state with the current wind bin — a sufficient statistic for a first-order wind process — restores the Markov property, and the augmented model is a finite-horizon MDP on which backward induction is exactly optimal (§4–§5).
3. The construction preserves the marginal wind distribution (climatology) identically at every stage; only the temporal correlation of wind is modified (§5.2).
4. The simulated information pattern matches the one assumed by the solver, and within-bin sampling is exact (§6).
5. The published i.i.d. model is recovered exactly as the one-bin special case, and asymptotically as the rank-one-chain special case (§7).
6. First order is the appropriate model order: a conditional-mutual-information diagnostic shows that deeper history contributes a negligible fraction of the remaining conditional entropy at the bin resolutions the policy can exploit (§8).

## 2. The Baseline Model

The endurance problem is a finite-horizon MDP over stages $t = 0, \dots, T-1$ with the following elements.

- **Endogenous state** $s = (\sigma, m)$: battery state of charge $\sigma$ on a discrete energy grid and operating mode $m \in \{\text{moored}, \text{flying}, \text{broken}\}$.
- **Action** $a \in \{0, 1\}$ (remain moored / fly).
- **Exogenous weather** at stage $t$: wind speed $W_t$ and solar irradiance $G_t$.
- **Reward** $r(s, a, s', t)$, consisting of an observation reward earned while flying and a penalty $c_f$ on the failure transition; **discount** $\gamma = 1$.

In the baseline model the weather is independent across stages, $W_t \sim \mathrm{Weibull}(k_t, \lambda_t)$ with stage-dependent (month- and hour-dependent) parameters, and the value function depends only on $(s, t)$:

$$
V_t(s) = \max_{a} \; \mathbb{E}_{W_t, G_t}\!\big[\, r + \gamma\, V_{t+1}(s') \,\big],
\qquad V_T(s) = 0. \tag{1}
$$

Here $V_t(s)$ is the optimal expected reward-to-go from state $s$ at stage $t$, $s'$ is the (random) successor state induced by the weather draw $(W_t, G_t)$ and the action $a$, $r = r(s, a, s', t)$ is the stage reward, $\gamma$ is the discount factor, and the terminal condition sets the value at the horizon $T$ to zero.

Equation (1) is valid because $W_t$ is independent of all information prior to stage $t$: the distribution of $s_{t+1}$ given $(s_t, a_t)$ does not depend on the past, so $s_t$ is a Markov state and (1) is the correct Bellman equation.

## 3. Loss of the Markov Property under Correlated Wind

Real wind is autocorrelated. Suppose the endogenous state $s = (\sigma, m)$ is retained but the true wind process is temporally correlated. The distribution of $W_t$ then depends on $W_{t-1}, W_{t-2}, \dots$, and since $W_t$ drives both the failure event and the energy transition, the transition kernel of $s$ satisfies

$$
\Pr\big(s_{t+1} \mid s_t, a_t, \underbrace{s_{t-1}, a_{t-1}, \dots}_{\text{history}}\big)
\;\ne\; \Pr\big(s_{t+1} \mid s_t, a_t\big).
$$

Here $s_t$ and $a_t$ denote the endogenous state and action at stage $t$, and the left-hand conditioning set includes the trajectory history, which carries information about past wind and hence about $W_t$.

The endogenous state $s$ is therefore no longer a Markov state, and (1) is not the correct optimality equation for correlated weather: it implicitly assumes that next-step wind is independent of current wind. A policy solved from (1) is optimal for the i.i.d. wind process, not for the correlated one.

The baseline model is not internally inconsistent — it is correct under its own independence assumption. The point is that on correlated wind, the state must be enlarged for the controlled process to remain Markov.

## 4. State Augmentation

The standard remedy for a non-Markov controlled process is to augment the state with a sufficient statistic of the exogenous process's memory. For a first-order wind model that statistic is the current wind regime. Discretize wind into $n_b$ bins and let $\beta_t \in \{0, \dots, n_b - 1\}$ denote the current bin index. Define the **augmented state**

$$
x_t = (\beta_t, s_t),
$$

where $\beta_t$ is the wind-bin index observed at stage $t$ and $s_t$ is the endogenous state of the baseline model (state of charge $\sigma$ and operating mode $m$).

Validity rests on two facts, made precise in §5:

1. **Sufficiency.** Under a first-order bin chain, given $\beta_t$ the future wind — and hence the future of $s$ — is conditionally independent of all history. Consequently $x_t$ is a Markov state.
2. **Observability.** The controller observes $\beta_t$ before acting (§6). An observed state component may be conditioned on by an optimal policy without loss of optimality; a Markov policy on the augmented state is well defined.

The wind bin is an exogenous state variable: the controller observes it and reacts to it but cannot influence it. The construction is the standard device of adjoining a weather or regime variable to a stochastic control problem.

## 5. Mathematical Formulation

### 5.1 Wind discretization and the conditional wind distribution

Partition the wind-speed axis into $n_b$ contiguous bins with edges $0 = e_0 < e_1 < \dots < e_{n_b} = \infty$, bin $b = [e_b, e_{b+1})$, and bin map $\beta(w) = \#\{i : e_i \le w\} - 1$. Interior edges are chosen either as equal-occupancy quantiles of the historical record or as physically motivated thresholds.

Persistence is carried by the bin, while the wind entering the failure computation remains continuous. Conditioned on bin $b$, stage-$t$ wind follows the stage-$t$ Weibull density truncated and renormalized to $[e_b, e_{b+1})$:

$$
f_t(w \mid b) = \frac{f_t^{\mathrm{W}}(w)}{F_t^{\mathrm{W}}(e_{b+1}) - F_t^{\mathrm{W}}(e_b)}\,
\mathbb{1}\{e_b \le w < e_{b+1}\},
\qquad
\pi_t(b) = F_t^{\mathrm{W}}(e_{b+1}) - F_t^{\mathrm{W}}(e_b), \tag{2}
$$

where $f_t^{\mathrm{W}}$ and $F_t^{\mathrm{W}}(e) = 1 - \exp\!\big[-(e/\lambda_t)^{k_t}\big]$ are the stage-$t$ Weibull density and CDF (with shape $k_t$ and scale $\lambda_t$), $\mathbb{1}\{\cdot\}$ is the indicator function, $f_t(w \mid b)$ is the within-bin wind density, and $\pi_t(b)$ is the bin occupancy mass — the probability that stage-$t$ wind falls in bin $b$.

### 5.2 Marginal preservation

By the law of total probability, the bin-weighted mixture of the truncated densities recovers the original Weibull density exactly: for every stage $t$ and every $w \ge 0$,

$$
\sum_{b=0}^{n_b - 1} \pi_t(b)\, f_t(w \mid b) \;=\; f_t^{\mathrm{W}}(w), \tag{3}
$$

where $\pi_t(b)$ and $f_t(w \mid b)$ are the bin mass and within-bin density of (2), $f_t^{\mathrm{W}}$ is the stage-$t$ Weibull density, and the sum runs over all $n_b$ bins. The identity holds since each term equals $\pi_t(b) \cdot \big[f_t^{\mathrm{W}}(w) / \pi_t(b)\big] \cdot \mathbb{1}\{w \in [e_b, e_{b+1})\} = f_t^{\mathrm{W}}(w)\, \mathbb{1}\{w \in [e_b, e_{b+1})\}$, and the indicators sum to one over the disjoint bins tiling $[0, \infty)$.

Identity (3) guarantees that binning and truncation leave the modeled wind distribution unchanged at every stage: the extension modifies only the temporal correlation of wind, never its marginal climatology.

### 5.3 The time-conditioned first-order chain

The bin evolves as a first-order Markov chain whose transition matrix is conditioned on the calendar (month $m_t$, hour $h_t$) of the stage:

$$
\Pr(\beta_{t+1} = b' \mid \beta_t = b) = P^{(m_t, h_t)}_{b, b'} \;\equiv\; P^{(t)}_{b, b'}. \tag{4}
$$

Here $b$ and $b'$ index the current and next wind bin, $P^{(m, h)}$ is the $n_b \times n_b$ stochastic (row-normalized) transition matrix for calendar stratum $(m, h)$, and $P^{(t)}$ abbreviates the matrix in force at stage $t$, namely $P^{(m_t, h_t)}$.

The transition matrices are estimated from the historical record by stratified transition counting with row normalization,

$$
P^{(m,h)}_{b,b'} =
\begin{cases}
N^{(m,h)}_{b,b'} \Big/ \displaystyle\sum_{b''} N^{(m,h)}_{b,b''}, & \text{if the row sum is positive},\\[2ex]
1/n_b, & \text{otherwise},
\end{cases}
$$

where $N^{(m,h)}_{b,b'}$ counts observed transitions $b \to b'$ within stratum $(m, h)$, $b''$ ranges over all bins in the row-sum normalization, and the uniform value $1/n_b$ is the fallback for calendar strata with no observed transitions out of bin $b$.

Because $(m_t, h_t)$ is a deterministic function of the stage index $t$, conditioning the transition on it does not compromise the Markov property of $x_t$: the chain is time-inhomogeneous, not history-dependent. Absorbing the diurnal and seasonal drift into the chain also matters for the model-order diagnostic of §8, where it isolates genuine higher-order memory from calendar effects.

### 5.4 The augmented transition kernel

Consider the one-step dynamics of $x_t = (\beta_t, s_t)$ under action $a$. Two exogenous draws occur: the within-step wind $W_t \sim f_t(\cdot \mid \beta_t)$, which determines the success/failure event and the energy update of $s$, and the next bin $\beta_{t+1} \sim P^{(t)}_{\beta_t, \cdot}$. Solar irradiance $G_t$ is drawn independently (i.i.d. across stages). The kernel is

$$
\Pr\big(\beta_{t+1} = b',\, s_{t+1} = s' \,\big|\, \beta_t = b,\, s_t = s,\, a\big)
\;=\;
\underbrace{P^{(t)}_{b, b'}}_{\text{regime transition}}
\cdot
\underbrace{\Pr\big(s_{t+1} = s' \mid s, a, \beta_t = b, t\big)}_{\text{endogenous transition}}. \tag{5}
$$

Here $b$ and $s$ are the current wind bin and endogenous state, $b'$ and $s'$ their successors, $a$ the chosen action, and $P^{(t)}$ the stage-$t$ transition matrix of (4); the second factor is the transition law of the battery/mode state given that the within-step wind is distributed as $f_t(\cdot \mid b)$.

The factorization (5) encodes one explicit modeling assumption of conditional independence: given the current bin $\beta_t = b$, the next bin $\beta_{t+1}$ is drawn from $P^{(t)}_{b, \cdot}$ independently of the realized within-step wind $W_t$ and of the success/failure outcome of the step.

This is the natural first-order assumption: the continuous within-bin position is a nuisance variable that keeps the failure integral sharp (§5.5) but does not feed back into the regime dynamics. The assumption is consistent with the marginal identity (3), and it is precisely what allows the future-value term in the Bellman recursion to separate into a regime average and an energy average (§5.6).

### 5.5 Failure probability

A step fails if the battery depletes (a solar/energy event of probability $p_B$) or a mechanical wind event occurs (probability $p_M$); the two causes combine independently:

$$
p_{\mathrm{fail}}(s, a, t, b) = p_B(s, a, t) + \big(1 - p_B(s, a, t)\big)\, p_M(s, a, t, b). \tag{6}
$$

Here $p_{\mathrm{fail}}$ is the total one-step failure probability, $p_B(s, a, t)$ the battery-depletion probability (independent of the wind bin), and $p_M(s, a, t, b)$ the mechanical failure probability given that stage-$t$ wind lies in bin $b$.

- **Solar/energy term.** $p_B$ is unchanged from the baseline model and does not depend on the bin (solar remains i.i.d.). With energy deficit $\Delta = E_{\mathrm{req}}(s, a) - E(\sigma)$ and normalizer $G_{\max} = \max(G^{\mathrm{cs}}_t, 10)$,
  $$
  p_B = I_u(\alpha_t, \beta_t), \qquad u = \operatorname{clip}(\Delta / G_{\max},\, 0,\, 1),
  $$
  where $E_{\mathrm{req}}(s, a)$ is the energy the chosen action requires, $E(\sigma)$ the energy stored at state of charge $\sigma$, $G^{\mathrm{cs}}_t$ the stage-$t$ clear-sky irradiance, $I_u$ the regularized incomplete beta function evaluated at the clipped normalized deficit $u$, and $\alpha_t, \beta_t$ the stage-fitted shape parameters of the solar distribution (distinct from the bin index $\beta_t$).
- **Mechanical/wind term.** $p_M$ is the expected one-step mechanical failure probability given that wind lies in bin $b$, integrated against the truncated density (2):
  $$
  p_M(s, a, t, b) = \int_{e_b}^{e_{b+1}} \big(1 - q(w, a, s)\big)\, f_t(w \mid b)\, \mathrm{d}w, \tag{7}
  $$
  where $q(w, a, s)$ is the mechanical success probability at wind speed $w$ (a logistic curve), $e_b$ and $e_{b+1}$ are the edges of bin $b$, and $f_t(w \mid b)$ is the truncated within-bin density of (2).

Substituting (7) and applying the marginal identity (3), the bin-mass-weighted average of the binned mechanical failure probability recovers the baseline full-Weibull failure probability:

$$
\sum_{b} \pi_t(b)\, p_M(s, a, t, b)
= \int_0^\infty \big(1 - q(w, a, s)\big)\, f_t^{\mathrm{W}}(w)\, \mathrm{d}w
= p_M^{\mathrm{iid}}(s, a, t),
$$

where $p_M^{\mathrm{iid}}$ denotes the mechanical failure probability of the baseline i.i.d. model, in which wind is integrated against the full stage-$t$ Weibull density $f_t^{\mathrm{W}}$.

The binned failure model is thus not a different failure physics: it is the same failure integral, partitioned by regime. With $n_b = 1$ it coincides with the baseline $p_M$.

### 5.6 Bellman recursion on the augmented state

The value function acquires a bin dimension, $V_t(b, s)$. Using the kernel factorization (5), the state–action value is

$$
Q_t(b, s, a) =
\Big[\, a\, O_t - c_f\, p_{\mathrm{fail}}(s, a, t, b) \,\Big]
+ \gamma\, \big(1 - p_{\mathrm{fail}}\big)
\sum_{e} \Delta P^{(t)}_{e}(s, a)\;
\underbrace{\sum_{b' = 0}^{n_b - 1} P^{(t)}_{b, b'}\, V_{t+1}\big(b', s'_e\big)}_{\textstyle \widetilde V_{t+1}(b,\, s'_e)}, \tag{8}
$$

where $Q_t(b, s, a)$ is the expected reward-to-go of taking action $a$ in wind bin $b$ and endogenous state $s$; $O_t$ is the stage observation reward (earned only when flying, $a = 1$); $c_f$ is the failure penalty and $p_{\mathrm{fail}}(s, a, t, b)$ the total failure probability of (6); $\gamma$ is the discount factor; the outer sum runs over successor states $s'_e$ indexed by energy bin $e$, with $\Delta P^{(t)}_e(s, a)$ the solar-driven probability of a surviving trajectory landing in energy bin $e$; and the inner sum — the effective next-stage value $\widetilde V_{t+1}(b, s'_e)$ — averages the next-stage value $V_{t+1}(b', s'_e)$ over the next bin $b'$ using the transition row $P^{(t)}_{b, \cdot}$. The optimality and terminal conditions are

$$
V_t(b, s) = \max_a Q_t(b, s, a),
\qquad V_T(b, s) = 0,
\qquad
\pi^\star_t(b, s) = \arg\max_a Q_t(b, s, a). \tag{9}
$$

Here $V_t(b, s)$ is the optimal value on the augmented state, $V_T(b, s) = 0$ is the terminal condition at the horizon, and $\pi^\star_t(b, s)$ is the optimal policy — the maximizing action for each stage, wind bin, and endogenous state.

Two structural points underlie (8):

1. **The double sum is exact under the conditional-independence assumption of §5.4.** The next energy state $s'_e$ (driven by i.i.d. solar) and the next bin $b'$ (driven by $P^{(t)}$) are conditionally independent given $(b, s, a)$, so the joint expectation of $V_{t+1}$ separates into the inner regime average $\widetilde V_{t+1}$ and the outer energy average. Surviving trajectories inherit the unconditional regime transition $P^{(t)}_{b, \cdot}$ because failure is conditionally independent of the next bin.
2. **The policy conditions on the current regime.** $\pi^\star_t(b, s)$ depends on $b$, which is optimal and legitimate because $b$ is an observed state component. This is the mechanism behind the empirical gains: the controller can pre-empt a persisting high-wind spell rather than treating each step as an independent draw.

Since $x_t = (\beta_t, s_t)$ is a Markov state (§4, §5.4), rewards depend only on the current transition, and the horizon is finite, backward induction on (8)–(9) yields the exactly optimal policy for the augmented model, by the standard finite-horizon dynamic-programming argument.

The augmentation itself introduces no approximation. The only approximations present are the discretizations (state-of-charge grid, wind-bin edges, quadrature grid) already present in the baseline model.

## 6. Observability and Simulation Consistency

For the policy to condition legitimately on $\beta_t$, the controller must observe it at decision time. The simulation enforces the following ordering at each stage:

1. **Observe.** The current bin $\beta_t$ is exposed to the policy, which selects $a = \pi^\star_t(\beta_t, s)$.
2. **Sample.** The within-step wind is drawn from the truncated distribution $f_t(\cdot \mid \beta_t)$ for the failure draw.
3. **Advance.** The next bin is drawn, $\beta_{t+1} \sim \operatorname{Categorical}\big(P^{(t)}_{\beta_t, \cdot}\big)$.

This observe-then-advance ordering makes the simulated information pattern identical to the one assumed by the solver. On historical evaluations the wind is the real record rather than a sampled trajectory; the policy then conditions on the realized bin $\beta_t = \beta(W_t)$, so the same solved value function is applied consistently to real weather.

Within-bin sampling is exact. To draw $W_t \sim f_t(\cdot \mid b)$, let $U \sim \operatorname{Uniform}\big(F_t^{\mathrm{W}}(e_b),\, F_t^{\mathrm{W}}(e_{b+1})\big)$ and set

$$
W_t = \lambda_t\, \big[-\ln(1 - U)\big]^{1/k_t}. \tag{10}
$$

Here $U$ is a uniform draw restricted to the CDF range of bin $b$, $k_t$ and $\lambda_t$ are the stage-$t$ Weibull shape and scale, and (10) is the Weibull inverse CDF applied to $U$.

Because the Weibull CDF is continuous and strictly increasing on $[e_b, e_{b+1})$, restricting $U$ to $\big[F_t^{\mathrm{W}}(e_b), F_t^{\mathrm{W}}(e_{b+1})\big)$ and inverting is the standard inverse-transform construction for a truncated distribution, so $W_t$ has density $f_t(\cdot \mid b)$ exactly.

The simulated within-bin wind therefore follows exactly the distribution against which the solver integrates in (7): solver and simulator are consistent by construction.

## 7. Reduction to the Baseline Model

The extension is a strict superset of the baseline model, with two independent reduction results.

1. **One-bin reduction.** With $n_b = 1$, the single bin is $[0, \infty)$, so $\pi_t(0) = 1$, $f_t(\cdot \mid 0) = f_t^{\mathrm{W}}$, and $P^{(t)} = [1]$. The bin sum in (8) collapses, $V_t(0, s) = V_t(s)$, and (8)–(9) reduce to (1) verbatim.

2. **Rank-one reduction.** If every row of $P^{(t)}$ is identical, then $\beta_{t+1}$ is independent of $\beta_t$; the bins are i.i.d. categorical draws, and by the marginal identity (3) the induced wind process is the i.i.d. Weibull process of the baseline model. The multi-bin recursion then reproduces the baseline solution.

Together these establish that enabling the extension cannot alter the baseline results except through genuine temporal correlation: with one bin the baseline model is recovered exactly, and with a correlation-free (rank-one) chain it is recovered in distribution.

## 8. Selection of the Model Order

A natural question is whether a first-order chain suffices, or whether a second-order chain is warranted. Because the chain already conditions on $(m, h)$, the only contribution a deeper chain could make is genuine higher-order memory beyond the calendar conditioning. This is quantified by the $(m, h)$-stratified **conditional mutual information** between the next and the previous bin given the current bin:

$$
I\big(\beta_{t+1};\, \beta_{t-1} \,\big|\, \beta_t,\, m,\, h\big). \tag{11}
$$

Here $\beta_{t-1}$, $\beta_t$, and $\beta_{t+1}$ are the previous, current, and next wind bins, $(m, h)$ is the calendar stratum (month, hour), and $I(\cdot\,;\cdot \mid \cdot)$ denotes conditional mutual information — the expected reduction in uncertainty about $\beta_{t+1}$ from additionally observing $\beta_{t-1}$, once $\beta_t$, $m$, and $h$ are known.

Quantity (11) is exactly the information a second-order chain would add over the first-order, calendar-conditioned chain. Conditioning on $(m, h)$ removes the diurnal and seasonal memory already modeled, isolating true higher-order structure. Evaluated on the historical record (lat 30°, lon −90°):

- At the coarse bin resolutions the policy can exploit (three quantile bins, or physical thresholds at 5 and 10 m/s), the previous bin contributes only about **1.4–2%** of the remaining conditional entropy $H(\beta_{t+1} \mid \beta_t)$ beyond the current bin. First order captures essentially all of the exploitable persistence.
- The residual conditional mutual information grows monotonically with bin resolution, not with history depth. The unmodeled structure therefore resides in the *resolution* of the wind representation — which a finer or continuous wind state would capture — rather than in the Markov *order*.
- The plug-in estimator of (11) is positively biased, as a separate second-order table is fit within up to 288 strata. The reported values are bias-corrected by subtracting an empirical permutation-null floor: shuffling $\beta_{t-1}$ within each $(\beta_t, m, h)$ stratum destroys genuine $\beta_{t-1} \to \beta_{t+1}$ information while preserving all marginal counts and the $\beta_t \to \beta_{t+1}$ structure. The appropriate summary is the bias-corrected information in bits (or as a fraction of conditional entropy); at this sample size the associated $p$-value is uninformative.

First order is therefore the appropriate model order for the exploitable signal: the measured higher-order memory is negligible at the relevant bin resolutions, and the residual structure is attributable to representation resolution rather than history.

## 9. Summary

| Question | Answer |
|---|---|
| Why is the extension needed? | Correlated wind makes the endogenous state non-Markov; (1) is no longer the correct Bellman equation (§3). |
| Why is the augmentation valid? | $\beta_t$ is a sufficient statistic for a first-order wind process and is observed at decision time, so $(\beta, s)$ is Markov and the policy may condition on $\beta$ without loss (§4, §5.4). |
| Is the modeled climatology changed? | No — the mixture identity (3) preserves the marginal wind distribution exactly; only temporal correlation is added (§5.2). |
| Is backward induction still exactly optimal? | Yes — the augmented state is a finite-horizon MDP state; (8)–(9) is the standard recursion, exact under the conditional-independence assumption of §5.4 (§5.6). |
| Is the simulator consistent with the solver? | Yes — observe-then-advance ordering, and exact inverse-CDF within-bin sampling (§6). |
| Does the extension subsume the baseline? | Yes — $n_b = 1$ reduces to the baseline exactly; a rank-one chain reduces to it in distribution (§7). |
| Is first order the right complexity? | Yes — measured higher-order memory is ~1–2% of the remaining conditional entropy at exploitable bin resolutions; the residual lies in resolution, not order (§8). |

## Appendix. Table of Symbols

**Decision process.**

| Symbol | Description |
|---|---|
| $t$, $T$ | Stage index and horizon; one stage per 15-minute control step, $t = 0, \dots, T-1$. |
| $s = (\sigma, m)$ | Endogenous state: battery state of charge $\sigma$ (discrete energy grid) and operating mode $m \in \{\text{moored}, \text{flying}, \text{broken}\}$. |
| $a \in \{0, 1\}$ | Action: remain moored ($0$) or fly ($1$). |
| $r(s, a, s', t)$ | Stage reward: observation reward earned while flying, minus the penalty on the failure transition. |
| $O_t$ | Observation reward accrued at stage $t$ when flying. |
| $c_f$ | Failure penalty, charged on the transition into the broken state. |
| $\gamma$ | Discount factor; $\gamma = 1$ throughout. |
| $x_t = (\beta_t, s_t)$ | Augmented state: current wind bin adjoined to the endogenous state. |

**Weather processes.**

| Symbol | Description |
|---|---|
| $W_t$ | Wind speed at stage $t$ (continuous, m/s). |
| $G_t$ | Solar irradiance at stage $t$; i.i.d. across stages in both models. |
| $k_t$, $\lambda_t$ | Shape and scale of the stage-$t$ Weibull wind distribution (month- and hour-dependent). |
| $f_t^{\mathrm{W}}$, $F_t^{\mathrm{W}}$ | Stage-$t$ Weibull density and CDF, $F_t^{\mathrm{W}}(e) = 1 - \exp[-(e/\lambda_t)^{k_t}]$. |
| $m_t$, $h_t$ | Calendar month and hour of stage $t$ (deterministic functions of $t$). |
| $G^{\mathrm{cs}}_t$ | Clear-sky irradiance at stage $t$, used to normalize the energy deficit. |

**Wind discretization and regime chain.**

| Symbol | Description |
|---|---|
| $n_b$ | Number of wind-speed bins. |
| $e_b$ | Bin edges, $0 = e_0 < e_1 < \dots < e_{n_b} = \infty$; bin $b$ is $[e_b, e_{b+1})$. |
| $\beta_t$; $\beta(w)$ | Wind-bin index at stage $t$; the bin map assigning a speed $w$ to its bin. |
| $b$, $b'$ | Generic current and next bin indices. |
| $f_t(w \mid b)$ | Within-bin wind density: the stage-$t$ Weibull truncated and renormalized to $[e_b, e_{b+1})$, eq. (2). |
| $\pi_t(b)$ | Bin occupancy mass, $\pi_t(b) = F_t^{\mathrm{W}}(e_{b+1}) - F_t^{\mathrm{W}}(e_b)$. |
| $P^{(m,h)}$; $P^{(t)}$ | Bin transition matrix for calendar stratum $(m, h)$; the matrix in force at stage $t$, eq. (4). |
| $N^{(m,h)}_{b,b'}$ | Count of observed transitions $b \to b'$ within stratum $(m, h)$ in the historical record. |
| $U$ | Uniform random variable used in the inverse-CDF within-bin draw, eq. (10). |
| $\mathbb{1}\{\cdot\}$ | Indicator function. |

**Failure model.**

| Symbol | Description |
|---|---|
| $p_B(s, a, t)$ | Solar/energy (battery-depletion) failure probability; independent of the wind bin. |
| $p_M(s, a, t, b)$ | Mechanical/wind failure probability given wind in bin $b$, eq. (7). |
| $p_{\mathrm{fail}}(s, a, t, b)$ | Total one-step failure probability, combining $p_B$ and $p_M$ as independent causes, eq. (6). |
| $q(w, a, s)$ | Mechanical success probability at wind speed $w$ (logistic in $w$). |
| $\Delta$ | Energy deficit, $\Delta = E_{\mathrm{req}}(s, a) - E(\sigma)$. |
| $E_{\mathrm{req}}(s, a)$; $E(\sigma)$ | Energy required by the chosen action; energy stored at state of charge $\sigma$. |
| $G_{\max}$ | Deficit normalizer, $G_{\max} = \max(G^{\mathrm{cs}}_t, 10)$. |
| $I_u(\alpha_t, \beta_t)$ | Regularized incomplete beta function evaluated at $u = \operatorname{clip}(\Delta / G_{\max}, 0, 1)$; here $\alpha_t, \beta_t$ are the stage-fitted shape parameters of the solar distribution (distinct from the bin index $\beta_t$). |

**Value functions and policy.**

| Symbol | Description |
|---|---|
| $V_t(s)$ | Baseline value function (i.i.d. model), eq. (1). |
| $V_t(b, s)$ | Augmented value function, indexed by wind bin and endogenous state, eq. (9). |
| $Q_t(b, s, a)$ | State–action value on the augmented state, eq. (8). |
| $\pi^\star_t(b, s)$ | Optimal policy on the augmented state. |
| $\widetilde V_{t+1}(b, s')$ | Effective next-stage value: the regime average $\sum_{b'} P^{(t)}_{b,b'} V_{t+1}(b', s')$. |
| $s'_e$; $\Delta P^{(t)}_e(s, a)$ | Successor state in energy bin $e$; the solar-driven probability of a surviving trajectory landing there. |

**Model-order diagnostic.**

| Symbol | Description |
|---|---|
| $I(\beta_{t+1}; \beta_{t-1} \mid \beta_t, m, h)$ | Conditional mutual information between next and previous bin given the current bin, stratified by calendar $(m, h)$, eq. (11). |
| $H(\beta_{t+1} \mid \beta_t)$ | Conditional entropy of the next bin given the current bin; the normalizer for reporting (11) as a percentage. |
