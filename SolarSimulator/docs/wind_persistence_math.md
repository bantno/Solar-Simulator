# Wind-Persistence Extension — Mathematical Formulation

*Companion to `wind_persistence_briefing.md`. This document specifies the math of the
Markov-modulated wind model and how it augments the published i.i.d. MDP. Section numbers in
brackets point to the implementing code, e.g. `[backward_induction_base.py]`.*

---

## 1. Notation and the i.i.d. baseline (published model)

The aircraft-endurance problem is a finite-horizon Markov decision process solved by
backward induction over stages $t = 0, 1, \dots, T-1$ (each stage = one 15-min control
step).

- **Endogenous state** $s = (\sigma, m)$: battery state-of-charge $\sigma$ (discretized into
  $n_\sigma$ energy levels) and operating mode $m \in \{\text{flying}, \text{moored},
  \text{broken}\}$. Write the flattened index set $\mathcal{S}$, $|\mathcal{S}| = n_\sigma
  \cdot n_m$.
- **Action** $a \in \mathcal{A} = \{0, 1\}$ (e.g. moor / fly).
- **Exogenous weather** at stage $t$: wind speed $W_t$ and solar irradiance $G_t$.
- **Reward** $r(s, a, s', t)$, with a failure penalty $c_f$ applied on the absorbing failure
  transition; **discount** $\gamma$.

In the published model the weather is **independent across stages**: $W_t \sim
\text{Weibull}(k_t, \lambda_t)$ with stage-dependent (month/hour) shape $k_t$ and scale
$\lambda_t$, and $G_t$ from its fitted distribution, each drawn i.i.d. The value function
depends only on $(s, t)$:

$$
V_t(s) \;=\; \max_{a \in \mathcal{A}} \; \mathbb{E}_{W_t,\,G_t}\!\Big[\, r + \gamma\, V_{t+1}(s') \,\Big],
\qquad V_T(s) = 0 .
$$

The expectation over $W_t$ is taken against the **full** stage-$t$ Weibull. The
wind-persistence extension changes only *how the weather evolves and what the policy
conditions on* — everything above is recovered exactly as a special case (§7).

---

## 2. Wind discretization

Partition wind speed into $n_b$ contiguous bins with edges

$$
0 = e_0 < e_1 < \dots < e_{n_b-1} < e_{n_b} = \infty,
\qquad
\text{bin } b = [\,e_b,\, e_{b+1}\,),\quad b = 0,\dots,n_b-1 .
$$

Two ways to choose the interior edges $\{e_1,\dots,e_{n_b-1}\}$
$[\,\texttt{create\_weather\_distributions.py}\,]$:

- **Equal-occupancy quantiles** (default): $e_i = Q_W\!\big(i / n_b\big)$, the empirical
  quantiles of the historical wind record. Each bin holds $\approx 1/n_b$ of the mass.
- **User-specified thresholds** (e.g. aircraft limits $\{5, 10\}$ m/s): fixed physical
  cutpoints supplied via `wind_chain.bin_edges`.

The bin index of a wind speed $w$ is $\beta(w) = \#\{i : e_i \le w\} - 1$.

---

## 3. Continuous within-bin distribution

Persistence is carried by the **bin**, but the failure integral keeps a **continuous**
within-bin wind so the steep failure curve stays resolved
$[\,\texttt{backward\_induction\_base.py: \_get\_wind\_grid\_bin}\,]$. Conditioned on bin
$b$, wind at stage $t$ is the stage-$t$ Weibull **truncated and renormalized** to
$[e_b, e_{b+1})$:

$$
f_t(w \mid b) \;=\;
\frac{f_t^{\mathrm{W}}(w)}{F_t^{\mathrm{W}}(e_{b+1}) - F_t^{\mathrm{W}}(e_b)}
\;\mathbb{1}\{\,e_b \le w < e_{b+1}\,\},
$$

where $f_t^{\mathrm{W}}, F_t^{\mathrm{W}}$ are the Weibull pdf/cdf with parameters
$(k_t, \lambda_t)$, and the bin mass is

$$
\pi_t(b) \;=\; F_t^{\mathrm{W}}(e_{b+1}) - F_t^{\mathrm{W}}(e_b),
\qquad
F_t^{\mathrm{W}}(e) = 1 - \exp\!\big[-(e/\lambda_t)^{k_t}\big].
$$

The mixture over bins reproduces the original marginal:
$\sum_{b} \pi_t(b)\, f_t(w\mid b) = f_t^{\mathrm{W}}(w)$ — the identity that
`verify_wind_chain.py` checks numerically.

---

## 4. Time-conditioned Markov chain

The bin evolves as a first-order Markov chain whose transition matrix is conditioned on the
calendar **(month, hour)** of the stage, so diurnal and seasonal drift are modeled by the
chain itself (this matters for the diagnostic in §9):

$$
\Pr\!\big(\beta_{t+1} = b' \mid \beta_t = b\big) \;=\; P^{(m_t,\,h_t)}_{b,\,b'},
$$

where $(m_t, h_t)$ is the month and hour at stage $t$. The full object is a tensor
$P \in \mathbb{R}^{13 \times 24 \times n_b \times n_b}$ — one $n_b \times n_b$ stochastic
matrix per (month, hour), i.e. $12 \times 24 = 288$ matrices
$[\,\texttt{run\_sim.py: EnvironmentLoader}\,]$.

**Fitting** $[\,\texttt{fit\_wind\_transition\_chain}\,]$. From the historical record
resampled to the model step, digitize each sample to a bin and count consecutive,
same-day-contiguous transitions stratified by (month, hour):

$$
N^{(m,h)}_{b,b'} \;=\; \sum_{\tau} \mathbb{1}\{\beta_\tau = b,\ \beta_{\tau+1} = b',\ m_\tau = m,\ h_\tau = h\},
$$

then row-normalize (uniform fallback for unobserved rows):

$$
P^{(m,h)}_{b,b'} \;=\;
\begin{cases}
\dfrac{N^{(m,h)}_{b,b'}}{\sum_{b''} N^{(m,h)}_{b,b''}}, & \text{if } \sum_{b''} N^{(m,h)}_{b,b''} > 0,\\[2ex]
1/n_b, & \text{otherwise.}
\end{cases}
$$

For brevity write $P^{(t)} \equiv P^{(m_t, h_t)}$ for the matrix in force at stage $t$.

---

## 5. Failure probability (binned)

A step fails if the battery depletes (a **solar / energy** event) **or** a mechanical
**wind** event occurs. The two are combined as independent causes
$[\,\texttt{\_compute\_failure\_probability\_bin}\,]$:

$$
p_{\mathrm{fail}}(s,a,t,b) \;=\; p_B(s,a,t) \;+\; \big(1 - p_B(s,a,t)\big)\, p_M(s,a,t,b).
$$

- **Solar / energy term** $p_B$ — unchanged from the published model and **not** binned
  (solar stays i.i.d.). With energy deficit $\Delta = E_{\text{req}}(s,a) - E(\sigma)$ and
  normalizer $G_{\max} = \max(G^{\text{cs}}_t, 10)$,

$$
p_B(s,a,t) \;=\; I_{u}\!\big(\alpha_t, \beta_t\big),
\qquad u = \operatorname{clip}\!\big(\Delta / G_{\max},\, 0,\, 1\big),
$$

  where $I_u(\alpha,\beta)$ is the regularized incomplete beta function and
  $(\alpha_t, \beta_t)$ the fitted stage-$t$ solar parameters.

- **Mechanical / wind term** $p_M$ — the expected one-step failure given wind is in bin
  $b$, integrated against the truncated within-bin density of §3:

$$
p_M(s,a,t,b) \;=\; \mathbb{E}_{\,W \sim f_t(\cdot\mid b)}\big[\,1 - q(W,a,s)\,\big]
\;=\; \int_{e_b}^{e_{b+1}} \big(1 - q(w,a,s)\big)\, f_t(w\mid b)\; \mathrm{d}w,
$$

  where $q(w,a,s)$ is the mechanical success probability. The integral is evaluated by
  trapezoidal quadrature on a per-(stage, bin) grid of the truncated support
  $[\,\texttt{\_mechanical\_failure\_probability\_bin}\,]$.

Setting $n_b = 1$ makes $f_t(\cdot\mid 0) = f_t^{\mathrm{W}}$, so $p_M$ becomes the original
full-Weibull failure integral.

---

## 6. Augmented MDP and Bellman recursion

The bin $b$ becomes an **exogenous state variable**. The value function gains a bin
dimension, $V_t(b, s)$, stored as a $n_b \times |\mathcal{S}| \times T$ table
$[\,\texttt{\_initialize\_future\_value\_table}\,]$. The state–action value is

$$
\boxed{\;
Q_t(b, s, a) \;=\;
\big(1 - p_{\mathrm{fail}}\big)\Big[\, r_{\text{succ}} \;+\; \gamma \sum_{b'=0}^{n_b-1} P^{(t)}_{b,b'}\, V_{t+1}\big(b',\, s'(s,a)\big) \Big]
\;+\; p_{\mathrm{fail}}\, r_{\text{fail}}
\;}
$$

with $p_{\mathrm{fail}} = p_{\mathrm{fail}}(s,a,t,b)$ from §5, $s'(s,a)$ the (deterministic)
energy/mode update on success, $r_{\text{succ}} = r(s,a,s',t)$, and
$r_{\text{fail}} = r(s,a,\text{fail},t)$ the penalized failure reward. The Bellman optimality
recursion and terminal condition are

$$
V_t(b, s) \;=\; \max_{a\in\mathcal{A}} Q_t(b,s,a),
\qquad
V_T(b, s) = 0,
\qquad
\pi_t^\star(b,s) \;=\; \arg\max_{a} Q_t(b,s,a).
$$

Two structural points:

1. **The next bin is independent of the failure event.** The bin transition $P^{(t)}$ is
   exogenous weather dynamics; the failure draw depends on the *current*-step wind via §5.
   Hence the future-value term factorizes as the bin-weighted average
   $\sum_{b'} P^{(t)}_{b,b'} V_{t+1}(b', s')$ $[\,\texttt{value\_function\_batch},$ computed
   as the contraction $\texttt{einsum}('nb,bn\!\to\! n')\,]$.
2. **The policy conditions on the current regime.** $\pi^\star_t(b,s)$ depends on $b$, so the
   aircraft can pre-empt a persisting high-wind spell — the mechanism behind the empirical
   gains in the briefing.

---

## 7. Exact reduction to the i.i.d. model

With $n_b = 1$: the single bin is $[0,\infty)$, $\pi_t(0)=1$, $f_t(\cdot\mid 0) =
f_t^{\mathrm{W}}$, and $P^{(t)} = [1]$. The bin sum collapses, $V_t(0,s) = V_t(s)$, and the
Bellman recursion of §6 is identical to §1. The implementation takes a **separate code
path** for $n_b = 1$ (a 2-D value table and the original solver), so the published results
are reproduced bit-for-bit, not merely in the limit
$[\,\texttt{solve}\to\texttt{\_solve\_iid}\,]$. A *rank-1* chain (all rows of $P^{(t)}$
equal) is the statistical analogue — it makes the bins i.i.d. and reproduces the i.i.d.
outcome within Monte-Carlo error, which `verify_wind_chain.py` asserts
($|\Delta\text{failure}\%| < 3$ pp, $|\Delta\text{reward}| < 0.5$).

---

## 8. Forward simulation (episode rollout)

To roll out a solved policy on stochastic or historical weather
$[\,\texttt{environment\_provider\_base.py}\,]$:

1. **Initialize** each lane's bin from the stage-0 bin masses:
   $\beta_0 \sim \operatorname{Categorical}\big(\pi_0(0), \dots, \pi_0(n_b-1)\big)$.
2. **Per step** $t$:
   - **Sample wind within the current bin** by inverse-CDF on the truncated support: draw
     $U \sim \mathrm{Uniform}\big(F_t^{\mathrm{W}}(e_{\beta_t}),\, F_t^{\mathrm{W}}(e_{\beta_t+1})\big)$
     and set
     $$
     W_t \;=\; \lambda_t \big[-\ln(1 - U)\big]^{1/k_t}.
     $$
   - **Expose** $\beta_t$ to the policy, which selects $a = \pi^\star_t(\beta_t, s)$.
   - **Advance the bin**: $\beta_{t+1} \sim \operatorname{Categorical}\big(P^{(t)}_{\beta_t, \cdot}\big)$.

   When no chain is supplied the provider falls back to ordinary i.i.d. Weibull sampling
   (single bin), so solver and rollout stay consistent. On *historical* rollouts the wind
   trajectory is taken from the real record (block bootstrap) rather than sampled, while the
   policy still conditions on the bin $\beta_t = \beta(W_t)$ implied by the realized wind.

---

## 9. The higher-order diagnostic (precheck)

The chain is **first order**: $\beta_{t+1}$ depends on $\beta_t$ but not on
$\beta_{t-1}$. Whether a *deeper* (history-augmented) chain could do better is tested
directly $[\,\texttt{wind\_persistence\_precheck.py}\,]$ by the (month, hour)-stratified
**conditional mutual information** between the next and previous bin given the current bin:

$$
I\big(\beta_{t+1};\, \beta_{t-1} \,\big|\, \beta_t,\, m,\, h\big)
\;=\!\!\sum_{m,h} \Pr(m,h)\!\!
\sum_{b_-,\,b_0,\,b_+}\!\!
\Pr(b_-, b_0, b_+ \mid m,h)\,
\log_2 \frac{\Pr(b_+ \mid b_0, b_-, m, h)}{\Pr(b_+ \mid b_0, m, h)} .
$$

This is exactly the information a second-order chain would add **beyond** the first-order,
(month, hour)-conditioned chain already in use. Conditioning on $(m,h)$ removes the diurnal
/ seasonal memory the chain models, isolating genuine higher-order wind memory. Reported as
bits and as a fraction of the conditional entropy $H(\beta_{t+1}\mid \beta_t)$, with the
verdict scale:

$$
\frac{I(\beta_{t+1};\beta_{t-1}\mid \beta_t, m, h)}{H(\beta_{t+1}\mid \beta_t)}
\;\approx\;
\begin{cases}
1\text{–}2\% & \text{coarse bins } (n_b = 3,\ \text{or } [5,10]\,\text{m/s}) \Rightarrow \text{first order suffices},\\
\nearrow & \text{rising monotonically with } n_b \Rightarrow \text{signal is in resolution.}
\end{cases}
$$

(High-$n_b$ magnitudes are inflated by finite-sample MI bias and need a Miller–Madow /
shuffle-null correction before publication.) The continuous analogue is the partial
autocorrelation of wind speed, significant out to lag $\approx 3$–$4$. **Conclusion:** the
exploitable structure lives in bin **resolution**, not Markov **order** — so first order is
the right complexity, and the next modeling lever is a finer/continuous wind state (and
solar persistence), not a deeper chain.

---

## 10. Computational cost

Let $G$ be the within-bin quadrature grid size. Relative to the i.i.d. model:

| Quantity | i.i.d. | wind chain | factor |
|---|---|---|---|
| Value table size | $|\mathcal{S}|\, T$ | $n_b\,|\mathcal{S}|\, T$ | $n_b$ |
| Backward-induction solve | $O(|\mathcal{S}|\,|\mathcal{A}|\,G\,T)$ | $O(n_b\,|\mathcal{S}|\,|\mathcal{A}|\,G\,T)$ | $n_b$ |
| Future-value lookup / step | $O(1)$ per lane | $O(n_b)$ per lane (bin contraction) | $n_b$ |
| Chain fit (offline, one-time) | — | $O(n_{\text{hist}} + 288\,n_b^2)$ | — |

So the model is a clean $O(n_b)$ multiplier in both solve and rollout, with a one-time
offline fit. It is **off by default** ($n_b = 1$), so the published pipeline is unaffected.

---

## Symbol reference

| Symbol | Meaning |
|---|---|
| $t,\ T$ | stage index; horizon |
| $s = (\sigma, m)$ | endogenous state: SoC level, mode |
| $a \in \{0,1\}$ | action |
| $\gamma,\ c_f$ | discount; failure penalty |
| $b,\ \beta_t,\ n_b$ | wind bin index; bin at stage $t$; number of bins |
| $e_b$ | bin edge ($e_0=0,\ e_{n_b}=\infty$) |
| $k_t,\ \lambda_t$ | stage-$t$ Weibull shape, scale (by month/hour) |
| $f_t(w\mid b),\ \pi_t(b)$ | truncated within-bin density; bin mass |
| $P^{(m,h)},\ P^{(t)}$ | (month,hour) transition matrix; the one in force at stage $t$ |
| $p_B,\ p_M,\ p_{\mathrm{fail}}$ | solar/energy, mechanical/wind, total one-step failure prob. |
| $V_t(b,s),\ Q_t,\ \pi^\star$ | value, action-value, optimal policy |

*Implementation: `BaseClasses/{backward_induction_base, environment_provider_base,
run_sim, simulation_base}.py`, `Scripts/create_weather_distributions.py`. Verification:
`Tests/verify_wind_chain.py`. Diagnostic: `Scripts/wind_persistence_precheck.py`.*
