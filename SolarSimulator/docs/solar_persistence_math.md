# Solar-Persistence Extension — Mathematical Formulation & Design Record

*Companion to the wind-persistence docs (`wind_persistence_math.md`,
`wind_persistence_validity.md`). This document specifies the Markov-modulated
clear-sky-index model, the two places its design intentionally departs from the wind
template, and the empirical prechecks that drove those decisions. Bracketed tags point
at the implementing code. Status: experimental, default **off**; with the chain off the
published i.i.d. model is untouched.*

---

## 1. The chain variable: the clear-sky index

Solar enters the published model as a **clear-sky index**

$$
K_t \;=\; \frac{\mathrm{GHI}_t}{G^{\mathrm{cs}}_t},
\qquad
G^{\mathrm{cs}}_t = A\,\cos z_t,\ A = 1150\ \mathrm{W/m^2},
$$

fitted per calendar slot as $K_t \sim \mathrm{Beta}(\alpha_t, \beta_t)$
[`weather_processor_cs_normalization.py: _fit_beta`] and consumed analytically by the
solver in two places: the energy-failure term $p_B = I_u(\alpha_t,\beta_t)$ and the
successor-SoC masses $\Delta P_e = I_{u_e^+} - I_{u_e^-}$ (regularized incomplete beta
differences across the energy-bin edges)
[`backward_induction_base.py: _compute_failure_probability_bin`,
`_compute_survival_contribution_batch`].

The extension makes $K_t$ persistent: a first-order, (month, hour)-conditioned Markov
chain over **index bins**, with the current bin an exogenous DP state variable —
exactly the wind construction, with the Beta playing the Weibull's role.

## 2. Departure #1 — stage-relative quantile bins (not global edges)

**Wind bins are global cutpoints; solar bins are the stage distribution's own quantile
bands.** Bin $g \in \{0,\dots,n_g-1\}$ at stage $t$ is

$$
K_t \in \big[\,F_t^{-1}(g/n_g),\; F_t^{-1}((g{+}1)/n_g)\,\big),
\qquad F_t = I_{\cdot}(\alpha_t, \beta_t),
$$

i.e. bin $g$ = "the sky is in the $g$-th $n_g$-tile of what this time-of-day/season
usually looks like". Bin masses are exactly $\pi_t(g) = 1/n_g$ at every stage by
construction.

**Why global edges fail for solar.** The historical GHI record is hour-averaged while
the clear-sky normalizer is instantaneous, so near sunrise/sunset (part of the
averaging window is dark) $K$ is biased low **under every weather regime**. With global
bin edges the dusk/dawn slots collapse into the bottom bin regardless of cloudiness —
measured dusk→dawn self-transition $[\approx 1, 0, 0]$, i.e. the day-to-day channel
carries *nothing* — while with rank (quantile) bins the same channel is strongly
informative ($[0.53, 0.36, 0.45]$ against $1/3$ memoryless; dawn-bin occupancy uniform).
The exploitable regime signal is *relative to the hour*, not absolute
[`Scripts/solar_persistence_precheck.py`, scratch comparison in the fitting docstring].

**The quantile parameterization also simplifies everything downstream.** Because the
bin edges are the stage Beta's own quantiles, the truncated, renormalized conditional
CDF needs no edges at all:

$$
\boxed{\;
F_t(x \mid g) \;=\; \operatorname{clip}\!\big(n_g\, I_x(\alpha_t,\beta_t) - g,\ 0,\ 1\big)
\;}
$$

[`backward_induction_base.py: _solar_cdf`]. Consequences:

- **$p_B$ and the successor-SoC masses condition on $g$** by substituting $F_t(\cdot|g)$
  for $I_\cdot$ — the only solver change (the wind term $p_M$ is untouched). Note the
  solar bin conditions *both* failure and the SoC transition, unlike wind which only
  conditions failure.
- **Marginal preservation is exact**: $\sum_g \frac{1}{n_g} f_t(x\mid g) = f_t(x)$ —
  same mixture identity as wind (§3.2 of the validity doc), so only temporal
  correlation is added, never the climatology.
- **No degenerate-bin guards.** Saturated slots (near-point-mass Betas, $\kappa=10^6$)
  give a valid conditional CDF automatically; there are no zero-mass bins in CDF space.
- **Exact within-bin sampling**: $u \sim U(g/n_g, (g{+}1)/n_g)$,
  $K = F_t^{-1}(u)$ via `scipy.stats.beta.ppf`
  [`environment_provider_base.py: _sample_solar_index_within_bin`].
- **Historical digitization**: the realized index maps to its bin through the stage
  CDF, $g = \lfloor n_g\, I_{K_t}(\alpha_t,\beta_t) \rfloor$
  [`HistoricalBootstrapEnvironmentProvider.sample_sunlight`].
- **Uniform initialization is exact**: the stage marginal over bins is uniform, so
  $\beta^{\mathrm{sol}}_0 \sim U\{0,\dots,n_g{-}1\}$.
- **The memoryless (rank-1) chain is the uniform matrix**, which is what
  `verify_solar_chain.py` uses for the i.i.d.-reduction check.

Fitting-side binning is the empirical analogue: each valid sample's rank within its
(month, hour) slot population across years
[`create_weather_distributions.py: fit_solar_transition_chain`]. (Fit-time rank bins
pool days within a month; run-time bins use the per-slot Beta — the same
approximation level as the wind chain's (month,hour) transition pooling.)

## 3. Departure #2 — night, and the dusk→dawn channel

The index is undefined when $G^{\mathrm{cs}}_t$ is small. A stage is **solar-valid**
when $G^{\mathrm{cs}}_t$ exceeds the artifact's gate (default **200 W/m²**, not
`_fit_beta`'s 50: raising it excludes the terminator slots whose hour-averaged bias is
worst, measurably improving the day channel — dawn MI 0.126 vs 0.097 bits — at the cost
of ~11% of valid slots, all twilight where little energy arrives). $G^{\mathrm{cs}}$ is
unimodal within a day, so valid slots are contiguous per day at any gate.

Everything night-related lives in the **per-stage transition stack** built by the
loader [`run_sim.py: _build_per_stage_solar_transition`]; the solver and provider have
no night special cases beyond "bypass conditioning at invalid stages":

- both $t$, $t{+}1$ valid → the fitted (month, hour) matrix
  ($P \in \mathbb{R}^{13\times24\times n_g\times n_g}$, wind-style);
- the **last invalid stage before each dawn** → the fitted per-month **dusk→dawn
  matrix** $D^{(m)} \in \mathbb{R}^{13\times n_g\times n_g}$ (last valid bin of day $d$
  → first valid bin of day $d{+}1$, stratified by the dawn day's month);
- all other stages (night hold, end of window, dawn with no in-window dusk) →
  **identity**.

Since `transition[t]` governs the move *out of* stage $t$ (provider advance and solver
contraction alike), the product of matrices across any night is
$I \cdots I \cdot D^{(m)} = D^{(m)}$: the chain state at dawn is exactly one fitted
day-scale transition from the dusk state. **This dusk→dawn matrix is the channel that
carries multi-day cloud-spell persistence** — the phenomenon that drives
battery-depletion failure and motivated the extension. At invalid stages $p_B$, the
SoC masses, and the within-step draw all use the unconditional stage Beta (the
published behavior); the bin is simply held as latent regime memory.

Stage-0-at-night: bins initialize uniformly (exact, §2) and the first dawn with no
in-window dusk is identity, so no fabricated dusk information enters.

## 4. Composition with the wind chain

The DP's exogenous state is the **joint regime index** $z = b \cdot n_g + g$ (wind bin
$b$, solar bin $g$), $n_z = n_b n_g$. The value table stays 3-D, $(n_z, |\mathcal S|,
T)$, and every existing lookup — `_vnext_eff`, the `einsum('nb,bn->n')` in
`value_function_batch`, `choose_action_batch`, `simulate_episode_batch` — works
unchanged with $z$ in place of the wind bin
[`environment_provider_base.py: use_exo_chain / n_exo_bins / last_exo_bins /
get_exo_transition`].

Under the **independence assumption** (the deliberate first-cut composition decision),

$$
P_z(t) \;=\; P^{\mathrm{wind}}(t) \otimes P^{\mathrm{sol}}(t),
$$

with an inactive chain contributing $[[1]]$; wind-only, solar-only, and both-off are
exact special cases (both-off keeps the separate 2-D `_solve_iid` path bit-for-bit). A
future **fitted joint** wind–solar chain (storms are windy *and* cloudy) replaces only
`get_exo_transition` with a fitted $(13,24,n_z,n_z)$ tensor in the same $z$ layout — no
solver refactor.

Bellman recursion on $x = (z, s)$, with $\Delta P_e(g)$ the bin-conditioned SoC masses:

$$
Q_t(z,s,a) = \big[aO_t - c_f\,p_{\mathrm{fail}}(s,a,t,b,g)\big]
+ \gamma\,(1-p_{\mathrm{fail}})
\sum_e \Delta P^{(t)}_e(g)\sum_{z'} P_z(t)_{z,z'}\, V_{t+1}(z', s'_e),
$$

$p_{\mathrm{fail}} = F_t(u\mid g) + (1 - F_t(u\mid g))\,p_M(s,a,t,b)$. (The
implementation reproduces the existing survival-mass convention — `deltaP` sums to
$1 - p_B$ and is multiplied by $(1-p_{\mathrm{fail}})$ — verbatim with the truncated
CDF substituted, so the $n_g=1$ reduction is exact. Whether that convention's apparent
extra $(1-p_B)$ factor versus Eq. (8) of the wind math doc is intended is a
**pre-existing** question in the published path, flagged separately; this extension
deliberately mirrors, not fixes, it.)

**Conditional-independence assumption.** As for wind: given the current bin, the next
bin is independent of the within-step draw, $g_{t+1} \perp K_t \mid g_t$ (and of the
failure event). For solar this is *stronger in consequence* than for wind, because
$K_t$ drives the successor SoC, not just failure — a lane at the top of its bin is
somewhat more likely to transition up than the chain admits. This is exactly what a
first-order bin chain is defined by; the precheck below bounds what it discards.

## 5. Precheck results (lat 30 / lon −90, 1950–2022, hourly)

[`Scripts/solar_persistence_precheck.py`; the CMI/permutation machinery is imported
from `wind_persistence_precheck.py` unchanged.]

- **Day-scale, first order (the prize):** $I(\text{next day};\text{current day}) =
  0.149$ bits = **9.4%** of daily regime entropy; daily-mean-$K$ ACF(1) $= 0.40$,
  e-folding $\approx 2$ days. The dusk→dawn channel is worth modeling.
- **Day-scale, higher order:** bias-corrected
  $I(\text{next};\text{prev}\mid\text{curr},m) = 0.007$ bits (**0.5%**) — first order
  is the right complexity at day scale.
- **Intra-day, higher order:** bias-corrected
  $I(\beta_{t+1};\beta_{t-1}\mid\beta_t,m,h) = 0.027$ bits (**3.2%** of
  $H(\beta_{t+1}\mid\beta_t)$) with rank bins — modest residual, somewhat above wind's
  1.4–2%; noted as a limitation rather than grounds for a deeper chain (same
  resolution-vs-order reading as wind, and the hourly→15-min interpolation caveat
  applies to the intra-day diagonal).
- Pooled intra-day rank-bin transition is strongly diagonal (0.83/0.69/0.86).

## 6. Reductions and verification (`Tests/verify_solar_chain.py`)

1. Chain off → 2-D table, `_solve_iid`, published path (and `verify_wind_chain.py`
   passes unchanged after the joint-index refactor — the wind regression gate).
2. **Uniform (memoryless) chain ⇒ i.i.d.** within MC error, plus the exact mixture
   identity: uniform bins + within-bin quantile sampling reproduce the stage-Beta
   marginal.
3. **Persistence effect**: with the fitted chain, $V$ rises with the solar bin at a
   valid mid-day stage (a clear regime is worth more).
4. **Joint plumbing**: wind+solar table has $n_b n_g$ regime rows; solar-memoryless
   joint ≈ wind-only chain within MC error (validates the Kronecker/index layout).
5. **Structure**: identity at held stages, one dawn matrix per in-window dawn;
   degenerate (saturated) stage Betas give finite, in-range conditional CDFs/samples.

## 7. Cost and scope

$O(n_b n_g)$ multiplier on the solve and the per-step lookup (3×3 bins ⇒ ~9× the
i.i.d. solve when both chains are on; solar-only is the same $O(n_g)$ as wind's
$O(n_b)$). Off by default; `solar_chain: {enabled: true, n_bins: 3}` in the run config
enables it, with artifact auto-provisioning mirroring the wind chain
(`<data>_solarchain.pkl`) [`harness/run_experiment.py: _ensure_location_data`].

Known limitations: one site/record so far; the hour-averaged GHI vs instantaneous
clear-sky mismatch is absorbed by rank bins rather than fixed at the normalization
level (an averaged-envelope normalizer would refit the expected-data Betas — out of
scope); the ~3% intra-day higher-order residual; independence of the wind and solar
chains (the fitted joint chain is the designed next step).
