# Backward-induction solver

The optimal policy comes from `mdpAnalyticalBackwardSolver` in
`BaseClasses/backward_induction_base.py` (API:
[backward_induction_base](../api/baseclasses/backward_induction_base.md)).

- **Method.** The solver fills a value table by backward induction over the finite horizon.
  It computes expectations over wind and solar **analytically** (a Beta CDF and quadrature)
  rather than by sampling.
- **Other solver in the module.** `mdpBackwardSolver`, a Monte Carlo variant, is also in the
  module. The harness and `SimulationFactory` only use the analytical solver.

Notation follows [The MDP](mdp.md).

## Per-stage quantities

For a state $s = (\sigma, m)$ with stored energy $C$ and action $a$ with energy cost $E$
(`get_required_energy`), the stage-$t$ solar budget is

$$G_\text{max} = \max\!\big(G^\text{cs}_t\ \text{in joules},\ 10\big).$$

**Battery-depletion probability.** The battery fails if the harvested solar can't cover the deficit:

$$p_B = I_{u}(\alpha_t, \beta_t), \qquad u = \mathrm{clip}\!\Big(\frac{E - C}{G_\text{max}}, 0, 1\Big),$$

where $I_u$ is the regularized incomplete Beta function (`scipy.special.betainc`), i.e. the CDF
of the stage's clear-sky index.

**Mechanical-failure probability.** This is the expected action-failure probability under the
stage's Weibull wind:

$$p_M = \int_0^{w_\text{max}} \big(1 - p_s(w, a, s)\big)\, f_{W_t}(w)\, dw,$$

computed with `np.trapz` on a 100-point grid up to the Weibull 0.9999999 quantile
(`_mechanical_failure_probability`, grid cached per stage in `_get_wind_grid`).

**Total failure probability:**

$$p_f = p_B + (1 - p_B)\, p_M.$$

## Bellman recursion (i.i.d. weather)

$$Q_t(s, a) = \underbrace{g(m,a)\,O_t - c_f\,p_f}_{\texttt{expected\_reward}} \;+\; \gamma \sum_{n=0}^{N-1} (1 - p_f)\,\Delta P_n\; V_{t+1}(n, a)$$

$$V_t(s) = \max_{a \in \{0,1\}} Q_t(s, a), \qquad V_{H-1}(s) = \max_a \big[g(m,a)\,O_{H-1} - c_f\,p_f\big]$$

- **Next state after a successful action.** The next mode equals the action, so the relevant
  slice of the next-stage table is the rows for mode $a$ (`_get_vnext_slice`).
- **$\Delta P_n$** is the probability that the post-step energy $C + G - E$ lands in SoC bin
  $n$. That bin covers $[n\Delta, (n+1)\Delta)$, where $\Delta$ is the bin width in joules;
  the top bin extends to $\infty$, so any surplus is capped at full:
  $$\Delta P_n = I_{u^+_n}(\alpha_t,\beta_t) - I_{u^-_n}(\alpha_t,\beta_t), \quad u^\pm_n = \mathrm{clip}\!\Big(\frac{e^\pm_n + E - C}{G_\text{max}}, 0, 1\Big).$$
- **Value of `broken`.** The broken state has value 0.
- **Implementation.** `_compute_survival_contribution_batch` evaluates all states at once as
  `(1 - p_f)[:, None] * ΔP @ V_next`.

## Vectorization

`_solve_iid` works from $t = H-1$ down to $0$. At each stage it scores **every** non-broken
state for each action in one batched call (`_value_batch`), then takes the elementwise
maximum. The original per-state scalar path (`_value`, `expected_future_value`,
`compute_survival_contribution`) is still in the source as a reference implementation.

| Measure | Result (from `Tests/verify_solver_speed.py`, horizon 200) |
|---|---|
| Speed-up over the per-state loop | ~28× (grows with horizon) |
| Difference between the two value tables | ~1e-15 |

See [Verification](../verification/index.md).

## Weather-chain recursion

When a wind and/or solar persistence chain is enabled, the provider exposes
$n_\text{bins} = n_\text{wind} \cdot n_\text{solar} > 1$ joint regimes, and `solve()`
dispatches to `_solve_chain`. For each stage and each current regime
$z = b \cdot n_g + g$:

- **Mechanical failure**, conditioned on wind bin $b$: the Weibull density is truncated to the
  bin and renormalized (`_get_wind_grid_bin`, `_mechanical_failure_probability_bin`).
- **Energy**, conditioned on solar bin $g$: the solar bins are the stage Beta's own quantile
  bands, so the conditional CDF is $F(u \mid g) = \mathrm{clip}(n_g\, I_u - g, 0, 1)$
  (`_solar_cdf`). It falls back to the unconditional CDF at stages where the clear-sky index is
  undefined.
- **Future value**, weighted by the stage transition row:
  $$\bar V_{t+1}(\cdot, a \mid z) = \sum_{z'} P_t[z, z']\, V_{t+1}(z', \cdot, a)$$
  (`_vnext_eff`). $P_t$ comes from the provider's `get_exo_transition(t)`, which is the
  Kronecker product $P^\text{wind}_t \otimes P^\text{solar}_t$ when both chains are on.

With both chains off ($n_\text{bins} = 1$) the solver takes the original `_solve_iid` path
unchanged. The chain maths is derived in
[Wind persistence: mathematics](../design/wind-persistence-math.md) and
[Solar persistence](../design/solar-persistence-math.md).

## Value table

| Weather | `future_value_table` shape | Saved `.npy` |
|---|---|---|
| i.i.d. | `(2·n_soc + 1, H)` | 2-D |
| chain | `(n_bins, 2·n_soc + 1, H)` | 3-D |

- **Rows.** Rows follow the state ordering in [The MDP](mdp.md): all moored SoC levels, then
  all flying levels, then `broken`.
- **Saving.** `solve()` saves the table with `np.save` under
  `<prefix>_<capacity>Wh_<H>h_<penalty>p_<start[0:12]>.npy`, where the prefix is the config
  basename. Example: `iid_small_300.0Wh_150h_5.0p_2020-04-01 0.npy`.
- **Location.** The harness points `output_dir` at the run directory's `solver_tables/`.
  Without it, the file goes to the current working directory.
- **Inspection.** `Scripts/plot_value_function.py` plots a saved table as a heatmap by mode.

!!! note "Tables are outputs, never inputs"
    Every run solves fresh. Saved tables are never loaded back to skip a solve; the
    maintainers ruled this out because it risks stale results.

## Using the table at run time

The optimal policy (see [Episode simulation](simulation.md)) looks up next-stage values with
`value_function_batch`:

- **i.i.d.:** an O(n) direct index (`_lookup_future_values_fast`); `broken` maps to 0.
- **Chain:** the lookup is weighted by `P[cur_bins]` over the next regimes.
