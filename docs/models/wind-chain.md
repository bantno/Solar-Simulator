# Wind persistence chain

A first-order, time-conditioned Markov chain over wind-speed bins. It replaces the iid
per-stage wind draw of the published model and adds the current **wind bin** to the state
the optimal policy conditions on, so the policy can anticipate persistence: calm tends to
follow calm, and gusty tends to follow gusty.

| | |
|---|---|
| Config block | `wind_chain` |
| Default | Off. With it off, the pipeline is the published model exactly. |
| Artifact | `<data_path>_windchain.pkl` |
| Builder | `Scripts/build_windchain.py` → `create_weather_distributions.fit_wind_transition_chain` |
| Verification | `Tests/verify_wind_chain.py` |
| Introduced | commit `d73b543` (2026-06-03); user bin edges `81e3d4d`; quantile-bin sweep `3598426` |

!!! abstract "Design records"
    This page is the practitioner summary. The full reasoning is in:

    - [Briefing](../design/wind-persistence-briefing.md): what changed, results, limitations
    - [Mathematics](../design/wind-persistence-math.md): formulation, Bellman recursion, cost
    - [Theory](../design/wind-persistence-theory.md): why augmenting the state keeps the Markov property ([PDF](../design/wind-persistence-theory.pdf))
    - [Validity and implementation](../design/wind-persistence-validity.md): correctness argument, with a before/after code diff in §9
    - [Threshold benchmark](../design/wind-threshold-benchmark.md): chain-optimal vs iid-optimal vs a tuned threshold policy

## What it models

The wind speed range is split into $n_b$ bins, $0 = e_0 < e_1 < \dots < e_{n_b} = \infty$.
The chain governs **which bin** the wind is in. Inside the bin, the continuous speed is
drawn from the stage's fitted Weibull **truncated to the bin**. Two properties follow:

- **Marginal preserved.** Mixing the truncated Weibulls with the stage bin masses gives back
  the original stage Weibull exactly. The climatology is unchanged; only the memory across
  time is new.
- **Exact reduction.** With one bin (or with the chain off) the value table is 2-D, the
  solver takes the unchanged `_solve_iid` path, and results match the published model.

Transitions are conditioned on `(month, hour)`: 288 matrices, one per month and hour of day.
This keeps the diurnal and seasonal structure. A stage uses the matrix for its own calendar
month and hour.

The design record gives the modelling rationale. Correlated wind would break the Markov
property of the published state $(\sigma, m)$; carrying the bin as a state variable
restores it. The model is first order because a conditional-mutual-information precheck
found little extra memory at second order when the bins are coarse; the exploitable
structure lies in bin resolution rather than chain order (`Scripts/wind_persistence_precheck.py`,
[Mathematics §9](../design/wind-persistence-math.md)).

## Configuration

```yaml
wind_chain:
  enabled: true
  n_bins: 4              # equal-occupancy quantile bins (default 3); used when bin_edges is absent
  # bin_edges: [5.0, 10.0]   # interior cutpoints in m/s; overrides n_bins
  # path: ...                # optional; default <data_path>_windchain.pkl
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn the chain on |
| `n_bins` | `3` | Number of **equal-occupancy quantile** bins, cut at the location's own historical wind quantiles |
| `bin_edges` | none | Interior cutpoints in m/s, e.g. aircraft-based thresholds. `0` and `inf` are added automatically. Takes precedence over `n_bins`. |
| `path` | derived | Explicit artifact path. Also turns off stale-artifact detection. |

If the configured binning no longer matches the artifact on disk, the harness rebuilds it:
changed edges, changed bin count, or quantile vs explicit mode (the artifact records
`quantile_derived`). See [Weather data pipeline](weather-data.md#automatic-provisioning).

### Choosing bins

The chain-vs-iid sweep now uses `n_bins: 4` quantile bins derived per location. Recorded
rationale ([Threshold benchmark §9](../design/wind-threshold-benchmark.md), commit `3598426`):

- With fixed `[5, 10]` m/s edges, chain-optimal beat the best tuned threshold policy in 30 of
  34 historical cells. The losses were at Hawaii, where the ~6 m/s decision band fell
  inside a single 5–10 bin.
- With 4 quantile bins (Hawaii edges 4.11 / 5.67 / 7.34 m/s) chain-optimal won all 34 cells,
  and no cell got worse by more than Monte Carlo noise.
- The recorded conclusion is that bin placement and resolution were the binding constraint.
  A first-order chain with 4 quantile bins is described as a sensible default that needs no
  tuning.

Because quantile edges differ by location, analysis scripts read the edges from each
location's `*_windchain.pkl` (`compare_chain_sweep.chain_edges_for`), never from the config.

## Artifact

| Key | Shape | Notes |
|---|---|---|
| `n_bins` | int | |
| `bin_edges` | `(n_bins + 1,)` | `[0, e1, …, inf]` |
| `quantile_derived` | bool | |
| `conditioning` | `("month", "hour")` | Only option implemented |
| `transition_by_month_hour` | `(13, 24, n_bins, n_bins)` | Row-stochastic; months 1–12 used; unobserved rows are uniform |

Fitting counts consecutive 15-min pairs from the interpolated historical record. As the
fitting docstring notes, interpolation inflates the diagonal.

## How the pipeline uses it

**Loading** (`run_sim.EnvironmentLoader._build_per_stage_transition`): looks up each stage's
`(month, hour)` and stacks the matrices into a per-stage array `wind_transition` of shape
`(horizon, n_bins, n_bins)`.

**Provider** (`StochasticWindSolarEnvironmentProvider`):

- `_init_wind_bins`: initial bins drawn from the stage-0 Weibull bin masses.
- `_sample_within_bin`: exact inverse-CDF draw from the truncated Weibull,
  $u \sim U(F(e_b), F(e_{b+1}))$, $w = \lambda(-\ln(1-u))^{1/k}$.
- `_advance_bins`: the next bin for each lane, drawn from row `P[bin]`.
- `last_wind_bins`: the bin each lane was in when the latest sample was drawn. The rollout
  reads it before choosing an action.

**Solver** (`mdpAnalyticalBackwardSolver`,
[`backward_induction_base`](../api/baseclasses/backward_induction_base.md)):

- The value table becomes `(n_bins, |S|, horizon)`, and `_solve_chain` runs in place of
  `_solve_iid`.
- The failure probability is computed on a wind grid restricted to the current bin
  (continuous within the bin), so the steep failure curve stays resolved.
- The next-stage value is contracted over the bin transition:
  $\sum_{z'} P_t[z, z']\, V_{t+1}(z', \cdot)$ (`_vnext_eff`).
- `solver_tables/*.npy` holds the 3-D table.

**Policy** ([Policies](policies.md)): the optimal policy's `choose_action_batch` receives each
lane's current bin and the stage matrix. Threshold policies ignore the bin.

**Historical rollouts**: realized wind is digitized against the same edges
([Weather modes](weather-modes.md)).

## Cost

An $O(n_b)$ multiplier on solve time, value-table size, and the per-step lookup, plus a
one-time offline fit ([Mathematics §10](../design/wind-persistence-math.md)). With both chains
on, the multiplier is $n_b \cdot n_g$.

## When to enable it

- To study or exploit weather persistence in the optimal policy.
- To compare against the published model: run the same config with the chain on and off. For
  an evaluation that doesn't rely on the chain's own assumptions, add
  `historical_weather` ([Weather modes](weather-modes.md)).
- Leave it off to reproduce journal results.

The briefing's stated limitations include:

- the original evaluation covered one site and one season window, since widened by the
  34-cell sweep;
- policies are compared at different failure operating points;
- solar remaining iid, since addressed by the [solar chain](solar-chain.md).
