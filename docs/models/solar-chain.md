# Solar persistence chain

A first-order, time-conditioned Markov chain over the **clear-sky index**
$K = \text{GHI} / G_{cs}$, which is the Beta-distributed variable the solver already
integrates. It mirrors the [wind chain](wind-chain.md) and carries cloud-regime
persistence both within a day and **from one day to the next**.

| | |
|---|---|
| Config block | `solar_chain` |
| Default | Off. With it off, or `n_bins: 1`, the pipeline is the published iid model exactly. |
| Status | Experimental |
| Artifact | `<data_path>_solarchain.pkl` |
| Builder | `Scripts/build_solarchain.py` → `create_weather_distributions.fit_solar_transition_chain` |
| Verification | `Tests/verify_solar_chain.py` (also re-runs the wind-chain checks unchanged) |
| Introduced | commit `59939eb` (2026-07-03) |

!!! abstract "Design record"
    [Solar persistence: mathematics and design record](../design/solar-persistence-math.md)
    covers the index definition, why global bin edges fail, the night and dawn handling,
    composition with wind, precheck numbers, and verification.

## Two differences from the wind chain

### 1. Stage-relative quantile bins, not global edges

At every stage, bin $g$ is the $[g/n, (g+1)/n)$ **quantile band of that stage's own
distribution**:

- fitting: the empirical rank within the `(month, hour)` slot across years;
- run time: the stage Beta's quantile band.

Every bin therefore has mass exactly $1/n_g$. There is no `bin_edges` option, and `n_bins`
is the only resolution setting.

The reason, as recorded in the design record and the `fit_solar_transition_chain` docstring:
the historical GHI is an hourly average, which biases $K$ low near sunrise and sunset.
With global edges, the dusk and dawn slots all fall into the bottom bin whatever the weather,
so the day-to-day channel carries no information (dusk→dawn diagonal ≈ [1, 0, 0]). Rank bins
recover the signal: [0.53, 0.36, 0.45] against 0.33 for a memoryless chain.

### 2. Night and the dusk→dawn channel

$K$ is undefined at night. A slot is **solar-valid** when $G_{cs}$ exceeds
`valid_threshold_wm2` (default 200 W/m²). The default is deliberately above the Beta fit's
50 W/m² gate: the `build_solar_chain_artifact` docstring reports that it cleans up the
dusk and dawn anchors (dawn-channel MI 0.126 vs 0.097 bits), at the cost of about 11% of
valid slots, all of them in twilight.

- **Intra-day** transitions are counted only between consecutive valid slots, per
  `(month, hour)`.
- **Day-to-day** persistence uses a separate per-month matrix that maps the last valid bin of
  day $d$ to the first valid bin of day $d+1$.

`EnvironmentLoader._build_per_stage_solar_transition` builds the per-stage stack
(`transition[t]` governs the move out of stage $t$):

| Stage *t* → *t+1* | Matrix |
|---|---|
| valid → valid | fitted `(month, hour)` matrix |
| last invalid stage before a dawn, when a dusk occurred earlier in the window | fitted dusk→dawn matrix for the dawn's month |
| anything else: night hold, dawn with no earlier dusk in the window, end of window | identity (bin held) |

Across a night, the product of the identity matrices and the one dawn matrix equals the
fitted day-to-day transition.

## Configuration

```yaml
solar_chain:
  enabled: true
  n_bins: 3        # stage-quantile bins (default 3)
  # path: ...      # optional; default <data_path>_solarchain.pkl
```

The harness rebuilds the artifact when `n_bins` or `clearsky_model` differs from what is
stored (unless `path` is explicit). The solar chain must be built with the same clear-sky
envelope as the expected-data file; see [Clear-sky models](clearsky.md).

## Artifact

| Key | Shape | Notes |
|---|---|---|
| `kind` | `"solar"` | |
| `n_bins` | int | |
| `bin_mode` | `"stage_quantile"` | |
| `conditioning` | `("month", "hour")` | |
| `transition_by_month_hour` | `(13, 24, n_bins, n_bins)` | Intra-day; unobserved rows are uniform |
| `dawn_transition_by_month` | `(13, n_bins, n_bins)` | Indexed by the dawn day's month |
| `valid_threshold_wm2` | float | Fit gate; the loader reuses it so fit, solver and provider agree on where $K$ exists |
| `clearsky_model` | str | |

## How the pipeline uses it

**Provider** (`StochasticWindSolarEnvironmentProvider`):

- Initial bins are uniform, which is exact because each bin has mass $1/n_g$.
- At valid stages the index is an exact inverse-CDF draw from the bin's band:
  $u \sim U(g/n, (g+1)/n)$, $K = F^{-1}_{\text{Beta}}(u)$.
- At invalid stages the draw is the unconditional stage Beta, the same as in iid mode
  (little energy arrives there anyway).
- The bin is exposed as `last_solar_bins`, then advanced with the stage matrix.

**Joint with wind**: the regime index is $z = b \cdot n_g + g$, with
$P_t = P^{wind}_t \otimes P^{solar}_t$ (`get_exo_transition`), which assumes the two chains
are independent. The value table remains 3-D with $n_b n_g$ regime rows. The code comments
describe a fitted joint chain as a drop-in replacement that would change only
`get_exo_transition`.

**Solver**: the solar bin conditions both the energy-failure term and the successor-SoC
masses, by substituting the truncated conditional Beta CDF
$\text{clip}(n\,I_x(\alpha,\beta) - g,\ 0,\ 1)$ (commit `59939eb`). The next-stage value is
contracted over $P_t[z,\cdot]$ exactly as for wind.

**Historical rollouts**: realized GHI at valid stages is digitized as
$\lfloor n_g F_{\text{Beta}}(K) \rfloor$ with the solve-side stage Beta. The bin is held
through the night. See [Weather modes](weather-modes.md).

## Evidence and limitations

Precheck at lat 30 / lon −90, 1950–2022 (`Scripts/solar_persistence_precheck.py`,
[design record §5](../design/solar-persistence-math.md)):

- **Day scale, first order**: 0.149 bits, 9.4% of daily regime entropy. Daily-mean-$K$
  ACF(1) = 0.40, e-folding time ≈ 2 days. Worth modelling.
- **Day scale, higher order**: 0.5%. First order is sufficient.
- **Intra-day, higher order**: 3.2%. A modest residual, recorded as a limitation.

Limitations recorded in the design record:

- only one site and record has been checked so far;
- the mismatch between the hour-averaged GHI and the instantaneous clear-sky value is absorbed
  by the rank bins rather than fixed in the normalization;
- the ~3% intra-day higher-order residual;
- the wind and solar chains are assumed independent.

## Cost

With both chains on, solve time and the per-step lookup grow by $O(n_b n_g)$: about 9× the
iid solve at 3 × 3 bins. Solar alone costs $O(n_g)$.
