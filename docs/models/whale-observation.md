# Whale observation

The MDP's reward is observation: on each stage the aircraft earns the per-stage **sighting
probability** $p(t)$, scaled by an observation gain that depends on the mode and action.
Failure subtracts the failure penalty. This page covers how $p(t)$ is built
([`whale_base`](../api/baseclasses/whale_base.md)) and how the gain is applied
([`mdp_base`](../api/baseclasses/mdp_base.md)).

| | |
|---|---|
| Series switch | `whale_series` (default `JoA2026`) |
| Options | `whale_block_values`, `whale_block_hours`, `whale_species`, `whale_platform`, `whale_light`, `locations[].density_per_km2` |
| Gain switches | `takeoff_cruise_fraction` (default 1.0), `landing_cruise_fraction` (default 0.0) |
| Verification | `Tests/verify_whale_model.py` |
| Commits | `f281dca` (block series, custom profile, cruise-fraction gain); `6855bd3` (literature and design doc) |

!!! abstract "Design records"
    - [Literature survey and sensor-model design](../design/whale-observation-literature.md):
      aerial visual and passive-acoustic detection, species-agnostic parameters, the tiered
      integration plan, and the provenance of every term of the encounter model (§3b)
    - [Model extensions](../design/whale-observation-extensions.md): occupancy vs detection,
      POMDP/MOMDP, Bayes-adaptive and robust formulations, sequencing

$p(t)$ is a **deterministic per-stage scalar**. It does not depend on the weather, and the
provider returns the same value on every episode lane. Changing the series therefore changes
the reward vector only; the solver, providers and rollout are untouched.

## Series types

`WhaleRewardSeriesFactory.create_series(series_type, horizon, **kwargs)`:

| `whale_series` | Class | $p(t)$ |
|---|---|---|
| `JoA2026` (default) | `JoA2026WhaleRewardSeries` | The journal-results profile: 12 two-hour blocks with the Scott et al. (2001) daytime values; 20:00–06:00 set to zero |
| `real` | alias of `JoA2026` | The name before the rename; still resolves, with a `DeprecationWarning` |
| `custom` | `CustomWhaleRewardSeries` | A block profile taken from config |
| `encounter` | `EncounterWhaleSeries` | The distance-sampling encounter model (below) |
| `constant` | `ConstantWhaleRewardSeries` | 0.5 on every stage |
| `sinusoidal` | `SinusoidalWhaleRewardSeries` | $0.5\sin(x) + 0.5$ over `linspace(π, 60π, horizon)` |

`JOA2026_BLOCK_VALUES` (00:00 start, one value per 2 h block):

| 00–06 | 06–08 | 08–10 | 10–12 | 12–14 | 14–16 | 16–18 | 18–20 | 20–24 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.082 | 0.098 | 0.095 | 0.217 | 0.215 | 0.183 | 0.278 | 0 |

The source (in the `whale_base` comments) is Scott et al. (2001), Table 2: the proportion of
time a single rehabilitated pygmy sperm whale (*Kogia breviceps*), VHF-tracked for 4 days,
spent at the surface, per two-hour block. The code comments call it
"an example profile, not a population model".

## Block series and start-time alignment

Block-based series (`JoA2026`, `custom`, and `encounter`'s availability profile) are expanded
by `block_series(block_values, horizon, block_hours, delta_t, start_offset_min)`:

1. Each block is repeated for `block_hours * 60 / delta_t` stages. This must be a whole
   number.
2. The daily series is **rolled** so that stage 0 holds the value for the mission's start
   time of day. `start_offset_min` is `hour*60 + minute` of `start_datetime`
   (`run_sim._start_offset_min`), and it must also be a whole number of stages.
3. The series is tiled and truncated to `horizon`.

Validation: `len(block_values) * block_hours` must equal 24, and every value must lie in
[0, 1]. With `delta_t = 15` and a midnight start, the output is identical to the original
`np.repeat(values, 8)` / `np.tile` construction; `verify_whale_model.py` check 1 tests this.

### `custom`

```yaml
whale_series: custom
whale_block_hours: 2                    # default 2.0
whale_block_values: [0.0, 0.0, 0.0, 0.082, 0.098, 0.095, 0.217, 0.215, 0.183, 0.278, 0.0, 0.0]
```

The profile applies to the whole experiment; it is not a swept dimension. With the values
above (the built-in profile), results are identical to `JoA2026`
(`harness/examples/whale_custom_small.yaml`).

## Encounter model

!!! warning "In progress"
    The `encounter` series, the species cards, `whale_species` / `whale_platform` /
    `whale_light` and `locations[].density_per_km2` exist in the current working tree but
    have **not been committed yet**. The design is Tier 1 in the
    [literature survey](../design/whale-observation-literature.md). Expect changes.

Conventional distance-sampling encounter theory (Buckland et al. 2001; Marsh & Sinclair
1989 for splitting availability from perception). Per stage:

$$
\lambda(t) = D \cdot A \cdot a(t) \cdot g_0 \cdot \text{light}(t),
\qquad
p(t) = 1 - e^{-\lambda(t)}
$$

| Term | Property of | Config source | Code |
|---|---|---|---|
| $D$ density, animals/km² | **location** (optionally by month) | `locations[].density_per_km2` | `resolve_density` |
| $A$ area swept per stage, km² | **platform** | `whale_platform: {strip_width_km, ground_speed_ms}` | `swept_area_km2` = `strip_width_km * ground_speed_ms * delta_t*60 / 1000` |
| $a(t)$ availability (fraction of time at surface) | **species** | card `availability_profile` | `availability_series` |
| $g_0$ perception, P(seen \| surfaced in strip) | **species** (and sensor) | card `g0` | card lookup |
| $\text{light}(t) \in \{0,1\}$ | **sensor** | `whale_light` | `light_gate` |

**Focal form.** With no density, $p(t) = \text{clip}(a(t)\, g_0\, \text{light}(t), 0, 1)$:
a whale is assumed to be under the aircraft. `JoA2026` is this form with the `kogia_spp`
availability, $g_0 = 1$, and the fixed 06:00–20:00 gate. The settings
`whale_species: {name: kogia_spp, g0: 1.0}` with no density reproduce `JoA2026` exactly
(`verify_whale_model.py` checks 4 and 8).

With a density, the per-stage probabilities are one to two orders of magnitude smaller.
For Gulf *Kogia*, $D\cdot A \approx 0.01$–$0.1$ for a 5 km² stage. The literature survey
frames this as the search scenario, and `JoA2026` as the focal-follow scenario.

### Species cards (`SPECIES_CARDS`)

Every number carries its citation in the code.

| Card | `availability_profile` | `g0` | Sources (from the card) |
|---|---|---|---|
| `kogia_spp` (default) | `scott2001` (12 two-hour blocks) | 0.35 | Scott et al. 2001 Table 2; $g_0$ is the calm-sea (Beaufort 0–2) ship-observer trackline value (Barlow & Forney 2007). No UAV-camera value exists, so the card says to treat it as an upper bound or sensitivity. |
| `humpback` | 0.63 (scalar) | 0.91 | Hodgson, Peel & Kelly (2017): fixed-wing UAV availability; $g_0$ from their 1.10 perception correction |

The cards also record example densities, e.g. northern Gulf of Mexico *Kogia*: 0.00046/km²
visual (uncorrected for $g(0)$), 0.004–0.016/km² acoustic at slope sites.

`whale_species` accepts:

- a card name: `whale_species: kogia_spp`;
- a card with overrides: `whale_species: {name: kogia_spp, g0: 1.0}`;
- a bare card: `{availability_profile: 0.5, g0: 0.8}`. Here `availability_profile` is
  required and `g0` defaults to 1.0.

`availability_profile` can be a named profile (`scott2001`), a scalar in [0, 1], or a
per-block list.

### Density (`locations[].density_per_km2`)

It can be a scalar, a list of 12 monthly values, or a `{month: value}` mapping. Monthly forms
are selected by the month of `start_datetime`. Density must be non-negative. Leave it out,
or set it to `null`, for the focal form. When a density is given, `whale_platform` must also
be given.

### Light gate (`whale_light`)

| Mode | Behaviour |
|---|---|
| `{mode: fixed_hours, day_start: 6, day_end: 20}` (default) | 1 when the stage's time of day falls in `[day_start, day_end)` hours; the window may wrap midnight |
| `{mode: clearsky_gate, threshold_wm2: 0.0}` | 1 when the location's per-stage clear-sky irradiance is above the threshold (sun above the horizon) |
| `{mode: none}` | Always 1 |

### Example

From `harness/examples/whale_encounter_small.yaml`:

```yaml
whale_series: encounter
whale_species:
  name: kogia_spp
whale_platform:
  strip_width_km: 0.4
  ground_speed_ms: 18.8
whale_light: {mode: fixed_hours, day_start: 6, day_end: 20}
locations:
- latitude: 30.0
  longitude: -90.0
  data_path: Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min.pkl
  density_per_km2: 0.01
```

The config's own comments justify the platform numbers: seaplane cruise speed is
17–20 m/s over 200–400 Wh (`seaplane_base.get_max_endurance_speed`), and the strip width
follows the camera footprint (Hodgson et al. 2017: 490 m at 732 m altitude).

The survey lists the model's own simplifications:

- $D$ is constant within a location, because the vehicle has no position;
- $A$ assumes transect-style motion;
- sightings on different stages are independent;
- sea state is deliberately left out.

## Observation gain (cruise fractions)

The per-stage reward is

$$
r = \text{gain}[m, a] \cdot p(t) \;-\; c_f\,\mathbb{1}[\text{failure}],
$$

where `AbstractMDP.observation_gain` is indexed `[mode, action]`:

| Mode | $a = 0$ | $a = 1$ |
|---|---|---|
| 0 moored | 0 (stay moored) | `takeoff_cruise_fraction` (take off) |
| 1 flying | `landing_cruise_fraction` (land) | 1 (keep flying) |
| 2 broken | 0 | 0 |

The fractions are the share of the takeoff or landing stage spent surveying at cruise. The
defaults `(1.0, 0.0)` encode the original simplification that a climb is short compared with
a stage, so a takeoff stage earns the full reward and a landing stage earns nothing. The
defaults reproduce the original model bit for bit.

Values must lie in [0, 1]; `AbstractMDP.__init__` rejects anything else. The gain is applied
in `stochasticMDP.reward` and at every solver call site, through `mdp.obs_gain(states, actions)`.

### `auto`

Either key accepts `"auto"`. `SimulationFactory._resolve_cruise_fractions` then derives

$$
f_{to} = \max\!\left(0,\ 1 - \frac{t_{climb}}{\Delta t}\right),
\qquad
f_{ld} = \max\!\left(0,\ 1 - \frac{t_{descent}}{\Delta t}\right)
$$

from `Seaplane.get_maneuver_times_s()`. That function times a climb from 0 to cruise altitude
at +2°, and a descent at −2°, both at cruise speed, using the same models the takeoff and
landing power estimates use (see [Aircraft power model](aircraft-power.md)). The fractions are
resolved per battery capacity, because the aircraft's mass depends on capacity.

## Where the series is built

Both places that construct weather in `run_sim` build the series with the same arguments:

- `EnvironmentLoader._extract_distributions`, for the solve-side provider;
- the historical-bootstrap branch of `SimulationFactory.__init__`, for rollouts.

The arguments are `delta_t`, `start_offset_min`, `start_month`, the per-stage clear-sky
series `solar_cs`, and the series options from `_whale_kwargs_from_config`, which maps the
config keys to factory keyword arguments:

| Config key | Factory keyword |
|---|---|
| `whale_block_values` | `block_values` (floats) |
| `whale_block_hours` | `block_hours` (float) |
| `whale_species` | `species` |
| `whale_platform` | `platform` |
| `whale_light` | `light` |
| `locations[].density_per_km2` | `density_per_km2` |

These keys are also written to `summary.csv` ([Outputs](../experiments/outputs.md)).

!!! todo "Owner input needed"
    Which species and survey scenario the mission targets, and therefore which series and
    density should be the default for new studies, is not recorded in the repository.
