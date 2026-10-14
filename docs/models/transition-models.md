# Transition models

A transition model gives the probability that a stage's action **succeeds**,
$P(S=1 \mid w_k, a, m)$, as a function of wind speed $w_k$, action $a$ and vehicle mode $m$.
If the action fails, the vehicle goes to the absorbing *broken* state $(-1, 2)$ and the MDP charges
the failure penalty. The models live in `SolarSimulator/BaseClasses/transition_model_base.py`
([API](../api/baseclasses/transition_model_base.md)). The YAML key `transition_model` picks one by
name; the default is `moderate`.

## Where it plugs in

`AbstractMDP.__init__` calls `ProbabilityModelFactory.select_probability_model(transition_model_name)`
and hands the result to `StochasticTransitionLogic`. The same `compute_probability` is used in two
places:

- **Solving.** `mdpAnalyticalBackwardSolver` weights the success and failure outcomes when
  computing each stage's value ([Solver](../architecture/solver.md)).
- **Rollouts.**
  - `StochasticTransitionLogic._apply_transition_probability` draws `rand() < p_success` for each
    episode.
  - `OptimalContinuousAnalyticalPolicySimulation` uses the same probability for its two-outcome
    expectation ([Policies](policies.md)).

## Actions and modes

| Mode $m$ | $a = 0$ | $a = 1$ |
|---|---|---|
| 0 moored | stay moored ("float") | take off |
| 1 flying | land | continue flying |
| 2 broken | no valid action | no valid action |

The mode after a successful transition is set by the action alone: $a=0$ leads to moored and
$a=1$ to flying. The only other way to reach broken is a battery going negative (SoC < 0).

## Named models

`ProbabilityModelFactory.models` maps names to classes. Lookup is case-insensitive.

| `transition_model` | Class | Form |
|---|---|---|
| `moderate` (default) | `ModerateSuccessProbability` | Sigmoid takeoff/landing, constant float/fly |
| `some` | `SomeSuccessProbability` | Same as `moderate` with a gentler landing slope |
| `realistic` | `RealisticSuccessProbability` | Sigmoid on takeoff, fly and landing, capped at 0.99 |
| `optimistic` | `OptimisticSuccessProbability` | Standard logistic, shifted to high winds |
| `linear` | `LinearSuccessProbability` | $1 - 0.05\,w$, clipped at 0 (all actions) |
| `nowind` | `WindIndependentSuccessProbability` | Constant per action, no wind dependence |
| `nofail` | `OnlySuccessProbability` | Always succeeds |
| `test` | `TestSuccessProbability` | Fixed low probabilities for unit tests |
| `wind based` | `DiscreteSuccessProbability` | Constant base probability × binned wind factor |

### Sigmoid family

`moderate`, `some` and `realistic` use a logistic with a 1.1 denominator, so the failure
probability saturates at $1/1.1 \approx 0.909$:

$$
\sigma_{1.1}(w;\,a,b) = \frac{1}{1.1 + e^{\,a - b w}}
$$

`optimistic` uses the standard logistic $\sigma(w;a,b) = 1/(1 + e^{a - bw})$.

| Model | Takeoff $P_\text{fail}$ | Stay moored $P_\text{fail}$ | Fly $P_\text{fail}$ | Land $P_\text{fail}$ |
|---|---|---|---|---|
| `moderate` | $\sigma_{1.1}(w;8,0.5)$ | $10^{-5}$ | $10^{-4}$ | $\sigma_{1.1}(w;7,0.7)$ |
| `some` | $\sigma_{1.1}(w;8,0.5)$ | $10^{-5}$ | $10^{-4}$ | $\sigma_{1.1}(w;7,0.5)$ |
| `realistic` | $0.01 + \sigma_{1.1}(w;5,0.5)$ | $10^{-3}$ | $0.01 + \sigma_{1.1}(w;5,0.35)$ | $0.01 + \sigma_{1.1}(w;4,0.7)$ |
| `optimistic` | $\sigma(w;17,0.5)$ | 0 | 0 | $\sigma(w;16,0.5)$ |

For `realistic`, success is coded as $0.99 - \sigma_{1.1}$ for takeoff, fly and land. The
constructor also takes `flying_failure=0.01`, but the fly case uses the sigmoid instead.

### Constant and binned models

| Model | Takeoff | Stay moored | Fly | Land |
|---|---|---|---|---|
| `nowind` (success) | 0.99 | 0.9999 | 0.999 | 0.95 |
| `nofail` (success) | 1 | 1 | 1 | 1 |
| `test` (success) | 0.25 | 1 | 0.75 | 0.25 |
| `wind based` base (success) | 0.999 | 1.0 | 0.9999 | 0.995 |

`wind based` multiplies the base success probability by a factor looked up from the wind bin.
The bin edges are `wind_bins = [0, 1, 5, 10, ∞)` m/s and the factors are
`wind_success_factors = [1.0, 0.98, 0.95, 0.90]`.

### Failure probability at sample wind speeds

Evaluated from the code (1 − `compute_probability`):

| Model | Action | 0 m/s | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| `moderate` | takeoff | 0.0003 | 0.0041 | 0.0472 | 0.364 | 0.810 |
| `moderate` | land | 0.0009 | 0.0292 | 0.476 | 0.885 | 0.908 |
| `some` | land | 0.0009 | 0.011 | 0.118 | 0.586 | 0.870 |
| `realistic` | takeoff | 0.017 | 0.085 | 0.486 | 0.856 | 0.914 |
| `realistic` | fly | 0.017 | 0.047 | 0.189 | 0.542 | 0.820 |
| `realistic` | land | 0.028 | 0.374 | 0.880 | 0.918 | 0.919 |
| `optimistic` | takeoff | 0 | 0 | 0 | 0.0001 | 0.0009 |
| `optimistic` | land | 0 | 0 | 0 | 0.0002 | 0.0025 |
| `linear` | any | 0 | 0.25 | 0.50 | 0.75 | 1.0 |

Under `moderate`, landing is the wind-limited maneuver. Its failure probability passes 0.5 just
above 10 m/s, while takeoff reaches that level near 16 m/s.

## Adding a model

1. Subclass `ActionSuccessProbabilityModel` (or `SigmoidSuccessProbability`) and implement
   `compute_probability(wind_speed, action, state)`. Begin with `check_and_broadcast` so that
   scalar and vector inputs are both accepted, and return an `(n,)` array.
2. Add the class to `ProbabilityModelFactory.models` under a lowercase name.
3. Refer to it from YAML with `transition_model: <name>`.

`visualize_failure_probability()` on any model plots $P_\text{fail}$ against wind speed for the four
action and mode combinations.

## State discretization helpers

`AbstractTransitionLogic` also provides the SoC ↔ energy mapping the solver uses:

- `soc_to_energy(soc)` $= \text{soc}/100 \times E_\text{cap}$ (J).
- `energy_to_soc(E)` floors $E$ onto $n = \text{round}(100/\Delta_\text{soc})$ equal energy bins,
  returns $\text{bin}\times\Delta_\text{soc}$, and maps negative energy to −1.

The SoC step $\Delta_\text{soc}$ comes from `energy_increment_wh` through
$\Delta_\text{soc} = 100\,\cdot$ `energy_increment_wh` / capacity (`SimulationFactory.build_mdp`).
Rollouts keep a continuous energy, clipped to $[-1000\text{ J}, E_\text{cap}]$ by
`_update_energy_and_state_continuous`, and derive the discrete SoC from it.
