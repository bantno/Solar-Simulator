# Clear-sky models

Solar input is modelled as a **clear-sky index** $K = G / G_\text{cs}(t)$, the measured GHI
divided by a clear-sky envelope. $K$ is fitted with a per-stage Beta distribution, or modelled as
a Markov chain when the [solar chain](solar-chain.md) is enabled. At run time the sampled $K$ is
multiplied back by $G_\text{cs}(t)$ to give GHI. `StochasticWindSolarEnvironmentProvider.beta_solar_energy_dist`
draws `rng.beta(alpha[t], beta[t]) * solar_cs[t]`.

`SolarSimulator/BaseClasses/clearsky_models.py` ([API](../api/baseclasses/clearsky_models.md))
is the only place $G_\text{cs}$ is computed. The YAML key `clearsky_model` selects the model.

## Models

All three models use pvlib solar position (apparent zenith $z$), clip to $\ge 0$, and are zero at
night.

| `clearsky_model` | Formula | Notes |
|---|---|---|
| `fatemi` (default) | $G_\text{cs} = A\,\max(\cos z, 0)$, $A = 1150$ W/m² | Fatemi–Kuh–Fripp (2018); the original model. Loosest envelope at noon |
| `haurwitz` | `pvlib.clearsky.haurwitz(z)` | Empirical GHI from zenith only |
| `ineichen` | `pvlib.location.Location(lat, lon, altitude=0).get_clearsky(model="ineichen")` | Ineichen–Perez with Linke-turbidity lookup. Tightest envelope, keeping $K \lesssim 1$ |

The function is `clearsky_ghi_series(times, lat, lon, model="fatemi", A=1150.0) -> np.ndarray`.
Unknown model names raise `ValueError`. `CLEARSKY_MODELS` lists the valid names.

## Where the envelope is used

Every stage of the solar pipeline must use the same model:

| Consumer | Code |
|---|---|
| Per-stage Beta fits and `clearsky_irradiance` in the expected-data pickle | `WeatherDataProcessor._fit_beta` in `weather_processor_cs_normalization.py` (`build_expected_data_artifact(..., clearsky_model=...)`) |
| Solar-chain fitting | `build_solar_chain_artifact(..., clearsky_model=...)` in `Scripts/create_weather_distributions.py` |
| Run-time `solar_cs` | Read from the expected-data pickle by `EnvironmentLoader` |

The Beta fit runs only when $G_\text{cs} > 50$ W/m². Observations are normalized by $G_\text{cs}$
and fitted by method of moments. Stages where every observation is at or above clear sky get a
near-degenerate Beta just below 1 ($\kappa = 10^6$).

## Artifact files per model

Artifacts normalized with different envelopes cannot be swapped for one another. Commit `76812fa`
therefore keeps them in separate files.

`run_sim._resolve_data_path(data_path, clearsky_model)` handles the naming:

- **`fatemi`:** the configured `data_path` is used unchanged, so the original pipeline stays byte-for-byte identical.
- **Any other model:** `_<model>` is inserted before the extension.
- **Derived chain and cube artifacts:** the `_windchain`, `_solarchain` and `_histcube` files inherit the suffix.

```text
Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min.pkl                 # fatemi
Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min_ineichen.pkl        # ineichen
Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min_ineichen_solarchain.pkl
```

Harness provisioning (`_ensure_location_data` in `harness/run_experiment.py`) resolves the suffixed
path and builds a missing expected-data artifact on demand from the location's historical pickle.
It also rebuilds an existing solar-chain artifact whose stored `clearsky_model` differs from the
configured one. The fatemi artifacts are never overwritten.

## Choosing a model

```yaml
clearsky_model: ineichen
solar_panel_model: pvwatts      # optional; independent switch
```

`harness/examples/pvlib_small.yaml` combines both switches (see [Example configs](../experiments/examples.md)).
The model changes the shape of $G_\text{cs}(t)$ and therefore the fitted $K$ distributions.

## Verification

`Tests/verify_pvlib_solar.py` checks four things:

- The fatemi path reproduces the original $A\cos z$ formula, and the `clearsky_irradiance` values stored in the existing expected-data pickle, exactly.
- haurwitz and ineichen are zero at night and positive at midday.
- Both sit below fatemi at noon.
- `_resolve_data_path` leaves fatemi paths unchanged and suffixes the others.

See [Verification](../verification/index.md).
