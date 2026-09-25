# Solar panel models

The panel turns irradiance into battery energy. Every model is a **linear** map from GHI to energy,
which the solver's integration over the clear-sky index relies on (commit `76812fa`):

$$
E_\text{gain} = G \cdot \Delta t_\text{s} \cdot A \cdot \eta
$$

- $G$ is the sampled GHI in W/m². It is treated as plane-of-array, with the panel horizontal.
- $\Delta t_\text{s}$ is the stage length in seconds.
- $A$ is `panel.area` and $\eta$ is `panel.efficiency`.

The environment providers compute this in `_energy_gain_from_solar` (`environment_provider_base.py`).

The code is in `SolarSimulator/BaseClasses/solar_panel_base.py`
([API](../api/baseclasses/solar_panel_base.md)).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `solar_panel_model` | `constant` | Registered model name |
| `solar_panel_kwargs` | none | Dict passed to the model constructor |

`SimulationFactory` passes both keys to every environment provider it builds: the distributional
provider used for solving, and the historical-bootstrap provider used for rollouts when that is
enabled. Each provider then calls `SolarPanelFactory.create_solar_panel(model, **kwargs)`.

```yaml
solar_panel_model: pvwatts
solar_panel_kwargs:
  cell_temperature: 45     # degC; PVWatts thermal derate applies above 25
```

## Models

| Name | Class | Constructor defaults | Efficiency |
|---|---|---|---|
| `constant` | `ConstantSolarPanel` | `area=0.65`, `efficiency=0.1` | fixed |
| `pvwatts` | `PVWattsSolarPanel` | `area=0.65`, `pdc0=65.0`, `gamma_pdc=-0.003`, `cell_temperature=25.0` | derived once from PVWatts |
| `variable` | `VariableEfficiencySolarPanel` | `area=0.65`, `base_efficiency=0.1`, `degradation=0.001` | falls on each access |

### `pvwatts`

`PVWattsSolarPanel` evaluates `pvlib.pvsystem.pvwatts_dc` once, at 1000 W/m² and the configured
cell temperature, and folds the result into an effective efficiency:

$$
P_\text{dc} = p_{dc0}\,\frac{G}{1000}\,\big[1 + \gamma_{pdc}(T_\text{cell} - 25)\big],
\qquad
\eta_\text{eff} = \frac{P_\text{dc}(1000,\,T_\text{cell})}{1000\,A}
$$

With the defaults, $\eta_\text{eff} = 65/(1000 \cdot 0.65) = 0.1$, so it **reproduces the
`constant` panel exactly**. At $T_\text{cell} = 45\,°$C, $\eta_\text{eff} = 0.094$ (a 6 % derate).
Cell temperature is fixed for the run, not modelled from irradiance and wind, so the
GHI → energy map stays linear.

### `variable`

`efficiency` returns $\max(0,\ \eta_0 - d\cdot n)$ and increments an internal counter $n$ each
time the property is read, so efficiency drops with every access.

## Registry

`SolarPanelMeta` registers every concrete subclass that sets a `model_name` with
`SolarPanelFactory` when the class is defined. To add a panel:

```python
class MyPanel(SolarPanel):
    model_name = "mypanel"
    def __init__(self, area=0.65, ...): ...
    @property
    def area(self): ...
    @property
    def efficiency(self): ...
```

Once the class is imported, `solar_panel_model: mypanel` resolves to it. Keep $E_\text{gain}$ linear
in $G$, because the solver integrates energy gain over the Beta-distributed clear-sky index in
closed form.

## Verification

`Tests/verify_pvlib_solar.py` checks three things:

- The `pvwatts` defaults equal the `constant` panel.
- A hotter cell derates by $1 + \gamma_{pdc}(T-25)$.
- Keyword arguments pass through the factory, and a provider built with `pvwatts` defaults
  produces bit-identical `sample_sunlight` draws to `constant` at the same seed.

See [Verification](../verification/index.md).

## Related

- [Clear-sky models](clearsky.md): the $G_\text{cs}$ envelope that scales the sampled clear-sky index into GHI.
- [Aircraft power model](aircraft-power.md): the consumption side of the energy balance.
