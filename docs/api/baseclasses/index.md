# BaseClasses

The core library. Every other package (harness, Scripts, Tests) builds on these modules.

| Module | Purpose |
|---|---|
| [`mdp_base`](mdp_base.md) | MDP definition: state space, actions, reward, observation gain. |
| [`backward_induction_base`](backward_induction_base.md) | Backward-induction value-function solver (iid and chain variants). |
| [`transition_model_base`](transition_model_base.md) | Action success/failure probability models and transition logic. |
| [`environment_provider_base`](environment_provider_base.md) | Weather providers: fitted-distribution, chain, historical bootstrap. |
| [`simulation_base`](simulation_base.md) | Policy simulations and batched episode rollout. |
| [`simulation_run_manager`](simulation_run_manager.md) | Runs a list of simulations (optionally in parallel) into one HDF5 file. |
| [`simulation_storage`](simulation_storage.md) | HDF5 result storage (columnar scalars + optional full histories). |
| [`run_sim`](run_sim.md) | YAML config loading, EnvironmentLoader, SimulationFactory, YAMLSimulationRunner. |
| [`seaplane_base`](seaplane_base.md) | Seaplane power, climb and maneuver model. |
| [`solar_panel_base`](solar_panel_base.md) | Solar panel models: constant, PVWatts, variable. |
| [`whale_base`](whale_base.md) | Whale observation-probability series and the encounter model. |
| [`clearsky_models`](clearsky_models.md) | Pluggable clear-sky GHI models (fatemi, haurwitz, ineichen). |
| [`weather_processor`](weather_processor.md) | Open-Meteo fetch and distribution fitting (original pipeline). |
| [`weather_processor_cs_normalization`](weather_processor_cs_normalization.md) | Clear-sky-normalized distribution fitting and EXPECTED_DATA artifact builder. |
| [`plot_base`](plot_base.md) | Result plotting helpers (matplotlib/plotly). |
| [`plotting_utils_base`](plotting_utils_base.md) | Plotting utilities for value tables and episodes. |
