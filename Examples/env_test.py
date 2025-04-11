#!/usr/bin/env python
import numpy as np

# ----------------------------
# Test 1: Direct Provider Sampling
# ----------------------------

# Create dummy data for solar and wind distributions and whale series.
# For each time step, the provider needs two parameters:
# - For solar: alpha and beta (here we use 2.0 and 5.0 for all times).
# - For wind: shape (k) and scale (λ) (here we use 1.5 and 3.0 for all times).
horizon = 10
solar_distributions = np.column_stack((np.full(horizon, 2.0), np.full(horizon, 5.0)))
wind_distributions = np.column_stack((np.full(horizon, 1.5), np.full(horizon, 3.0)))
whale_reward_series = np.linspace(0, 1, horizon)

# Fixed simulation parameters
delta_t_min = 15
seed = 42

# Import your environment provider
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider

# Create first instance of the provider with a dedicated RNG
rng1 = np.random.default_rng(seed)
provider1 = StochasticWindSolarEnvironmentProvider(
    solar_distributions=solar_distributions,
    wind_distributions=wind_distributions,
    whale_reward_series=whale_reward_series,
    delta_t_min=delta_t_min,
    rng=rng1
)

# Sample the environmental data for each time step from provider1
samples1 = []
for t in range(horizon):
    s = provider1.sample_sunlight(t, 1)[0]
    w = provider1.sample_wind_speed(t, 1)[0]
    whale = provider1.sample_whale_observation(t, 1)[0]
    samples1.append((s, w, whale))

# Recreate the provider with the same seed
rng2 = np.random.default_rng(seed)
provider2 = StochasticWindSolarEnvironmentProvider(
    solar_distributions=solar_distributions,
    wind_distributions=wind_distributions,
    whale_reward_series=whale_reward_series,
    delta_t_min=delta_t_min,
    rng=rng2
)

samples2 = []
for t in range(horizon):
    s = provider2.sample_sunlight(t, 1)[0]
    w = provider2.sample_wind_speed(t, 1)[0]
    whale = provider2.sample_whale_observation(t, 1)[0]
    samples2.append((s, w, whale))

print("=== Direct Provider Sampling Test ===")
for i, (s1, s2) in enumerate(zip(samples1, samples2)):
    print(f"Time step {i}:")
    print("  Provider 1:", s1)
    print("  Provider 2:", s2)
    # Check (using np.isclose for floating-point comparisons)
    assert np.isclose(s1[0], s2[0]), "Sunlight samples differ!"
    assert np.isclose(s1[1], s2[1]), "Wind samples differ!"
    assert np.isclose(s1[2], s2[2]), "Whale observation samples differ!"
print("Direct provider sampling test passed.\n")

# ----------------------------
# Test 2: Dummy Simulation Test
# ----------------------------
# We define a minimal dummy simulation that simply collects the environmental samples.
# It inherits from the simulation base class without advancing state or executing transitions.

from BaseClasses.simulation_base import AbstractSimulation

class DummySimulation(AbstractSimulation):
    def choose_action(self, **kwargs):
        # Dummy action (not important for this test)
        return 0

    def simulate_episode(self):
        # Instead of simulating state transitions, simply record the environmental samples.
        solar_samples = []
        wind_samples = []
        whale_samples = []
        for t in range(self.horizon):
            solar = self.env_provider.sample_sunlight(t, 1)[0]
            wind = self.env_provider.sample_wind_speed(t, 1)[0]
            whale = self.env_provider.sample_whale_observation(t, 1)[0]
            solar_samples.append(solar)
            wind_samples.append(wind)
            whale_samples.append(whale)
        return solar_samples, wind_samples, whale_samples

# Create simulation instance 1 with an environment provider built with a fixed seed.
rng_sim1 = np.random.default_rng(seed)
provider_sim1 = StochasticWindSolarEnvironmentProvider(
    solar_distributions=solar_distributions,
    wind_distributions=wind_distributions,
    whale_reward_series=whale_reward_series,
    delta_t_min=delta_t_min,
    rng=rng_sim1
)
dummy_sim1 = DummySimulation(mdp=None, horizon=horizon, initial_state=np.array([0, 0]), env_provider=provider_sim1)
sim_samples1 = dummy_sim1.simulate_episode()

# Create simulation instance 2 with another provider built with the same seed.
rng_sim2 = np.random.default_rng(seed)
provider_sim2 = StochasticWindSolarEnvironmentProvider(
    solar_distributions=solar_distributions,
    wind_distributions=wind_distributions,
    whale_reward_series=whale_reward_series,
    delta_t_min=delta_t_min,
    rng=rng_sim2
)
dummy_sim2 = DummySimulation(mdp=None, horizon=horizon, initial_state=np.array([0, 0]), env_provider=provider_sim2)
sim_samples2 = dummy_sim2.simulate_episode()

print("=== Dummy Simulation Sampling Test ===")
print("Simulation 1 Samples:")
print("  Solar:", sim_samples1[0])
print("  Wind:", sim_samples1[1])
print("  Whale:", sim_samples1[2])
print("Simulation 2 Samples:")
print("  Solar:", sim_samples2[0])
print("  Wind:", sim_samples2[1])
print("  Whale:", sim_samples2[2])

# Check that the two simulation sample arrays are identical.
for arr1, arr2, label in zip(sim_samples1, sim_samples2, ["Solar", "Wind", "Whale"]):
    assert np.allclose(arr1, arr2), f"{label} series do not match!"

print("Dummy simulation sampling test passed.")

if __name__ == "__main__":
    print("\nAll tests passed successfully!")
