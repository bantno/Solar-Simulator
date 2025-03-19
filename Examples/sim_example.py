import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.simulation_base import AlwaysFlySimulation, AlwaysFloatSimulation
from BaseClasses.environment_provider_base import DeterministicEnvironmentProvider

# ----- Setup dummy parameters for the MDP -----
solar_rate_series = np.full(10, 50)      # Constant solar rate
wind_series = np.full(10, 5.0)                 # Constant wind speed
whale_reward_series = np.full(10, 1)           # Constant whale reward

battery_capacity_wh = 200 * 60 * 60 * 2 / 3600  # Battery capacity in watt-hours.
idle_power = 0                                  # Consumption when moored.
cruise_power = 200                              # Consumption while flying.
takeoff_power = 200                             # Additional consumption for takeoff.
failure_penalty = 1000                          # Penalty for failed transition.
delta_t = 15                                    # Duration (minutes) per time step.
gamma = 1.0                                   # Discount factor.
transition_model_name = "moderate"              # Transition model name.
soc_increment = 5.0                             # Increment for SoC.

# Create a deterministic environment provider.
env_provider = DeterministicEnvironmentProvider(solar_rate_series, wind_series, whale_reward_series, delta_t)

# ----- Instantiate the MDP with the environment provider -----
mdp = DeterministicMDP(
    battery_capacity_wh, idle_power, cruise_power, takeoff_power,
    solar_rate_series, wind_series, whale_reward_series,
    failure_penalty, delta_t, gamma, transition_model_name, soc_increment,
    env_provider=env_provider
)

horizon = 10
initial_state = np.array([100, 0])  # [SoC, mode]

# ----- Simulation with AlwaysFlySimulation -----
sim = AlwaysFlySimulation(mdp, horizon, initial_state, env_provider=env_provider)
trajectory, actions, rewards = sim.simulate_episode()
print("AlwaysFlySimulation Episode:")
print("Trajectory:", trajectory)
print("Actions:", actions)
print("Rewards:", rewards)

# ----- Simulation with AlwaysFloatSimulation -----
initial_state = np.array([20, 0])
sim_float = AlwaysFloatSimulation(mdp, horizon, initial_state, env_provider=env_provider)
trajectory, actions, rewards = sim_float.simulate_episode()
print("\nAlwaysFloatSimulation Episode:")
print("Trajectory:", trajectory)
print("Actions:", actions)
print("Rewards:", rewards)
