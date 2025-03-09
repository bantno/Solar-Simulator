import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.backward_induction_base import DeterministicMDPBackwardSolver
from BaseClasses.environment_provider_base import DeterministicEnvironmentProvider
from BaseClasses.simulation_base import OptimalPolicySimulation

horizon = 30

# ----- Setup dummy parameters for the MDP -----
solar_rate_series = np.full(horizon, 4000)      # Constant solar rate for 10 time steps
wind_series = np.full(horizon, 5.0)                 # Constant wind speed

x = np.linspace(0, np.pi, horizon)

# Compute the sine values for x, which forms a half sine wave
whale_reward_series = np.sin(x)
print(whale_reward_series)
# whale_reward_series = np.full(horizon, 1)           # Constant whale reward

battery_capacity_wh = 200 * 60 * 60 * 2 / 3600  # Battery capacity in watt-hours.
idle_power = 0                                  # Energy consumption when moored.
cruise_power = 200                              # Energy consumption while flying.
takeoff_power = 200                             # Additional consumption for takeoff.
failure_penalty = 1                          # Penalty for failed transitions.
delta_t = 15                                    # Duration per time step (minutes).
gamma = 1.0                                   # Discount factor.
transition_model_name = "nofail"              # Name of the transition model.
soc_increment = 1.0                             # SoC increment.

# ----- Create a deterministic environment provider -----
env_provider = DeterministicEnvironmentProvider(solar_rate_series, wind_series, whale_reward_series, delta_t)

# ----- Instantiate the MDP with the environment provider -----
mdp = DeterministicMDP(
    battery_capacity_wh, idle_power, cruise_power, takeoff_power,
    solar_rate_series, wind_series, whale_reward_series,
    failure_penalty, delta_t, gamma, transition_model_name, soc_increment,
    env_provider=env_provider
)

# ----- Define simulation horizon and initial state -----

initial_state = np.array([100, 0])  # full battery (SoC 100) and mode 0 (moored)

# ----- Create and solve the backward induction solver -----
solver = DeterministicMDPBackwardSolver(mdp, horizon)

# ----- Instantiate the OptimalPolicySimulation using the solver -----
optimal_sim = OptimalPolicySimulation(solver, horizon, initial_state, env_provider=env_provider)

# ----- Run a simulation episode -----
trajectory, actions, rewards = optimal_sim.simulate_episode()

print("Optimal Policy Simulation Episode:")
print("Trajectory:", trajectory)
print("Actions:", actions)
print("Rewards:", rewards)
print("Total Reward:", sum(rewards))
