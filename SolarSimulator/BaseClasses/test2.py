import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.backward_induction_base import DeterministicMDPBackwardSolver
from BaseClasses.environment_provider_base import DeterministicEnvironmentProvider
from BaseClasses.simulation_base import OptimalPolicySimulation, ObservationThresholdSimulation

# Assume SimulationStorage is defined in simulation_storage.py
from BaseClasses.simulation_storage import SimulationStorage

# ----- Simulation and environment setup -----
horizon = 150

# Dummy parameters for the MDP.
solar_rate_series = np.full(horizon, 4000)  # Constant solar rate.
wind_series = np.full(horizon, 5.0)           # Constant wind speed.
x = np.linspace(0, np.pi, horizon)
whale_reward_series = np.sin(x)               # Half sine wave for whale reward.

battery_capacity_wh = 200 * 60 * 60 * 2 / 3600  # Battery capacity (in Wh).
idle_power = 0                                   # Energy consumption when moored.
cruise_power = 200                               # Energy consumption while flying.
takeoff_power = 200                              # Additional consumption for takeoff.
failure_penalty = 1                              # Penalty for failed transitions.
delta_t = 15                                     # Time step duration (minutes).
gamma = 1.0                                   
transition_model_name = "moderate"
soc_increment = 1.0

# Create a deterministic environment provider.
env_provider = DeterministicEnvironmentProvider(solar_rate_series, wind_series, whale_reward_series, delta_t)

# Instantiate the MDP.
mdp = DeterministicMDP(
    battery_capacity_wh, idle_power, cruise_power, takeoff_power,
    solar_rate_series, wind_series, whale_reward_series,
    failure_penalty, delta_t, gamma, transition_model_name, soc_increment,
    env_provider=env_provider
)

# Define simulation horizon and initial state.
initial_state = np.array([100, 0])  # Full battery and mode 0 (moored).

# ----- Create and solve the backward induction solver -----
solver = DeterministicMDPBackwardSolver(mdp, horizon)


# ----- Set up the simulation storage -----
# Here, we use a small batch size for demonstration.
storage = SimulationStorage(storage_dir="simulation_results", batch_size=2)

# ----- Run multiple simulation episodes and store them -----
simulations = [
    OptimalPolicySimulation(solver, horizon, initial_state, env_provider=env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.0, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.5, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.9, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.99, 10, env_provider)
]

for sim in simulations:
    trajectory, actions, rewards = sim.simulate_episode()
    # Include simulation metadata for identification.
    episode_data = {
        'metadata': {
            'simulation_type': sim.__class__.__name__,
            # Include parameters specific to ObservationThresholdSimulation.
            'observation_threshold': getattr(sim, 'observation_threshold', None),
            'wind_threshold': getattr(sim, 'wind_threshold', None),
        },
        'trajectory': trajectory,
        'actions': actions,
        'rewards': rewards,
        'total_reward': sum(rewards)
    }
    storage.store_episode(episode_data)
    
    print("Simulation Episode:")
    print("Metadata:", episode_data['metadata'])
    print("Total Reward:", episode_data['total_reward'])
    print("-" * 40)

# Flush any remaining episodes in the buffer to disk.
storage.flush_buffer()

# ----- Load and inspect stored episodes -----
all_episodes = storage.load_all_episodes()
print(f"Loaded {len(all_episodes)} episodes from storage.")
for idx, ep in enumerate(all_episodes):
    print(f"Episode {idx} ({ep['metadata']['simulation_type']}): Total Reward = {ep['total_reward']}")