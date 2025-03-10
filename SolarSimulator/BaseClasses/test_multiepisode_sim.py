import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.environment_provider_base import DeterministicEnvironmentProvider
from BaseClasses.simulation_base import ObservationThresholdSimulation
from BaseClasses.simulation_file_io_base import SimulationStorage  # The file I/O class


# ----- Simulation and environment setup -----
horizon = 100

# Dummy parameters for the MDP.
solar_rate_series = np.full(horizon, 4000)  # Constant solar rate.
wind_series = np.full(horizon, 5.0)           # Constant wind speed.
x = np.linspace(0, np.pi, horizon)
whale_reward_series = np.sin(x)               # Half sine wave for whale reward.

battery_capacity_wh = 200 * 60 * 60 * 2 / 3600  # Battery capacity (in Wh).
idle_power = 0                                  # Energy consumption when moored.
cruise_power = 200                              # Energy consumption while flying.
takeoff_power = 200                             # Additional consumption for takeoff.
failure_penalty = 1                             # Penalty for failed transitions.
delta_t = 15                                    # Time step duration (minutes).
gamma = 1.0                                   
transition_model_name = "nofail"
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

# Define the initial state.
initial_state = np.array([100, 0])  # Full battery and mode 0 (moored).

# ----- Set up the simulation storage -----
# For demonstration, use a small batch size.
storage = SimulationStorage(storage_dir="simulation_results", batch_size=5)

# ----- Simulation configurations -----
# Each tuple represents (observation_threshold, wind_threshold)
simulation_params = [
    (0.5, 10),
    (0.0, 10),
    (0.9, 10),
    (0.99, 10)
]

# Specify the number of episodes to run per simulation instance.
episodes_per_simulation = 1000

# ----- Run simulations using the built-in multiple episodes method -----
for obs_threshold, wind_threshold in simulation_params:
    # Create a simulation instance with specific parameters.
    sim = ObservationThresholdSimulation(mdp, horizon, initial_state, obs_threshold, wind_threshold, env_provider)
    # Run multiple episodes using the simulation's method.
    episodes = sim.simulate_multiple_episodes(episodes_per_simulation)
    
    # Process and store each episode with added metadata.
    for episode_index, ep in enumerate(episodes):
        # Attach metadata to help later identify the source of the episode.
        ep['metadata'] = {
            'simulation_type': sim.__class__.__name__,
            'observation_threshold': obs_threshold,
            'wind_threshold': wind_threshold,
            'episode_index': episode_index,
        }
        storage.store_episode(ep)
        print(f"Stored {sim.__class__.__name__} episode {episode_index} "
              f"(obs_threshold={obs_threshold}, wind_threshold={wind_threshold}) "
              f"with total reward: {ep.get('total_reward', sum(ep.get('rewards', [])))}")

# Flush any remaining episodes in the buffer to disk.
storage.flush_buffer()

# ----- Loading and Inspecting Episodes -----
all_episodes = storage.load_all_episodes()
print(f"\nLoaded {len(all_episodes)} episodes from storage.")
for idx, ep in enumerate(all_episodes):
    meta = ep['metadata']
    print(f"Episode {idx} ({meta['simulation_type']} - "
          f"obs_threshold: {meta['observation_threshold']}, "
          f"episode index: {meta['episode_index']}): Total Reward = {ep['total_reward']}")
