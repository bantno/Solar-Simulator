import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
# Import your AbstractSimulation from wherever you defined it.
from BaseClasses.simulation_base import AbstractSimulation  

# Define a concrete simulation subclass.
class AlwaysFlySimulation(AbstractSimulation):
    def choose_action(self, state: np.ndarray, t: int) -> int:
        """
        A simple policy that always chooses to fly (action 1).
        """
        return 1

class AlwaysFloatSimulation(AbstractSimulation):
    def choose_action(self, state: np.ndarray, t: int) -> int:
        """
        A simple policy that always chooses to fly (action 1).
        """
        return 0

# ----- Setup dummy parameters for the MDP -----

# Create simple time series arrays for solar and wind data (10 time steps each).
solar_rate_series = np.full(10, 100000)       # Constant solar rate of 0.5
wind_series = np.full(10, 5.0)               # Constant wind speed of 5.0

battery_capacity_wh = 200*60*60*5/3600        # Battery capacity in watt-hours.
idle_power = 0                  # Power consumption (when moored) per time step.
cruise_power = 200               # Power consumption while flying.
takeoff_power = 200              # Additional power consumption for takeoff.
whale_reward_series = np.full(10, 1)       # Constant whale reward of 100
failure_penalty = 1000                       # Penalty for a failed transition
delta_t = 15                                 # Duration of each time step in minutes
gamma = 1.0                                  # Discount factor
transition_model_name = "moderate"           # Name of the transition model to use
soc_increment = 5.0                          # Increment for state-of-charge (SOC)

# ----- Instantiate the MDP -----
mdp = DeterministicMDP(
    battery_capacity_wh, idle_power, cruise_power, takeoff_power,
    solar_rate_series, wind_series, whale_reward_series,
    failure_penalty, delta_t, gamma, transition_model_name, soc_increment
)

# Define an initial state for the simulation.
# Here, the state is represented as [SoC, mode]: full battery (100%) and mode 0 (moored).
initial_state = np.array([100, 0])

# Set the simulation horizon (e.g., 10 time steps).
horizon = 10

# ----- Instantiate and run the simulation -----
sim = AlwaysFlySimulation(mdp, horizon, initial_state)

# Run a single simulation episode.
trajectory, actions, rewards = sim.simulate_episode()
print("Single Episode Simulation:")
print("Trajectory:", trajectory)
print("Actions:", actions)
print("Rewards:", rewards)

# Run multiple episodes (e.g., 3 episodes).
episodes = sim.simulate_multiple_episodes(3)
print("\nMultiple Episodes Simulation:")
for idx, ep in enumerate(episodes):
    print(f"Episode {idx + 1}:")
    print("Trajectory:", ep['trajectory'])
    print("Actions:", ep['actions'])
    print("Rewards:", ep['rewards'])

initial_state = np.array([20, 0])
sim = AlwaysFloatSimulation(mdp, horizon, initial_state)
# Run a single simulation episode.
trajectory, actions, rewards = sim.simulate_episode()
print("Single Episode Simulation:")
print("Trajectory:", trajectory)
print("Actions:", actions)
print("Rewards:", rewards)

# Run multiple episodes (e.g., 3 episodes).
episodes = sim.simulate_multiple_episodes(3)
print("\nMultiple Episodes Simulation:")
for idx, ep in enumerate(episodes):
    print(f"Episode {idx + 1}:")
    print("Trajectory:", ep['trajectory'])
    print("Actions:", ep['actions'])
    print("Rewards:", ep['rewards'])
