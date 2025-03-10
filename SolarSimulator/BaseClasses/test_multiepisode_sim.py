import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.environment_provider_base import DeterministicEnvironmentProvider
from BaseClasses.simulation_base import ObservationThresholdSimulation, OptimalPolicySimulation
from BaseClasses.simulation_run_manager import SimulationRunManager
from BaseClasses.backward_induction_base import DeterministicMDPBackwardSolver

# ----- Simulation and Environment Setup -----
horizon = 100
solar_rate_series = np.full(horizon, 4000)
wind_series = np.full(horizon, 5.0)
x = np.linspace(0, np.pi*12, horizon)
whale_reward_series = np.sin(x)
solar_rate_series = np.clip(np.sin(x)*4000,0,4000)

battery_capacity_wh = 200 * 60 * 60 * 4 / 3600
idle_power = 0
cruise_power = 200
takeoff_power = 200
failure_penalty = 1
delta_t = 15
gamma = 1.0
transition_model_name = "nofail"
soc_increment = 1.0

env_provider = DeterministicEnvironmentProvider(solar_rate_series, wind_series,
                                                whale_reward_series, delta_t)

mdp = DeterministicMDP(
    battery_capacity_wh, idle_power, cruise_power, takeoff_power,
    solar_rate_series, wind_series, whale_reward_series,
    failure_penalty, delta_t, gamma, transition_model_name, soc_increment,
    env_provider=env_provider
)

initial_state = np.array([100, 0])

# ----- Create Simulation Instances -----
# solver = DeterministicMDPBackwardSolver(mdp, horizon)
# sim_opt = OptimalPolicySimulation(solver, horizon, initial_state, env_provider)

# A list of simulation instances (they can be of different types).
# simulation_list = [sim_obs, sim_opt]
# simulation_list = [sim_obs]
simulation_list = [
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.0, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.1, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.2, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.3, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.4, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.5, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.6, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.7, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.8, 10, env_provider),
    ObservationThresholdSimulation(mdp, horizon, initial_state, 0.9, 10, env_provider),
    # sim_opt
]

# ----- Set Up and Run the SimulationRunManager -----
episodes_per_simulation = 1  # Number of episodes per simulation run
run_manager = SimulationRunManager(episodes_per_simulation, storage_dir="simulation_results")

# Run all simulations provided in the list. Each simulation run is stored as a batch.
run_manager.run_simulations(simulation_list)
