import numpy as np
import pandas as pd
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.simulation_base import OptimalSimulation, ObservationThresholdSimulation
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider as sep
from BaseClasses.simulation_run_manager import SimulationRunManager

if __name__ == "__main__":
    # ----- Define dummy data -----

    # Create wind_distributions array with shape (T, 2):
    # For each time step t, the first value is the Weibull shape (k)
    # and the second value is the Weibull scale (λ), here varying with a diurnal pattern.

    horizon = 10
    battery_capacity_wh = 844
    idle_power = 0
    cruise_power = 200
    takeoff_power = 200
    failure_penalty = 5
    delta_t = 15
    gamma = 1.0
    transition_model_name = "moderate"
    soc_increment = 1.0

    # Define weather distrubution for wind
    data = pd.read_pickle(rf"Data\EXPECTED_DATA\data_expected_lat0_lon-90_15min.pkl")
    wind_shape = data['weibull_k'].values[:horizon]
    wind_scale = data['weibull_scale'].values[:horizon]
    wind_distributions = np.column_stack((wind_shape, wind_scale))
    
    # Define weather distribution for solar
    solar_alpha = data['beta_alpha']
    solar_beta = data['beta_beta']
    solar_distributions = np.column_stack((solar_alpha,solar_beta))

    x = np.linspace(np.pi, np.pi*30, horizon)
    whale_reward_series = 0.5*np.sin(x)+0.5
    

    # ----- Instantiate the custom environment provider -----
    env_provider = sep(
        solar_distributions=solar_distributions,
        wind_distributions=wind_distributions,
        whale_reward_series=whale_reward_series,
        delta_t=delta_t
    )

    # ----- Instantiate the MDP using the stochastic environment provider -----
    mdp = stochasticMDP(
        battery_capacity_wh=battery_capacity_wh,
        idle_power=idle_power,
        cruise_power=cruise_power,
        takeoff_power=takeoff_power,
        failure_penalty=failure_penalty,
        delta_t=delta_t,
        gamma=gamma,
        transition_model_name=transition_model_name,
        soc_increment=soc_increment,
        env_provider=env_provider
    )

    # ----- Setup and run a simulation episode -----
    # Define an initial state, for example, 50% battery (SoC) and mode 0.
    initial_state = np.array([100.0, 0])

    # ----- Instantiate the Backward Induction Solver -----
    mdp_solver = mdpBackwardSolver(mdp, horizon)

    # ----- Instantiate the Optimal Simulation -----
    optimal_simulation = OptimalSimulation(
        mdp_solver=mdp_solver,
        horizon=horizon,
        initial_state=initial_state,
        env_provider=env_provider
    )

    # # ----- Run a Simulation Episode -----
    # trajectory, actions, rewards, solar_samples, wind_samples, whale_samples = optimal_simulation.simulate_episode()

    # # ----- Output the Results -----
    # print("Trajectory:", trajectory)
    # print("Actions:", actions)
    # print("Rewards:", rewards)
    # print("Solar Samples:", solar_samples)
    # print("Wind Samples:", wind_samples)
    # print("Whale Samples:", whale_samples)
    # print("Total Reward: ", sum(rewards))

    # ----- Create Simulation Instances -----
    sim_opt = optimal_simulation

    # A list of simulation instances (they can be of different types).
    simulation_list = [
        ObservationThresholdSimulation(mdp, horizon, initial_state, 0.0, 5, env_provider),
        ObservationThresholdSimulation(mdp, horizon, initial_state, 0.5, 5, env_provider),
        ObservationThresholdSimulation(mdp, horizon, initial_state, 0.9, 5, env_provider),
        sim_opt
    ]

    # ----- Set Up and Run the SimulationRunManager -----
    episodes_per_simulation = 5000  # Number of episodes per simulation run
    run_manager = SimulationRunManager(episodes_per_simulation, storage_dir="simulation_results")

    # Run all simulations provided in the list. Each simulation run is stored as a batch.
    run_manager.run_simulations(simulation_list,True,4)