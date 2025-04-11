import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

# Import the necessary classes from your base modules.
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider as EnvProv
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import OptimalContinuousAnalyticalPolicySimulation, ObservationThresholdContinuousSimulation
from BaseClasses.whale_base import WhaleRewardSeriesFactory

# Import the SimulationRunManager from the simulation_run_manager module.
from BaseClasses.simulation_run_manager import SimulationRunManager

def main():
    # -------------------------------
    # Configuration parameters
    # -------------------------------
    battery_capacity = 400.0  # in Wh
    horizon = 1000            # number of time steps per episode
    initial_state = np.array([100.0, 0])  # [SoC, mode] where mode 0 = safe, mode 2 = failure

    # -------------------------------
    # Build environment distributions
    # -------------------------------
    data_path = r"Data/EXPECTED_DATA/data_expected_lat0_lon-90_15min.pkl"
    data = pd.read_pickle(data_path)
    wind_shape = data['weibull_k'].values[:horizon]
    wind_scale = data['weibull_scale'].values[:horizon]
    solar_alpha = data['beta_alpha'].values[:horizon]
    solar_beta = data['beta_beta'].values[:horizon]
    
    # Create distribution arrays.
    wind_distributions = np.column_stack((wind_shape, wind_scale))
    solar_distributions = np.column_stack((solar_alpha, solar_beta))
    whale_reward_series = WhaleRewardSeriesFactory.create_series("real", horizon)

    # -------------------------------
    # Create the environment provider
    # -------------------------------
    env_provider = EnvProv(
        solar_distributions=solar_distributions,
        wind_distributions=wind_distributions,
        whale_reward_series=whale_reward_series,
        delta_t_min=15
    )

    # -------------------------------
    # Set up the seaplane and MDP power parameters
    # -------------------------------
    seaplane = Seaplane(30, -90, "none", capacity=battery_capacity / 22.2)
    power_params = seaplane.get_mdp_power_params()

    # -------------------------------
    # Instantiate the MDP and backward solver
    # -------------------------------
    mdp = stochasticMDP(
        battery_capacity_wh=battery_capacity,
        idle_power=power_params["idle_power"],
        cruise_power=power_params["cruise_power"],
        takeoff_power=power_params["takeoff_power"],
        failure_penalty=15,
        delta_t=15,
        gamma=1.0,
        transition_model_name="moderate",
        soc_increment=1.0,
        env_provider=env_provider
    )
    solver = mdpBackwardSolver(mdp, horizon)
    solver.solve()

    # -------------------------------
    # Create the simulation objects
    # -------------------------------
    # Both simulation objects should implement simulate_multiple_episodes(num_episodes).
    # optimal_policy_sim = OptimalPolicySimulation(
    #     mdp_solver=solver,
    #     horizon=horizon,
    #     initial_state=initial_state,
    #     env_provider=env_provider
    # )
    optimal_analytical_sim = OptimalContinuousAnalyticalPolicySimulation(
        mdp_solver=solver,
        horizon=horizon,
        initial_state=initial_state,
        env_provider=env_provider
    )
    simulation_list = [optimal_analytical_sim]
    for i in range(5):
        for j in range(5):
            i=i+1
            j=j+1
            simulation_list.append(
                ObservationThresholdContinuousSimulation(
                mdp=mdp,
                horizon=horizon,
                initial_state=initial_state,
                observation_threshold=0.3/j,
                wind_threshold=i*2.,
                env_provider=env_provider
            ))
        
    # -------------------------------
    # Set up the SimulationRunManager to run and store episodes
    # -------------------------------
    
    # Here we choose to run 5000 episodes per simulation.
    # The storage_dir will be used to save simulation data (handled by SimulationStorage within the manager).
    episodes_per_simulation = 5000
    storage_dir = "simulation_results_testing_penal3"
    sim_manager = SimulationRunManager(episodes_per_simulation, storage_dir)
    
    # Optionally, you can enable multiprocessing by setting use_multiprocessing=True.
    # If you do not wish to use multiprocessing, leave it as False.
    sim_manager.run_simulations(simulation_list, use_multiprocessing=True)
    
    print(f"Simulations complete. Results stored in directory: {storage_dir}")

if __name__ == "__main__":
    main()
