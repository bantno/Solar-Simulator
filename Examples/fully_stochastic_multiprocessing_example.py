import multiprocessing
import numpy as np
import pandas as pd
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider as EnvProv
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.simulation_base import ObservationThresholdSimulation, OptimalPolicySimulation
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.simulation_run_manager import SimulationRunManager
from BaseClasses.seaplane_base import Seaplane

def create_simulation(sim_type, battery_capacity, threshold, horizon, data_path):
    """
    Create a simulation based on sim_type:
      - "threshold" creates an ObservationThresholdSimulation with a given threshold.
      - "optimal" creates an OptimalSimulation.
    """
    # Load data for environment distributions
    data = pd.read_pickle(data_path)
    wind_shape = data['weibull_k'].values[:horizon]
    wind_scale = data['weibull_scale'].values[:horizon]
    wind_distributions = np.column_stack((wind_shape, wind_scale))
    solar_alpha = data['beta_alpha'].values[:horizon]
    solar_beta = data['beta_beta'].values[:horizon]
    solar_distributions = np.column_stack((solar_alpha, solar_beta))
    x = np.linspace(np.pi, np.pi * 60, horizon)
    whale_reward_series = 0.5 * np.sin(x) + 0.5

    # Create environment provider
    env_provider = EnvProv(
        solar_distributions=solar_distributions,
        wind_distributions=wind_distributions,
        whale_reward_series=whale_reward_series,
        delta_t=15
    )
    
    # Create seaplane instance
    seaplane = Seaplane(30,
                        -90,
                        "none",
                        capacity=battery_capacity/22.2)

    # Get power parameters for the MDP
    power_params = seaplane.get_mdp_power_params()
    print(power_params)

    # Create the MDP
    mdp = stochasticMDP(
        battery_capacity_wh=battery_capacity,
        idle_power=power_params["idle_power"],
        cruise_power=power_params["cruise_power"],
        takeoff_power=power_params["takeoff_power"],
        failure_penalty=5,
        delta_t=15,
        gamma=1.0,
        transition_model_name="moderate",
        soc_increment=1.0,
        env_provider=env_provider
    )
    
    initial_state = np.array([100.0, 0])
    if sim_type == "threshold":
        # Build a threshold simulation
        sim = ObservationThresholdSimulation(
            mdp=mdp,
            horizon=horizon,
            initial_state=initial_state,
            observation_threshold=threshold,
            wind_threshold=5,
            env_provider=env_provider
        )
    elif sim_type == "optimal":
        # Build an optimal simulation
        solver = mdpBackwardSolver(mdp, horizon)
        sim = OptimalPolicySimulation(
            mdp_solver=solver,
            horizon=horizon,
            initial_state=initial_state,
            env_provider=env_provider
        )
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")
    
    return sim

def build_param_list(battery_capacities, threshold_values, horizon, data_path):
    """
    Return a list of parameter tuples for all battery capacities.
    For each battery capacity, create simulations for each threshold and one optimal simulation.
    """
    params = []
    for bc in battery_capacities:
        # Create threshold simulations
        for th in threshold_values:
            params.append(("threshold", bc, th, horizon, data_path))
        # Create one optimal simulation for each battery capacity
        params.append(("optimal", bc, None, horizon, data_path))
    return params

if __name__ == "__main__":
    # Define the parameter ranges and common values
    # battery_capacities = [400,600,800,1000,1200,1400]
    # threshold_values = [0.0, 0.5, 0.9]
    battery_capacities = [640]
    threshold_values = []
    horizon = 300
    data_path = r"Data\EXPECTED_DATA\data_expected_lat0_lon-90_15min.pkl"

    # Build a list of parameter tuples
    param_list = build_param_list(battery_capacities, threshold_values, horizon, data_path)

    # Create the simulations in parallel using starmap
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        simulations = pool.starmap(create_simulation, param_list)
    
    print(f"Created {len(simulations)} simulation objects.")

    # Use the SimulationRunManager to run these simulations
    run_manager = SimulationRunManager(episodes_per_simulation=10, storage_dir="simulation_results")
    # Optionally use multiprocessing again for running simulations
    run_manager.run_simulations(simulations, use_multiprocessing=False, num_workers=4)
