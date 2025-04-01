import argparse
import yaml
import multiprocessing
import numpy as np
import pandas as pd
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider as EnvProv
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.simulation_base import ObservationThresholdSimulation, OptimalPolicySimulation
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.simulation_run_manager import SimulationRunManager
from BaseClasses.seaplane_base import Seaplane

def create_simulation(sim_type, battery_capacity, threshold, wind_threshold, horizon, data_path):
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
    seaplane = Seaplane(30, -90, "none", capacity=battery_capacity/22.2)
    power_params = seaplane.get_mdp_power_params()
    print(power_params)

    # Create the MDP
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
    
    initial_state = np.array([100.0, 0])
    if sim_type == "threshold":
        sim = ObservationThresholdSimulation(
            mdp=mdp,
            horizon=horizon,
            initial_state=initial_state,
            observation_threshold=threshold,
            wind_threshold=wind_threshold,
            env_provider=env_provider
        )
    elif sim_type == "optimal":
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

def build_param_list(battery_capacities, threshold_values, wind_thresholds, horizon, data_path):
    params = []
    for bc in battery_capacities:
        for th in threshold_values:
            for w_th in wind_thresholds:
                params.append(("threshold", bc, th, w_th, horizon, data_path))
        params.append(("optimal", bc, None, None, horizon, data_path))
    return params

def main(config_file):
    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters from the configuration
    battery_capacities = config["battery_capacities"]
    threshold_values = config["threshold_values"]
    wind_thresholds = config["wind_thresholds"]
    horizon = config["horizon"]
    data_path = config["data_path"]
    episodes = config.get("episodes", 3000)  # Default to 3000 if not provided

    # Build the parameter list
    param_list = build_param_list(battery_capacities, threshold_values, wind_thresholds, horizon, data_path)

    # Create simulations in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        simulations = pool.starmap(create_simulation, param_list)
    
    print(f"Created {len(simulations)} simulation objects.")

    # Run the simulations using the episodes parameter from YAML
    run_manager = SimulationRunManager(episodes_per_simulation=episodes, storage_dir="simulation_results")
    run_manager.run_simulations(simulations, use_multiprocessing=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations using configuration from a YAML file")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    main(args.config)
