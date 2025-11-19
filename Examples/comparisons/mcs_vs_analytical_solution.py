import numpy as np
import pandas as pd
from tqdm import tqdm
from BaseClasses.environment_provider_base_cs_normalization import StochasticWindSolarEnvironmentProvider as EnvProv
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver, mdpAnalyticalBackwardSolver
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import OptimalPolicySimulation, OptimalAnalyticalPolicySimulation
from BaseClasses.whale_base import WhaleRewardSeriesFactory

def main():
    # -------------------------------
    # Configuration parameters
    # -------------------------------
    battery_capacity = 400.0  # in Wh
    horizon = 100           # number of time steps per episode
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
    # Instantiate the MDP
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

    # -------------------------------
    # Set up the backward induction solver
    # -------------------------------
    solver_mcs = mdpBackwardSolver(mdp, horizon)
    solver_mcs.solve()
    # solver_analytical = mdpNumericalBackwardSolver(mdp, horizon)
    # solver_analytical.solve()

if __name__ == "__main__":
    main()