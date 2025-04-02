import numpy as np
from tqdm import tqdm
import pandas as pd
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider as EnvProv
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import OptimalPolicySimulation, OptimalAnalyticalPolicySimulation
from BaseClasses.whale_base import WhaleRewardSeriesFactory

def main():
    # -------------------------------
    # Configuration parameters
    # -------------------------------
    battery_capacity = 300.0  # in Wh (adjust as needed)
    horizon = 100              # number of time steps in each episode
    initial_state = np.array([100.0, 0])  # [SoC, mode] with mode 0 = safe, mode 2 = failure

    # -------------------------------
    # Build dummy environment distributions
    # -------------------------------
    # For solar energy gain: assume a Beta distribution with fixed parameters over the horizon.
    data_path = r"Data/EXPECTED_DATA/data_expected_lat0_lon-90_15min.pkl"
    data = pd.read_pickle(data_path)
    wind_shape = data['weibull_k'].values[:horizon]
    wind_scale = data['weibull_scale'].values[:horizon]
    wind_distributions = np.column_stack((wind_shape, wind_scale))
    solar_alpha = data['beta_alpha'].values[:horizon]
    solar_beta = data['beta_beta'].values[:horizon]
    solar_distributions = np.column_stack((solar_alpha, solar_beta))
    x = np.linspace(np.pi, np.pi * 5, horizon)
    whale_reward_series = WhaleRewardSeriesFactory.create_series("real", horizon)


    # -------------------------------
    # Create the environment provider
    # -------------------------------
    env_provider = EnvProv(
        solar_distributions=solar_distributions,
        wind_distributions=wind_distributions,
        whale_reward_series=whale_reward_series,
        delta_t=15
    )

    # -------------------------------
    # Create the seaplane and obtain power parameters
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
    solver = mdpBackwardSolver(mdp, horizon)

    # -------------------------------
    # Create the simulation objects
    # -------------------------------
    # Simulation using Monte Carlo sampling in the choose_action method.
    mcs_sim = OptimalPolicySimulation(
        mdp_solver=solver,
        horizon=horizon,
        initial_state=initial_state,
        env_provider=env_provider
    )
    # Simulation using numerical integration (the revised policy simulation).
    # analytical_sim = OptimalAnalyticalPolicySimulation(
    #     mdp_solver=solver,
    #     horizon=horizon,
    #     initial_state=initial_state,
    #     env_provider=env_provider
    # )

    # -------------------------------
    # Run a set number of episodes and compare outcomes.
    # -------------------------------
    episodes = 5000  # number of episodes to simulate with each method
    mcs_rewards = []
    analytical_rewards = []

    for _ in tqdm(range(episodes)):
        # Run one episode with the Monte Carlo simulation method.
        traj, acts, rews, solar, wind, whale = mcs_sim.simulate_episode()
        mcs_rewards.append(sum(rews))

        # # Run one episode with the numerical integration method.
        # traj, acts, rews, solar, wind, whale = analytical_sim.simulate_episode()
        # analytical_rewards.append(sum(rews))

    print("Monte Carlo Simulation (MCS) average total reward:", np.mean(mcs_rewards))
    print("Numerical Integration Simulation average total reward:", np.mean(analytical_rewards))


if __name__ == "__main__":
    main()
