import numpy as np
import pandas as pd
from tqdm import tqdm
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider as EnvProv
from BaseClasses.mdp_base import stochasticMDP
from BaseClasses.backward_induction_base import mdpBackwardSolver
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import OptimalAnalyticalPolicySimulation, OptimalContinuousAnalyticalPolicySimulation
from BaseClasses.whale_base import WhaleRewardSeriesFactory

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
        delta_t=15
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
    solver = mdpBackwardSolver(mdp, horizon)

    # -------------------------------
    # Create simulation objects
    # -------------------------------
    # Discrete simulation using numerical integration.
    optimal_analytical_sim = OptimalAnalyticalPolicySimulation(
        mdp_solver=solver,
        horizon=horizon,
        initial_state=initial_state,
        env_provider=env_provider
    )
    # Continuous energy simulation using numerical integration.
    optimal_continuous_sim = OptimalContinuousAnalyticalPolicySimulation(
        mdp_solver=solver,
        horizon=horizon,
        initial_state=initial_state,
        env_provider=env_provider
    )

    # -------------------------------
    # Run episodes and collect results.
    # -------------------------------
    episodes = 1000
    results = []

    for episode in tqdm(range(episodes)):
        # Run discrete analytical simulation episode.
        traj_anal, acts_anal, rews_anal, solar_anal, wind_anal, whale_anal = optimal_analytical_sim.simulate_episode()
        total_reward_anal = sum(rews_anal)

        # Run continuous analytical simulation episode.
        traj_cont, acts_cont, rews_cont, solar_cont, wind_cont, whale_cont, energies_cont = optimal_continuous_sim.simulate_episode()
        total_reward_cont = sum(rews_cont)

        # Save all episode data in a dictionary.
        results.append({
            "Episode": episode + 1,
            "Analytical_Trajectory": traj_anal,
            "Analytical_Actions": acts_anal,
            "Analytical_Rewards": rews_anal,
            "Analytical_Solar": solar_anal,
            "Analytical_Wind": wind_anal,
            "Analytical_Whale": whale_anal,
            "Analytical_TotalReward": total_reward_anal,
            "Continuous_Trajectory": traj_cont,
            "Continuous_Actions": acts_cont,
            "Continuous_Rewards": rews_cont,
            "Continuous_Solar": solar_cont,
            "Continuous_Wind": wind_cont,
            "Continuous_Whale": whale_cont,
            "Continuous_Energies": energies_cont,
            "Continuous_TotalReward": total_reward_cont
        })

    # Create a DataFrame from the results.
    results_df = pd.DataFrame(results)

    # Optionally, compute and print average total rewards.
    avg_analytical = np.mean(results_df["Analytical_TotalReward"])
    avg_continuous = np.mean(results_df["Continuous_TotalReward"])
    print("OptimalAnalyticalPolicySimulation average total reward:", avg_analytical)
    print("OptimalContinuousAnalyticalPolicySimulation average total reward:", avg_continuous)

    # -------------------------------
    # Save the DataFrame as a pickle file.
    # -------------------------------
    results_df.to_pickle("continuous_vs_analytical_results.pkl")
    print("Results saved to continuous_vs_analytical_results.pkl")

if __name__ == "__main__":
    main()
