import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.backward_induction_base import DeterministicMDPBackwardSolver
from BaseClasses.simulation_base import OptimalSimulation, ObservationThresholdSimulation
from BaseClasses.environment_provider_base import StochasticWindEnvironmentProvider
from BaseClasses.simulation_run_manager import SimulationRunManager

if __name__ == "__main__":
    # ----- Define dummy data -----

    # Create wind_distributions array with shape (T, 2):
    # For each time step t, the first value is the Weibull shape (k)
    # and the second value is the Weibull scale (λ), here varying with a diurnal pattern.

    horizon = 1000
    solar_rate_series = np.full(horizon, 4000)
    wind_series = np.full(horizon, 5.0)
    x = np.linspace(0, np.pi*48, horizon)
    whale_reward_series = np.sin(x)
    solar_rate_series = np.clip(np.sin(x)*4000,0,4000)
    t_indices = np.arange(horizon)
    wind_shape = np.full(horizon, 2.0)  # Constant shape parameter
    wind_scale = 4.0 + 3.0 * np.sin(2 * np.pi * t_indices / 24)  # Scale varies with time
    wind_distributions = np.column_stack((wind_shape, wind_scale))

    battery_capacity_wh = 200 * 60 * 60 * 10 / 3600
    idle_power = 0
    cruise_power = 200
    takeoff_power = 200
    failure_penalty = 15
    delta_t = 15
    gamma = 1.0
    transition_model_name = "moderate"
    soc_increment = 1.0

    # ----- Instantiate the custom environment provider -----
    env_provider = StochasticWindEnvironmentProvider(
        solar_rate_series=solar_rate_series,
        wind_distributions=wind_distributions,
        whale_reward_series=whale_reward_series,
        delta_t=delta_t
    )

    # A dummy wind series is provided to the MDP (its samples come from env_provider)
    dummy_wind_series = np.zeros_like(solar_rate_series)

    # ----- Instantiate the MDP using the stochastic environment provider -----
    mdp = DeterministicMDP(
        battery_capacity_wh=battery_capacity_wh,
        idle_power=idle_power,
        cruise_power=cruise_power,
        takeoff_power=takeoff_power,
        solar_rate_series=solar_rate_series,
        wind_series=dummy_wind_series,
        whale_reward_series=whale_reward_series,
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
    mdp_solver = DeterministicMDPBackwardSolver(mdp, horizon)

    # ----- Instantiate the Optimal Simulation -----
    optimal_simulation = OptimalSimulation(
        mdp_solver=mdp_solver,
        horizon=horizon,
        initial_state=initial_state,
        env_provider=env_provider
    )

    # ----- Run a Simulation Episode -----
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
        ObservationThresholdSimulation(mdp, horizon, initial_state, 0.0, 10, env_provider),
        ObservationThresholdSimulation(mdp, horizon, initial_state, 0.5, 10, env_provider),
        ObservationThresholdSimulation(mdp, horizon, initial_state, 0.9, 10, env_provider),
        sim_opt
    ]

    # ----- Set Up and Run the SimulationRunManager -----
    episodes_per_simulation = 5000  # Number of episodes per simulation run
    run_manager = SimulationRunManager(episodes_per_simulation, storage_dir="simulation_results")

    # Run all simulations provided in the list. Each simulation run is stored as a batch.
    run_manager.run_simulations(simulation_list)