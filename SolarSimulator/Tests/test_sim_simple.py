import unittest
import numpy as np
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.simulation_base import AbstractSimulation, AlwaysFloatSimulation, AlwaysFlySimulation, ObservationThresholdSimulation

class TestSimulationSOCBounds(unittest.TestCase):
    def setUp(self):
        # Setup dummy parameters for the MDP.
        self.solar_rate_series = np.full(10, 100000)       # Solar rate for 10 timesteps.
        self.wind_series = np.full(10, 5.0)                  # Constant wind speed.
        self.battery_capacity_wh = 200 * 60 * 60 * 5 / 3600    # Battery capacity in Wh.
        self.idle_power = 0                                  # Power consumption when moored.
        self.cruise_power = 200                              # Power consumption when flying.
        self.takeoff_power = 200                             # Additional power consumption for takeoff.
        self.whale_reward_series = np.full(10, 1)            # Constant whale reward.
        self.failure_penalty = 1000                          # Penalty for failed transition.
        self.delta_t = 15                                    # Time step duration in minutes.
        self.gamma = 1.0                                   # Discount factor.
        self.transition_model_name = "moderate"              # Transition model name.
        self.soc_increment = 5.0                             # SOC increment.
        self.horizon = 10                                    # Simulation horizon (10 timesteps).

        # Instantiate the MDP.
        self.mdp = DeterministicMDP(
            self.battery_capacity_wh, self.idle_power, self.cruise_power, self.takeoff_power,
            self.solar_rate_series, self.wind_series, self.whale_reward_series,
            self.failure_penalty, self.delta_t, self.gamma,
            self.transition_model_name, self.soc_increment
        )

    def check_soc_bounds(self, trajectory):
        """Helper method that asserts each state in the trajectory has an SOC between -1 and 100."""
        for state in trajectory:
            soc = state[0]  # Assuming state is [SOC, mode]
            self.assertGreaterEqual(soc, -1.0, "SOC is below -1.")
            self.assertLessEqual(soc, 100.0, "SOC exceeds 100.")

    def test_always_fly_simulation_soc_bounds(self):
        # Use an initial state with full battery.
        initial_state = np.array([100, 0])
        solar_data = np.tile(self.solar_rate_series,(3,1))
        wind_data = np.tile(self.wind_series,(3,1))
        whale_data = np.tile(self.whale_reward_series,(3,1))
        initial_state = np.array([20, 0])
        sim = AlwaysFlySimulation(self.mdp, self.horizon, initial_state)
        trajectory, actions, rewards = sim.simulate_episode(solar_data[0,:],wind_data[0,:],whale_data[0,:])
        self.check_soc_bounds(trajectory)

    def test_always_float_simulation_soc_bounds(self):
        # Use an initial state with a lower SOC.
        solar_data = np.tile(self.solar_rate_series,(3,1))
        wind_data = np.tile(self.wind_series,(3,1))
        whale_data = np.tile(self.whale_reward_series,(3,1))
        initial_state = np.array([20, 0])
        sim = AlwaysFloatSimulation(self.mdp, self.horizon, initial_state)
        trajectory, actions, rewards = sim.simulate_episode(solar_data[0,:],wind_data[0,:],whale_data[0,:])
        self.check_soc_bounds(trajectory)

    def test_observation_threshold_simulation_soc_bounds(self):
        # Use an initial state with a lower SOC.
        solar_data = np.tile(self.solar_rate_series,(3,1))
        wind_data = np.tile(self.wind_series,(3,1))
        whale_data = np.tile(self.whale_reward_series,(3,1))
        initial_state = np.array([20, 0])
        sim = ObservationThresholdSimulation(self.mdp, self.horizon, initial_state,0.5,10)
        trajectory, actions, rewards = sim.simulate_episode(solar_data[0,:],wind_data[0,:],whale_data[0,:])
        self.check_soc_bounds(trajectory)

    def test_multiple_episodes_soc_bounds(self):
        # Test that in multiple episodes, the SOC remains within bounds.
        # For AlwaysFlySimulation.
        initial_state = np.array([100, 0])
        solar_data = np.tile(self.solar_rate_series,(3,1))
        wind_data = np.tile(self.wind_series,(3,1))
        whale_data = np.tile(self.whale_reward_series,(3,1))

        sim = AlwaysFlySimulation(self.mdp, self.horizon, initial_state)

        episodes = sim.simulate_multiple_episodes(solar_data,wind_data,whale_data,3)
        for ep in episodes:
            self.check_soc_bounds(ep['trajectory'])
        
        # For AlwaysFloatSimulation.
        initial_state = np.array([20, 0])
        sim = AlwaysFloatSimulation(self.mdp, self.horizon, initial_state)
        episodes = sim.simulate_multiple_episodes(solar_data,wind_data,whale_data,3)
        for ep in episodes:
            self.check_soc_bounds(ep['trajectory'])

if __name__ == '__main__':
    unittest.main()
