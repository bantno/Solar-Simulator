import unittest
import numpy as np
from BaseClasses.mdp_base import MDP
from BaseClasses.transition_model_base import ProbabilityModelFactory

# A fake probability model that always succeeds.
class FakeProbabilityModel:
    def compute_probability(self, wind_speeds, actions, states):
        # Always return an array of ones so that the transition always succeeds.
        return np.ones(states.shape[0])

# Test class for the DeterministicMDP
class TestDeterministicMDP(unittest.TestCase):
    def setUp(self):
        # Patch the probability model factory so that our fake model is used.
        # Save the original method to restore it later.
        self.original_select = ProbabilityModelFactory.select_probability_model
        ProbabilityModelFactory.select_probability_model = lambda model_name: FakeProbabilityModel()
        
        # Set up test parameters.
        self.battery_capacity_wh = 100
        self.idle_power = 10
        self.cruise_power = 20
        self.takeoff_power = 5
        self.solar_rate_series = np.array([100, 200, 300])
        self.wind_series = np.array([4, 5, 6])
        self.whale_reward_series = np.array([0.5, 1, 0.5])
        self.failure_penalty = 50
        self.delta_t = 1  # in minutes
        self.gamma = 0.9
        self.soc_increment = 10
        
        # Create an instance of DeterministicMDP.
        self.mdp = DeterministicMDP(
            self.battery_capacity_wh,
            self.idle_power,
            self.cruise_power,
            self.takeoff_power,
            self.solar_rate_series,
            self.wind_series,
            self.whale_reward_series,
            self.failure_penalty,
            self.delta_t,
            self.gamma,
            "fake",   # transition_model_name (dummy, as we patch it)
            self.soc_increment
        )
        
    def tearDown(self):
        # Restore the original probability model factory method.
        ProbabilityModelFactory.select_probability_model = self.original_select
        
    def test_ensure_vectorized_input_valid(self):
        # Passing a numpy array should not raise an error.
        try:
            self.mdp._ensure_vectorized_input(np.array([1, 2, 3]), "test")
        except Exception as e:
            self.fail("Unexpected exception raised for valid numpy array input.")
            
    def test_ensure_vectorized_input_invalid(self):
        # Passing a non-numpy array should raise a TypeError.
        with self.assertRaises(TypeError):
            self.mdp._ensure_vectorized_input([1, 2, 3], "test")
            
    def test_sample_sunlight(self):
        # For t=0 and n=3, sample_sunlight should return an array filled with solar_rate_series[0] * delta_t.
        result = self.mdp.sample_sunlight(0, 3)
        expected = np.full((3,), self.solar_rate_series[0] * self.delta_t)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_sample_sunlight_index_error(self):
        # When t is out of range, sample_sunlight should raise an IndexError.
        with self.assertRaises(IndexError):
            self.mdp.sample_sunlight(10, 3)
            
    def test_sample_wind_speed(self):
        # For t=0 and n=2, sample_wind_speed should return an array filled with wind_speed_series[0].
        result = self.mdp.sample_wind_speed(0, 2)
        expected = np.full((2,), self.mdp.wind_speed_series[0])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_sample_wind_speed_index_error(self):
        # When t is out of range, sample_wind_speed should raise an IndexError.
        with self.assertRaises(IndexError):
            self.mdp.sample_wind_speed(10, 2)
            
    def test_soc_to_energy_full_battery(self):
        # Test that 100% SOC converts to the full battery energy in joules.
        energy = self.mdp.soc_to_energy(np.array([100]))
        expected = np.array([self.battery_capacity_wh * 3600])
        np.testing.assert_array_almost_equal(energy, expected)

    def test_soc_to_energy_half_battery(self):
        energy = self.mdp.soc_to_energy(np.array([50]))
        expected = np.array([self.battery_capacity_wh*3600/2])
        np.testing.assert_array_almost_equal(energy, expected)

    def test_soc_to_energy_negative_battery(self):
        energy = self.mdp.soc_to_energy(np.array([-1]))
        expected = np.array([-1])
        np.testing.assert_array_almost_equal(energy, expected)

    def test_energy_to_soc(self):
        # For energy equal to half the battery capacity, SOC should be 50.
        half_energy = np.array([self.battery_capacity_wh * 3600 / 2])
        soc = self.mdp.energy_to_soc(half_energy)
        expected = np.array([50.0])
        np.testing.assert_array_almost_equal(soc, expected)
        # Test flooring: if the raw SOC is 55, flooring to the nearest increment (10) gives 50.
        energy = np.array([self.battery_capacity_wh * 3600 * 0.55])
        soc = self.mdp.energy_to_soc(energy)
        expected = np.array([50.0])
        np.testing.assert_array_almost_equal(soc, expected)
        
    def test_min_to_seconds(self):
        # 2 minutes should convert to 120 seconds.
        seconds = self.mdp.min_to_seconds(2)
        self.assertEqual(seconds, 120.)
        
    def test_transition_success(self):
        # To test the transition function deterministically,
        # we patch np.random.rand to return zeros so that all transitions succeed.
        original_rand = np.random.rand
        np.random.rand = lambda n: np.zeros(n)
        try:
            # Create two simple states: one moored and one flying.
            states = np.array([[50, 0], [80, 1]])
            # Actions: 0 (moored) and 1 (fly).
            actions = np.array([0, 1])
            t = 0
            
            next_states = self.mdp.transition(states, actions, t)
            # Verify that the returned array has the same shape as the states.
            self.assertEqual(next_states.shape, states.shape)
            
            # Compute expected energy consumption.
            moored_float_energy = self.idle_power * self.mdp.min_to_seconds(self.delta_t)  # 10*60 = 600
            takeoff_energy = (self.cruise_power + self.takeoff_power) * self.mdp.min_to_seconds(self.delta_t)  # (20+5)*60 = 1500
            land_energy = self.cruise_power * self.mdp.min_to_seconds(self.delta_t)  # 20*60 = 1200
            continue_flight_energy = self.cruise_power * self.mdp.min_to_seconds(self.delta_t)  # 20*60 = 1200
            
            energy_lookup = np.array([[moored_float_energy, takeoff_energy],
                                      [land_energy, continue_flight_energy]])
            energy_consumption = np.array([
                energy_lookup[int(states[i, 1]), actions[i]]
                for i in range(len(states))
            ])
            energy_gain = np.full((len(states),), self.solar_rate_series[t] * self.delta_t)
            current_energy = self.mdp.soc_to_energy(states[:, 0])
            next_energy = current_energy + energy_gain - energy_consumption
            next_soc = self.mdp.energy_to_soc(next_energy)
            expected_modes = np.where(next_soc < 0, 2, np.where(actions == 0, 0, 1))
            expected_next_states = np.column_stack((next_soc, expected_modes))
            np.testing.assert_array_almost_equal(next_states, expected_next_states)
        finally:
            np.random.rand = original_rand
            
    def test_reward(self):
        # Test that reward is computed correctly.
        # For flying (action==1), the reward should equal the whale reward at time t,
        # and if the resulting state is broken (mode==2), the failure penalty should be subtracted.
        states = np.array([[50, 0], [80, 1]])
        actions = np.array([0, 1])
        # Manually set next_states: first state remains moored (mode 0), second state is marked as broken (mode 2).
        next_states = np.array([[40, 0], [70, 2]])
        t = 0
        rewards = self.mdp.reward(states, actions, next_states, t)
        expected = np.array([0.0, self.whale_reward_series[t] - self.failure_penalty])
        np.testing.assert_array_almost_equal(rewards, expected)
        
if __name__ == '__main__':
    unittest.main()
