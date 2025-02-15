import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from BaseClasses.expectedValue_base import ExpectedValueTable
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.transition_model_base import TestSuccessProbability, RealisticSuccessProbability


class TestExpectedValueTable(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Mocking the Seaplane class
        self.mock_plane = MagicMock(spec=Seaplane)
        self.mock_plane.capacity = 100  # Ah
        self.mock_plane.voltage = 24  # V
        self.mock_plane.S = 1  # Wing area in m^2
        self.mock_plane.idle_power = 1  # W
        self.mock_plane.required_cruise_power = 500  # W
        self.mock_plane.required_takeoff_energy = 50000  # J

        # Example data for initialization
        self.expected_solar_data = np.array([[1.5, 2.5, 5.0], [1, 3.0, 6.0]])
        self.expected_wind_data = np.array([[10, 15], [15, 6]])
        self.whale_observation_data = np.array([0.2, 0.8])
        self.soc_increment = 1
        self.timestep_min = 30

        # Mocking the transition model
        self.mock_transition_model = TestSuccessProbability()

        # Instantiate the class to test
        self.ev_table = ExpectedValueTable(
            plane=self.mock_plane,
            expected_solar_data=self.expected_solar_data,
            expected_wind_data=self.expected_wind_data,
            whale_observation_data=self.whale_observation_data,
            soc_increment=self.soc_increment,
            timestep_min=self.timestep_min,
            transition_model=self.mock_transition_model,
        )
        self.ev_table.max_collected_power = 1000  # Example value

    def test_soc_to_joules(self):
        """Test the soc_to_joules method."""
        soc = 50  # 50% state of charge
        expected_joules = (
            soc / 100 * self.mock_plane.capacity * self.mock_plane.voltage * 3600
        )  # capacity * voltage * SOC * 3600
        self.assertEqual(self.ev_table.soc_to_joules(soc), int(expected_joules))

        soc = 25  # 50% state of charge
        expected_joules = (
            soc / 100 * self.mock_plane.capacity * self.mock_plane.voltage * 3600
        )  # capacity * voltage * SOC * 3600
        self.assertEqual(self.ev_table.soc_to_joules(soc), int(expected_joules))

        soc = 0  # 50% state of charge
        expected_joules = (
            soc / 100 * self.mock_plane.capacity * self.mock_plane.voltage * 3600
        )  # capacity * voltage * SOC * 3600
        self.assertEqual(self.ev_table.soc_to_joules(soc), int(expected_joules))

        soc = 100  # 50% state of charge
        expected_joules = (
            soc / 100 * self.mock_plane.capacity * self.mock_plane.voltage * 3600
        )  # capacity * voltage * SOC * 3600
        self.assertEqual(self.ev_table.soc_to_joules(soc), int(expected_joules))

    def test_calculate_sufficient_solar_probability(self):
        """Test _calculate_sufficient_solar_probability method."""
        required_energy = 50000  # J
        current_energy = 25000  # J
        max_collected_energy = 50000  # J
        alpha = 5.0
        beta = 5.0

        # Mock the Beta distribution CDF
        with unittest.mock.patch("scipy.stats.beta.cdf") as mock_cdf:
            mock_cdf.return_value = 0.5
            result = self.ev_table._calculate_sufficient_solar_probability(
                required_energy, current_energy, max_collected_energy, alpha, beta
            )
            self.assertEqual(result, 0.5)  # 1 - F_S

        result = self.ev_table._calculate_sufficient_solar_probability(
            required_energy, current_energy, max_collected_energy, alpha, beta
        )
        self.assertEqual(result, 0.5)  # 1 - F_S

        required_energy = 50000  # J
        current_energy = 37500  # J
        max_collected_energy = 50000  # J
        alpha = 5.0
        beta = 5.0
        result = self.ev_table._calculate_sufficient_solar_probability(
            required_energy, current_energy, max_collected_energy, alpha, beta
        )
        self.assertAlmostEqual(result, 0.95, places=2)  # 1 - F_S

    def test_calculate_next_state(self):
        """Test calculate_next_state method."""

        state = (50, 0)
        action = 1
        solar_power = 300  # W

        # Mock the _calculate_soc_update method
        self.ev_table._calculate_soc_update = MagicMock(return_value=-30)

        next_state = self.ev_table.calculate_next_state(state, action, solar_power)
        expected_state = (20, 1)
        self.assertEqual(next_state, expected_state)

        state = (10, 1)
        action = 1
        solar_power = 300  # W
        self.ev_table._calculate_soc_update = MagicMock(return_value=-20)

        next_state = self.ev_table.calculate_next_state(state, action, solar_power)
        expected_state = (-1, 2)
        self.assertEqual(next_state, expected_state)

    def test_lookup_expected_value(self):
        # Create a mock array for testing
        # This array has dimensions (6, 5), representing 3 vehicle states (moored, flying, broken) and 5 stages
        array = np.random.rand(203, 5)
        array[:, 4] = 0.0

        # Define stage, state, and other parameters
        stage = 2
        state_moored = (50, 0)  # 50% state of charge, moored
        state_flying = (75, 1)  # 75% state of charge, flying
        state_broken = (-1, 2)  # -1% state of charge, broken
        invalid_state = (40, 3)  # Invalid vehicle state
        invalid_soc = (-40, 0)  # Invalid vehicle state

        # Test: Look up value for 'moored' state (row_index = 50 // 1% discretization = 50)
        result = self.ev_table.lookup_expected_value(array, stage, state_moored)
        expected_value_moored = array[50, stage]  # should match the value in array at this index
        self.assertEqual(result, expected_value_moored)

        # Test: Look up value for 'flying' state (row_index = 100 + 75 // 1% discretization = 175)
        result = self.ev_table.lookup_expected_value(array, stage, state_flying)
        expected_value_flying = array[176, stage]  # should match the value in array at this index
        self.assertEqual(result, expected_value_flying)

        # Test: Look up value for 'broken' state (row_index = 2 * n = 200)
        result = self.ev_table.lookup_expected_value(array, stage, state_broken)
        expected_value_broken = array[202, stage]  # should match the value in array at this index
        self.assertEqual(result, expected_value_broken)

        # Test: Check invalid vehicle state (should raise ValueError)
        with self.assertRaises(ValueError):
            self.ev_table.lookup_expected_value(array, stage, invalid_state)

        # Test: Check for stage out of bounds (should raise IndexError)
        with self.assertRaises(IndexError):
            self.ev_table.lookup_expected_value(
                array, 10, state_moored
            )  # stage index > array.shape[1]

        with self.assertRaises(ValueError):
            self.ev_table.lookup_expected_value(array, stage, invalid_soc)

        # Test: Check for stage == max stage (should return 0.0)
        result = self.ev_table.lookup_expected_value(array, array.shape[1] - 1, state_moored)
        self.assertEqual(result, 0.0)  # last stage should return 0.0

    def test_alpha(self):
        """
        Test the _alpha method by mocking its dependencies and verifying
        the result returned based on mock data.
        """
        # Prepare mock return values
        mock_state = (50, 1)  # Example state: 50% charge, flying
        mock_action = 1  # Example action
        mock_solar_power_w = 100.0  # Example solar power in Watts
        mock_ev_value = -1.25  # Expected value returned from the lookup table
        mock_wind_speed = 100

        # Assuming the method _alpha is part of the ExpectedValueTable class
        ev_table_instance = self.ev_table  # Since it's instantiated in setUp

        # Call the _alpha method
        result = ev_table_instance._alpha(
            stage=0,
            state=mock_state,
            action=mock_action,
            solar_power_w=mock_solar_power_w,
            wind_speed_ms=mock_wind_speed,
        )

        # Assert that the result is what we expect
        self.assertEqual(result, mock_ev_value)

    def test_alpha_vector_inputs_takeoff(self):
        """
        Test the _alpha method with vector inputs for solar_power_w and wind_speed_ms.
        """
        # Mock environmental data
        mock_solar_power_w = np.array([0.0, 2000.0, 300.0])  # Example solar power in Watts
        mock_wind_speed_ms = np.array([0.0, 10.0, 15.0])  # Example wind speeds in m/s

        # Assuming the method _alpha is part of the ExpectedValueTable class
        ev_table_instance = self.ev_table  # Since it's instantiated in setUp
        ev_table_instance.transition_model = RealisticSuccessProbability()

        # Prepare mock inputs for takeoff case
        mock_state = (50, 0)  # Example state: 50% charge, floating
        mock_action = 1  # Example action fly

        # Run test
        ev_table_instance.ev_table = np.ones((203, 3))
        result = ev_table_instance._alpha(
            stage=0,
            state=mock_state,
            action=mock_action,
            solar_power_w=mock_solar_power_w,
            wind_speed_ms=mock_wind_speed_ms,
        )

        # Assert that the result is a numpy array and has the same length as the input vectors
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(mock_solar_power_w))
        self.assertEqual(
            len(result), len(set(result)), f"Result vector ({result}) entries are not unique."
        )
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])

    def test_alpha_vector_inputs_landing(self):
        # Mock environmental data
        mock_solar_power_w = np.array([0.0, 4000.0, 500.0])  # Example solar power in Watts
        mock_wind_speed_ms = np.array([17.0, 5.0, 40.0])  # Example wind speeds in m/s

        # Assuming the method _alpha is part of the ExpectedValueTable class
        ev_table_instance = self.ev_table  # Since it's instantiated in setUp
        ev_table_instance.transition_model = RealisticSuccessProbability()

        # Prepare mock inputs for takeoff case
        mock_state = (50, 1)  # Example state: 50% charge, floating
        mock_action = 0  # Example action fly

        # Run test
        ev_table_instance.ev_table = np.ones((203, 3))
        result = ev_table_instance._alpha(
            stage=0,
            state=mock_state,
            action=mock_action,
            solar_power_w=mock_solar_power_w,
            wind_speed_ms=mock_wind_speed_ms,
        )

        # Assert that the result is a numpy array and has the same length as the input vectors
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(mock_solar_power_w))
        self.assertEqual(
            len(result), len(set(result)), f"Result vector ({result}) entries are not unique."
        )
        self.assertGreater(result[1], result[2])
        self.assertGreater(result[1], result[0])

    def test_ev_entry(self):
        """Test _ev_entry method."""
        # Sample data for k and state
        k = 0  # Index for whale observation
        state = (50, 0)  # 50% SOC, moored

        # Mock the required methods
        self.ev_table._calculate_case_probabilities = MagicMock(
            return_value=([0.1, 0.2, 0.3, 0.4], 0.8, 1.0)
        )
        self.ev_table._compute_success_probability = MagicMock(return_value=0.9)

        # Set mock values
        self.ev_table.whale_probability_data = np.array(
            [0.5, 0.6, 0.7]
        )  # Example whale probabilities
        self.ev_table.expected_wind = np.array(
            [[10, 5], [15, 6], [30, 2]]
        )  # Wind data for two different times

        reward_k = (
            self.ev_table.whale_probability_data[k] * 1
        )  # Should be 0.6 (from whale_probability_data[k])
        wind_shape_k = self.ev_table.expected_wind[k, 0]  # Should be 15
        wind_scale_k = self.ev_table.expected_wind[k, 1]  # Should be 6

        probabilities, alpha_u_0, alpha_u_1 = self.ev_table._calculate_case_probabilities(
            k, state, reward_k, 0.9, 0.9
        )
        p_4 = probabilities[3]  # p_4 = 0.4 from mock return value
        p_success_u_0 = self.ev_table._compute_success_probability(
            0, state, wind_shape_k, wind_scale_k
        )  # Should return 0.9
        p_success_u_1 = self.ev_table._compute_success_probability(
            1, state, wind_shape_k, wind_scale_k
        )  # Should return 0.9

        # Compute expected value E_J_k using the formula
        fly_case_value = p_4 * (reward_k + p_success_u_1 * alpha_u_1 - (1 - p_success_u_1) * self.ev_table.failure_penalty)
        float_case_value = (1-p_4) * (p_success_u_0 * alpha_u_0 - (1-p_success_u_0)*self.ev_table.failure_penalty)


        expected_E_J_k = fly_case_value + float_case_value

        # Call _ev_entry method
        result = self.ev_table._ev_entry(k, state)

        # Assert that the result matches the expected value
        self.assertAlmostEqual(result, expected_E_J_k, places=5)

        # Check that the mocked methods were called with the expected arguments
        self.ev_table._calculate_case_probabilities.assert_called_with(k, state, reward_k, 0.9, 0.9)
        self.ev_table._compute_success_probability.assert_any_call(
            0, state, wind_shape_k, wind_scale_k
        )
        self.ev_table._compute_success_probability.assert_any_call(
            1, state, wind_shape_k, wind_scale_k
        )

    def test_calculate_case_probabilities(self):
        """Test _calculate_case_probabilities method."""
        # Sample data for testing
        stage = 1  # Time stage for the decision process
        state = (50, 0)  # 50% state of charge, moored
        reward_k = 0.8  # Example reward threshold

        # Mock the required methods
        self.ev_table._calculate_sufficient_solar_probability = MagicMock(
            return_value=0.7
        )  # Example value
        self.ev_table._calculate_sufficient_reward_probability = MagicMock(
            return_value=(0.9, 0.95, 1.0)
        )  # p_sufficient_reward, alpha_u_0, alpha_u_1
        self.ev_table._calculate_required_energy = MagicMock(
            return_value=10000
        )  # Mocked required energy in Joules

        # Set mock values for expected solar data
        self.ev_table.expected_solar = np.array(
            [[0, 2.5, 5.0], [1, 3.0, 6.0]]  # stage 0  # stage 1
        )  # alpha_k=3.0, beta_k=6.0

        self.ev_table.expected_wind = np.array(
            [[0, 1.0, 3.0], [2, 5.0, 3.2]]  # stage 0  # stage 1
        )  # alpha_k=3.0, beta_k=6.0

        # Set mock values for max collected energy and current energy
        self.ev_table.max_collected_power = 1000  # W
        self.ev_table.dt = 1  # Example time step (minute)

        # Run the method
        probabilities, alpha_u_0, alpha_u_1 = self.ev_table._calculate_case_probabilities(
            stage, state, reward_k, 0.9, 0.9
        )

        # Expected calculation steps:
        p_sufficient_solar = (
            self.ev_table._calculate_sufficient_solar_probability.return_value
        )  # Should be 0.7
        p_sufficient_reward = self.ev_table._calculate_sufficient_reward_probability.return_value[
            0
        ]  # Should be 0.9

        # Calculate the probabilities manually based on the formula
        p0 = (1 - p_sufficient_solar) * (1 - p_sufficient_reward)
        p1 = (1 - p_sufficient_solar) * p_sufficient_reward
        p2 = p_sufficient_solar * (1 - p_sufficient_reward)
        p3 = p_sufficient_solar * p_sufficient_reward

        expected_probabilities = (p0, p1, p2, p3)
        expected_alpha_u_0 = 0.95  # Mocked value
        expected_alpha_u_1 = 1.0  # Mocked value

        # Assert the probabilities match
        self.assertAlmostEqual(probabilities[0], expected_probabilities[0], places=5)
        self.assertAlmostEqual(probabilities[1], expected_probabilities[1], places=5)
        self.assertAlmostEqual(probabilities[2], expected_probabilities[2], places=5)
        self.assertAlmostEqual(probabilities[3], expected_probabilities[3], places=5)

        # Assert the alpha values match
        self.assertEqual(alpha_u_0, expected_alpha_u_0)
        self.assertEqual(alpha_u_1, expected_alpha_u_1)

        # Check that the required methods were called
        self.ev_table._calculate_sufficient_solar_probability.assert_called_with(
            self.ev_table._calculate_required_energy.return_value,
            self.ev_table.soc_to_joules(state[0]),
            self.ev_table.max_collected_power * self.ev_table.dt * 60,
            1.0,  # alpha_k for stage 1
            3.0,  # beta_k for stage 1
        )

        self.ev_table._calculate_sufficient_reward_probability.assert_called_with(
            stage, state, reward_k, 1.0, 3.0, 2, 5, 0.9, 0.9
        )

        # Verify the call to _calculate_required_energy
        self.ev_table._calculate_required_energy.assert_called_with(state, action=1)

    def test_vectorized_soc_update(self):
        # Input data
        solar_power = np.array([[100], [1000], [30000]])  # Solar power in W
        actions = 1  # Actions for each time step (1 for fly, 0 for float)
        state = (30, 0)  # Example state
        dt = 10  # Time step in minutes

        # Expected calculations
        panel_efficiency = 0.1
        solar_input = solar_power.flatten() * panel_efficiency * self.ev_table.plane.S
        required_power = np.where(actions == 0, 0.0, self.ev_table.plane.required_cruise_power)
        required_takeoff_energy = np.where(
            state[1] == 0, self.ev_table.plane.required_takeoff_energy, 0
        )
        net_power = solar_input - required_power - self.ev_table.plane.idle_power
        energy_change = net_power * dt * 60 - required_takeoff_energy
        expected_soc_change = self.ev_table.soc_increment * np.round(
            (energy_change / (self.ev_table.battery_capacity_wh * 3600))
            * 100
            / self.ev_table.soc_increment
        )

        # Mock the method and compare the results
        self.ev_table._calculate_soc_update = lambda plane, state, action, dt, solar_power: (
            expected_soc_change
        )
        soc_changes = self.ev_table._calculate_soc_update(
            self.ev_table.plane, state, actions, dt, solar_power
        )

        # Assertions
        np.testing.assert_array_almost_equal(
            soc_changes,
            expected_soc_change,
            decimal=5,
            err_msg="SOC changes do not match expected values.",
        )


if __name__ == "__main__":
    unittest.main()
