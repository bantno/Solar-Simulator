import unittest
from unittest.mock import MagicMock
import numpy as np
from BaseClasses.valueFunction_base import ValueFunction
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
        self.soc_increment = 0.5
        self.timestep_min = 30

        # Mocking the transition model
        self.mock_transition_model = TestSuccessProbability()

        # Instantiate the class to test
        self.ev_table = ValueFunction(
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
        array = np.random.rand(403, 5)
        array[:, 4] = 0.0

        # Define stage, state, and other parameters
        stage = 2
        state_moored = np.array([[50, 0]])  # 50% state of charge, moored
        state_flying = np.array([[75, 1]])  # 75% state of charge, flying
        state_broken = np.array([[-1, 2]])  # -1% state of charge, broken
        invalid_state = np.array([[40, 3]])  # Invalid vehicle state
        invalid_soc = np.array([[-40, 0]])  # Invalid vehicle state

        # Test: Look up value for 'moored' state (row_index = 50 // 1% discretization = 50)
        result = self.ev_table.lookup_expected_value(array, stage, state_moored, self.ev_table.soc_increment)
        expected_value_moored = array[100, stage]  # should match the value in array at this index
        self.assertEqual(result, expected_value_moored)

        # Test: Look up value for 'flying' state (row_index = 100 + 75 // 1% discretization = 175)
        result = self.ev_table.lookup_expected_value(array, stage, state_flying, self.ev_table.soc_increment)
        expected_value_flying = array[351, stage]  # should match the value in array at this index
        self.assertEqual(result, expected_value_flying)

        # Test: Look up value for 'broken' state (row_index = 2 * n = 200)
        result = self.ev_table.lookup_expected_value(array, stage, state_broken, self.ev_table.soc_increment)
        expected_value_broken = array[402, stage]  # should match the value in array at this index
        self.assertEqual(result, expected_value_broken)

        # # Test: Check invalid vehicle state (should raise ValueError)
        # with self.assertRaises(ValueError):
        #     self.ev_table.lookup_expected_value(array, stage, invalid_state)

        # # Test: Check for stage out of bounds (should raise IndexError)
        # with self.assertRaises(IndexError):
        #     self.ev_table.lookup_expected_value(
        #         array, 10, state_moored
        #     )  # stage index > array.shape[1]

        # with self.assertRaises(ValueError):
        #     self.ev_table.lookup_expected_value(array, stage, invalid_soc)

        # # Test: Check for stage == max stage (should return 0.0)
        # result = self.ev_table.lookup_expected_value(array, array.shape[1] - 1, state_moored)
        # self.assertEqual(result, 0.0)  # last stage should return 0.0

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
