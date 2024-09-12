import sys
import os
import unittest
from unittest.mock import MagicMock

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))

from seaplane_base import Seaplane
from mdp import mdp

class TestMDPCalculateSOCUpdate(unittest.TestCase):

    def setUp(self):
        # Mock plane with necessary attributes and methods
        self.plane_mock = MagicMock()
        self.plane_mock.voltage = 12  # Volts
        self.plane_mock.capacity = 10000  # mAh, so 12V * 10000mAh = 120000 Joules
        self.plane_mock.get_required_power = MagicMock(return_value=150)

        # Actions and vehicle states
        soc_increment = 20
        vehicle_states = ['floating', 'flying']
        actions = ['float', 'fly']
        stm = [0, -20, 20, -40]  # state transition matrix
        max_stages = 100

        # Create mdp object with mock plane
        self.mdp_instance = mdp(self.plane_mock, soc_increment, vehicle_states, max_stages, actions, stm)

    def test_soc_update_floating_full_sunlight(self):
        """Test the SoC update when floating during full sunlight."""
        dt = 60  # 60 minutes timestep
        stage = 12  # Assume midday
        self.plane_mock.get_required_power.return_value = 0  # Floating requires no energy

        soc_update = self.mdp_instance.calculate_soc_update(0, dt, stage)

        # Check that SoC increased by the correct amount
        self.assertGreater(soc_update, 0, f"Expected positive SoC update, got {soc_update}")

    def test_soc_update_flying_cloudy(self):
        """Test the SoC update when flying with some cloud cover."""
        dt = 60  # 60 minutes timestep
        stage = 6  # Morning
        self.plane_mock.get_required_power.return_value = 50  # Flying requires 50 watts

        soc_update = self.mdp_instance.calculate_soc_update(1, dt, stage)

        # Check that SoC decreases due to high energy consumption
        self.assertLess(soc_update, 0, f"Expected negative SoC update, got {soc_update}")

    def test_soc_update_floating_night(self):
        """Test the SoC update when floating at night (no solar power)."""
        dt = 60  # 60 minutes timestep
        stage = 0  # Midnight (no sunlight)
        self.plane_mock.get_required_power.return_value = 0  # Floating requires no energy

        soc_update = self.mdp_instance.calculate_soc_update(0, dt, stage)

        # Check that SoC doesn't change as no energy is consumed or generated
        self.assertEqual(soc_update, 0, f"Expected no change in SoC, got {soc_update}")

    def test_soc_update_flying_no_sunlight(self):
        """Test the SoC update when flying at night (no solar power, high energy consumption)."""
        dt = 60  # 60 minutes timestep
        stage = 0  # Midnight
        self.plane_mock.get_required_power.return_value = 60  # Flying requires 60 watts

        soc_update = self.mdp_instance.calculate_soc_update(1, dt, stage)

        # Check that SoC decreases significantly due to energy consumption without solar power
        self.assertLess(soc_update, 0, f"Expected large negative SoC update, got {soc_update}")


if __name__ == "__main__":
    unittest.main()
