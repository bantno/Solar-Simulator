import unittest
import pandas as pd
from unittest.mock import patch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))
from seaplane_base import Seaplane
from simulation_base import Simulation
from solar_grib_base import SolarRadiationProcessor

class TestSimulation(unittest.TestCase):
    
    @patch('solar_grib_base.SolarRadiationProcessor')
    def test_calculate_expected_solar_power(self, MockSolarRadiationProcessor):
        # Setup the mock GRIB file processor and return a mock beta distribution
        mock_processor = MockSolarRadiationProcessor.return_value
        mock_beta_params = {
            (29.5, -85.25): (2.0, 5.0, 1000)  # Mock parameters: (alpha, beta, max_irradiance)
        }
        mock_processor.process_grib_file.return_value = mock_beta_params

        # Create a seaplane object
        mock_plane = Seaplane(-29.5,-87.25,'EST',pdc0=100, gamma=-0.004)  # Example parameters for the plane

        # Initialize the simulation with the mock plane and coordinates
        sim = Simulation(plane=mock_plane, lat=29.5, lon=-85.25, tz='EST')

        # Test parameters for the simulation
        year = (2024, 2024)
        month = (1, 1)
        day = (1, 1)
        periods = 10  # Define the number of time periods for the test
        frequency = '1H'  # Frequency set to 1 hour

        # Run the calculate_expected_solar_power method
        sim.calculate_expected_solar_power(year, month, day, periods, frequency)

        # Assertions to check that the SolarRadiationProcessor was called correctly
        mock_processor.process_grib_file.assert_called_once()

        # Verify that the correct parameters were retrieved from the beta distribution
        beta_dist = mock_processor.process_grib_file.return_value[(29.5, -85.25)]
        expected_value = beta_dist[0] / (beta_dist[0] + beta_dist[1]) * beta_dist[2]
        
        # Example assertion for expected value calculation
        self.assertAlmostEqual(expected_value, 285.714, places=3)

        # Additional tests can include checking the generated times array and behavior

if __name__ == '__main__':
    unittest.main()
