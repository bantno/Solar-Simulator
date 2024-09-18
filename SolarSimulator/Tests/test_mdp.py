import sys
import os
import numpy as np
import unittest

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))

from seaplane_base import Seaplane
from mdp import mdp


class TestMDP(unittest.TestCase):
    
    def setUp(self):
        # Example setup for testing
        self.soc_increment = 20
        self.vehicle_states = ["moored", "flying"]
        self.max_stages = 3
        self.actions = ["float", "fly"]
        self.stm = [0, -40, 20, -20]

            # Define constant parameters
        lat = 29.02291491363789
        lon = -90.23223029442693
        tz = "Etc/GMT+6"
        pdc0 = 0  # nameplate power rating [W]
        gamma = -0.0047  # Temperature coefficient of power [1/deg Celsius]

        # Airplane params
        capacity_ah = 5.0
        voltage = 22.2
        Cdtot = 0.0
        Cd0 = 0.02584
        S = 0.653  # from OpenVSP model
        af_mass = 8.8  # TODO: Read in AF mass from VSPAero, multiply by safety factor
        cruise_speed = 20.0  # m/s
        rho = 1.19  # air density (dependent on altitude)
        U = cruise_speed
        N_PROP = 0.82  # from Raymer
        N_ESC = 0.9  # esc efficiency estimate

        # Create plane
        plane = Seaplane(
            lat,
            lon,
            tz,
            pdc0,
            gamma,
            cd0=Cd0 * 1.5,
            cs=True,
            tracking=False,
            cdtot=Cdtot,
            n_tot=N_PROP * N_ESC,
            S=S,
            af_mass=af_mass,
            voltage=voltage,
            capacity=capacity_ah
            )

        self.plane = plane
        self.mdp_instance = mdp(self.plane,self.soc_increment, self.vehicle_states, self.max_stages, self.actions, self.stm)
    
    def test_create_states(self):
        expected_states = [
            (0, 'moored'), (20, 'moored'), (40, 'moored'), (60, 'moored'), (80, 'moored'), (100, 'moored'),
            (0, 'flying'), (20, 'flying'), (40, 'flying'), (60, 'flying'), (80, 'flying'), (100, 'flying')
        ]
        self.assertEqual(self.mdp_instance.states, expected_states)

    def test_time_of_day_func(self):
        # Test case 1: Stage 0, timestep 60 (1 hour)
        # Expected: Sin value of 0 should be 0
        assert np.isclose(self.mdp_instance.time_of_day_func(0, 60), 0.0), "Test case 1 failed"

        # Test case 2: Stage 60, timestep 1 (1 hour)
        # Expected: Stage 60 corresponds to 1 hour, sin(pi/24)
        assert np.isclose(self.mdp_instance.time_of_day_func(60, 1), np.sin(np.pi/24)), "Test case 2 failed"

        # Test case 3: Stage 720, timestep 1 (12 hours)
        # Expected: Stage 720 corresponds to 12 hours, sin(pi/2) = 1
        assert np.isclose(self.mdp_instance.time_of_day_func(720, 1), 1.0), "Test case 3 failed"

        # Test case 4: Stage 1440, timestep 1 (24 hours)
        assert np.isclose(self.mdp_instance.time_of_day_func(1440, 1), 0.0), "Test case 4 failed"

        # Test case 5: Stage 30, timestep 30 (0.5 hour)
        # Expected: Stage 30 corresponds to 0.5 hour, sin(pi/48) ~ 0.1305
        assert np.isclose(self.mdp_instance.time_of_day_func(30, 1), np.sin(np.pi/48)), "Test case 5 failed"

        # Test case 6: Stage 15, timestep 15 (0.25 hour)
        # Expected: Stage 15 corresponds to 0.25 hour, sin(pi/96) ~ 0.064
        assert np.isclose(self.mdp_instance.time_of_day_func(15, 1), np.sin(np.pi/96)), "Test case 6 failed"

        # Test case 7: Stage 10000, timestep 60 (Constant offset)
        # Expected: Stage 10000 is beyond one day, so should wrap around. Should give same result as stage 10000 % 1440
        assert np.isclose(self.mdp_instance.time_of_day_func(10000, 1), self.mdp_instance.time_of_day_func(10000 % 1440, 1)), "Test case 7 failed"

        print("All test cases passed!")
    
    def test_expected_solar_power_clear_day(self):
        """Test a clear day with no clouds and full sun."""
        irradiance_mean = 1000  # W/m^2
        cloud_prob = 0.0  # No cloud cover
        time_of_day_factor = 1.0  # Full sunlight (no attenuation)
        max_solar_power = 80  # W
        
        expected_power = mdp.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power)
        
        self.assertEqual(expected_power, max_solar_power, f"Expected full power output, got {expected_power}")

    def test_expected_solar_power_partial_cloud(self):
        """Test a partially cloudy day with some attenuation."""
        irradiance_mean = 800  # W/m^2
        cloud_prob = 0.5  # 50% cloud probability
        time_of_day_factor = 1.0  # Full sunlight
        max_solar_power = 5  # W
        
        expected_power = mdp.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power)
        
        expected_clear_power = (800 / 1000) * max_solar_power
        expected_cloud_reduction = expected_clear_power * (1 - 0.5 * cloud_prob)
        
        self.assertAlmostEqual(expected_power, expected_cloud_reduction, places=3, 
                               msg=f"Expected {expected_cloud_reduction}, got {expected_power}")

    def test_expected_solar_power_low_sunlight(self):
        """Test with low sunlight (early morning or late afternoon)."""
        irradiance_mean = 600  # W/m^2
        cloud_prob = 0.0  # No clouds
        time_of_day_factor = 0.5  # Low sunlight intensity
        max_solar_power = 5  # kW
        
        expected_power = mdp.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power)
        
        expected_clear_power = (600 / 1000) * max_solar_power * 0.5
        
        self.assertAlmostEqual(expected_power, expected_clear_power, places=3, 
                               msg=f"Expected {expected_clear_power}, got {expected_power}")

    def test_expected_solar_power_full_cloud(self):
        """Test with full cloud cover and full sunlight (clouds block half of the power)."""
        irradiance_mean = 1000  # W/m^2
        cloud_prob = 1.0  # 100% cloud probability
        time_of_day_factor = 1.0  # Full sunlight
        max_solar_power = 5  # kW
        
        expected_power = mdp.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power)
        
        expected_clear_power = max_solar_power
        expected_cloud_reduction = expected_clear_power * 0.5
        
        self.assertAlmostEqual(expected_power, expected_cloud_reduction, places=3, 
                               msg=f"Expected {expected_cloud_reduction}, got {expected_power}")

    def test_expected_solar_power_zero_irradiance(self):
        """Test when there's zero irradiance (nighttime or complete darkness)."""
        irradiance_mean = 0.0  # W/m^2
        cloud_prob = 0.0  # No cloud cover
        time_of_day_factor = 0.0  # No sunlight
        max_solar_power = 5  # kW
        
        expected_power = mdp.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power)
        
        self.assertEqual(expected_power, 0.0, f"Expected 0 power output, got {expected_power}")



if __name__ == '__main__':
    unittest.main()
    # soc_increment = 5
    # vehicle_states = ["moored", "flying"]
    # max_stages = 200
    # actions = ["float", "fly"]
    # stm = [0, -40, 20, -20]
    # mdp_instance = mdp(soc_increment, vehicle_states, max_stages, actions, stm)
    # print(mdp_instance.states)
    # mdp_instance.ev_table.to_csv('EV_table.csv')
    # print(mdp_instance.ev_table)
    # plot_surfaces_by_state(mdp_instance.ev_table)

    

