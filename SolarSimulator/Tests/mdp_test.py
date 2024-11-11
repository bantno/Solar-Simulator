import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from datetime import datetime
from BaseClasses.mdp import mdp  # Adjust the import based on your file structure

class TestMDP(unittest.TestCase):
    
    def setUp(self):
        """
        Initialize necessary variables and instances for testing.
        """
        # Sample data to initialize the MDP
        plane = Mock()
        plane.get_required_power.return_value = 200  # Example required power value
        plane.idle_power = 0  
        plane.voltage = 24  # Example voltage
        plane.capacity = 50  # Example capacity in Amp Hours
        plane.S = 0.5  # Example solar panel area in square meters

        soc_increment = 1
        dt = 30
        vehicle_states = ["moored", "flying"]
        actions = ["float", "fly"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        # Creating dummy expected data
        expected_data = pd.DataFrame({
            "expected_solar_rad": [100] * 49,  # Sample solar data
            "expected_wind_speed": [10] * 49   # Sample wind data
        })
        self.time_intervals = [
            (0, 120),     # 0000-0200
            (120, 240),   # 0200-0400
            (240, 360),   # 0400-0600
            (360, 480),   # 0600-0800
            (480, 600),   # 0800-1000
            (600, 720),   # 1000-1200
            (720, 840),   # 1200-1400
            (840, 960),   # 1400-1600
            (960, 1080),  # 1600-1800
            (1080, 1200), # 1800-2000
            (1200, 1320), # 2000-2200
            (1320, 1440)  # 2200-2400
        ]
        
        # Define the sighting probabilities
        self.sighting_probabilities = [0.073, 0.093, 0.065, 0.082, 0.098, 0.217, 0.183, 0.278, 0.183, 0.204, 0.090, 0.090]

        # Create a pandas DataFrame to store the data
        self.df = pd.DataFrame({
            'Time Interval': self.time_intervals,
            'Sighting Probability': self.sighting_probabilities
        })
        whale_prob=self.df

        
        self.mdp = mdp(
            plane=plane,
            soc_increment=soc_increment,
            vehicle_states=vehicle_states,
            actions=actions,
            start_date=start_date,
            end_date=end_date,
            expected_data=expected_data,
            whale_prob=whale_prob,
            dt=dt,
            mission_success_prob=0.99,
        )

    def test_create_states(self):
        """
        Test if states are created correctly based on SOC increments and vehicle states.
        """
        states = self.mdp.create_states(self.mdp.soc_increment, self.mdp.vehicle_states)
        expected_states = [
            (0, "moored"), (1, "moored"), (2, "moored"), (3, "moored"), (4, "moored"), (5, "moored"),
            (95, "flying"), (96, "flying"), (97, "flying"), (98, "flying"), (99, "flying"), (100, "flying")
            # Continue based on increments
        ]
        self.assertEqual(states[:4], expected_states[:4])  # Only checking the first few for brevity
        self.assertEqual(states[-5:], expected_states[-5:])  # Only checking the first few for brevity
        # print("Passed create_states test.")

    def test_transition_probabilities(self):
        """
        Test if the transition probabilities (success and failure) are calculated correctly.
        """
        state = (50, "flying")
        action = "fly"
        stage = 10
        success_prob, failure_prob = self.mdp.calculate_maneuver_probabilities(state[1], action, stage)
        self.assertAlmostEqual(success_prob + failure_prob, 1.0, places=5)
        # print("Passed transition probabilities test.")

    def test_reward_function(self):
        """
        Test the reward function for specific state-action combinations.
        """
        state = (50, "moored")
        action = "fly"
        stage = 10
        reward = self.mdp.R(state, action, stage)
        # Assert some condition based on expected reward value (placeholder)
        self.assertIsInstance(reward, float)
        # print("Passed reward function test.")

    def test_soc_update(self):
        """
        Test the calculate_soc_update function for different actions and solar power levels.
        """
        soc_change_float = self.mdp.calculate_soc_update(self.mdp.plane, "float", self.mdp.dt, 50)
        soc_change_fly = self.mdp.calculate_soc_update(self.mdp.plane, "fly", self.mdp.dt, 50)
        # Validate SOC changes based on expected behavior
        self.assertTrue(soc_change_float >= 0)
        self.assertTrue(soc_change_fly <= 0)
        # print("Passed soc_update test.")

    def test_generate_potential_states(self):
        """
        Test the generation of potential states for MCS simulation
        """
        state = (50,"moored")
        action = "fly"
        solar_alpha = 8
        solar_beta = 5
        N = 100
        samples = self.mdp._generate_potential_states(state,action,solar_alpha,solar_beta,1367,N)

        self.assertTrue(len(samples)==N)
        socs = np.array([s[0] for i,s in enumerate(samples)]).astype(float)
        self.assertTrue(all(socs>=0) and all(socs<100))
        print(socs)


if __name__ == '__main__':
    unittest.main()
