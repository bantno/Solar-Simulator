from datetime import datetime
import unittest
from unittest.mock import Mock
import pandas as pd
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
            whale_surface_probs=whale_prob,
            dt=dt,
            mission_success_prob=0.99,
        )

    def test_expected_reward(self):
        # P = 50
        # C = 100
        # I = 50
        # k = 1
        # l = 1
        # solar_alpha = 20
        # solar_beta = 20
        # P_H = 0
        # reward = self.mdp.expected_reward(P,C,I,k,l,solar_alpha,solar_beta,P_H)
        # self.assertAlmostEqual(reward,0.0)

        P = 110
        C = .5*100
        I = 10
        k = 1
        l = 1
        solar_alpha = 4
        solar_beta = 4
        P_H = 0
        reward = self.mdp.expected_reward(P,C,I,k,l,solar_alpha,solar_beta,P_H)
        self.assertAlmostEqual(reward,0.0)

if __name__ == '__main__':
    unittest.main()
