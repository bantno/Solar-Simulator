import sys
import os
import numpy as np
import unittest

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))

from mdp import mdp
from seaplane_base import Seaplane

####

import unittest
from unittest.mock import MagicMock
import numpy as np

class TestMDP(unittest.TestCase):
    
    def setUp(self):
        # Use MagicMock to mock the Plane object and its methods
        self.plane = MagicMock()
        self.plane.voltage = 22.2
        self.plane.capacity = 5
        
        # Mock the get_required_power method of the plane
        self.plane.get_required_power.return_value = 200  # Example value for required power
        
        self.soc_increment = 1
        self.vehicle_states = ["moored", "flying"]
        self.max_stages = 144  # Assuming 144 stages (10 minutes per timestep for a full day)
        self.actions = ["float", "fly"]
        self.stm = {}  # Stub for state transition matrix
        
        # Initialize mdp object with the mocked plane
        self.mdp_instance = mdp(self.plane, self.soc_increment, self.vehicle_states, self.max_stages, self.actions, self.stm)

    def test_soc_update_noon_vs_midnight(self):
        # Test the state of charge (SoC) update for noon and midnight comparison
        
        stage_noon = 72  # Noon stage (halfway through the day for a 144-stage day)
        stage_midnight = 0  # Midnight stage (start of the day)
        soc_i = self.mdp_instance.soc_increment
        
        soc_update_noon = self.mdp_instance.calculate_soc_update(self.plane, action="fly", dt=10, stage=stage_noon,soc_increment=soc_i)
        soc_update_midnight = self.mdp_instance.calculate_soc_update(self.plane, action="fly", dt=10, stage=stage_midnight,soc_increment=soc_i)

        print(f"SoC update at stage {stage_noon} is {soc_update_noon}")
        print(f"SoC update at stage {stage_midnight} is {soc_update_midnight}")

        self.assertLess(soc_update_midnight, soc_update_noon, "SoC update at noon should be less negative than at midnight.")

    # def test_soc_update_before_and_after_noon(self):
    #     # Test that SoC update slightly after noon is slightly more negative than before noon
        
    #     stage_before_noon = 60  # Stage just before noon
    #     stage_after_noon = 80  # Stage just after noon
        
    #     soc_update_before_noon = self.mdp_instance.calculate_soc_update(self.plane, action="fly", dt=10, stage=stage_before_noon)
    #     soc_update_after_noon = self.mdp_instance.calculate_soc_update(self.plane, action="fly", dt=10, stage=stage_after_noon)
        
    #     self.assertLess(soc_update_after_noon, soc_update_before_noon, "SoC update slightly after noon should be more negative than just before noon.")

if __name__ == "__main__":
    unittest.main()
