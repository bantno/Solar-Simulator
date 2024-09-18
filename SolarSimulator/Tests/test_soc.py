import sys
import os
import numpy as np
import unittest

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))

from mdp import mdp
from seaplane_base import Seaplane


class TestMDP(unittest.TestCase):

    def setUp(self):
        # Example setup for testing
        self.soc_increment = 1
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

    def test_calculate_soc_update_fly(self):
        """
        Test that the SoC update is negative when the action is "fly", timestep is 10, and stage is 0.
        """
        action = "fly"
        timestep = 10  # 10 minutes
        stage = 0

        soc_update = self.mdp_instance.calculate_soc_update(action, timestep, stage)
        print(f"SoC update at stage {stage} is {soc_update}.")
        self.assertLess(soc_update, 0, "SoC update should be negative when the action is 'fly'.")
    
    def test_soc_update_noon_vs_midnight(self):
        """
        Test that the SoC update is less negative at noon (stage 720) compared to midnight (stage 0)
        when the action is 'fly'.
        """
        action = "fly"
        timestep = 10

        # Midnight case (stage 0)
        stage_midnight = 0
        soc_update_midnight = self.mdp_instance.calculate_soc_update(action, timestep, stage_midnight)

        # Noon case (stage 720 represents noon when timestep is 10 minutes)
        stage_noon = int((12 * 60) / timestep)  # 12 hours * 60 minutes = 720 minutes, divided by 10
        soc_update_noon = self.mdp_instance.calculate_soc_update(action, timestep, stage_noon)

        print(f"SoC update at stage {stage_midnight} is {soc_update_midnight}.")
        print(f"SoC update at stage {stage_noon} is {soc_update_noon}.")
        # Assert that the SoC update at noon is less negative than at midnight
        self.assertLess(soc_update_midnight,soc_update_noon, "SoC update should be less negative at noon than at midnight.")

    def test_soc_update_before_and_after_noon(self):
        """
        Test that the SoC update is slightly more negative just after noon than just before noon
        when the action is 'fly'.
        """
        action = "fly"
        timestep = 10

        # Calculate the stage for slightly before noon (11:00 AM)
        stage_before_noon = int(((12 * 60) - 60) / timestep)  
        soc_update_before_noon = self.mdp_instance.calculate_soc_update(action, timestep, stage_before_noon)

        # Calculate the stage for slightly after noon (1:0 PM)
        stage_after_noon = int(((12 * 60) + 60) / timestep)  
        soc_update_after_noon = self.mdp_instance.calculate_soc_update(action, timestep, stage_after_noon)

        # Assert that the SoC update just after noon is slightly more negative than just before noon
        self.assertLess(soc_update_after_noon, soc_update_before_noon,
                        "SoC update should be slightly more negative just after noon than just before noon.")


if __name__ == "__main__":
    unittest.main()
