import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))

# import pandas as pd
import numpy as np
import unittest
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from mdp import mdp

# def plot_surface(df, title):
#     # Extracting the multiindex levels
#     X = df.index.get_level_values(0).values.astype(float)
#     Y = df.columns.values.astype(float)
    
#     # Converting the DataFrame values into a 2D numpy array
#     Z = df.values
    
#     # Creating a meshgrid for X and Y
#     X, Y = np.meshgrid(Y, X)
    
#     # Plotting the surface
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
    
#     surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
#     # Adding labels
#     ax.set_xlabel('Stages')
#     ax.set_ylabel('State of Charge')
#     ax.set_zlabel('Value')
#     ax.set_title(f'Surface Plot for state: {title}')
    
#     # Adding a color bar to show the color scale
#     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#     plt.tight_layout()
#     plt.show()

# def plot_surfaces_by_state(df):
#     # Get the unique values in the second level of the MultiIndex
#     states = df.index.get_level_values(1).unique()
    
#     for state in states:
#         # Filter the DataFrame based on the state
#         df_state = df.xs(state, level=1)
        
#         # Plot the surface for this state
#         plot_surface(df_state, state)

class TestMDP(unittest.TestCase):
    
    def setUp(self):
        # Example setup for testing
        self.soc_increment = 20
        self.vehicle_states = ["moored", "flying"]
        self.max_stages = 3
        self.actions = ["float", "fly"]
        self.stm = [0, -40, 20, -20]
        self.mdp_instance = mdp(self.soc_increment, self.vehicle_states, self.max_stages, self.actions, self.stm)
    
    def test_create_states(self):
        expected_states = [
            (0, 'moored'), (20, 'moored'), (40, 'moored'), (60, 'moored'), (80, 'moored'), (100, 'moored'),
            (0, 'flying'), (20, 'flying'), (40, 'flying'), (60, 'flying'), (80, 'flying'), (100, 'flying')
        ]
        self.assertEqual(self.mdp_instance.states, expected_states)

    # def test_get_control_reward(self):
    #     weights = [1, 2, 3]
        
    #     # Test 'float' action
    #     self.assertEqual(self.mdp_instance.get_control_reward("float", weights), 0)
        
    #     # Test 'fly' action
    #     self.assertEqual(self.mdp_instance.get_control_reward("fly", weights), 1 * 1 * 2 * 3)

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

    

