import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseClasses')))

import pandas as pd
import numpy as np
import unittest
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mdp import mdp

def plot_surface(df, title):
    # Extracting the multiindex levels
    X = df.index.get_level_values(0).values.astype(float)
    Y = df.columns.values.astype(float)
    
    # Converting the DataFrame values into a 2D numpy array
    Z = df.values
    
    # Creating a meshgrid for X and Y
    X, Y = np.meshgrid(Y, X)
    
    # Plotting the surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    # Adding labels
    ax.set_xlabel('Stages')
    ax.set_ylabel('State of Charge')
    ax.set_zlabel('Value')
    ax.set_title(f'Surface Plot for state: {title}')
    
    # Adding a color bar to show the color scale
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.show()

def plot_surfaces_by_state(df):
    # Get the unique values in the second level of the MultiIndex
    states = df.index.get_level_values(1).unique()
    
    for state in states:
        # Filter the DataFrame based on the state
        df_state = df.xs(state, level=1)
        
        # Plot the surface for this state
        plot_surface(df_state, state)

class TestMDP(unittest.TestCase):
    
    def setUp(self):
        # Example setup for testing
        self.soc_increment = 20
        self.vehicle_states = ["moored", "flying"]
        self.max_stages = 3
        self.actions = ["float", "fly"]
        self.mdp_instance = mdp(self.soc_increment, self.vehicle_states, self.max_stages, self.actions)
    
    def test_create_states(self):
        expected_states = [
            (0, 'moored'), (20, 'moored'), (40, 'moored'), (60, 'moored'), (80, 'moored'), (100, 'moored'),
            (0, 'flying'), (20, 'flying'), (40, 'flying'), (60, 'flying'), (80, 'flying'), (100, 'flying')
        ]
        self.assertEqual(self.mdp_instance.states, expected_states)
    
    def test_get_activation_vector(self):
        # Test for `w = 0`
        self.assertEqual(self.mdp_instance.get_activation_vector(0, [0]), 0)
        self.assertEqual(self.mdp_instance.get_activation_vector(1, [0]), -40)
        
        # Test for `w = 1`
        self.assertEqual(self.mdp_instance.get_activation_vector(0, [1]), 20)
        self.assertEqual(self.mdp_instance.get_activation_vector(1, [1]), -20)

    def test_get_control_reward(self):
        weights = [1, 2, 3]
        
        # Test 'float' action
        self.assertEqual(self.mdp_instance.get_control_reward("float", weights), 0)
        
        # Test 'fly' action
        self.assertEqual(self.mdp_instance.get_control_reward("fly", weights), 1 * 1 * 2 * 3)
        
    def test_daylight(self):
        # Test different hours
        self.assertEqual(self.mdp_instance.daylight(0), 0)
        self.assertEqual(self.mdp_instance.daylight(6), 1)
        self.assertEqual(self.mdp_instance.daylight(18), 0)
        self.assertEqual(self.mdp_instance.daylight(24), 0)

if __name__ == '__main__':
    # unittest.main()s
    soc_increment = 1
    vehicle_states = ["moored", "flying"]
    max_stages = 355
    actions = ["float", "fly"]
    mdp_instance = mdp(soc_increment, vehicle_states, max_stages, actions)
    # print(mdp_instance.states)
    # print(mdp_instance.ev_table)
    plot_surfaces_by_state(mdp_instance.ev_table)

    

