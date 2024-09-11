import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseClasses')))

# import pandas as pd
import numpy as np
import unittest
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mdp import mdp
from seaplane_base import Seaplane

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
    plt.tight_layout()
    plt.show()

def plot_surfaces_by_state(df):
    # Get the unique values in the second level of the MultiIndex
    states = df.index.get_level_values(1).unique()
    
    for state in states:
        # Filter the DataFrame based on the state
        df_state = df.xs(state, level=1)
        
        # Plot the surface for this state
        plot_surface(df_state, state)

if __name__ == '__main__':
    soc_increment = 5
    vehicle_states = ["moored", "flying"]
    max_stages = 200
    actions = ["float", "fly"]
    stm = [0, -40, 20, -20]

    # Define constant parameters
    lat = 29.02291491363789
    lon = -90.23223029442693
    tz = "Etc/GMT+6"
    pdc0 = 0  # nameplate power rating [W]
    gamma = -0.0047  # Temperature coefficient of power [1/deg Celsius]

    # Airplane params
    capacity_ah = 0.0
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
        capacity=capacity_ah,
)

    mdp_instance = mdp(plane,soc_increment, vehicle_states, max_stages, actions, stm)    
    plot_surfaces_by_state(mdp_instance.ev_table)

    # print(mdp_instance.states)
    # print(mdp_instance.ev_table)
    # mdp_instance.ev_table.to_csv('EV_table.csv')

