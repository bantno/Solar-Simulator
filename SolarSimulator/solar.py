import os
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from BaseClasses.seaplane_base import Seaplane
from Utilities import ParetoFront
from Tools import plotting, stl_slice
from tqdm import tqdm

# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
tz = 'Etc/GMT+6'
pdc0 = 0 # nameplate power rating [W]
gamma = -0.0047 # Temperature coefficient of power [1/deg Celsius]

# Airplane params
capacity_ah = 0.0
voltage = 22.2
Cdtot = 0.0
Cd0 = 0.02584
S = 0.653 # from OpenVSP model
af_mass = 0.0 #TODO: Read in AF mass from VSPAero, multiply by safety factor
cruise_speed = 20.0 # m/s
rho = 1.19 # air density (dependent on altitude)
U = cruise_speed
N_PROP = 0.82 # from Raymer
N_ESC = 0.9 # esc efficiency estimate

# Create plane
plane = Seaplane(lat,
                 lon,
                 tz,
                 pdc0,
                 gamma,
                 cd0=Cd0*1.5,
                 cs=True,
                 tracking=False,
                 cdtot = Cdtot,
                 n_tot=N_PROP*N_ESC,
                 S=S,
                 af_mass=af_mass,
                 voltage=voltage,
                 capacity=capacity_ah)

plane_AIAA = plane

plotting.plot_solar(plane,2019,6,1,1,filename="SolarTest")