import os
import datetime
import calendar

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
af_mass = 8.8 #TODO: Read in AF mass from VSPAero, multiply by safety factor
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

# Create Pareto Plots
# plotting.make_pareto_classic(plane,(1,25),250)
# plotting.make_pareto(plane)

# Set date for simulation
year = 2019
month = 6
day = 1
days_list = [1,3,7,30,365]

# Define capacities to investigate
cap = np.linspace(5,30,50)
title = "Battery Capacity Sweep for Various Mission Durations"
fig = -1
for i,days in enumerate(days_list):
    # Run battery sweep
    duty,num_takeoffs = plotting.battery_sweep(plane,cap,month=month,days=days)

    # Create battery sweep plot
    FILENAME = f"BatterySweep_{year}_{month}_{day}-combined"
    label = f"Sim Length = {days}"

    fig = plotting.plot_battery_sweep(cap,duty,label,FILENAME,fig,title)
    
    # Print optimal battery capacity
    data = data = {'Capacity': cap, 'Duty': duty}
    df = pd.DataFrame(data)
    max_duty = df['Duty'].max()
    max_duty_row = df[df['Duty'] == max_duty]
    max_cap= max_duty_row['Capacity'].values[0]
    print("Capacity for max duty cycle: {0}".format('%.2f'%max_cap))
    print("Maximum Duty Cycle: {0}".format('%.2f'%max_duty))

plt.close()

###########################################################################


# Set date for simulation
year = 2019
months_list = [1,3,6,9]
day = 1
days = 30

# Define capacities to investigate
cap = np.linspace(5,30,50)
title = "Battery Capacity Sweep for Various Months"
fig = -1
for i,month in enumerate(months_list):
    # Run battery sweep
    duty,num_takeoffs = plotting.battery_sweep(plane,cap,month=month,days=days)

    # Create battery sweep plot
    FILENAME = f"BatterySweep_{year}_combined_{day}-{days}"
    label = calendar.month_abbr[month]
    fig = plotting.plot_battery_sweep(cap,duty,label,FILENAME,fig,title)
    
    # Print optimal battery capacity
    data = data = {'Capacity': cap, 'Duty': duty}
    df = pd.DataFrame(data)
    max_duty = df['Duty'].max()
    max_duty_row = df[df['Duty'] == max_duty]
    max_cap= max_duty_row['Capacity'].values[0]
    print("Capacity for max duty cycle: {0}".format('%.2f'%max_cap))
    print("Maximum Duty Cycle: {0}".format('%.2f'%max_duty))


plt.close()

# ########################################################

# Set date for simulation
year = 2019
month = 6
day = 1
days = 30
lat_list = [-50, -25, 0, 25, 50]
plane.lon = 14.5

# Define capacities to investigate
cap = np.linspace(5,30,50)

fig = -1
title = "Battery Capacity Sweep for Various Lattitudes"
for i,lat in enumerate(lat_list):
    
    # Update plane location
    plane.update_location(lat)
    # Run battery sweep
    duty,num_takeoffs = plotting.battery_sweep(plane,cap,month=month,days=days)

    # Create battery sweep plot
    FILENAME = f"BatterySweep_{year}_{month}_{day}-{days}_lat-combined"
    label = f"{lat}\u00B0 Lattitude"
    fig = plotting.plot_battery_sweep(cap,duty,label,FILENAME,fig,title)
    
    # Print optimal battery capacity
    data = data = {'Capacity': cap, 'Duty': duty}
    df = pd.DataFrame(data)
    max_duty = df['Duty'].max()
    max_duty_row = df[df['Duty'] == max_duty]
    max_cap= max_duty_row['Capacity'].values[0]
    print("Capacity for max duty cycle: {0}".format('%.2f'%max_cap))
    print("Maximum Duty Cycle: {0}".format('%.2f'%max_duty))

plt.close()