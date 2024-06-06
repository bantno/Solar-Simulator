import os
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from BaseClasses.seaplane_base import Seaplane
from Utilities import ParetoFront
from Tools import plotting
from tqdm import tqdm

# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

# Airplane params
capacity_ah = 10.6
voltage = 22.2
Cdtot = 0.02616
Cd0 = 0.01487
S = 0.653
af_mass = 6 #TODO: Read in AF mass from VSPAero, multiply by safety factor
cruise_speed = 20
rho = 1.1 # air density (dependent on altitude)
U = cruise_speed

# Create plane
plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0*1.5,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,af_mass=af_mass,voltage=voltage,capacity=capacity_ah)

# Create Pareto Plots
# plotting.make_pareto_classic(plane,(1,25),250)
# plotting.make_pareto(plane)

# Set date for simulation
YEAR = 2019
MONTH = 6
DAY = 1
DAYS = 31

# Define capacities to investigate
cap = np.linspace(1,15,25)

# Run battery sweep
duty,num_takeoffs = plotting.battery_sweep(plane,cap,month=MONTH,days=DAYS)

# Create battery sweep plot
FILENAME = f"BatterySweep_{YEAR}_{MONTH}_{DAY}-{DAYS}"
plotting.plot_battery_sweep(cap,duty,FILENAME)

# Print optimal battery capacity
data = data = {'Capacity': cap, 'Duty': duty}
df = pd.DataFrame(data)
max_duty = df['Duty'].max()
max_duty_row = df[df['Duty'] == max_duty]
max_cap= max_duty_row['Capacity'].values[0]
print("Capacity for max duty cycle: {0}".format('%.2f'%max_cap))
print("Maximum Duty Cycle: {0}".format('%.2f'%max_duty))


# TODO: Make Function
# Run simulation for optimal battery size(s)
# capacities = np.linspace(1,18,3).tolist()
# capacities.append(max_cap)
# capacities = [5.9]
# fig = -1
# duty_cycle = []

# year = 2019
# month = 5
# day = 30
# days = 10
# filename = "SimResults_{0}_{1}_{2}-{3}".format(year,month,day,days)
# for cap in capacities:
#     plane.capacity = cap
#     label = "{0} Ah".format(plane.capacity)
#     times,e_h,P_solar,states,dc = plotting.run_simulation(plane,year,month,day,days)
#     fig = plotting.plot_simulation(plane,times,e_h,P_solar,states,filename,fig=fig)
#     duty_cycle.append(dc)
# print(duty_cycle)



# # TODO: Make Function
# # Run simulation for optimal battery size(s)
# capacities = np.linspace(1,18,3).tolist()
# capacities.append(max_cap)
# cap = max_cap
# # months = list(range(1,13))
# months = [1]
# fig = -1
# duty_cycle = []
# year = 2019
# day = 1
# days = 30

# filename = f"SimResults_{YEAR}_{months[0]}-{months[-1]}_{day}-{days}"
# for i in tqdm(range(len(months))):
#     plane.capacity = cap
#     month = months[i]
#     label = f"Month: {month}"
#     times,e_h,P_solar,states,dc = plotting.run_simulation(plane,year,month,day,days)
#     times = np.linspace(1,days+1,len(e_h))
#     fig = plotting.plot_simulation(plane,times,e_h,P_solar,states,filename,fig=fig,label=label)
#     duty_cycle.append(dc)
# print(duty_cycle)

# # Step 2: Extract the data from all lines on the first axis using the figure object
# first_ax = fig.axes[0]  # Get the first axis from the figure
# lines = first_ax.get_lines()  # Get all line objects

# # Collect the data for each line
# data = [(line.get_xdata(), line.get_ydata(), line.get_label()) for line in lines]

# # Step 3: Create a new figure and plot the extracted data
# fig_new, ax_new = plt.subplots()

# for x_data, y_data, label in data:
#     ax_new.scatter(x_data, y_data, label=label, s=0.2)
# ax_new.set_title("Battery Charge Level")
# ax_new.set_xlabel("Dates")
# ax_new.set_ylabel("State of Charge [%]")
# ax_new.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.tight_layout()

# plt.show()


# #-------------------------------------
# # YEARLY DUTY CYCLE #
# year = 2019
# month = 1
# day = 1
# days = 365
# plane.capacity = 5.9
# plotting.plot_yearly_dc(plane,year,month,day,days)


# # Endurance and P_req calculations
# plane = Seaplane(lat,lon,tz,pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,af_mass=6,voltage=22.2,capacity=28.82)

# # List Parameters for different wing area values
# S =          [0.65340, 0.79]
# Cd0 =        [0.01487, 0.015]
# Cdtot =      [0.02572, 0.03]
# af_mass =    [6.0    , 6.0 ]
# capacities = [28.82, 28.82]
# rho = 1.1

# plotting.plot_endurance(plane,S,Cd0,af_mass,capacities,rho)


from datetime import datetime, timedelta

def day_to_month_day(day_number, year):
    day_number = int(day_number)
    # Create a datetime object for the given year and day number
    date_obj = datetime(year, 1, 1) + timedelta(day_number - 1)
    
    # Extract month and day from the datetime object as numbers
    month = date_obj.month
    day = date_obj.day
    
    return month, day


# TODO: Make function
# lat = [26.0857 , 29.1615 , 32.9148 , 35.3777 , 39.7020 , 42.07658 , 44.66028]
# lon = [-80.0695, -80.8963, -79.7388, -75.4398, -74.0343, -69.81796, -66.9243]
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,af_mass=6,voltage=15.2,capacity=20.3)
rho = 1.225
U = 20
days = 7

N_LAT = 20
N_DAYS = 73


lat = np.linspace(-70,70,N_LAT)
day = np.linspace(1, 365, N_DAYS).astype(int)
duty_cycle = np.zeros((N_LAT,N_DAYS))
plane.capacity = max_cap

# Create a meshgrid from the data
X, Y = np.meshgrid(day, lat)

for i in tqdm(range(X.shape[0])):
    for j in range(X.shape[1]):
        plane.update_location(Y[i, j])
        month,day = day_to_month_day(X[i,j],YEAR)
        _,_,_,_,dc = plotting.run_simulation(plane,YEAR,month,day,days)
        duty_cycle[i, j] = dc

# Plot the contour
plt.figure(figsize=(10, 6))
levels = np.linspace(0, 59, 20)
contour = plt.contourf(X, Y, duty_cycle, levels=levels, cmap='viridis')
plt.colorbar(contour, label='Duty Cycle [%]')

# Add labels and title
plt.xlabel('Days of the Year')
plt.ylabel('Latitude')
plt.title('Duty Cycle Contour Plot')

# Show plot
filename = "dc_contour_plot"
plot_path = os.path.join("Figures", f"{filename}.png")
plt.savefig(plot_path)