import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaseClasses.Seaplane_base import Seaplane
import datetime
from Utilities import ParetoFront
from Tools import plotting
from tqdm import tqdm
import os

# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

# plot_endurance(plane,S,Cd0,weight,capacity,rho) # TODO: Fix weight estimation



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
year = 2019
month = 1
day = 1
days = 31

# Define capacities to investigate
cap = np.linspace(1,25,50)

# Run battery sweep
duty,num_takeoffs = plotting.battery_sweep(plane,cap,month=month,days=days)

# Create battery sweep plot
filename = "BatterySweep_{0}_{1}_{2}-{3}".format(year,month,day,days)
plotting.plot_battery_sweep(cap,duty,filename)

# Print optimal battery capacity
data = data = {'Capacity': cap, 'Duty': duty}
df = pd.DataFrame(data)
max_duty = df['Duty'].max()
max_duty_row = df[df['Duty'] == max_duty]
max_cap= max_duty_row['Capacity'].values[0]
print("Capacity for max duty cycle: {0}".format('%.2f'%max_cap))
print("Maximum Duty Cycle: {0}".format('%.2f'%max_duty))


# # TODO: Make Function
# # Run simulation for optimal battery size(s)
# # capacities = np.linspace(1,18,3).tolist()
# # capacities.append(max_cap)
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

# TODO: Make Function
# Run simulation for optimal battery size(s)
capacities = np.linspace(1,18,3).tolist()
capacities.append(max_cap)
cap = max_cap
months = [1,3,5,7]
fig = -1
duty_cycle = []

# year = 2019
# month = 7
day = 1
days = 30
filename = "SimResults_{0}_{1}-{2}_{3}-{4}".format(year,months[0],months[-1],day,days)
for i in tqdm(range(len(months))):
    plane.capacity = cap
    month = months[i]
    label = "Month: {0}".format(month)
    times,e_h,P_solar,states,dc = plotting.run_simulation(plane,year,month,day,days)
    times = np.linspace(1,days+1,len(e_h))
    fig = plotting.plot_simulation(plane,times,e_h,P_solar,states,filename,fig=fig,label=label)
    duty_cycle.append(dc)
print(duty_cycle)



# TODO: Make Function

# YEARLY DUTY CYCLE ##########################################
year = 2019
month = 1
day = 1
days = 365

plane.capacity = 5.9
# label = "{0} Ah".format(plane.capacity)
# times,e_h,P_solar,states,dc = plotting.run_simulation(plane,year,month,day,days)
# df = pd.DataFrame(e_h,index=times,columns=['battery'])
# df['P_solar'] = P_solar
# df['states'] = states
# df['duty cycle'] = dc

# df_daylight_hours = df.between_time('08:00', '18:00')
# daily_avg = df_daylight_hours['states'].resample('D').sum()/120.0*100

# print("Plane Capacity: {0}, Average Duty Cycle: {1}".format(plane.capacity,np.mean(dc)))

# plt.clf()
# plt.plot(range(0,days),daily_avg)
# plt.scatter(range(0,days),daily_avg,s=7)
# plt.title('Daily Duty Cycle')
# plt.xlabel("Day of Year")
# plt.ylabel("Duty Cycle [%]")
# plt.tight_layout()
# plt.savefig("YearlyDutyCycle.png")
# filename = "YearSweep"
# plot_path = os.path.join("Figures", f"{filename}.png")
# plt.savefig(plot_path)

plotting.plot_yearly_dc(plane,year,month,day,days)

########################################

# TODO: Make plotting function for this plot
# # Battery Power Required Plot
# plane_cs = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,weight=weight,voltage=voltage,capacity=capacity_ah)
# times_cs, P_solar_cs = plane_cs.calc_collected_energy((2019,2019),(1,1),(1,3))

# plane_w  = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=False,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,weight=weight,voltage=voltage,capacity=capacity_ah)
# times_w, P_solar_w = plane_w.calc_collected_energy((2019,2019),(1,1),(1,3))

# U = cruise_speed
# P_req_cruise = plane_cs.get_required_power(U,rho)
# P_bat_cs = P_req_cruise-P_solar_cs
# P_bat_w = P_req_cruise-P_solar_w

# plt.figure(figsize=(10, 6))
# plt.plot(times_cs,P_bat_cs,label="Clear Sky")
# plt.plot(times_w,P_bat_w,label="TMY")
# plt.xlabel("Date")
# plt.ylabel("Required Battery Power")
# plt.title("Required Battery Power vs Date")
# plt.legend()
# plt.show()



# Endurance and P_req calculations
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=22.2,capacity=28.82)

# List Parameters for different wing area values
# S =      [0.65340, 0.79]
# Cd0 =    [0.01487, 0.015]
# Cdtot =  [0.02572, 0.03]
# weight = [weight, 149.169]
# rho = 1.1

# plot_endurance(plane,[0.65340, 0.79],[0.01487, 0.015],[weight, 149.169],rho)



# TODO: Make endurance contour plot
# lat = [26.0857 , 29.1615 , 32.9148 , 35.3777 , 39.7020 , 42.07658 , 44.66028]
# lon = [-80.0695, -80.8963, -79.7388, -75.4398, -74.0343, -69.81796, -66.9243]
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=False,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=15.2,capacity=5.3)
# rho = 1.225
# U = 15
# E = np.empty((len(lat),365))

# pdc,power_kWh = plane.calc_tmy_energy(2019)

# days_of_year = np.linspace(1, 365, 365)
# latitude = lat
# endurance = E  

# # Create a meshgrid from the data
# X, Y = np.meshgrid(days_of_year, latitude)

# # Plot the contour
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(X, Y, endurance, cmap='viridis')
# plt.colorbar(contour, label='Endurance')

# # Add labels and title
# plt.xlabel('Days of the Year')
# plt.ylabel('Latitude')
# plt.title('Endurance Contour Plot')

# # Show plot
# plt.show()

# lat = 29.02291491363789
# lon = -90.23223029442693



