import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaseClasses.Seaplane_base import Seaplane
import datetime
from Utilities import ParetoFront
from Tools import plotting
from tqdm import tqdm
import os

########################################################


# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
# lat = 44.3655
# lon = -68.0818
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

# # Airplane params
# capacity_ah = 10.6
# voltage = 22.2
# Cdtot = 0.02616
# Cd0 = 0.01487
# S = 0.653
# af_mass = 6
# cruise_speed = 20
# rho = 1.1 # air density (dependent on altitude)
# U = cruise_speed
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,af_mass=af_mass,voltage=voltage,capacity=capacity_ah)


# # List Parameters for different wing area values
# S =      [0.65340, 0.79]
# Cd0 =    [0.01487, 0.015]
# Cdtot =  [0.02572, 0.03]
# af_mass = [af_mass, 149.169/9.81]
# capacity = [capacity_ah, 28.8]

# plot_endurance(plane,S,Cd0,weight,capacity,rho) # TODO: Fix weight estimation

############################################

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

plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0*1.5,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,af_mass=af_mass,voltage=voltage,capacity=capacity_ah)

# make_pareto(plane)


year = 2019
month = 1
day = 1
days = 31

cap = np.linspace(1,25,50)
duty,num_takeoffs = plotting.battery_sweep(plane,cap,month=month,days=days)
filename = "BatterySweep_{0}_{1}_{2}-{3}".format(year,month,day,days)
plotting.plot_battery_sweep(cap,duty,filename)

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
# capacities = [max_cap]
# fig = -1
# duty_cycle = []

# # year = 2019
# month = 6
# # day = 1
# days = 7
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
months = [4,6]
fig = -1
duty_cycle = []

# year = 2019
# month = 7
day = 10
days = 2
filename = "SimResults_{0}_{1}-{2}_{3}-{4}".format(year,months[0],months[1],day,days)
for i in range(len(months)):
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
duty = []
monthly_duty = []
daily_duty = []
days = 1
year = 2019
plane.capacity = max_cap
for j in tqdm(range(1,13)):
    if j==12:
        days_in_month=31
    else:
        days_in_month = (datetime.date(year, j % 12 + 1, 1) - datetime.date(year, j, 1)).days
    for i in range(1, days_in_month+1):
        times, P_solar = plane.calc_collected_energy((year,year),(j,j),(i,i),periods=12*24*days,frequency='5min')
        duty_cycle,e_h,state,_ = plane.simulate_deployment(U,rho,.100,.15,P_solar,5)
        duty.append(duty_cycle)
        daily_duty.append(duty_cycle)
    monthly_duty.append(np.mean(duty))
    duty = []

print("Plane Capacity: {0}, Average Duty Cycle: {1}".format(plane.capacity,np.mean(daily_duty)))


# Pretty sure this method of doing this is wack
# plt.clf()
# plt.plot(daily_duty)
# plt.title('Daily Duty Cycle')
# plt.xlabel("Day of Year")
# plt.ylabel("Duty Cycle")
# plt.savefig("YearlyDutyCycle.png")
# filename = "YearSweep"
# plot_path = os.path.join("Figures", f"{filename}.png")
# plt.savefig(plot_path)

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

# plot_endurance(plane,S,Cd0,weight,rho)



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



