import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Seaplane_base import Seaplane

def plot_endurance(plane,S,Cd0,weight,rho):
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
    ax1.set_title('Endurance vs Forward Flight Speed')
    ax1.set_xlabel('Forward Flight Speed [m/s]')
    ax1.set_ylabel('Endurance [H]')
    ax2.set_title('Required Power vs Forward Flight Speed')
    ax2.set_xlabel("Forward Flight Speed [m/s]")
    ax2.set_ylabel('Required Power [W]')

    # Get endurance and required power

    if isinstance(S,float):
        E = []
        P_req = []
        U = range(5,40)    
        for v in U:
            E.append(plane.get_endurance(v,rho))
            P_req.append(plane.get_required_power(U=v,rho=rho))
        label_1 = "S = {0}".format(plane.S)
        ax1.plot(U, E,label=label_1)
        # Plot Required Power
        ax2.plot(U, P_req,label=label_1)
    else:
        for i in range(0,len(S)):
            E = []
            P_req = []
            U = range(5,40)
            for v in U:
                plane.S = S[i]
                plane.cd0 = Cd0[i]
                plane.weight = weight[i]
                E.append(plane.get_endurance(v,rho))
                P_req.append(plane.get_required_power(U=v,rho=rho))
            # Plot endurance
            label_1 = "S = {0}".format(plane.S)
            ax1.plot(U, E,label=label_1)
            # Plot Required Power
            ax2.plot(U, P_req,label=label_1)

    # Display the plots
    ax1.legend()
    ax2.legend()
    plt.show()
    return

# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
# lat = 44.3655
# lon = -68.0818
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

# Airplane params
capacity_ah = 10.6
voltage = 22.2
Cdtot = 0.025
Cd0 = 0.0145
S = 0.63
weight = 6*9.81
cruise_speed = 20
rho = 1.1 # air density (dependent on altitude)
U = 15

plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=Cd0,cs=True ,tracking=False,cdtot = Cdtot,n_tot=.75,S=S,weight=weight,voltage=voltage,capacity=capacity_ah)
duty = []

for j in range(1,12):
    for i in range(1,31):
        times, P_solar = plane.calc_collected_energy((2019,2019),(j,j+1),(i,i+1),24*5,'5min')
        duty_cycle,e_h,state = plane.simulate_deployment(U,rho,.85,.3,10,P_solar,5)
        duty.append(duty_cycle)

plt.plot(duty)
plt.show()

# # Plot Results
# # TODO: Make plotting function for this plot
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10,5))
# # ax1.set_title('')
# ax1.set_xlabel('Dates')
# ax1.set_ylabel('Battery Charge [%]')
# # ax2.set_title('')
# ax2.set_xlabel('Dates')
# ax2.set_ylabel('Power [W]')
# # ax3.set_title('')
# ax3.set_xlabel('Dates')
# ax3.set_ylabel('State')

# ax1.plot(times,e_h[0:-1])
# ax2.plot(times,P_solar)
# ax3.plot(times,state[0:-1])

plt.tight_layout() 
plt.show()



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



# # Endurance and P_req calculations
# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=22.2,capacity=28.82)

# # List Parameters for different wing area values
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



