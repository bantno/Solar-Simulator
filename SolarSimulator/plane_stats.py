import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Seaplane_base import Seaplane
import datetime

def day_of_year_to_month(day_of_year):
    # Create a datetime object for the given day of the year in a non-leap year (e.g., 2022)
    date = datetime.datetime.strptime(str(day_of_year), "%j")
    
    # Extract the month from the datetime object and return as a number
    month_number = date.month
    
    return month_number

# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
# lat = 44.3655
# lon = -68.0818
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

# Battery Power Required Plot
plane_cs = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=15.2,capacity=5.3)
times, P_solar_cs = plane_cs.calc_collected_energy((2019,2019),(1,1),(1,30))
rho = 1.225
U = 15
P_req_cruise = plane_cs.get_required_power(U,rho)
P_bat = P_req_cruise#-P_solar_cs
available = plane_cs.capacity*plane_cs.voltage
consumed = P_req_cruise*1/plane_cs.n_tot*2 #np.trapz(P_bat[0:4],dx=60*60)*1/plane_cs.n_tot
print(available)
print(consumed)

print(consumed/available)
plt.plot(P_bat)
plt.show()

# Endurance and P_req calculations
plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=15.2,capacity=5.3)

# List Parameters for different wing area values
S =      [0.3800, 0.5700, 0.7600]
Cd0 =    [0.0145, 0.01334, 0.0127]
Cdtot =  [0.0250, 0.02471, 0.0245]
weight = [4*9.81, 4.4*9.81, 5.0*9.81]

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))
ax1.set_title('Endurance vs Forward Flight Speed')
ax1.set_xlabel('Forward Flight Speed [m/s]')
ax1.set_ylabel('Endurance [H]')
ax2.set_title('Required Power vs Forward Flight Speed')
ax2.set_xlabel("Forward Flight Speed [m/s]")
ax2.set_ylabel('Required Power [W]')

# Get endurance and required power
for i in range(0,len(S)):
    E = []
    P_req = []
    U = range(5,40)    
    for v in U:
        plane.S = S[i]
        plane.cd0 = Cd0[i]
        plane.cdtot = Cdtot[i]
        plane.weight = weight[i]
        rho = 1.225
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




# rho = 1.225
# U = 15

# lat = 29.02291491363789
# lon = -90.23223029442693

# lat = [26.0857 , 29.1615 , 32.9148 , 35.3777 , 39.7020 , 42.07658 , 44.66028]
# lon = [-80.0695, -80.8963, -79.7388, -75.4398, -74.0343, -69.81796, -66.9243]

# plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=15.2,capacity=5.3)
# rho = 1.225
# U = 15
# E = np.empty((len(lat),365))

# for i in range(0,len(lat)):
#     plane.lat = lat[i]
#     plane.lon = lon[i]
#     times, P_solar = plane.calc_collected_energy((2019,2019),(1,1),(1,3))
#     for j in range(0,365):
#         # month = day_of_year_to_month(j+1)
        
#         P_req_cruise = plane.get_required_power(U,rho)
#         P_bat = P_req_cruise#-P_solar_cs
#         E[i,j] = (plane.get_weather_endurance(P_bat,15))
# print(E)

# # Generate some sample data (replace this with your actual data)
# days_of_year = np.linspace(1, 365, 365)
# latitude = lat
# endurance = E  # Random endurance values for demonstration

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

