import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Seaplane_base import Seaplane


# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
tz = 'Etc/GMT+6'
pdc0 = 80
gamma = -0.0043

plane_t = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,tracking=True,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=37.0,capacity=10)
plane_f = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,weight=4*9.81,voltage=37.0,capacity=10)

times, P_solar_f = plane_f.calc_collected_energy((2019,2019),(1,1),(1,30))
times, P_solar_t = plane_t.calc_collected_energy((2019,2019),(1,1),(1,30))

# DC Power Absorbed Plot
fig = plt.plot(P_solar_f)
fig = plt.plot(P_solar_t)
plt.show()

# # Endurance and P_req calculations
# E = []
# P_req = []
# U = range(10,40)
# for v in U:
#     # U = 30
#     rho = 1.225
#     E.append(plane.get_endurance(v,rho))
#     P_req.append(plane.get_required_power(U=v,rho=rho))

# # Create a figure and two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

# # Plot endurance
# ax1.plot(U, E)
# ax1.set_title('Endurance vs Forward Flight Speed')
# ax1.set_xlabel('Forward Flight Speed [m/s]')
# ax1.set_ylabel('Endurance [H]')

# # Plot Required Power
# ax2.plot(U, P_req)
# ax2.set_title('Required Power vs Forward Flight Speed')
# ax2.set_xlabel("Forward Flight Speed [m/s]")
# ax2.set_ylabel('Required Power [W]')

# # Display the plots
# plt.show()
