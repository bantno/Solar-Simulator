import pandas as pd
import pvlib.iotools
from pvlib import location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem
import numpy as np
import matplotlib.pyplot as plt
import warnings
from SolarSimulator.BaseClasses.Seaplane_base import Seaplane

# supressing shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action='ignore', module='pvfactors')


# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
tz = 'Etc/GMT+5'


# TZ2 - massachusetts
lat2 = 41.228766304923454
lon2 = -69.91957622209218
tz2 = 'Etc/GMT+5'

pdc0 = 80
gamma = -0.0043
# Create months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months_n = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# ### Create comparison between CS and TMY
# # Determine times for Gulf of Mexico
# plane_cs = Seaplane(lat,lon,tz,pdc0,gamma,tracking=False,cs=True)
# total_energy_cs,power_kWh_cs = plane_cs.calc_tmy_energy(2021)

# plane_w = Seaplane(lat,lon,tz,pdc0,gamma,tracking=False,cs=False)
# total_energy_w,power_kWh_w = plane_w.calc_tmy_energy(2021)

# print("Total Energy (CS,TMY)")
# print(total_energy_cs,total_energy_w)

# # Creating side-by-side bar chart
# bar_width = 0.35
# index = range(len(months))

# plt.bar(index, np.divide(power_kWh_cs,months_n), bar_width, label='Clear Sky')
# plt.bar([i + bar_width for i in index], np.divide(power_kWh_w,months_n), bar_width, label='TMY')

# plt.xlabel('Months')
# plt.ylabel('Values')
# plt.title('Energy Collected by 80W Panel CS vs TMY')
# plt.xticks([i + bar_width / 2 for i in index], months)
# plt.legend()

# plt.tight_layout()
# plt.show()

# ### Create comparison between untracked and tracked TMY results
# plane_utr = Seaplane(lat,lon,tz,pdc0,gamma,tracking=False,cs=False)
# total_energy_utr,power_kWh_utr = plane_utr.calc_tmy_energy(2021)

# plane_tr = Seaplane(lat,lon,tz,pdc0,gamma,tracking=True,cs=False)
# total_energy_tr,power_kWh_tr = plane_tr.calc_tmy_energy(2021)

# print("Daily Collected Energy [kWh] (TMY - Untracked,TMY - Tracked)")
# print(total_energy_utr,total_energy_tr)

# # Sample data for two categories (e.g., A and B) across months
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# months_n = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# # Creating side-by-side bar chart
# bar_width = 0.35
# index = range(len(months))

# plt.bar(index, np.divide(power_kWh_utr,months_n), bar_width, label='Untracked')
# plt.bar([i + bar_width for i in index], np.divide(power_kWh_tr,months_n), bar_width, label='Tracked')

# plt.xlabel('Months')
# plt.ylabel('Values')
# plt.title("80W Panel - Daily Collected Energy [kWh] (TMY - Untracked,TMY - Tracked)")
# plt.xticks([i + bar_width / 2 for i in index], months)
# plt.legend()

# plt.tight_layout()
# plt.show()


# ## Create comparison between different lattitudes
# # Determine times for Gulf of Mexico
# plane_1 = Seaplane(lat,lon,tz,pdc0,gamma,tracking=True,cs=False)
# total_energy_1,power_kWh_1 = plane_1.calc_tmy_energy(2021)

# plane_2 = Seaplane(lat2,lon2,tz2,pdc0,gamma,tracking=True,cs=False)
# total_energy_2,power_kWh_2 = plane_2.calc_tmy_energy(2021)

# print("Total Energy (CS,TMY)")
# print(total_energy_1,total_energy_2)

# # Creating side-by-side bar chart
# bar_width = 0.35
# index = range(len(months))

# plt.bar(index, np.divide(power_kWh_1,months_n), bar_width, label='Gulf of Mexico')
# plt.bar([i + bar_width for i in index], np.divide(power_kWh_2,months_n), bar_width, label='Massachusetts Bay')

# plt.xlabel('Months')
# plt.ylabel('Values')
# plt.title('Energy Collected by 80W Panel Gulf vs Mass')
# plt.xticks([i + bar_width / 2 for i in index], months)
# plt.legend()

# plt.tight_layout()
# plt.show()

## Create comparison between different lattitudes
# Determine times for Gulf of Mexico
plane_1 = Seaplane(lat2,lon2,tz2,pdc0,gamma,tracking=False,cs=False)
total_energy_1,power_kWh_1 = plane_1.calc_tmy_energy(2021)

plane_2 = Seaplane(lat2,lon2,tz2,pdc0,gamma,tracking=True,cs=False)
total_energy_2,power_kWh_2 = plane_2.calc_tmy_energy(2021)

print("Total Energy (CS,TMY)")
print(total_energy_1,total_energy_2)

# Creating side-by-side bar chart
bar_width = 0.35
index = range(len(months))

plt.bar(index, np.divide(power_kWh_1,months_n), bar_width, label='Fixed')
plt.bar([i + bar_width for i in index], np.divide(power_kWh_2,months_n), bar_width, label='Tracking')

plt.xlabel('Months')
plt.ylabel('Values')
plt.title('Energy Collected by 80W Panel, Massachussetts (Fixed vs Tracking)')
plt.xticks([i + bar_width / 2 for i in index], months)
plt.legend()

plt.tight_layout()
plt.show()