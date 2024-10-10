import os
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation
from Tools import plotting, stl_slice


# Define constant parameters
lat = 29.02291491363789
lon = -90.23223029442693
tz = "Etc/GMT+6"
pdc0 = 0  # nameplate power rating [W]
gamma = -0.0047  # Temperature coefficient of power [1/deg Celsius]

# Airplane params
capacity_ah = 50.0
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

plane_AIAA = plane

sim = Simulation(plane, lat, lon, tz, cs=True)

# # Create Pareto Plots
# plotting.make_pareto_classic(plane,(1,25),250)
# plotting.make_pareto(plane)

# Set date for simulation
YEAR = 2019
MONTH = 6
DAY = 1
DAYS = 30

# Define capacities to investigate
# cap = np.linspace(5, 30, 50)
# cap = [25]


# # #######################################################
# # Run battery sweep
# duty, num_takeoffs = plotting.battery_sweep(sim, cap, month=MONTH, days=DAYS)

# # Create battery sweep plot
# FILENAME = f"BatterySweep_{YEAR}_{MONTH}_{DAY}-{DAYS}"

# # Print optimal battery capacity
# data = data = {"Capacity": cap, "Duty": duty}
# df = pd.DataFrame(data)
# max_duty = df["Duty"].max()
# max_duty_row = df[df["Duty"] == max_duty]
# max_cap = max_duty_row["Capacity"].values[0]
# print("Capacity for max duty cycle: {0}".format("%.2f" % max_cap))
# print("Maximum Duty Cycle: {0}".format("%.2f" % max_duty))
# # #######################################################


# TODO: Make Function
# Run simulation for optimal battery size(s)
# capacities = np.linspace(1,18,3).tolist()
# capacities.append(max_cap)
plt.close()
capacities = [50]
fig = -1
duty_cycle = []

year = 2019
month = 6
day = 1
days = 15

current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"SimResults_{year}_{month}_{day}-{days}__{time_string}"
solar_file = r"Data\DISTRIBUTIONS\2022_solar_data.pkl"

for cap in capacities:
    plane.capacity = cap
    print(f"Required Power: {plane.get_required_power(20,1.2)} W")
    label = f"{plane.capacity:.2f} Ah"
    # times,e_h,P_solar,states,dc = plotting.run_simulation(sim,year,month,day,days,algo='Greedy')
    # fig = plotting.plot_simulation(sim,times,e_h,P_solar,states,filename,fig=fig,label=label)
    # duty_cycle.append(dc)
    times,e_h,P_solar,states,dc = plotting.run_simulation(sim,solar_file,U,rho,algo='MDP')
    fig = plotting.plot_simulation(sim,times,e_h,P_solar,states,filename,fig=fig,label=label)
    duty_cycle.append(dc)
print(duty_cycle)

plt.tight_layout()
plot_path = os.path.join("Figures", f"{filename}.png")
plt.savefig(plot_path)


# TODO: Make Function
# # Run simulation for optimal battery size(s)
# cap = max_cap
# # months = list(range(1,13))
# months = [1,6]
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
#     times,e_h,P_solar,states,dc = plotting.run_simulation(sim,year,month,day,days)
#     times = np.linspace(1,days+1,len(e_h))
#     fig = plotting.plot_simulation(plane,times,e_h,P_solar,states,filename,fig=fig,label=label)
#     duty_cycle.append(dc)

# first_ax = fig.axes[0]
# lines = first_ax.get_lines()
# data = [(line.get_xdata(), line.get_ydata(), line.get_label()) for line in lines]
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


# ##################### Endurance and P_req calculations########################
# plane = Seaplane(
#     lat,
#     lon,
#     tz,
#     pdc0,
#     gamma,
#     cd0=0.0145,
#     cs=True,
#     tracking=False,
#     cdtot=0.025,
#     n_tot=0.75,
#     S=0.38,
#     af_mass=6,
#     voltage=22.2,
#     capacity=28.82,
# )

# # List Parameters for different wing area values
# S = [0.4, 0.5, 0.65, 0.79, 0.95]
# Cd0 = [0.030, 0.027, 0.024, 0.021, 0.020]
# Cdtot = [0.025, 0.025, 0.02572, 0.03, 0.032]
# af_mass = [9.71, 9.86, 10.09, 10.30, 10.54]  # TODO: Update airframe mass calculation
# capacities = [10, 10, 10, 10, 10]
# rho = 1.1

# plotting.plot_endurance(plane, S, Cd0, af_mass, capacities, rho, filename="Endurance")


# # TODO: Make function
# # lat = [26.0857 , 29.1615 , 32.9148 , 35.3777 , 39.7020 , 42.07658 , 44.66028]
# # lon = [-80.0695, -80.8963, -79.7388, -75.4398, -74.0343, -69.81796, -66.9243]
# # plane = Seaplane(lat, lon, tz, pdc0,gamma,cd0=0.0145,cs=True,tracking=False,cdtot = 0.025,n_tot=.75,S=0.38,af_mass=6,voltage=15.2,capacity=20.3)


# ######################## CONTOUR PLOT ###############################
# rho = 1.19
# U = 20
# days = 30

# plane = plane_AIAA
# print("Contour Plane Cap: {plane.capacity}")
# N_LAT = 30
# N_DAYS = 50
# N_LEVELS = 19


# lat = np.linspace(-60, 60, N_LAT)
# day = np.linspace(1, 365, N_DAYS).astype(int)
# duty_cycle = np.zeros((N_LAT, N_DAYS))
# plane.capacity = max_cap

# # Create a meshgrid from the data
# X, Y = np.meshgrid(day, lat)

# for i in tqdm(range(X.shape[0])):
#     for j in range(X.shape[1]):
#         plane.update_location(Y[i, j])
#         month, day = plotting.day_to_month_day(X[i, j], YEAR)
#         _, _, _, _, dc = plotting.run_simulation(sim, YEAR, month, day, days)
#         duty_cycle[i, j] = dc

# # Plot the contour
# plt.figure(figsize=(10, 6))
# levels = np.linspace(0, np.max(duty_cycle), N_LEVELS)
# contour = plt.contourf(X, Y, duty_cycle, levels=levels, cmap="viridis")
# plt.colorbar(contour, label="Duty Cycle [%]")

# # Add labels and title
# plt.xlabel("Day of the Year")
# plt.ylabel("Latitude")
# # plt.title('Duty Cycle Contour Plot')

# # Show plot
# filename = "dc_contour_plot"
# plot_path = os.path.join("Figures", f"{filename}.png")
# plt.savefig(plot_path)


# ######################## BUOYANCY############################

# # Load the STL file
# FILE_PATH = r"SampleData\STL\WhalePlaneSkinny.stl"  # Adjust the path if necessary


# # Transverse Stability
# # Define the plane for the cross-section
# plane_origin = [0.52, 0.0, 0.0]  # Origin of the plane
# plane_normal = [1.0, 0.0, 0.0]  # Normal to the plane (XY plane)

# # TODO: write function to determine waterline
# # WATERLINE = 0.00 # = -0.3048/2+0.1328928
# PLANE_DIRECTION = "y"  # or "y"
# CUT_DIRECTION = "below"  # or "right" for plane "x", "below" or "above" for plane "y"
# WEIGHT = 8.8  # [kg]
# RHO_W = 1020  # density of water [kg/m^3]
# # h_cb = 0.2286-0.13289/2 #h_cg-0.066 # needs to be the vertical distance between the CG and the CB
# # h_cg = (0.3048)/4 # height of center of gravity
# CG = (0.49, 0.000, 0.046)


# draft, WATERLINE = stl_slice.calculate_draft(8.0, FILE_PATH)

# print(f"Draft: {draft} m")

# stl_slice.calculate_hstab(
#     FILE_PATH,
#     "lateral",
#     plane_origin,
#     plane_normal,
#     WATERLINE,
#     PLANE_DIRECTION,
#     CUT_DIRECTION,
#     WEIGHT,
#     CG,
# )


# # Longitudinal Stability
# # Define the plane for the cross-section
# plane_origin = [0.0, 0.0, 0.0]  # Origin of the plane
# plane_normal = [0.0, 1.0, 0.0]  # Normal to the plane (XY plane)

# stl_slice.calculate_hstab(
#     FILE_PATH,
#     "longitudinal",
#     plane_origin,
#     plane_normal,
#     WATERLINE,
#     PLANE_DIRECTION,
#     CUT_DIRECTION,
#     WEIGHT,
#     CG,
# )
