import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import matplotlib.pyplot as plt

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation
from BaseClasses.plotting_base import SolarChargePlotter


from Tools import plotting, stl_slice


# Define constant parameters
lat = 25
lon = -90
tz = "Etc/GMT-5"
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


sim = Simulation(plane, lat, lon, tz, cs=False)
fig = -1
duty_cycle = []

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

utc_offset = timezone(timedelta(hours=-6))
start_date = pd.to_datetime(datetime(2019,1,2).replace(tzinfo=utc_offset))
end_date = pd.to_datetime(datetime(2019,6,2).replace(tzinfo=utc_offset))

filename = f"SimResults_{time_string}"


capacities = [20,50,80]
# capacities = [50]
mdp_probs = [0.5,0.9,1.0]
success_prob=0.9
visualize = True
dt=60
actual_data, expected_data = sim.get_weather_data(start_date,end_date,dt=dt)
NUM_RUNS = 1

# Run simulation
for cap in tqdm(capacities, desc="Processing capacities"):
    sim.plane.capacity = cap
    solar_data_expected = expected_data["expected_solar_rad"].values
    solar_data_actual = actual_data["shortwave_radiation"].values
    
    # Greedy Simulation
    algo='Greedy'
    times,data = sim.run_simulation(start_date,end_date,dt,algo=algo,mdp_success_prob=0.9,true_success_prob=success_prob,runs=NUM_RUNS)
    data.to_pickle(f"Greedy_Data_c{cap}_p{0.9}_{dt}min.pkl")

    for mdp_success_prob in tqdm(mdp_probs, desc=f"Processing probabilities for cap={cap}", leave=False):
        # MDP Simulation
        algo='MDP'
        times,data = sim.run_simulation(start_date,end_date,dt,algo=algo,mdp_success_prob=mdp_success_prob,true_success_prob=success_prob,runs=NUM_RUNS)
        data.to_pickle(f"MDP_Data_c{cap}_p{mdp_success_prob}_{dt}min.pkl")


if visualize:
    plotter = SolarChargePlotter(r".",start_date=start_date,time_step=f"{dt}min")
    plotter.plot_data()
