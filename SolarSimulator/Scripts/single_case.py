from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import SingleCaseSimulation
import pandas as pd
from datetime import datetime, timezone, timedelta


# Define simulation parameters
lat = 30
lon = -90
tz = "Etc/GMT-5"
pdc0 = 0
gamma = -0.0047
capacity_ah = 30.0
voltage = 22.2
Cdtot = 0.0
Cd0 = 0.02584
S = 0.653
af_mass = 8.8
cruise_speed = 20.0
rho = 1.19
N_PROP = 0.82
N_ESC = 0.9
n_tot = N_ESC * N_PROP
start_date = "2019-07-01"
end_date = "2019-08-02"
dt = 30
num_runs = 1
visualize = False
save_dir = "."
show = False
use_expected = False

plane = Seaplane(
    lat=30,
    lon=-90,
    tz=0,
    pdc0=0,
    gamma=0,
    cd0=Cd0,
    cs=True,
    tracking=False,
    cdtot=Cdtot,
    n_tot=n_tot,
    S=S,
    af_mass=af_mass,
    voltage=voltage,
    capacity=capacity_ah,
)

expected_file = r"Data\TEST_CASES\Wind\expected_fake_weather_data_alternating.pkl"

actual_file = r"Data\TEST_CASES\Wind\fake_weather_data_alternating.pkl"

# Create the simulation instance
simulation = SingleCaseSimulation(
    plane, lat, lon, tz, expected_file, actual_file, save_history=True
)

# Define time range
utc_offset = timedelta(hours=-6)
start_date = datetime(2025, 1, 1, 0, 0).replace(tzinfo=timezone(utc_offset))
end_date = datetime(2025, 1, 5, 0, 0).replace(tzinfo=timezone(utc_offset))
dt = 15  # Time step in minutes

# Run the simulation
optimal_data = simulation.run_single_case(
    start_date=start_date,
    end_date=end_date,
    dt=dt,
    algo="Optimal",
    mdp_success_prob=0.99995,
    true_success_prob=0.9995,
)
optimal_data.to_pickle(
    r"Data/TEST_CASES/Wind/Meeting-Results/varyWind-varyWhale/single_case_data_optimal_constant_wind_and_whale.pkl"
)

# Run the simulation
threshold_data = simulation.run_single_case(
    start_date=start_date,
    end_date=end_date,
    dt=dt,
    algo="Threshold",
    mdp_success_prob=0.99995,
    true_success_prob=0.9995,
    threshold=0.25,
)

threshold_data.to_pickle(
    r"Data/TEST_CASES/Wind/Meeting-Results/varyWind-varyWhale/single_case_data_threshold_constant_wind_and_whale.pkl"
)
