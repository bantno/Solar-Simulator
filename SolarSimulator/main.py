from datetime import datetime, timedelta, timezone

import pandas as pd

from tqdm import tqdm

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation
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

utc_offset = timezone(timedelta(hours=-5))
start_date = pd.to_datetime(datetime(2019,1,2).replace(tzinfo=utc_offset))
end_date = pd.to_datetime(datetime(2019,6,2).replace(tzinfo=utc_offset))

filename = f"SimResults_{time_string}"
actual_data, expected_data = sim.get_weather_data(start_date,end_date)


# capacities = [5,10,20,30,40,50,60,70,80,90,100]
capacities = [10,55,70,100]
mdp_probs = [0.5,0.9,1.0]
success_prob=0.9
visualize = False

NUM_RUNS = 1000

for cap in tqdm(capacities, desc="Processing capacities"):
    sim.plane.capacity = cap
    solar_data_expected = expected_data["expected_solar_rad"].values
    solar_data_actual = actual_data["shortwave_radiation"].values
    
    # Greedy Simulation
    algo='Greedy'
    times,data = sim.run_simulation(start_date,end_date,algo=algo,mdp_success_prob=0.9,true_success_prob=success_prob,runs=NUM_RUNS)
    data.to_pickle(f"Greedy_Data_c{cap}_p{0.9}.pkl")
    
    if visualize :
        reward = data["Reward"]
        state_history = data["StateHistory"]
        label = f"{algo}: {sim.plane.capacity:.2f} Ah, P(S)={success_prob}, R: {round(reward[0])}"
        fig = plotting.plot_simulation_results(times,state_history,solar_data_expected,solar_data_actual,filename,fig=fig,label=label)

    for mdp_success_prob in tqdm(mdp_probs, desc=f"Processing probabilities for cap={cap}", leave=False):
        # MDP Simulation
        algo='MDP'
        times,data = sim.run_simulation(start_date,end_date,algo=algo,mdp_success_prob=0.9,true_success_prob=success_prob,runs=NUM_RUNS)
        data.to_pickle(f"MDP_Data_c{cap}_p{mdp_success_prob}.pkl")
        
        if visualize :
            reward = data["Reward"]
            state_history = data["StateHistory"]
            label = f"{algo}: {sim.plane.capacity:.2f} Ah, P(S)={success_prob}, R: {round(reward[0])}"
            fig = plotting.plot_simulation_results(times,state_history,solar_data_expected,solar_data_actual,filename,fig=fig,label=label)