import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation
from BaseClasses.plotting_base import SolarChargePlotter

if __name__ == '__main__':
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


    
    fig = -1
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    utc_offset = timezone(timedelta(hours=0))
    start_date = pd.to_datetime(datetime(2024,7,1).replace(tzinfo=utc_offset))
    end_date = pd.to_datetime(datetime(2024,7,7).replace(tzinfo=utc_offset))

    
    # start_date = pd.to_datetime(datetime(2024,7,1))
    # end_date = pd.to_datetime(datetime(2024,7,2))

    capacities = [50]
    mdp_probs = [0.9]
    thresholds = [0.2]
    success_prob=1.0
    visualize = True
    dt=10
    NUM_RUNS = 1
    sim = Simulation(plane, lat, lon, tz, save_history=visualize)

    # Run simulation
    for cap in tqdm(capacities, desc="Processing capacities"):
        sim.plane.capacity = cap

        for threshold in tqdm(thresholds, desc=f"Processing thresholds for cap={cap}", leave=False):
            # Threshold Simulation
            algo='Threshold'
            times,data = sim.run_simulation(start_date,end_date,dt,algo=algo,mdp_success_prob=0.9,true_success_prob=success_prob,runs=NUM_RUNS,threshold=threshold)
            data.to_pickle(f"{algo}_Data_c{cap}_t{threshold}_{dt}min_{start_date.day_of_year}-{end_date.day_of_year}_{NUM_RUNS}.pkl")

        for mdp_success_prob in tqdm(mdp_probs, desc=f"Processing probabilities for cap={cap}", leave=False):
            # MDP Simulation
            algo='Optimal'
            times,data = sim.run_simulation(start_date,end_date,dt,algo=algo,mdp_success_prob=mdp_success_prob,true_success_prob=success_prob,runs=NUM_RUNS)
            data.to_pickle(f"{algo}_Data_c{cap}_p{mdp_success_prob}_{dt}min_{start_date.day_of_year}-{end_date.day_of_year}_{NUM_RUNS}.pkl")


    if visualize:
        plotter = SolarChargePlotter(r".",start_date=start_date,time_step=f"{dt}min")
        plotter.plot_data()
