import gc
import os
import re
import signal

from datetime import datetime, timedelta, timezone
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from seaplane_base import Seaplane
from simulation_base import Simulation
from plotting_base import SolarChargePlotter

class SolarPlaneSimulation:
    def __init__(self, lat=25, lon=-90, tz="Etc/GMT-5", pdc0=0, gamma=-0.0047,
                 capacity_ah=50.0, voltage=22.2, Cdtot=0.0, Cd0=0.02584, S=0.653,
                 af_mass=8.8, cruise_speed=20.0, rho=1.19, N_PROP=0.82, N_ESC=0.9,
                 start_date="2019-07-01", end_date="2019-08-02", dt=30,
                 num_runs=10000, visualize=False, save_dir=".", show=False):
        
        # Define plane parameters
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.capacity_ah = capacity_ah
        self.voltage = voltage
        self.S = S
        self.af_mass = af_mass
        self.cruise_speed = cruise_speed
        self.rho = rho
        self.Cd0 = Cd0 * 1.5
        self.Cdtot = Cdtot
        self.n_tot = N_PROP * N_ESC
        self.show = show

        # Define simulation parameters
        self.dt = dt
        self.num_runs = num_runs
        self.visualize = visualize
        self.save_dir = save_dir

        # Time settings
        utc_offset = timezone(timedelta(hours=0))
        self.start_date = pd.to_datetime(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=utc_offset))
        self.end_date = pd.to_datetime(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=utc_offset))

        # Initialize the plane
        self.plane = Seaplane(
            lat=self.lat, lon=self.lon, tz=self.tz, pdc0=pdc0, gamma=gamma,
            cd0=self.Cd0, cs=True, tracking=False, cdtot=self.Cdtot,
            n_tot=self.n_tot, S=self.S, af_mass=self.af_mass,
            voltage=self.voltage, capacity=self.capacity_ah
        )

        # Initialize the simulation
        self.simulation = Simulation(self.plane, self.lat, self.lon, self.tz, save_history=self.visualize)
        self.results = []  # To store processed results

    def run_single(self, capacities=[50], thresholds=[0.1], mdp_probs=[0.9], success_prob=1.0):
        """
        Run the simulation without multiprocessing. 
        Sequentially processes capacities, thresholds, and MDP probabilities.

        Parameters:
        - capacities: List of battery capacities to simulate.
        - thresholds: List of thresholds for 'Threshold' algorithm.
        - mdp_probs: List of MDP success probabilities for 'Optimal' algorithm.
        - success_prob: True success probability for simulations.
        """
        for cap in tqdm(capacities, desc="Processing capacities"):
            self.simulation.plane.capacity = cap

            # Run Threshold Algorithm
            for threshold in tqdm(thresholds, desc=f"Processing thresholds for cap={cap}", leave=False):
                algo = "Threshold"
                times, data = self.simulation.run_simulation(
                    self.start_date, self.end_date, self.dt,
                    algo=algo, mdp_success_prob=0.9,
                    true_success_prob=success_prob, runs=self.num_runs,
                    threshold=threshold
                )
                filename = f"{self.save_dir}/{algo}_Data_c{cap}_t{threshold}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl"
                self._save_data(data, filename)

            # Run Optimal Algorithm
            for mdp_prob in tqdm(mdp_probs, desc=f"Processing probabilities for cap={cap}", leave=False):
                algo = "Optimal"
                times, data = self.simulation.run_simulation(
                    self.start_date, self.end_date, self.dt,
                    algo=algo, mdp_success_prob=mdp_prob,
                    true_success_prob=success_prob, runs=self.num_runs
                )
                filename = f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl"
                self._save_data(data, filename)

    def _save_data(self, data, filename):
        """
        Save the data to the specified filename.

        Parameters:
        - data: The data to save.
        - filename: The full path for saving the file.
        """
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            data.to_pickle(filename)
        except Exception as e:
            print(f"Error saving file {filename}: {e}")
        finally:
            del data  # Free memory
            gc.collect()

    def run(self, capacities=[50], thresholds=[0.1], mdp_probs=[0.9], success_prob=1.0):
        for cap in tqdm(capacities, desc="Processing capacities"):
            self.simulation.plane.capacity = cap

            for threshold in tqdm(thresholds, desc=f"Processing thresholds for cap={cap}", leave=False):
                algo = 'Threshold'
                times, data = self.simulation.run_simulation(
                    self.start_date, self.end_date, self.dt, algo=algo,
                    mdp_success_prob=0.9, true_success_prob=success_prob,
                    runs=self.num_runs, threshold=threshold
                )
                data.to_pickle(f"{self.save_dir}/{algo}_Data_c{cap}_t{threshold}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl")
                del data
                gc.collect()

            for mdp_success_prob in tqdm(mdp_probs, desc=f"Processing probabilities for cap={cap}", leave=False):
                algo = 'Optimal'
                times, data = self.simulation.run_simulation(
                    self.start_date, self.end_date, self.dt, algo=algo,
                    mdp_success_prob=mdp_success_prob, true_success_prob=success_prob,
                    runs=self.num_runs
                )
                data.to_pickle(f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_success_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl")
                del data
                gc.collect()

        if self.visualize:
            plotter = SolarChargePlotter(self.save_dir, start_date=self.start_date, time_step=f"{self.dt}min")
            plotter.plot_data()


    def _run_simulation(self, args):
        """Helper function to execute a single simulation run."""
        cap, algo, threshold, mdp_success_prob, success_prob = args
        self.simulation.plane.capacity = cap
        
        if algo == "Threshold":
            times, data = self.simulation.run_simulation(
                self.start_date, self.end_date, self.dt, algo=algo,
                mdp_success_prob=0.9, true_success_prob=success_prob,
                runs=self.num_runs, threshold=threshold
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_t{threshold}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl"
        elif algo == "Optimal":
            times, data = self.simulation.run_simulation(
                self.start_date, self.end_date, self.dt, algo=algo,
                mdp_success_prob=mdp_success_prob, true_success_prob=success_prob,
                runs=self.num_runs
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_success_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}.pkl"
        else:
            return None

        data.to_pickle(filename)
        del data
        gc.collect()

        return filename

    def run(self, capacities=[50], thresholds=[0.1], mdp_probs=[0.9], success_prob=1.0):
        tasks = []
        for cap in capacities:
            for threshold in thresholds:
                tasks.append((cap, "Threshold", threshold, None, success_prob))
            for mdp_prob in mdp_probs:
                tasks.append((cap, "Optimal", None, mdp_prob, success_prob))

        num_cores_to_use = max(1, os.cpu_count() - 1)  # Leave 1 core free

        try:
            with Pool(processes=num_cores_to_use) as pool:
                # Graceful termination on Ctrl+C
                signal.signal(signal.SIGINT, lambda sig, frame: pool.terminate())

                # Run tasks with progress bar
                for _ in tqdm(pool.imap_unordered(self._run_simulation, tasks), total=len(tasks), desc="Running simulations"):
                    pass

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user. Cleaning up...")
            pool.terminate()  # Kill remaining processes
            pool.join()       # Ensure all processes exit cleanly
            print("All processes terminated.")
        except Exception as e:
            print(f"An error occurred: {e}")
            pool.terminate()
            pool.join()
        finally:
            pool.close()
            pool.join()


# Example usage
if __name__ == "__main__":
    # Initialize the SolarPlaneSimulation with relevant parameters
    simulation = SolarPlaneSimulation(
        lat=25, lon=-90, tz="Etc/GMT-5",  # Location parameters
        start_date="2024-06-01",          # Simulation start date
        end_date="2024-7-01",            # Simulation end date
        dt=30,                            # Time step in minutes
        num_runs=100,                     # Number of simulation runs
        visualize=False,                   # Enable visualization
        save_dir=r"Results\1monthRun-0.75fail", # Directory to save results
        show=False                        # Suppress immediate plot display
    )

    # Define simulation parameters
    capacities = [10,20,30,40,50,60,70,80]  # Battery capacities in Amp-hours
    thresholds = [0.05,0.1,0.15,0.2,0.25] # Threshold values for 'Threshold' algorithm
    mdp_probs = [0.9]                              # MDP success probabilities for 'Optimal' algorithm
    success_prob = 0.75                             # True success probability

    # Run the simulation
    simulation.run(capacities=capacities, thresholds=thresholds, mdp_probs=mdp_probs, success_prob=success_prob)
    # simulation.run_single(capacities=capacities, thresholds=thresholds, mdp_probs=mdp_probs, success_prob=success_prob)

