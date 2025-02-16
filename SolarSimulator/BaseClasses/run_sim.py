import gc
import os
import signal

from datetime import datetime, timedelta, timezone
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.simulation_base import Simulation


class SolarPlaneSimulation:
    def __init__(
        self,
        lat=25,
        lon=-90,
        tz="Etc/GMT-5",
        pdc0=0,
        gamma=-0.0047,
        capacity_ah=50.0,
        voltage=22.2,
        Cdtot=0.0,
        Cd0=0.02584,
        S=0.653,
        af_mass=8.8,
        cruise_speed=20.0,
        rho=1.19,
        N_PROP=0.82,
        N_ESC=0.9,
        start_date="2019-07-01",
        end_date="2019-08-02",
        dt=30,
        num_runs=100,
        visualize=False,
        save_dir=".",
        show=False,
        use_expected=False,
        simulate_failure=True,
        transition_model=None,
        use_multiprocessing=True,
        failure_penalty=None,
    ):

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
        self.use_multiprocessing = use_multiprocessing
        self.failure_penalty = failure_penalty

        # Time settings

        utc_offset = timezone(self._get_utc_offset(self.lat, self.lon))
        self.start_date = pd.to_datetime(
            datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=utc_offset)
        )
        self.end_date = pd.to_datetime(
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=utc_offset)
        )

        # Initialize the plane
        self.plane = Seaplane(
            lat=self.lat,
            lon=self.lon,
            tz=self.tz,
            pdc0=pdc0,
            gamma=gamma,
            cd0=self.Cd0,
            cs=True,
            tracking=False,
            cdtot=self.Cdtot,
            n_tot=self.n_tot,
            S=self.S,
            af_mass=self.af_mass,
            voltage=self.voltage,
            capacity=self.capacity_ah,
        )

        # Initialize the simulation
        self.simulation = Simulation(
            self.plane,
            self.lat,
            self.lon,
            self.tz,
            save_history=self.visualize,
            use_expected=use_expected,
            simulate_failure=simulate_failure,
            failure_penalty=failure_penalty
        )
        self.results = []  # To store processed results
        self.transition_model = transition_model

    @staticmethod
    def _get_utc_offset(lat, lon, date=None):
        # Determine the timezone from latitude and longitude
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if timezone_str is None:
            raise ValueError("Could not determine timezone for the given location.")

        # Use the date provided or the current time
        date = date or datetime.now()

        # Calculate the UTC offset
        timezone = ZoneInfo(timezone_str)
        utc_offset_seconds = date.astimezone(timezone).utcoffset().total_seconds()
        utc_offset = timedelta(seconds=utc_offset_seconds)

        return utc_offset

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

    def run(
        self,
        capacities=[],
        thresholds=[],
        mdp_probs=[],
        charge_thresholds=[],
        success_prob=1.0,
    ):
        """
        Assign tasks for a simulation run.
        """
        tasks = []
        for cap in capacities:
            for threshold in thresholds:
                tasks.append((cap, "Threshold", threshold, None, success_prob))
            for mdp_prob in mdp_probs:
                tasks.append((cap, "Optimal", None, mdp_prob, success_prob))
            for charge_threshold in charge_thresholds:
                tasks.append((cap, "Charge Threshold", None, charge_threshold, success_prob))

        if self.use_multiprocessing:
            num_cores_to_use = max(1, os.cpu_count()-1) # Leave one core to prevent freezing
            print(f"Running with {num_cores_to_use} cores.")

            try:
                with Pool(processes=num_cores_to_use) as pool:
                    # Graceful termination on Ctrl+C
                    signal.signal(signal.SIGINT, lambda sig, frame: pool.terminate())

                    # Run tasks with progress bar
                    for _ in tqdm(
                        pool.imap_unordered(self._simulation_task, tasks),
                        total=len(tasks),
                        desc="Running simulations",
                    ):
                        pass

            except KeyboardInterrupt:
                print("\nSimulation interrupted by user. Cleaning up...")
                pool.terminate()  # Kill remaining processes
                pool.join()  # Ensure all processes exit cleanly
                print("All processes terminated.")
            except Exception as e:
                print(f"An error occurred: {e}")
                pool.terminate()
                pool.join()
            finally:
                pool.close()
                pool.join()
        else:
            for task in tqdm(tasks, desc="Running simulations"):
                self._simulation_task(task)

    def _simulation_task(self, args):
        """Helper function to execute a single simulation run."""
        cap, algo, threshold, mdp_success_prob, success_prob = args
        self.simulation.plane.capacity = cap

        if algo == "Threshold":
            times, data = self.simulation.run_simulation(
                self.start_date,
                self.end_date,
                self.dt,
                algo=algo,
                mdp_success_prob=0.0,
                true_success_prob=success_prob,
                runs=self.num_runs,
                threshold=threshold,
                transition_model=self.transition_model,
                failure_penalty=self.failure_penalty
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_t{threshold}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}_lat{self.lat}.pkl"
        elif algo == "Optimal":
            times, data = self.simulation.run_simulation(
                self.start_date,
                self.end_date,
                self.dt,
                algo=algo,
                mdp_success_prob=mdp_success_prob,
                true_success_prob=success_prob,
                runs=self.num_runs,
                transition_model=self.transition_model,
                failure_penalty=self.failure_penalty
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_success_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}_lat{self.lat}.pkl"
        elif algo == "Charge Threshold":
            times, data = self.simulation.run_simulation(
                self.start_date,
                self.end_date,
                self.dt,
                algo=algo,
                mdp_success_prob=mdp_success_prob,
                true_success_prob=success_prob,
                runs=self.num_runs,
                transition_model=self.transition_model,
                failure_penalty=self.failure_penalty,
            )
            filename = f"{self.save_dir}/{algo}_Data_c{cap}_p{mdp_success_prob}_{self.dt}min_{self.start_date.day_of_year}-{self.end_date.day_of_year}_{self.num_runs}_lat{self.lat}.pkl"
        else:
            return None

        data.to_pickle(filename)
        del data
        gc.collect()
        return filename


# Example usage
if __name__ == "__main__":
    #TODO: Write Example use case
    pass
