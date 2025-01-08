import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.stats import weibull_min, beta as beta_dist
from pvlib import location, tracking, temperature, pvsystem
from pvlib.bifacial.pvfactors import pvfactors_timeseries

# # Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from BaseClasses.mdp import mdp
from BaseClasses.whale_sighting_base import WhaleSighting
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.autonomy_base import Autonomy



# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)



class Simulation:
    """
    A class to simulate the operations of a seaplane using solar energy collection 
    and behavioral algorithms. The class handles the initialization of the simulation 
    environment, retrieves weather data, calculates solar energy collection, and 
    simulates the vehicle's deployment.
    """

    def __init__(self, plane: Seaplane, lat: float, lon: float, tz: str, save_history:bool=False, use_expected:bool=False) -> None:
        self.plane = plane
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.whale_table = WhaleSighting().probability_map
        self.save_history = save_history
        self.use_expected = False

    def run_simulation(self,
                    start_date:datetime,
                    end_date:datetime,
                    dt,
                    algo=None,
                    mdp_success_prob=0,
                    true_success_prob=0,
                    runs=1,
                    threshold=None):
    
        """
        Simulates and plots the duty cycle for the given plane, solar data file, cruise speed, air density, and algorithm.
        """        

        # NEED TO FIX THIS TO BE ABLE TO RUN 3rd algorithm type.

        plane = self.plane
        plane.update_plane()
        if isinstance(threshold,float):
            data = self.simulate_deployment(
                    start_date=start_date,
                    end_date=end_date,
                    dt=dt,
                    algo=algo,
                    mdp_success_prob=mdp_success_prob,
                    true_success_prob=true_success_prob,
                    num_runs=runs,
                    threshold=threshold)
        else:
            data = self.simulate_deployment(
                    start_date=start_date,
                    end_date=end_date,
                    dt=dt,
                    algo=algo,
                    mdp_success_prob=mdp_success_prob,
                    true_success_prob=true_success_prob,
                    num_runs=runs,
                    threshold=0.1)
                    
        times = pd.date_range(start_date,end_date,freq=f"{dt}min")
        
        return times,data

    def simulate_deployment(self,start_date, end_date, dt, algo: str, mdp_success_prob, true_success_prob, num_runs, threshold):
        self.plane.update_plane()
        expected_data = self.get_expected_weather_data(start_date=start_date,end_date=end_date,dt=dt,lat=self.lat)
        vehicle_states = ["moored", "flying"]
        actions = ["float", "fly"]
        mdp_model = mdp(self.plane,
                        soc_increment=1,
                        vehicle_states=vehicle_states,
                        actions=actions,
                        start_date=start_date,
                        end_date=end_date,
                        expected_data=expected_data,
                        whale_surface_probs=self.whale_table,
                        dt=dt,
                        mission_success_prob=mdp_success_prob
                        )
        auto = Autonomy(dt,mdp_model=mdp_model,data=None,whale_probabilities=self.whale_table)
        mdp_model.show_progress=True
        data = {}
        loc = rf"Data\SYNTHETIC_DATA\lat{int(self.lat)}"

        # Simulate the behavior
        if algo == "Threshold":
            for i in tqdm(range(num_runs), desc=f"{algo} Simulation", leave=False, mininterval=1):
                actual_data = self._load_weather_data(dt,directory=loc,i=i)
                actual_data = actual_data.loc[start_date:end_date]
                auto.data=actual_data
                if self.save_history:
                    reward, last_step,state_history_list,solar_list,whale_list = auto.simulate_simple_behavior(
                        initial_state=(100, "moored"),
                        true_success_prob=true_success_prob,
                        simulate_failure=True,
                        save_history = self.save_history,
                        threshold=threshold
                    )

                    data[i] = {
                        "Iteration": i,
                        "StateHistory": state_history_list,
                        "SolarHistory":solar_list,
                        "ExpectedSolarHistory":expected_data["expected_solar_rad"].values,
                        "WhaleHistory":whale_list,
                        "Reward": reward,
                        "LastStep": last_step
                    }
                else:
                    reward, last_step, flight_minutes = auto.simulate_simple_behavior(
                        initial_state=(100, "moored"),
                        true_success_prob=true_success_prob,
                        simulate_failure=True,
                        save_history = self.save_history,
                        threshold=threshold
                    )

                    # Store the data in a multilevel dictionary: {iteration: {reward: value, last_step: value}}
                    data[i] = {
                        "Reward": reward,
                        "LastStep": last_step,
                        "FlightHours": flight_minutes/(60/dt)
                    }
    
        elif algo == "Optimal": 
            mdp_model.create_ev_table()
            for i in tqdm(range(num_runs), desc=f"{algo} Simulation", leave=False, mininterval=1):
                actual_data = self._load_weather_data(dt,directory=loc,i=i)
                actual_data = actual_data.loc[start_date:end_date]
                auto.data=actual_data
                # Simulate the behavior
                if self.save_history:
                    reward, last_step,state_history_list,solar_list,whale_list = auto.simulate_mdp_behavior(
                        initial_state=(100, "moored"),
                        true_success_prob=true_success_prob,
                        simulate_failure=True,
                        save_history = self.save_history
                    )

                    data[i] = {
                        "Iteration": i,
                        "StateHistory": state_history_list,
                        "SolarHistory":solar_list,
                        "ExpectedSolarHistory":expected_data["expected_solar_rad"].values,
                        "WhaleHistory":whale_list,
                        "Reward": reward,
                        "LastStep": last_step
                    }
                else:
                    reward, last_step, flight_minutes = auto.simulate_mdp_behavior(
                        initial_state=(100, "moored"),
                        true_success_prob=true_success_prob,
                        simulate_failure=True
                    )

                    # Store the data in a multilevel dictionary: {iteration: {reward: value, last_step: value}}
                    data[i] = {
                        "Reward": reward,
                        "LastStep": last_step,
                        "FlightHours": flight_minutes/(60/dt)
                    }
        elif algo == "Charge Threshold": 
            for i in tqdm(range(num_runs), desc=f"{algo} Simulation", leave=False, mininterval=1):
                actual_data = self._load_weather_data(dt,directory=loc,i=i)
                actual_data = actual_data.loc[start_date:end_date]
                auto.data=actual_data
                # Simulate the behavior
                if self.save_history:
                    reward, last_step,state_history_list,solar_list,whale_list = auto.simulate_fullcharge_behavior(
                        initial_state=(100, "moored"),
                        true_success_prob=true_success_prob,
                        simulate_failure=True,
                        save_history = self.save_history
                    )

                    data[i] = {
                        "Iteration": i,
                        "StateHistory": state_history_list,
                        "SolarHistory":solar_list,
                        "ExpectedSolarHistory":expected_data["expected_solar_rad"].values,
                        "WhaleHistory":whale_list,
                        "Reward": reward,
                        "LastStep": last_step
                    }
                else:
                    reward, last_step, flight_minutes = auto.simulate_fullcharge_behavior(
                        initial_state=(100, "moored"),
                        true_success_prob=true_success_prob,
                        simulate_failure=True
                    )

                    # Store the data in a multilevel dictionary: {iteration: {reward: value, last_step: value}}
                    data[i] = {
                        "Reward": reward,
                        "LastStep": last_step,
                        "FlightHours": flight_minutes/(60/dt)
                    }
        else:
            raise ValueError(f"Unknown algorithm: {algo}. Use 'Threshold' or 'Optimal'.")

        # Convert the multilevel dictionary to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')  # 'index' means the outer keys become the rows
        return df
 
        
    def get_expected_weather_data(self,start_date:datetime,end_date:datetime,dt:int,lat):
        """Return expected and actual solar and wind data for given indices."""
        df = self._load_expected_weather_data(dt,lat)
        # Define start and end dates (month, day, hour, minute)
        start_month = start_date.month
        start_day = start_date.day
        start_hour = start_date.hour
        start_minute = start_date.minute

        end_month = end_date.month
        end_day = end_date.day
        end_hour = end_date.hour
        end_minute = end_date.minute

        # Filter the DataFrame by checking each component of the date individually
        expected_data = df[
            ((df['month'] > start_month) | 
            ((df['month'] == start_month) & (df['day'] > start_day)) |
            ((df['month'] == start_month) & (df['day'] == start_day) & (df['hour'] > start_hour)) |
            ((df['month'] == start_month) & (df['day'] == start_day) & (df['hour'] == start_hour) & (df['minute'] >= start_minute))
            ) & 
            ((df['month'] < end_month) | 
            ((df['month'] == end_month) & (df['day'] < end_day)) |
            ((df['month'] == end_month) & (df['day'] == end_day) & (df['hour'] < end_hour)) |
            ((df['month'] == end_month) & (df['day'] == end_day) & (df['hour'] == end_hour) & (df['minute'] <= end_minute))
            ) #&
            # ~((df['month'] == 2) & (df['day'] == 29))  # Exclude leap day
        ]
        # if len(actual_data) != len(expected_data):
            # raise ValueError(f"Actual data and expected data have different lengths. Actual: {len(actual_data)}, Expected: {len(expected_data)}")
        
        return expected_data

    def _load_weather_data(self, dt: int, directory: str=None, i=None):
        """Load actual solar and wind data from pickle files with a specific timestep in the filename."""
        # Create regex pattern to match files with the specified timestep (in minutes)
        if i is None :
            pattern = rf"data_{dt}min\.pkl$"
        else:
            pattern = rf"data_{dt}min_{i}\.pkl$"
        
        # Search for the file in the specified directory
        actual_file = None
        for file in os.listdir(directory):
            if re.search(pattern, file):
                actual_file = os.path.join(directory, file)
                break

        if actual_file:
            actual_data = pd.read_pickle(actual_file)
            # actual_data = actual_data[~((actual_data.index.month == 2) & (actual_data.index.day == 29))]
            return actual_data
        else:
            raise FileNotFoundError(f"No weather file found in '{directory}' with timestep '{dt}' minutes.")

    def _load_expected_weather_data(self, dt: int, lat, directory: str=r"Data\EXPECTED_DATA"):
        """Load expected solar and wind data from pickle files with a specific timestep in the filename."""
        # Create regex pattern to match files with the specified timestep (in minutes)
        directory = rf"Data\EXPECTED_DATA\lat{lat}"
        pattern = rf"data_expected_{dt}min_lat{lat}\.pkl$"
        
        # Search for the file in the specified directory
        expected_file = None
        for file in os.listdir(directory):
            if re.search(pattern, file):
                expected_file = os.path.join(directory, file)
                break

        if expected_file:
            expected_data = pd.read_pickle(expected_file)
            return expected_data
        else:
            raise FileNotFoundError(f"No expected weather file found in '{directory}' with timestep '{dt}' minutes and latitude '{lat}'.")

