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
from BaseClasses.new_mdp import ExpectedValueTable
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
        self.use_expected = use_expected

    def run_simulation(self, start_date: datetime, end_date: datetime, dt, algo=None, mdp_success_prob=0, true_success_prob=0, runs=1, threshold=None):
        """Simulates and plots the duty cycle for the given parameters."""
        self.plane.update_plane()
        threshold = threshold if isinstance(threshold, float) else 0.1

        data = self.simulate_deployment(
            start_date=start_date,
            end_date=end_date,
            dt=dt,
            algo=algo,
            mdp_success_prob=mdp_success_prob,
            true_success_prob=true_success_prob,
            num_runs=runs,
            threshold=threshold
        )

        times = pd.date_range(start_date, end_date, freq=f"{dt}min")
        return times, data

    def simulate_deployment(self, start_date, end_date, dt, algo: str, mdp_success_prob, true_success_prob, num_runs, threshold):
        self.plane.update_plane()
        times = pd.date_range(
                start=start_date,
                end=end_date,
                freq=pd.Timedelta(minutes=dt),
                inclusive="both"
            )
        expected_data = self.get_expected_weather_data(start_date, end_date, dt, self.lat, self.lon)
        
        solar_columns = ['beta_alpha', 'beta_beta', 'expected_solar_rad']
        expected_solar_data = expected_data[solar_columns].to_numpy()

        wind_columns = ['weibull_k', 'weibull_scale', 'expected_wind_speed']
        expected_wind_data = expected_data[wind_columns].to_numpy()


        loc = rf"Data\SYNTHETIC_DATA\lat{int(self.lat)}"

        whale_observation_data = self.get_whale_observation_probabilities(times)
        mdp_model = ExpectedValueTable(self.plane,
                                       expected_solar_data,
                                       expected_wind_data,
                                       whale_observation_data,
                                       soc_increment=1,
                                       timestep_min=dt,
                                       floating_failure_prob=1-true_success_prob)

        auto = Autonomy(dt, mdp_model)
        mdp_model.show_progress = True

        data = {}
        simulation_methods = {
            "Threshold": auto.simulate_simple_behavior,
            "Optimal": auto.simulate_mdp_behavior,
            "Charge Threshold": auto.simulate_fullcharge_behavior
        }

        if algo not in simulation_methods:
            raise ValueError(f"Unknown algorithm: {algo}. Use 'Threshold', 'Optimal', or 'Charge Threshold'.")

        simulate_method = simulation_methods[algo]
        if algo == "Optimal":
            mdp_model.generate_ev_table()

        for i in tqdm(range(num_runs), desc=f"{algo} Simulation", leave=False, mininterval=1):
            actual_data = self._load_weather_data(dt, directory=loc, i=i, lat=self.lat, lon=self.lon) if not self.use_expected else expected_data
            actual_data = actual_data.loc[start_date:end_date]

            result = simulate_method(
                initial_state=(100, 0),
                solar_data=actual_data["shortwave_radiation"].values,
                wind_data=actual_data["wind_speed_10m"].values,
                whale_data=whale_observation_data,
                true_success_prob=true_success_prob,
                simulate_failure=True,
                save_history=self.save_history,
                threshold=threshold if algo == "Threshold" else None
            )

            data[i] = self._format_simulation_result(result, expected_data)

        return pd.DataFrame.from_dict(data, orient='index')
    
    def _format_simulation_result(self, result, expected_data):
        if self.save_history:
            reward, last_step, state_history, solar_list, whale_list = result
            return {
                "Reward": reward,
                "LastStep": last_step,
                "StateHistory": state_history,
                "SolarHistory": solar_list,
                "ExpectedSolarHistory": expected_data["expected_solar_rad"].values,
                "WhaleHistory": whale_list
            }
        else:
            reward, last_step, flight_minutes = result
            return {
                "Reward": reward,
                "LastStep": last_step,
                "FlightHours": flight_minutes / 60
            }
 
        
    def get_expected_weather_data(self,start_date:datetime,end_date:datetime,dt:int,lat,lon):
        """Return expected and actual solar and wind data for given indices."""
        df = self._load_expected_weather_data(dt,lat,lon)
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
            ) & ~((df['month'] == 2) & (df['day'] == 29))  # Exclude leap day
            
        ]
        
        return expected_data

    def _load_weather_data(self, dt: int, directory: str=None, i=None, lat=None, lon=None):
        """Load actual solar and wind data from pickle files with a specific timestep and optional index in the filename."""
        # Create regex pattern to match files with the specified timestep, latitude, longitude, and index
        if i is None:
            if lat is not None and lon is not None:
                pattern = rf"data_lat{lat}_lon{lon}_{dt}min(_{i})?\.pkl$"
            else:
                pattern = rf"data_{dt}min(_{i})?\.pkl$"
        else:
            if lat is not None and lon is not None:
                pattern = rf"data_lat{lat}_lon{lon}_{dt}min_{i}\.pkl$"
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
            # actual_data = actual_data[~((actual_data.index.month == 2) & (actual_data.index.day == 29))]  # Uncomment if needed
            return actual_data
        else:
            raise FileNotFoundError(f"No weather file found in '{directory}' with timestep '{dt}' minutes, latitude '{lat}', longitude '{lon}', and index '{i}'.")


    def _load_expected_weather_data(self, dt: int, lat: float, lon: float, directory: str=r"Data\EXPECTED_DATA"):
        """Load expected solar and wind data from pickle files with a specific timestep in the filename."""
        # Create regex pattern to match files with the specified timestep, latitude, and longitude
        pattern = rf"data_expected_lat{lat}_lon{lon}_{dt}min.pkl$"
        
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
            raise FileNotFoundError(f"No expected weather file found in '{directory}' with timestep '{dt}' minutes, latitude '{lat}', and longitude '{lon}'.")

    @staticmethod
    def get_whale_observation_probabilities(time_index: pd.DatetimeIndex):
        # Define the time intervals and probabilities
        time_intervals = [
            # ("0600", "0800", 0.082),
            ("0800", "1000", 0.098),
            ("1000", "1200", 0.095),
            ("1200", "1400", 0.217),
            ("1400", "1600", 0.215),
            ("1600", "2000", 0.278)
        ]
        
        # Function to get the whale observation probability for a given time
        def get_probability_for_time(time):
            time_str = time.strftime("%H%M")
            for interval_start, interval_end, prob in time_intervals:
                if interval_start <= time_str < interval_end:
                    return prob
            return 0.0  # Return 0.0 if the time doesn't fall into any interval

        # Apply the function to the DatetimeIndex and return the probabilities
        probabilities = np.array(time_index.map(get_probability_for_time))
        return probabilities