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
from BaseClasses.whale_sighting_base import WhaleSightingProbability
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

    Attributes
    ----------
    plane : Seaplane
        An instance of the Seaplane class representing the vehicle.
    lat : float
        The latitude of the simulation location.
    lon : float
        The longitude of the simulation location.
    tz : str
        The timezone of the simulation location (e.g., 'UTC', 'US/Eastern').
    cs : bool
        A flag indicating whether to use clearsky weather data (default is False).
    location : pvlib.location.Location
        A pvlib Location object initialized with the provided latitude, longitude, 
        and timezone.

    Methods
    -------
    set_location(latitude: float, longitude: float, timezone: str) -> pvlib.location.Location
        Sets the location for the simulation based on provided latitude, longitude, 
        and timezone.
    get_DateTimeIndex(years, months, days, hours, minutes) -> pd.DatetimeIndex
        Creates a DateTimeIndex from provided date components.
    get_weather(cs: bool, times=-1) -> pd.DataFrame
        Retrieves weather data based on the specified parameters.
    get_azimuth(cs: bool, weather: pd.DataFrame) -> float
        Determines the azimuthal position of the plane based on the weather data.
    get_windspeed(cs: bool, weather: pd.DataFrame) -> float
        Retrieves the wind speed from the weather data.
    get_air_temp(cs: bool, weather: pd.DataFrame) -> float
        Retrieves the air temperature from the weather data.
    calc_collected_energy(year, month, day, periods, frequency, cs: bool) -> tuple
        Calculates the solar energy collected during the specified period.
    simulate_deployment(U, rho, takeoff_capacity, landing_capacity, start_index, 
                        end_index, dt, algo) -> tuple
        Simulates the deployment of the vehicle and determines its duty cycle based 
        on the specified parameters and chosen algorithm.
    """

    def __init__(self, plane: Seaplane, lat: float, lon: float, tz: str, cs: bool = False) -> None:
        self.plane = plane
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.cs = cs
        self.whale_table = WhaleSightingProbability().df

    def run_simulation(self,
                    start_date:datetime,
                    end_date:datetime,
                    dt,
                    algo="MDP",
                    mdp_success_prob=0,
                    true_success_prob=0,
                    runs=1):
    
        """
        Simulates and plots the duty cycle for the given plane, solar data file, cruise speed, air density, and algorithm.
        """        

        plane = self.plane
        plane.update_plane()
        print(f"Capacity in Ah: {plane.capacity}")
        print(f"Required Power: {plane.get_required_power(20,1.2)} W")
        data = self.simulate_deployment(
                start_date=start_date,
                end_date=end_date,
                dt=dt,
                algo=algo,
                mdp_success_prob=mdp_success_prob,
                true_success_prob=true_success_prob,
                num_runs=runs)
        times = pd.date_range(start_date,end_date,periods=len(data["StateHistory"][0]))
        if runs == 1:
            reward = data["Reward"][0]
            state_history = data["StateHistory"]
            print(f"Reward {reward} for algorithm {algo}.")
            
            # self.generate_simulation_summary_table(start_date,end_date,total_failure_prob=(1-mdp_success_prob),time_step=60)
        return times,data
    
    def generate_simulation_summary_table(self, start_date: tuple, end_date: tuple, 
                                    total_failure_prob: float,time_step: int) -> pd.DataFrame:
        """
        Generate a summary table for the simulation with key parameters formatted for PowerPoint.

        Parameters:
        ----------
        start_date : datetime
            The start date of the simulation.
        end_date : datetime
            The end date of the simulation.
        total_failure_prob : float
            The overall probability of failure across the simulation.
        stepwise_failure_prob : float
            The probability of failure at each simulation step.
        time_step : int
            The time step of the simulation in minutes.

        Returns:
        -------
        pd.DataFrame
            A formatted DataFrame containing the summary of the simulation.
        """

        # Format the start and end dates
        start_date_str = f"{start_date[1]:04d}-{start_date[0]:02d} {start_date[2]:02d}:00"
        end_date_str = f"{end_date[1]:04d}-{end_date[0]:02d} {end_date[2]:02d}:00"

        # Construct the summary data as a dictionary
        summary_data = {
            "Simulation Parameter": [
                "Battery Capacity (Ah)",
                "Start Date", "End Date", "Cumulative Failure Probability", 
                "Time Step (minutes)", "Latitude", "Longitude"
            ],
            "Value": [
                self.plane.capacity,
                start_date_str,
                end_date_str,
                f"{total_failure_prob:.2%}",  # Format as percentage
                time_step,
                self.lat,
                self.lon
            ]
        }

        # Create a DataFrame from the dictionary
        summary_table = pd.DataFrame(summary_data)
        # print(summary_table)
        return summary_table


    def simulate_deployment(self,start_date, end_date, dt, algo: str, mdp_success_prob, true_success_prob, num_runs):
        self.plane.calculate_weight()
        actual_data, expected_data = self.get_weather_data(start_date, end_date, dt) # move this outside loop
        vehicle_states = ["moored", "flying"]
        actions = ["float", "fly"]
        mdp_model = mdp(self.plane,
                        soc_increment=1,
                        vehicle_states=vehicle_states,
                        actions=actions,
                        start_date=start_date,
                        end_date=end_date,
                        expected_data=expected_data,
                        whale_prob=self.whale_table,
                        dt=dt,
                        mission_success_prob=mdp_success_prob
                        )
        
        auto = Autonomy(dt,mdp_model=mdp_model,data=actual_data,whale_probabilities=self.whale_table)
        mdp_model.show_progress=True
        if algo == "Greedy":
            data = []
            for i in tqdm(range(0,num_runs),desc=f"{algo} Simulation",leave=False):
                state_history_list,solar_list,reward,last_step = self._simulate_greedy_behavior(auto=auto,true_success_prob=true_success_prob)
                data.append({
                    "Iteration": i,
                    "StateHistory": state_history_list,
                    "SolarHistory": solar_list,
                    "Reward": reward,
                    "LastStep": last_step
                })
            df = pd.DataFrame(data)
            return df
        elif algo == "MDP":
            mdp_model.create_ev_table()
            data = []
            for i in tqdm(range(0,num_runs),desc=f"{algo} Simulation"):
                state_history_list,solar_list,reward,last_step = self._simulate_mdp_behavior(auto=auto,true_success_prob=true_success_prob)
                data.append({
                    "Iteration": i,
                    "StateHistory": state_history_list,
                    "SolarHistory":solar_list,
                    "Reward": reward,
                    "LastStep": last_step
                })
            df = pd.DataFrame(data)
            return df
        else:
            raise ValueError(f"Unknown algorithm: {algo}. Use 'Greedy' or 'MDP'.")
        
    def _simulate_greedy_behavior(self, auto, true_success_prob):
        """Simulate the 'Greedy' behavior."""
        return auto.simulate_simple_behavior(initial_state=(100,"moored"),
                                            true_success_prob = true_success_prob,
                                            simulate_failure = True)

    def _simulate_mdp_behavior(self, auto, true_success_prob):
        """Simulate the 'MDP' behavior."""  
        return auto.simulate_mdp_behavior(initial_state=(100,"moored"),
                                            true_success_prob = true_success_prob,
                                            simulate_failure = True)
        
    def get_weather_data(self,start_date:datetime,end_date:datetime,dt:int):
        """Return expected and actual solar and wind data for given indices."""
        actual_data = self._load_weather_data(dt)
        actual_data = actual_data.loc[start_date:end_date]
        df = self._load_expected_weather_data(dt)
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
            ) &
            ~((df['month'] == 2) & (df['day'] == 29))  # Exclude leap day
        ]
        if len(actual_data) != len(expected_data):
            raise ValueError(f"Actual data and expected data have different lengths. Actual: {len(actual_data)}, Expected: {len(expected_data)}")
        
        return actual_data,expected_data

    def _load_weather_data(self, dt: int, directory: str=r"Data\HISTORICAL_DATA"):
        """Load actual solar and wind data from pickle files with a specific timestep in the filename."""
        # Create regex pattern to match files with the specified timestep (in minutes)
        pattern = rf"data_{dt}min\.pkl$"
        
        # Search for the file in the specified directory
        actual_file = None
        for file in os.listdir(directory):
            if re.search(pattern, file):
                actual_file = os.path.join(directory, file)
                break

        if actual_file:
            actual_data = pd.read_pickle(actual_file)
            return actual_data
        else:
            raise FileNotFoundError(f"No file found in '{directory}' with timestep '{dt}' minutes.")

    def _load_expected_weather_data(self, dt: int, directory: str=r"Data\EXPECTED_DATA"):
        """Load expected solar and wind data from pickle files with a specific timestep in the filename."""
        # Create regex pattern to match files with the specified timestep (in minutes)
        pattern = rf"data_expected_{dt}min\.pkl$"
        
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
            raise FileNotFoundError(f"No file found in '{directory}' with timestep '{dt}' minutes.")