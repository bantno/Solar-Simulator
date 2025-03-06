from BaseClasses.seaplane_base import Seaplane
from BaseClasses.valueFunction_base import ValueFunction
from BaseClasses.transition_model_base import ProbabilityModelFactory
import os
import re
import datetime
import numpy as np
import pandas as pd

class weather_data:
    def __init__(self, lat, lon, dt, directory, i=None):
        self.lat = lat
        self.lon = lon
        self.dt = dt
        self.directory = directory
        self.i = i

    def get_expected_weather_data(
            self, start_date: datetime, end_date: datetime, dt: int, lat, lon
        ):
            """Return expected and actual solar and wind data for given indices."""
            df = self._load_expected_weather_data(dt, lat, lon)
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
                (
                    (df["month"] > start_month)
                    | ((df["month"] == start_month) & (df["day"] > start_day))
                    | (
                        (df["month"] == start_month)
                        & (df["day"] == start_day)
                        & (df["hour"] > start_hour)
                    )
                    | (
                        (df["month"] == start_month)
                        & (df["day"] == start_day)
                        & (df["hour"] == start_hour)
                        & (df["minute"] >= start_minute)
                    )
                )
                & (
                    (df["month"] < end_month)
                    | ((df["month"] == end_month) & (df["day"] < end_day))
                    | ((df["month"] == end_month) & (df["day"] == end_day) & (df["hour"] < end_hour))
                    | (
                        (df["month"] == end_month)
                        & (df["day"] == end_day)
                        & (df["hour"] == end_hour)
                        & (df["minute"] <= end_minute)
                    )
                )
                & ~((df["month"] == 2) & (df["day"] == 29))  # Exclude leap day
            ]

            return expected_data

    def _load_weather_data(self, dt: int, directory: str = None, i=None, lat=None, lon=None):
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
            raise FileNotFoundError(
                f"No weather file found in '{directory}' with timestep '{dt}' minutes, latitude '{lat}', longitude '{lon}', and index '{i}'."
            )

    def _load_expected_weather_data(
        self, dt: int, lat: float, lon: float, directory: str = r"Data\EXPECTED_DATA"
    ):
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
            raise FileNotFoundError(
                f"No expected weather file found in '{directory}' with timestep '{dt}' minutes, latitude '{lat}', and longitude '{lon}'."
            )

        # Define the time intervals and probabilities
    def get_whale_observation_probabilities(self, time_index: pd.DatetimeIndex, solar_radiation: np.ndarray) -> np.ndarray:
        """
        Get whale observation probabilities based on percentages outlined in PAPER.
        Also assume that nothing can be observed with no sunlight.
        
        Parameters:
        - time_index (pd.DatetimeIndex): List of times to get whale observation probabilities.
        - solar_radiation (np.ndarray): Corresponding solar radiation values.

        Returns:
        - np.ndarray: Whale observation probabilities for each time index.
        """

        # Define the time intervals and probabilities
        # time_intervals = [
        #     ("06:00", "08:00", 0.082),
        #     ("08:00", "10:00", 0.098),
        #     ("10:00", "12:00", 0.095),
        #     ("12:00", "14:00", 0.217),
        #     ("14:00", "16:00", 0.215),
        #     ("16:00", "20:00", 0.278),
        # ]

        time_intervals = [
            ("06:00", "08:00", 0.5),
            ("08:00", "10:00", 0.5),
            ("10:00", "12:00", 0.75),
            ("12:00", "14:00", 0.75),
            ("14:00", "16:00", 0.25),
            ("16:00", "20:00", 0.25),
        ]

        # Initialize probabilities with zeros
        probabilities = np.zeros_like(solar_radiation, dtype=float)

        # Convert string intervals to time objects for easy comparison
        for start, end, prob in time_intervals:
            mask = (time_index.time >= pd.to_datetime(start).time()) & (time_index.time < pd.to_datetime(end).time())
            probabilities[mask] = prob

        # Apply sunlight condition (if solar radiation is zero, probability is zero)
        probabilities[solar_radiation == 0] = 0.0

        return probabilities

    @staticmethod
    def is_data_same_length(solar_data, wind_data, whale_data) -> bool:
        return len(np.unique([len(solar_data), len(wind_data), len(whale_data)])) == 1


seaplane = Seaplane(30,-90, "Etc/GMT-6",0,1.0)
weatherdata = weather_data(30,-90, 15, "Data/EXPECTED_DATA")
start_date = pd.to_datetime(datetime.datetime(2025, 3, 1))
end_date = pd.to_datetime(datetime.datetime(2025, 3, 2))
dt=15
lat = 30
lon = -90
soc_increment = 1
timestep_min = dt
expected_data = weatherdata.get_expected_weather_data(start_date, end_date, dt, lat, lon)

solar_columns = ["beta_alpha", "beta_beta", "expected_solar_rad"]
expected_solar_data = expected_data[solar_columns].to_numpy()

wind_columns = ["weibull_k", "weibull_scale", "expected_wind_speed"]
expected_wind_data = expected_data[wind_columns].to_numpy()

loc = rf"Data\SYNTHETIC_DATA\lat{int(lat)}"

times = pd.date_range(
    start=start_date,
    end=end_date,
    freq=pd.Timedelta(minutes=dt),
    inclusive="both",
)

whale_observation_data = weatherdata.get_whale_observation_probabilities(times,expected_solar_data[:,2])
transition_model = ProbabilityModelFactory.select_probability_model("Moderate")
failure_penalty = 1


valuefunc = ValueFunction(seaplane,
                        expected_solar_data,
                        expected_wind_data,
                        whale_observation_data,
                        soc_increment,
                        timestep_min,
                        transition_model ,
                        failure_penalty,      
              )

stage = int(24*4/2)
state = np.array([100,0])
valuefunc.expected_reward_function(stage,state,1)
valuefunc.expected_reward_function(stage,state,0)

stage = int(24*4/2)
state = np.array([50,0])
valuefunc.expected_reward_function(stage,state,1)
valuefunc.expected_reward_function(stage,state,0)

stage = int(24*4/2)
state = np.array([25,0])
valuefunc.expected_reward_function(stage,state,1)
valuefunc.expected_reward_function(stage,state,0)

stage = int(24*4/2)
state = np.array([10,0])
valuefunc.expected_reward_function(stage,state,1)
valuefunc.expected_reward_function(stage,state,0)


stage = int(24*4/2)
state = np.array([5,0])
valuefunc.expected_reward_function(stage,state,1)
valuefunc.expected_reward_function(stage,state,0)