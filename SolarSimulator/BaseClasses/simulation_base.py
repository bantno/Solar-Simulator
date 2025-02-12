import os
import sys
import re
from suntime import Sun
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder
import cProfile

# # Add the project root directory to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

from BaseClasses.expectedValue_base import ExpectedValueTable
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.autonomy_base import Autonomy
from BaseClasses.transition_model_base import ActionSuccessProbabilityModel


# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


class Simulation:
    """
    A class to simulate the operations of a seaplane using solar energy collection
    and behavioral algorithms. The class handles the initialization of the simulation
    environment, retrieves weather data, calculates solar energy collection, and
    simulates the vehicle's deployment.
    """

    def __init__(
        self,
        plane: Seaplane,
        lat: float,
        lon: float,
        tz: str,
        save_history: bool = False,
        use_expected: bool = False,
        simulate_failure: bool = True,
        transition_model: ActionSuccessProbabilityModel = None,
    ) -> None:
        self.plane = plane
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.simulate_failure = simulate_failure
        self.save_history = save_history
        self.use_expected = use_expected
        self.transition_model = transition_model

    def run_simulation(
        self,
        start_date: datetime,
        end_date: datetime,
        dt,
        algo=None,
        mdp_success_prob=0,
        true_success_prob=0,
        runs=1,
        threshold=None,
        transition_model=None,
    ):
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
            threshold=threshold,
            transition_model=transition_model,
        )

        times = pd.date_range(start_date, end_date, freq=f"{dt}min")
        return times, data

    def simulate_deployment(
        self,
        start_date,
        end_date,
        dt,
        algo: str,
        mdp_success_prob,
        true_success_prob,
        num_runs,
        threshold,
        transition_model,
    ):
        
        # profiler = cProfile.Profile()
        # profiler.enable()
        
        self.plane.update_plane()
        times = pd.date_range(
            start=start_date,
            end=end_date,
            freq=pd.Timedelta(minutes=dt),
            inclusive="both",
        )
        expected_data = self.get_expected_weather_data(
            start_date, end_date, dt, self.lat, self.lon
        )

        solar_columns = ["beta_alpha", "beta_beta", "expected_solar_rad"]
        expected_solar_data = expected_data[solar_columns].to_numpy()

        wind_columns = ["weibull_k", "weibull_scale", "expected_wind_speed"]
        expected_wind_data = expected_data[wind_columns].to_numpy()

        loc = rf"Data\SYNTHETIC_DATA\lat{int(self.lat)}"

        whale_observation_data = self.get_whale_observation_probabilities(times)
        mdp_model = ExpectedValueTable(
            self.plane,
            expected_solar_data,
            expected_wind_data,
            whale_observation_data,
            soc_increment=1,
            timestep_min=dt,
            transition_model=transition_model,
        )

        auto = Autonomy(dt, mdp_model, use_expected_reward=self.use_expected)
        mdp_model.show_progress = True

        data = {}
        simulation_methods = {
            "Threshold": auto.simulate_observation_threshold_mission,
            "Optimal": auto.simulate_optimal_mission,
            "Charge Threshold": auto.simulate_charge_threshold_mission,
        }

        if algo not in simulation_methods:
            raise ValueError(
                f"Unknown algorithm: {algo}. Use 'Threshold', 'Optimal', or 'Charge Threshold'."
            )

        simulate_method = simulation_methods[algo]
        if algo == "Optimal":
            mdp_model.generate_ev_table()

        for i in tqdm(
            range(num_runs), desc=f"{algo} Simulation", leave=False, mininterval=1
        ):
            if not self.use_expected:
                actual_data = self._load_weather_data(
                    dt, directory=loc, i=i, lat=self.lat, lon=self.lon
                )
                actual_data = actual_data.loc[start_date:end_date]
                sim_solar_data = actual_data["shortwave_radiation"].values
                sim_wind_data = actual_data["wind_speed_10m"].values

            else:
                sim_solar_data = expected_data["expected_solar_rad"].values
                sim_wind_data = expected_data["expected_wind_speed"].values

            result = simulate_method(
                initial_state=(100, 0),
                solar_data=sim_solar_data,
                wind_data=sim_wind_data,
                whale_data=whale_observation_data,
                true_success_prob=true_success_prob,
                simulate_failure=self.simulate_failure,
                save_history=self.save_history,
                threshold=threshold if algo == "Threshold" else None,
            )

            data[i] = self._format_simulation_result(result, expected_data)

        # profiler.disable()
        # profiler.dump_stats("sim_profile_output.prof")

        return pd.DataFrame.from_dict(data, orient="index")

    def _format_simulation_result(self, result, expected_data):
        if self.save_history:
            (
                reward,
                last_step,
                state_history,
                action_history,
                failure_prob_history,
                solar_list,
                wind_history,
                whale_list,
                flight_minutes,
            ) = result
            return {
                "Reward": reward,
                "LastStep": last_step,
                "ActionHistory": action_history,
                "FailureProbHistory": failure_prob_history,
                "StateHistory": state_history,
                "SolarHistory": solar_list,
                "ExpectedSolarHistory": expected_data["expected_solar_rad"].values,
                "WindHistory": wind_history,
                "ExpectedWindHistory": expected_data["expected_wind_speed"].values,
                "WhaleHistory": whale_list,
                "FlightHours": flight_minutes / 60,
            }
        else:
            reward, last_step, flight_minutes = result
            return {
                "Reward": reward,
                "LastStep": last_step,
                "FlightHours": flight_minutes / 60,
            }

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
                | (
                    (df["month"] == end_month)
                    & (df["day"] == end_day)
                    & (df["hour"] < end_hour)
                )
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

    def _load_weather_data(
        self, dt: int, directory: str = None, i=None, lat=None, lon=None
    ):
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

    def get_whale_observation_probabilities(self, time_index: pd.DatetimeIndex):
        # Define the time intervals and probabilities
        time_intervals = [
            ("0600", "0800", 0.082),
            ("0800", "1000", 0.098),
            ("1000", "1200", 0.095),
            ("1200", "1400", 0.217),
            ("1400", "1600", 0.215),
            ("1600", "2000", 0.278),
        ]
        tz_finder = TimezoneFinder()
        timezone_str = tz_finder.timezone_at(lng=self.lon, lat=self.lat)
        tz = pytz.timezone(timezone_str)

        def get_day_night_flag(dt, latitude, longitude):
            # Create Sun object for the location
            sun = Sun(latitude, longitude)

            # Get the sunrise and sunset times for the given day
            sunrise = sun.get_sunrise_time(dt).astimezone(tz)
            sunset = sun.get_sunset_time(dt).astimezone(tz)

            # Determine if current time is day or night
            if sunrise <= dt <= sunset:
                return 1
            else:
                return 0

        # Function to get the whale observation probability for a given time
        def get_probability_for_time(time):
            time_str = time.strftime("%H%M")
            for interval_start, interval_end, prob in time_intervals:
                if interval_start <= time_str < interval_end:
                    return prob * get_day_night_flag(time, self.lat, self.lon)
            return 0.0  # Return 0.0 if the time doesn't fall into any interval

        # Apply the function to the DatetimeIndex and return the probabilities
        probabilities = np.array(time_index.map(get_probability_for_time))
        return probabilities


class SingleCaseSimulation(Simulation):
    """
    A class that runs a single case of the seaplane simulation, allowing specification of
    the expected and actual weather data files.
    """

    def __init__(
        self,
        plane: Seaplane,
        lat: float,
        lon: float,
        tz: str,
        expected_file: str,
        actual_file: str,
        save_history: bool = False,
        use_expected: bool = False,
        simulate_failure: bool = True,
    ) -> None:
        super().__init__(
            plane, lat, lon, tz, save_history, use_expected, simulate_failure
        )
        self.expected_file = expected_file
        self.actual_file = actual_file

    def run_single_case(
        self,
        start_date: datetime,
        end_date: datetime,
        dt: int,
        algo: str,
        mdp_success_prob: float,
        true_success_prob: float,
        threshold: float = None,
    ):
        """Runs a single case of the simulation."""
        self.plane.update_plane()
        threshold = threshold if isinstance(threshold, float) else 0.1

        times = pd.date_range(
            start=start_date, end=end_date, freq=pd.Timedelta(minutes=dt)
        )
        expected_data = pd.read_pickle(self.expected_file)
        actual_data = pd.read_pickle(self.actual_file).loc[start_date:end_date]

        solar_columns = ["beta_alpha", "beta_beta", "expected_solar_rad"]
        expected_solar_data = expected_data[solar_columns].to_numpy()

        wind_columns = ["weibull_k", "weibull_scale", "expected_wind_speed"]
        expected_wind_data = expected_data[wind_columns].to_numpy()

        # whale_observation_data = self.get_whale_observation_probabilities(times)
        whale_observation_data = np.ones(len(times))*0.25
        whale_observation_data[len(times) // 2 :] = 0.75
        whale_observation_data[0: len(times) // 4] = 0.00

        mdp_model = ExpectedValueTable(
            self.plane,
            expected_solar_data,
            expected_wind_data,
            whale_observation_data,
            soc_increment=1,
            timestep_min=dt,
            floating_failure_prob=1 - true_success_prob,
        )

        auto = Autonomy(dt, mdp_model, use_expected_reward=self.use_expected)
        mdp_model.show_progress = True

        simulation_methods = {
            "Threshold": auto.simulate_observation_threshold_mission,
            "Optimal": auto.simulate_optimal_mission,
            "Charge Threshold": auto.simulate_charge_threshold_mission,
        }

        if algo not in simulation_methods:
            raise ValueError(
                f"Unknown algorithm: {algo}. Use 'Threshold', 'Optimal', or 'Charge Threshold'."
            )

        if algo == "Optimal":
            mdp_model.generate_ev_table()

        sim_solar_data = expected_data["expected_solar_rad"].values
        sim_wind_data = expected_data["expected_wind_speed"].values

        simulate_method = simulation_methods[algo]

        result = simulate_method(
            initial_state=(100, 0),
            solar_data=sim_solar_data,
            wind_data=sim_wind_data,
            whale_data=whale_observation_data,
            true_success_prob=true_success_prob,
            simulate_failure=self.simulate_failure,
            save_history=True,
            threshold=threshold if algo == "Threshold" else None,
        )
        data = {}
        data[0] = self._format_simulation_result(result, expected_data)
        return pd.DataFrame.from_dict(data, orient="index")
