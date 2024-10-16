# import warnings

# import numpy as np
# import pandas as pd

# from scipy.stats import weibull_min
# from scipy.stats import beta as beta_dist

# from BaseClasses.whale_sighting_base import WhaleSightingProbability

# from pvlib import location, tracking, temperature, pvsystem
# from pvlib.bifacial.pvfactors import pvfactors_timeseries

# import sys
# import os

# # Add the project root directory to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)
# from BaseClasses.seaplane_base import Seaplane
# from BaseClasses.autonomy_base import Autonomy
# from BaseClasses.solar_grib_base import SolarRadiationProcessor

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import weibull_min, beta as beta_dist
from pvlib import location, tracking, temperature, pvsystem
from pvlib.bifacial.pvfactors import pvfactors_timeseries

# # Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from BaseClasses.whale_sighting_base import WhaleSightingProbability
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.autonomy_base import Autonomy
from BaseClasses.solar_grib_base import SolarRadiationProcessor



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
        self.location = self.set_location(lat, lon, tz)

    def set_location(self, latitude: float, longitude: float, timezone: str) -> location.Location:
        """Set the simulation's location using latitude, longitude, and timezone."""
        return location.Location(latitude, longitude, tz=timezone)

    def generate_datetime_index(self, years, months, days, hours, minutes) -> pd.DatetimeIndex:
        """Generate a DateTimeIndex for the provided date components."""
        dates = [
            pd.to_datetime(f"{y}-{m}-{d} {h}:{min:02d}:00")
            for y, m, d, h, min in zip(years, months, days, hours, minutes)
        ]
        return pd.DatetimeIndex(dates)

    def get_weather(self, cs: bool, times=-1):
        """Retrieve clearsky or TMY weather data."""
        if cs and times != -1:
            return self.location.get_clearsky(times)
        raise NotImplementedError("Non-clearsky weather data is not yet implemented.")

    def get_azimuth(self, cs: bool, wthr) -> float:
        """Get the azimuthal position of the plane based on weather data."""
        return np.mod(wthr['wind_direction'] + 90, 360) if not cs and wthr is not None else 0

    def get_weather_param(self, cs: bool, wthr, param: str, default=0) -> float:
        """Retrieve specific weather parameters, returning defaults if unavailable."""
        return wthr.get(param, default) if not cs and wthr is not None else default

    def calculate_collected_energy(self, year, month, day, periods, frequency, cs: bool):
        """
        Calculate the energy collected by the solar array over a given period.

        This method retrieves the solar position data, calculates the orientation of 
        the solar array, retrieves weather data, and computes the energy output from 
        the solar panels based on the irradiance and temperature conditions.

        Parameters
        ----------
        year : int
            The year for which to calculate the collected energy.
        month : int
            The month for which to calculate the collected energy.
        day : int
            The day for which to calculate the collected energy.
        periods : int
            The number of periods to simulate for energy collection.
        frequency : str
            The frequency of the time intervals for the simulation (e.g., '1H' for hourly).
        cs : bool
            A flag indicating whether to use clearsky weather data (True) or real weather data (False).

        Returns
        -------
        tuple
            A tuple containing:
            - times : pd.DatetimeIndex
                A DatetimeIndex object representing the times for which energy collection is calculated.
            - pdc : pd.Series
                A Pandas Series representing the direct current (DC) power output of the solar array 
                over the specified period, with values set to 0 for times with no energy collection.

        Notes
        -----
        This method utilizes the `pvlib` library for solar position calculations, irradiance 
        modeling, and DC power output calculations. It requires a properly initialized `location`
        attribute and assumes the `plane` attribute has properties like `pdc0` and `gamma` 
        defined.
        """

        times = self._get_simulation_times(year, month, day, periods, frequency, cs)
        solar_position = self.location.get_solarposition(times)

        orientation = tracking.singleaxis(
            solar_position['apparent_zenith'], solar_position['azimuth'],
            max_angle=0, backtrack=True, gcr=0.01
        )

        wthr = self.get_weather(cs, times)
        irrad = self._calculate_irradiance(solar_position, orientation, wthr)

        temp_air = self.get_weather_param(cs, wthr, 'temp_air', 25)
        wind_speed = self.get_weather_param(cs, wthr, 'wind_speed')
        temp_cell = temperature.faiman(irrad['total_abs_front'], temp_air, wind_speed)

        pdc = pvsystem.pvwatts_dc(
            irrad['total_abs_front'], temp_cell,
            self.plane.pdc0, gamma_pdc=self.plane.gamma
        ).fillna(0)

        return times, pdc

    def _get_simulation_times(self, year, month, day, periods, frequency, cs):
        """Generate simulation times based on the input parameters."""
        if cs:
            start = f"{year[0]}-{month[0]}-{day[0]}"
            return pd.date_range(start, periods=periods, freq=frequency, tz=self.tz)
        wthr = self.get_weather(cs)
        wthr.query('Month >= @month[0] and Month <= @month[1]', inplace=True)
        wthr.query('Day >= @day[0] and Day <= @day[1]', inplace=True)
        return wthr.index

    def _calculate_irradiance(self, solar_position, orientation, wthr):
        """
        Calculate the irradiance on the solar array using PVFactors.

        This method computes the total irradiance received by the solar array based on
        the solar position, array orientation, and weather data. It utilizes the 
        `pvfactors_timeseries` function to model the effects of direct normal irradiance (DNI),
        diffuse horizontal irradiance (DHI), and other parameters.

        Parameters
        ----------
        solar_position : pd.DataFrame
            A DataFrame containing the solar position data, which must include:
            - 'azimuth': The azimuth angle of the sun.
            - 'apparent_zenith': The apparent zenith angle of the sun.

        orientation : pd.DataFrame
            A DataFrame containing the orientation data of the solar array, which must include:
            - 'surface_azimuth': The azimuth angle of the solar array surface.
            - 'surface_tilt': The tilt angle of the solar array surface.

        wthr : pd.DataFrame
            A DataFrame containing weather data with the following columns:
            - 'dni': Direct normal irradiance (W/m²).
            - 'dhi': Diffuse horizontal irradiance (W/m²).
            The index should represent time.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated irradiance on the solar array, 
            which includes the components of irradiance based on the provided solar position,
            orientation, and weather data.

        Notes
        -----
        The method uses fixed parameters such as ground cover ratio (gcr), row height, 
        row width, albedo, number of PV rows, and the index of the observed PV row. 
        These parameters may need to be adjusted based on the specific configuration 
        of the solar installation.
        """
        return pvfactors_timeseries(
            solar_position['azimuth'], solar_position['apparent_zenith'],
            orientation['surface_azimuth'], orientation['surface_tilt'],
            self.get_azimuth(self.cs, wthr), wthr.index,
            wthr['dni'], wthr['dhi'], gcr=0.01, pvrow_height=10, pvrow_width=10,
            albedo=0.06, n_pvrows=1, index_observed_pvrow=0
        ).pipe(pd.concat, axis=1)

    def simulate_deployment(self, U, rho, takeoff_capacity, landing_capacity,
                            start_index, end_index, dt, algo: str):
        """
        Simulate the vehicle's deployment and determine its duty cycle.

        This method calculates the vehicle's duty cycle based on the specified algorithm
        ('Greedy' or 'MDP') and the weather data (solar and wind) within the given time frame.

        Parameters:
        - U (float): The maximum velocity of the vehicle.
        - rho (float): The air density (kg/m^3) to be used in the simulation.
        - takeoff_capacity (float): The maximum weight capacity for takeoff (kg).
        - landing_capacity (float): The maximum weight capacity for landing (kg).
        - start_index (int): The index in the weather data where the simulation starts.
        - end_index (int): The index in the weather data where the simulation ends.
        - dt (float): The time step for the simulation (in seconds).
        - algo (str): The algorithm to be used for the simulation. Options are "Greedy" or "MDP".

        Returns:
        - list: A list containing the results of the simulation. The structure of the list 
        will depend on the chosen algorithm. Each entry typically includes data about 
        the vehicle's duty cycle during the specified time frame.

        Raises:
        - ValueError: If the specified algorithm is not recognized. Only 'Greedy' or 'MDP'
        are accepted.
        """
        self.plane.calculate_weight()
        solar_data, wind_data = self._load_weather_data(start_index, end_index)

        if algo == "Greedy":
            return self._simulate_greedy_behavior(solar_data, wind_data, start_index, end_index)
        elif algo == "MDP":
            return self._simulate_mdp_behavior(solar_data, wind_data, start_index, end_index)
        else:
            raise ValueError(f"Unknown algorithm: {algo}. Use 'Greedy' or 'MDP'.")

    def _load_weather_data(self, start_index, end_index):
        """Load solar and wind data from pickle files."""
        solar_file = r"Data\DISTRIBUTIONS\2022_solar_data.pkl"
        wind_file = r"Data\DISTRIBUTIONS\2022_wind_mag.pkl"

        solar_data = pd.read_pickle(solar_file)[(29.25, -85.0)].loc[start_index:end_index]
        wind_data = pd.read_pickle(wind_file)[(29.25, -85.0)].sort_index().loc[start_index:end_index]
        return solar_data, wind_data

    def _simulate_greedy_behavior(self, solar_data, wind_data, start_index, end_index):
        """Simulate the 'Greedy' behavior."""
        return Autonomy.simulate_simple_behavior(
            self, plane=self.plane, soc_increment=1, start_index=start_index,
            end_index=end_index, max_stages=len(solar_data) - 1,
            initial_state=(100, "moored"), actual_solar_power=solar_data,
            avail_wind_mag=wind_data, whale_probabilities=WhaleSightingProbability().df
        )

    def _simulate_mdp_behavior(self, solar_data, wind_data, start_index, end_index):
        """Simulate the 'MDP' behavior."""
        return Autonomy.simulate_mdp_behavior(
            self, plane=self.plane, soc_increment=1, start_index=start_index,
            end_index=end_index, max_stages=len(solar_data) - 1,
            initial_state=(100, "moored"), actual_solar_power=solar_data,
            avail_wind_mag=wind_data, whale_probabilities=WhaleSightingProbability().df
        )
