import warnings

import numpy as np
import pandas as pd

from scipy.stats import weibull_min
from scipy.stats import beta as beta_dist

from pvlib import location, tracking, temperature, pvsystem
from pvlib.bifacial.pvfactors import pvfactors_timeseries

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from BaseClasses.seaplane_base import Seaplane
from BaseClasses.autonomy_base import Autonomy
from BaseClasses.solar_grib_base import SolarRadiationProcessor

class Simulation:
    def __init__(self,plane:Seaplane,lat,lon,tz,cs:bool=False) -> None:
        self.plane = plane
        
        # Define solar parameters
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.location = self.set_location(lat,lon,tz)
        self.cs = cs

    def set_location(self,latitude,longitude,timezone):
        return location.Location(latitude, longitude, tz=timezone)

    def get_DateTimeIndex(self,years,months,days,hours,minutes) -> pd.DatetimeIndex:
        """Returns DateTimeIndex for provided dates

        Parameters
        ----------
        years : list of int
            Year of date
        months : list of int
            Month of date
        days : list of int
            Day of date
        hours : list of int
            Hour of date
        minutes : list of int
            minute of date

        """
        dates=[]
        for year, month, day, hour, minute in zip(years, months, days, hours, minutes):
            date = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute:02d}:00")
            dates.append(date)
        return pd.DatetimeIndex(dates)

    def get_weather(self,cs:bool,times = -1):
        """Retrieve clearsky or TMY weather data
        
        If cs is false, times are ignored

        Parameters
        ----------
        cs : bool
            True if simulation uses clearsky weather
        times : DateTimeIndex
            Times at which to pull clearsky weather information
        
        """

        if cs :
            # Get clearsky weather
            if not (times is -1):
                wthr = self.location.get_clearsky(times)
        else :
            # Get weather based on TMY data
            raise NotImplementedError
        return wthr

    def get_azimuth(self,cs,wthr):
        "Get azimuthal position of plane for each time step"
        if cs or wthr is None:
            axis_azimuth = 0
        else:
            # Set windspeed and azimuth based on TMY data
            #tilt axis is 90 degrees offset from wind direction
            axis_azimuth = np.mod(wthr['wind_direction']+90,360)
        return axis_azimuth
    
    def get_windspeed(self,cs,wthr):
        """Get wind speed"""
        if cs or wthr is None:
            wind_speed = 0
        else:
            wind_speed = wthr['wind_speed'] # m/s
        return wind_speed
    
    def get_air_temp(self,cs,wthr):
        """Get the air temperature for use in solar panel model"""
        if cs or wthr is None:
            temp_air = 25
        else:
            temp_air = wthr['temp_air']
        return temp_air

    def calc_collected_energy(self,year,month,day,periods,frequency,cs):
        """Calculate energy collected by solar array for given period
        
        Parameters
        ----------
        year : tuple, int
            Start and end year
        month : tuple, int
            Start and end month
        day : tuple, int
            Start and end day
        periods : int
            Number of times at which power should be determined
        frequency : char
            Period between times Ex. '5min'

    
        """
        if cs:
            start = f"{year[0]}-{month[0]}-{day[0]}"
            times = pd.date_range(start, periods=periods, freq=frequency, tz=self.tz)
            wthr = self.get_weather(self.cs,times)
        else:
            wthr = self.get_weather(self.cs)
            month_s = month[0]
            month_e = month[1]
            day_s = day[0]
            day_e = day[1]

            wthr.query('Month >= @month_s and Month <= @month_e',inplace=True)
            wthr.query('Day >= @day_s and Day <= @day_e',inplace=True)
            times = wthr.index

        solar_position = self.location.get_solarposition(times)

        # set ground coverage ratio and max tilt angle
        gcr = 0.01
        max_phi = 0

        orientation = tracking.singleaxis(solar_position['apparent_zenith'],
                                        solar_position['azimuth'],
                                        max_angle=max_phi,
                                        backtrack=True,
                                        gcr=gcr)

        # set axis_azimuth, albedo, pvrow width and height, and use
        # the pvfactors engine for absorbed irradiance
        pvrow_height = 10
        pvrow_width = 10
        albedo = 0.06

        axis_azimuth = self.get_azimuth(cs,wthr)
        wind_speed = self.get_windspeed(cs,wthr)
        temp_air = self.get_air_temp(cs,wthr)

        # explicity simulate on pvarray with sensor placed in middle row
        # users may select different values depending on needs
        irrad = pvfactors_timeseries(solar_position['azimuth'],
                                    solar_position['apparent_zenith'],
                                    orientation['surface_azimuth'],
                                    orientation['surface_tilt'],
                                    axis_azimuth,
                                    wthr.index,
                                    wthr['dni'],
                                    wthr['dhi'],
                                    gcr,
                                    pvrow_height,
                                    pvrow_width,
                                    albedo,
                                    n_pvrows=1,
                                    index_observed_pvrow=0
                                    )
        # turn into pandas DataFrame
        irrad = pd.concat(irrad, axis=1)
        effective_irrad_mono = irrad['total_abs_front']
        
        temp_cell = temperature.faiman(effective_irrad_mono, temp_air=temp_air,
                                    wind_speed=wind_speed)
        # Create pvsystem using pvwatts model for single face solar cell
        pdc = pvsystem.pvwatts_dc(effective_irrad_mono,
                                    temp_cell,
                                    self.plane.pdc0,
                                    gamma_pdc=self.plane.gamma
                                    ).fillna(0)

        return times, pdc
    
    def calculate_expected_solar_power(self,
                                       year,
                                       month,
                                       day,
                                       periods,
                                       frequency):
        grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\GRIB\January\solar_radiation.grib'
        processor = SolarRadiationProcessor(grib_file_path)
        beta_params = processor.process_grib_file()
        beta_dist = beta_params[(29.5, -85.25)]
        start = f"{year[0]}-{month[0]}-{day[0]}"
        times = pd.date_range(start, periods=periods, freq=frequency, tz=self.tz)

        for i in len(times):
            expected_value = param[0]/(param[0]+param[1])*param[2]
        

    @staticmethod
    def generate_irradiance_timeseries(num_timesteps, num_samples, alpha, beta, max_irradiance):
        """
        Generate solar irradiance time series using a beta distribution.
        
        Parameters:
        - num_timesteps: Number of time steps in each simulation
        - num_samples: Number samples to generate per step of the timeseries
        - alpha: beta shape parameter 
        - scale: beta scale parameter
        - max_irradiance: Maximum irradiance 
        
        Returns:
        - timeseries: Array of shape (num_simulations, num_timesteps) with wind speed values
        """
        timeseries = np.zeros((num_samples, num_timesteps))
        
        for t in range(num_timesteps):
            samples = beta_dist.rvs(alpha[t], beta[t], size=num_samples)
            timeseries[:, t] = samples * max_irradiance[t]
        
        return timeseries
    
    @staticmethod
    def generate_windspeed_timeseries(num_timesteps, num_simulations, shape, scale):
        """
        Generate wind speed time series using a Weibull distribution.
        
        Parameters:
        - num_timesteps: Number of time steps in each simulation
        - num_simulations: Number of simulations to run
        - shape: Weibull shape parameter (k)
        - scale: Weibull scale parameter (c)
        
        Returns:
        - timeseries: Array of shape (num_simulations, num_timesteps) with wind speed values
        """
        timeseries = np.zeros((num_simulations, num_timesteps))
        
        for t in range(num_timesteps):
            # Generate samples from Weibull distribution
            samples = weibull_min.rvs(shape, scale=scale, size=num_simulations)
            timeseries[:, t] = samples
        
        return timeseries


    def simulate_deployment(self,
                            U,
                            rho,
                            takeoff_capacity,
                            landing_capacity,
                            avail_solar_w,
                            expected_solar_w,
                            dt,
                            algo):
        
        """
        Simulates the deployment of a vehicle and determines the duty cycle for a specified time period.

        Parameters
        ----------
        U : float
            Cruise speed of the vehicle (m/s).
        rho : float
            Air density at cruise altitude in kg/m^3.
        takeoff_capacity : float
            Minimum percentage of total battery capacity required for takeoff.
        landing_capacity : float
            Minimum percentage of total battery capacity required to initiate landing.
        avail_solar_w : numpy.ndarray
            Array of actual solar power collected by the vehicle's photovoltaic system (W) for each time step.
        expected_solar_w : numpy.ndarray
            Array of expected solar power (W) used for decision-making in the MDP algorithm.
        dt : int
            Time step duration in minutes between each sample in the solar power data.
        algo : str
            The algorithm to use for simulating behavior, either 'Greedy' or 'MDP'.

        Returns
        -------
        duty_cycle : float
            The percentage of the total time period the vehicle spends in the air.
        energy_j : list of float
            List of energy stored in the battery (in joules) at each time step of the simulation.

        Raises
        ------
        ValueError
            If the lengths of `expected_solar_w` and `avail_solar_w` do not match or if an unknown algorithm is specified.
        """

        
        def daytime(start_time: int, time_step: int = 10, stage: int = 0) -> int:
            """
            Determines if the current stage is during the day or night, accounting for simulations
            that span multiple days.

            Args:
                start_time (int): The time in minutes from the start of the day (0-1439).
                                For example, 0 is 12:00 AM, 720 is 12:00 PM, and 1439 is 11:59 PM.
                time_step (int): The time step duration in minutes. Default is 10 minutes.
                stage (int): The current stage of the simulation.

            Returns:
                int: 1 if the stage is during the day (6 AM to 6 PM), otherwise 0.
            """
            # Number of minutes in a day
            minutes_per_day = 24 * 60
            
            # Calculate the current time in minutes, accounting for multiple days
            total_time = start_time + time_step * stage
            current_time = total_time % minutes_per_day
            
            # Convert minutes to determine day (6:00 AM = 360 minutes, 6:00 PM = 1080 minutes)
            if 360 <= current_time < 1080:
                return 1
            else:
                return 0

        # Ensure weight estimate is accurate
        self.plane.calculate_weight()

        # Calculate required parameters
        P_cruise = self.plane.get_required_power(U, rho)
        capacity_j = self.plane.voltage * self.plane.capacity * 3600
        # is_daytime = pd.DataFrame(columns=['daytime'])
        is_daytime = []
        for i in range(len(avail_solar_w-1)):
            # is_daytime.iloc[i] = daytime(0,10,i)
            is_daytime.append(daytime(0,10,i))
        min_flight_hr = 0.5  # Minimum flight time (hours) after takeoff
        
        if algo == "Greedy":
            # Call the simple behavior for the "Greedy" algorithm
            return Autonomy.simulate_simple_behavior(self,
                                                     avail_solar_w,
                                                     is_daytime,
                                                     P_cruise,
                                                     capacity_j,
                                                     landing_capacity,
                                                     takeoff_capacity,
                                                     dt,
                                                     min_flight_hr,
                                                     self.plane.calc_takeoff_penalty)
        
        elif algo == "MDP":
            # Set MDP-specific parameters
            soc_increment = 1  # State of Charge increments
            max_stages = len(avail_solar_w-1)  # Set max stages based on the length of the solar power data
            start_state = (100, "moored")  # Initial state (100% SoC, moored)

            # Call the mdp_behavior function from the Autonomy class
            if len(expected_solar_w) == len(avail_solar_w):
                return Autonomy.simulate_mdp_behavior(self,
                                                      plane=self.plane,
                                                      soc_increment=soc_increment,
                                                      max_stages=max_stages,
                                                      initial_state=start_state,
                                                      expected_solar_power=expected_solar_w,
                                                      actual_solar_power=avail_solar_w,
                                                      is_daytime=is_daytime)
            else:
                raise ValueError(f"Lengths do not match: {len(expected_solar_w)} != {len(avail_solar_w)}")
        
        else:
            raise ValueError(f"Unknown algorithm: {algo}. Use 'Greedy' or 'MDP'.")
