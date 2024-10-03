import warnings

import numpy as np
import pandas as pd

from scipy.stats import weibull_min
from scipy.stats import beta as beta_dist

from pvlib import location, tracking, temperature, pvsystem
from pvlib.bifacial.pvfactors import pvfactors_timeseries

from BaseClasses.seaplane_base import Seaplane
from BaseClasses.autonomy_base import Autonomy

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
    
    def calculate_expected_solar_power():
        

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


    def simulate_deployment(self, U, rho, takeoff_capacity, landing_capacity, P_solar, dt, algo):
        """Determines duty cycle for specified period
        
        Parameters
        ----------
        U : float
            Cruise speed of the vehicle.
        rho : float
            Air density at cruise altitude in kg/m^3.
        takeoff_capacity : float
            Percentage of total capacity at which the vehicle is sufficiently charged to takeoff.
        landing_capacity : float
            Percentage of total capacity at which the vehicle must land.
        P_solar : numpy array
            Array of solar power collected by the vehicle's photovoltaic system [W].
        dt : int
            Time in minutes between each sample in P_solar.
        algo : str
            The algorithm to use, either "Greedy" or "MDP".
        
        Returns
        -------
        duty_cycle : float
            Percentage of time period spent in air.
        energy_j : list, float
            Energy stored in battery for each simulated time.
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
        for i in range(len(P_solar-1)):
            # is_daytime.iloc[i] = daytime(0,10,i)
            is_daytime.append(daytime(0,10,i))
        min_flight_hr = 0.5  # Minimum flight time (hours) after takeoff
        
        if algo == "Greedy":
            # Call the simple behavior for the "Greedy" algorithm
            return Autonomy.simulate_simple_behavior(self, P_solar, is_daytime, P_cruise, capacity_j, landing_capacity, takeoff_capacity, dt, min_flight_hr, self.plane.calc_takeoff_penalty)
        
        elif algo == "MDP":
            # Set MDP-specific parameters
            soc_increment = 1  # State of Charge increments
            max_stages = len(P_solar-1)  # Set max stages based on the length of the solar power data
            start_state = (100, "moored")  # Initial state (100% SoC, moored)

            # Call the mdp_behavior function from the Autonomy class
            return Autonomy.simulate_mdp_behavior(self,
                                                  plane=self.plane,
                                                  soc_increment=soc_increment,
                                                  max_stages=max_stages,
                                                  initial_state=start_state,
                                                  expected_solar_power=expected_solar_power,
                                                  actual_solar_power=actual_solar_power,
                                                  is_daytime=is_daytime)
        
        else:
            raise ValueError(f"Unknown algorithm: {algo}. Use 'Greedy' or 'MDP'.")
