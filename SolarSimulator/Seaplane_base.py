import pandas as pd

from pvlib import location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# supressing shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action='ignore', module='pvfactors')

class Seaplane:
    """Class representing a seaplane"""
    def __init__(self, lat, lon, tz, pdc0,gamma,tracking:bool=False,cs:bool=False,cd0=0.01,cdtot = 0.06,n_tot=.75,S=1,weight=10,voltage=24.4,capacity=150):
        
        # Define solar parameters
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.tracking = tracking
        self.location = location.Location(lat, lon, tz=tz, name = 'Gulf of Mexico')
        self.pdc0 = pdc0
        self.gamma = gamma
        self.collected_energy = 0 #kWh
        self.cs = cs

        # Define airframe and motion parameters
        self.cd0 = cd0
        self.n_tot = n_tot
        self.S = S
        self.weight = weight
        self.voltage = voltage
        self.capacity = capacity
        self.Rt = 1.0
        self.n = 1.0
        self.AR = 6.0 #remove hardcode
        self.e = 0.8
        self.k = 1.0/(np.pi*self.AR*self.e)
        self.cdtot = cdtot

    def convert_to_datetime_index(self, day_number, base_year=2024):
        # Calculate the base date for the given year
        base_date = datetime(base_year, 1, 1)
        
        # Add the day_number - 1 to the base date (since day_number starts from 1)
        target_date = base_date + timedelta(days=day_number - 1)
        
        # Convert the target date to a Pandas datetime index
        datetime_index = pd.to_datetime(target_date)
        
        return datetime_index

    def get_endurance(self,U,rho):
        """Returns endurance estimate according to Traub
        
        Parameters
        ----------
        U : float
            Cruise speed [m/s]
        rho : float
            Air density [kg/m^3]
        
        """
        
        E = self.Rt**(1.0-self.n)*(self.n_tot*self.voltage*self.capacity)/self.get_required_power(U,rho)
        return E
            
    def get_dynamic_pressure(self,U,rho):
        """Returns dynamic pressure

        Parameters
        ----------
        U : float
            Cruise speed [m/s]
        rho : float
            Air density [kg/m^3]
        
        """
        return 0.5*rho*U**2
    
    def get_required_power(self,U,rho):
        """Returns required cruise power consumption
        
        Parameters
        ----------
        U : float
            Cruise speed [m/s]
        rho : float
            Air density [kg/m^3]
        
        """
        q = self.get_dynamic_pressure(U,rho)
        D = q*self.S*self.cdtot*U

        # return D*U # U in m/s
        return .5*rho*U**3*self.S*self.cd0 + 2*self.weight**2*self.k/(rho*U*self.S)
    
    def get_times(self,year,month,day,tz):
        """Returns hourly daterange for the times between given date and subsequent day"""
        # get times of interest
        start = "{0}-{1}-{2}".format(year,month,day)
        day = day + 1
        end = "{0}-{1}-{2}".format(year,month,day)
        return pd.date_range(start, end, freq='60min', tz=tz) # Get times for entire study period      

    def get_DateTimeIndex(self,years,months,days,hours,minutes):
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
            # get clearsky weather
            # TODO: throw error if times is -1
            wthr = self.location.get_clearsky(times)
        else :
            wthr = pd.read_csv(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\Solar Sim\2019TMY.csv")
            times = self.get_DateTimeIndex(wthr['Year'],
                                        wthr['Month'],
                                        wthr['Day'],
                                        wthr['Hour'],
                                        wthr['Minute'],
                                        )
            wthr.index = times
        return wthr

    def calc_collected_energy(self,year,month,day,periods,frequency):
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
        if self.cs:
            start = "{0}-{1}-{2}".format(year[0],month[0],day[0])
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
        if self.tracking:
            max_phi = 30
        else:
            max_phi = 0

        orientation = tracking.singleaxis(solar_position['apparent_zenith'],
                                        solar_position['azimuth'],
                                        max_angle=max_phi,
                                        backtrack=True,
                                        gcr=gcr)

        # set axis_azimuth, albedo, pvrow width and height, and use
        # the pvfactors engine for absorbed irradiance
        pvrow_height = 1
        pvrow_width = 1
        albedo = 0.06

        #TODO: Create separate function to set these parameters
        if self.cs:
            axis_azimuth = np.random.rand(1)*360
            wind_speed = np.random.rand(1)*10
            temp_air = 30
        else:
            # Set windspeed and azimuth based on TMY data
            axis_azimuth = np.mod(wthr['wind_direction']+90,360) #tilt axis is 90 degrees offset from wind direction
            wind_speed = wthr['wind_speed'] # m/s
            temp_air = wthr['temp_air']

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
                                    self.pdc0,
                                    gamma_pdc=self.gamma
                                    ).fillna(0)
        
        return times, pdc

    def simulate_deployment(self,U,rho,takeoff_capacity,landing_capacity,period,P_solar,dt):
        """Determines duty cycle for specified period
        
        Parameters
        ----------
        U : float
            Cruise speed of the vehicle
        rho : float
            Air density at cruise altitude in kg/m^3
        takeoff_voltage : float
            Percentage of total capacity at which the vehicle is sufficiently charged to takeoff
        landing_voltage : float
            Percentage of total capacity at which the vehicle must land (%)
        period : int
            Time in days over which the simulation should run
        P_solar : numpy arr
            Array of solar power collected by the vehicle's photovoltaic system [W]
        dt : int
            Time in minutes between each sample in P_solar

        Returns
        -------
        duty_cycle : float
            Percentage of time period spent in air
        energy_j : list, float
            Energy stored in battery for each simulated time

        """
        # Get cruise power
        P_cruise = self.get_required_power(U,rho)

        state = "Moored"
        flying = 0
        state_history = [0]

        # TODO: Add correction for energy gained from solar panels
        capacity_j = self.voltage*self.capacity*dt*60
        energy_j = capacity_j*.8
        energy_history = [energy_j/capacity_j]

        # Define the daytime range (e.g., from 7 AM to 7 PM)
        # TODO: Set daylight hours based on when sun rises and sets, not hard coded
        daytime_start = pd.to_datetime('07:00:00').time()
        daytime_end = pd.to_datetime('19:00:00').time()
        is_daytime = (P_solar.index.time >= daytime_start) & (P_solar.index.time <= daytime_end)
        
        for i in range(0,len(P_solar)):
            
            if state == "Flying":
                state_history.append(1)
                flying += 1
                energy_j-= (P_cruise - P_solar.iloc[i])*dt*60
                if energy_j <= capacity_j*landing_capacity or not is_daytime[i]:
                    state = "Moored"
            elif state == "Moored":
                state_history.append(0)
                if energy_j <= capacity_j:
                    energy_j+= (P_solar.iloc[i])*dt*60
                if energy_j>=takeoff_capacity*capacity_j and is_daytime[i]:
                    state = "Flying"
                    energy_j -= 1000*30
            if energy_j > capacity_j :
                energy_j = capacity_j
            energy_history.append(energy_j/capacity_j*100)

        return flying/np.sum(is_daytime),energy_history,state_history


        
    def calc_takeoff_penalty(self):
        pass

