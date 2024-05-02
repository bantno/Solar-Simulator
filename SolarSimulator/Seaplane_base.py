import pandas as pd
import pvlib.iotools
from pvlib import location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem
import numpy as np
import matplotlib.pyplot as plt
import warnings

# supressing shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action='ignore', module='pvfactors')

class Seaplane:
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
        print("Plane Initialized")

    def get_endurance(self,U,rho):
        E = self.Rt**(1.0-self.n)*(self.n_tot*self.voltage*self.capacity)/self.get_required_power(U,rho)
        return E
    
    def get_weather_endurance(self,P_req,dt=15):
        
        joules = self.voltage*self.capacity*3600
        E = 0
        for i in range(5,len(P_req)):
            joules-=P_req[i]
            if joules < 0:
                return E
            E += dt
            
    
    def get_dynamic_pressure(self,U,rho):
        return 0.5*rho*U**2
    
    def get_required_power(self,U,rho):
        q = self.get_dynamic_pressure(U,rho)
        D = q*self.S*self.cdtot*U

        # return D*U # U in m/s
        return .5*rho*U**3*self.S*self.cd0 + 2*self.weight**2*self.k/(rho*U*self.S)
    
    def get_times(self,year,month,day,tz):
        # get times of interest
        start = "{0}-{1}-{2}".format(year,month,day)
        day = day + 1
        end = "{0}-{1}-{2}".format(year,month,day)
        return pd.date_range(start, end, freq='60min', tz=tz) # Get times for entire study period      

    def get_weather(self,cs:bool,times=0):
        if cs:
            # get clearsky weather
            wthr = self.location.get_clearsky(times) # Get clearsky weather data
            print("Retrieved CS Weather Data")
        else:
            wthr = self.get_tmy()
            print("Retrieved TMY Weather Data")
        return wthr

    def get_tmy(self):
        # Weather Data Info
        api_key = 'unURaXbAGeMjP8359wy5gyQfWIKq1g1y7hdhUmNo'
        email = 'bepstein8@gatech.edu'
        keys = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed', 'wind_direction',
                'albedo']

        psm3, psm3_metadata = pvlib.iotools.get_psm3(self.lat, self.lon, api_key,
                                                email, interval=60, names='tmy',
                                                map_variables=True, leap_day=False,
                                                attributes=keys)
        return psm3
    
    def calc_collected_energy(self,year,month,day):
        
        if self.cs:
            start = "{0}-{1}-{2}".format(year[0],month[0],day[0])
            end = "{0}-{1}-{2}".format(year[1],month[1],day[1])
            times = pd.date_range(start, end, freq='15min', tz=self.tz)
            
            wthr = self.get_weather(self.cs,times)
        else:
            wthr = self.get_weather(self.cs)
            month_s = month[0]
            month_e = month[1]
            day_s = day[0]
            day_e = day[1]

            wthr.query('Month >= @month_s and Month <= @month_e',inplace=True)
            wthr.query('Day >= @day_s and Day <= @day_e',inplace=True)
            
            start = "{0}-{1}-{2}".format(year[0],month[0],day[0])
            end = "{0}-{1}-{2}".format(year[1],month[1],day[1])
            times = wthr.index

        # get solar position data
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

        #TODO: Create separate function
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
        irrad_cs = pvfactors_timeseries(solar_position['azimuth'],
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
        irrad_cs = pd.concat(irrad_cs, axis=1)
        effective_irrad_mono = irrad_cs['total_abs_front']
        temp_cell = temperature.faiman(effective_irrad_mono, temp_air=temp_air,
                                    wind_speed=wind_speed) 
        # Create pvsystem using pvwatts model for single face solar cell
        pdc = pvsystem.pvwatts_dc(effective_irrad_mono,
                                    temp_cell,
                                    self.pdc0,
                                    gamma_pdc=self.gamma
                                    ).fillna(0)
        
        return times, pdc



    def calc_tmy_energy(self,year):

        power_kWh = []
        pdc_list = []

        start = "{0}-{1}-{2}".format(year,1,1)
        end = "{0}-{1}-{2}".format(year,12,31)
        times = pd.date_range(start, end, freq='60min', tz=self.tz) # Get times for entire study period

        psm3 = self.get_weather(self.cs,times)
        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_counter = 0

        for j in range(0,12) :
            month = j+1
            day = 1
            total_energy = 0
            for i in range(0,months[j]-1) : # need to change this so that every day of the year is accounted for
                times = self.get_times(year,month,day+i,self.tz)
                wthr = psm3.iloc[day_counter*24:day_counter*24+len(times)] # update to only be the correct times

                # get solar position data
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
                                                gcr=gcr
                                                )

                # set axis_azimuth, albedo, pvrow width and height, and use
                # the pvfactors engine for both front and rear-side absorbed irradiance
                pvrow_height = 1
                pvrow_width = 1
                albedo = 0.06


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
                irrad_cs = pvfactors_timeseries(solar_position['azimuth'],
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
                irrad_cs = pd.concat(irrad_cs, axis=1)
                
                effective_irrad_mono = irrad_cs['total_abs_front']
                
                temp_cell = temperature.faiman(effective_irrad_mono, temp_air=temp_air,
                                            wind_speed=wind_speed) 
                
                # Create pvsystem using pvwatts model for single face solar cell
                pdc = pvsystem.pvwatts_dc(effective_irrad_mono,
                                            temp_cell,
                                            self.pdc0,
                                            gamma_pdc=self.gamma
                                            ).fillna(0)
                
                total_energy += np.trapz(pdc,dx=60*60)/3600000 # energy in kWh
                day_counter+=1
            power_kWh.append(total_energy)
            pdc_list.append(pdc)

        total_energy = np.sum(power_kWh)
        return total_energy,power_kWh


