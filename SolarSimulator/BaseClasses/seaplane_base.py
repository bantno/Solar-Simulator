import os
import warnings

import numpy as np
import pandas as pd

from pvlib import location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem



# supressing shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action='ignore', module='pvfactors')

class Seaplane:
    """Class representing a seaplane"""
    def __init__(self, lat, lon, tz, pdc0, gamma,tracking:bool=False,cs:bool=False,
                 cd0=0.01,cdtot = 0.06,n_tot=.75,S=1,af_mass=6,voltage=22.2,capacity=150):

        # Define solar parameters
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.tracking = tracking
        self.location = location.Location(lat, lon, tz=tz)
        
        self.gamma = gamma
        self.collected_energy = 0 #kWh
        self.cs = cs
        self.idle_power = 1.0 # W

        # Define airframe and motion parameters
        self.cd0 = cd0
        self.n_tot = n_tot
        self.S = S

        path = r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data"
        self.af_mass = self.get_total_mass(path)

        self.voltage = voltage
        self.capacity = capacity
        self.Rt = 1.0
        self.n = 1.3
        self.AR = 6.0 #remove hardcode
        self.e = 0.8
        self.k = 1.0/(np.pi*self.AR*self.e)
        self.cdtot = cdtot
        
        self.calculate_pdc0()
        self.calculate_weight()
        self.required_cruise_power = self.get_required_power(20,1.2)
        self.required_takeoff_energy = self.get_required_takeoff_energy(1)
        

    def get_total_mass(self,directory):
        # Search for a file with 'mass' in its name in the given directory
        for filename in os.listdir(directory):
            if 'mass' in filename.lower():
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    for line in file:
                        if 'Total Mass' in line:
                            total_mass = float(line.split()[0])
                            return total_mass
        return None
        
    def update_plane(self):
        self.calculate_pdc0()
        self.calculate_weight()

    def update_location(self,lat):
        self.lat = lat
        self.location = location.Location(self.lat, self.lon, tz=self.tz)

    def calculate_pdc0(self):
        self.pdc0 = self.S*3.3/.034 # based on ascent solar bare module - mid scale

    def calculate_weight(self,energy_density=150) -> float:
        """Estimates weight of the aircraft based on fixed airframe mass and variable battery mass.
        
        Assumes that the airframe mass will not change but the installed battery capacity will.
        
        Parameters:
        self: Seaplane
            Requires seaplane object
        """

        battery_mass = self.capacity*self.voltage/energy_density
        payload_mass = 1.35
        pv_mass = self.S*.0039/.034
        fcs_mass = 0.2
        propulsion_mass = 0.0002*4000 # k_ps*P_ps
        k_str = 0.6

        mass = (payload_mass + pv_mass  + fcs_mass + propulsion_mass)/(1-k_str) + battery_mass
        # mass = self.af_mass # Set mass to imported mass from openVSP
        self.weight = 9.81*(mass)

    def get_endurance(self,u,rho) -> float:
        """Returns endurance estimate according to Traub
        
        Parameters
        ----------
        U : float
            Cruise speed [m/s]
        rho : float
            Air density [kg/m^3]
        
        """
        p_req = self.get_required_power(u,rho)
        e=self.Rt**(1.-self.n)*(self.n_tot*self.voltage*self.capacity)/p_req
        return e

    def get_dynamic_pressure(self,U,rho) -> float:
        """Returns dynamic pressure

        Parameters
        ----------
        U : float
            Cruise speed [m/s]
        rho : float
            Air density [kg/m^3]
        
        """
        return 0.5*rho*U**2

    def get_required_power(self,U,rho) -> float:
        """Returns required cruise power consumption
        
        Parameters
        ----------
        U : float
            Cruise speed [m/s]
        rho : float
            Air density [kg/m^3]
        
        """
        # q = self.get_dynamic_pressure(U,rho)
        # D = q*self.S*self.cdtot*U
        # D = self.cdtot*rho*0.5*self.S*(U**2) # U in m/s
        D = .5*rho*U**3*self.S*self.cd0 + 2*self.weight**2*self.k/(rho*U*self.S) # From Traub
        return D

    def get_required_takeoff_energy(self,takeoff_time_min) -> float:
        """Determine energy cost of taking off in Joules"""
        required_power_w = 4000.
        seconds_to_takeoff = 60.
        required_takeoff_energy_J = required_power_w*seconds_to_takeoff
        return required_takeoff_energy_J
