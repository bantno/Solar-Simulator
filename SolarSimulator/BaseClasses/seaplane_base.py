import os
import warnings
import numpy as np
from pvlib import location

# Suppress shapely warnings that occur on import of pvfactors
warnings.filterwarnings(action="ignore", module="pvfactors")


class Seaplane:
    """Class representing a seaplane modeled with a propulsion power formulation
    inspired by Dantsker et al. (2018). This implementation provides methods to
    estimate steady-state (level) cruise power as well as the increased power and
    energy requirements during a climb maneuver.
    """
    
    def __init__(
        self,
        lat,
        lon,
        tz,
        cd0=0.01,
        cdtot=0.06,
        n_tot=0.65,
        S=0.66,
        af_mass=6,
        voltage=22.2,
        capacity=150,
    ):
        # Define solar parameters
        self.lat = lat
        self.lon = lon
        self.tz = tz
        self.collected_energy = 0  # kWh
        self.idle_power = 0.0      # W
        self.mean_chord = 0.33        # Mean aerodynamic chord (m)

        # Define airframe and motion parameters
        self.cd0 = cd0          # Parasite drag coefficient at zero lift
        self.n_tot = n_tot
        self.S = S              # Wing area (in m²)

        # Battery and airframe properties
        self.voltage = voltage
        self.capacity = capacity
        self.capacity_Wh = voltage*capacity
        self.capacity_J = self.capacity_Wh * 3600
        self.af_mass = af_mass   # (Unused in weight calculation here)

        # Additional aerodynamic and propulsion parameters
        self.Rt = 1.0
        self.n = 1.3
        self.AR = 8.0          # Aspect ratio (could be parameterized)
        self.e = 0.8           # Oswald efficiency factor
        self.k = 1.0 / (np.pi * self.AR * self.e)  # Induced drag factor
        self.cdtot = cdtot
        self.solar_panel_efficiency = 0.1

        # Propulsion efficiency factors (assumed typical values)
        self.motor_efficiency = 0.9     # η_m
        self.propeller_efficiency = 0.8 # η_p

        self.INSTALLED_PROPULSION_POWER = 4000 # Watts
        self.cruise_altitude = 300 # meters


        # Perform initial calculations
        self.calculate_pdc0()
        self.calculate_weight()
        self.update_plane()

    def get_total_mass(self, directory):
        """
        Searches for a file containing 'mass' in its name within the specified directory.
        If found, reads the file to extract the total mass of the aircraft.
        """
        # Search for a file with 'mass' in its name in the given directory.
        for filename in os.listdir(directory):
            if "mass" in filename.lower():
                file_path = os.path.join(directory, filename)
                with open(file_path, "r") as file:
                    for line in file:
                        if "Total Mass" in line:
                            total_mass = float(line.split()[0])
                            return total_mass
        return None

    def update_plane(self):
        """Update the plane's parameters based on the current state."""
        self.calculate_pdc0()
        self.calculate_weight()
        self.cruise_speed = self.get_max_endurance_speed()
        self._cached_takeoff_power = self.get_average_takeoff_power()
        self._cached_landing_power = self.get_average_landing_power()
        


    def update_location(self, lat):
        """Update the plane's location based on the given latitude."""
        self.lat = lat
        self.location = location.Location(self.lat, self.lon, tz=self.tz)

    def calculate_pdc0(self):
        """Calculate the initial power output of the solar panels."""
        # Based on an ascent solar bare module - mid scale
        self.pdc0 = self.S * 3.3 / 0.034

    def calculate_weight(self, energy_density=150) -> float:
        """Estimates the aircraft's weight based on battery mass and fixed components.
        
        The weight (in Newtons) is updated in self.weight.
        """
        battery_mass = (self.capacity * self.voltage) / energy_density
        payload_mass = 1.35
        pv_mass = self.S * 0.0039 / 0.034
        fcs_mass = 0.2
        propulsion_mass = 0.288*self.INSTALLED_PROPULSION_POWER/1500  # k_ps*P_ps
        k_str = 0.6

        mass = (payload_mass + pv_mass + fcs_mass + propulsion_mass + battery_mass) / (1 - k_str) 
        self.weight = 9.81 * mass
        return self.weight

    def get_dynamic_pressure(self, U, rho) -> float:
        """Returns the dynamic pressure."""
        return 0.5 * rho * U**2

    def get_lift_coefficient(self, rho, U):
        """Calculate the lift coefficient required for flight."""
        self.calculate_weight()  # Update weight if necessary.
        C_l = self.weight / (0.5 * rho * U**2 * self.S)
        return C_l

    def get_max_endurance_speed(self):
        W = self.weight
        rho = self.rho(self.cruise_altitude)
        S = self.S
        k = self.k
        C_D0 = self.cd0
        U_E = np.sqrt(2*W/(rho*S)*np.sqrt(k/(3*C_D0)))
        return U_E

    def get_propulsion_power(self, U: float, rho: float) -> float:
        """Calculates the steady-state propulsion power (Watts) required for level flight at speed U.
        
        This method computes the power needed to overcome both parasite drag (scaling with U³)
        and induced drag (scaling inversely with U). The computed power is then adjusted for the
        overall propulsion system efficiency.
        
        Args:
            U (float): Flight speed (m/s).
            rho (float): Air density (kg/m³).
        
        Returns:
            float: Steady-state propulsion power in Watts.
        """
        eta_m = self.motor_efficiency
        eta_p = self.propeller_efficiency

        power_parasite = 0.5 * rho * U**3 * self.S * self.cd0
        power_induced = 2 * self.weight**2 * self.k / (rho * U * self.S)
        P_propulsion = (power_parasite + power_induced) / (eta_m * eta_p)
        return P_propulsion

    def get_takeoff_power(self, U: float, rho: float) -> float:
        """Estimate the propulsion power (W) needed during takeoff.
        
        """
        return 

    @property
    def cruise_power(self) -> float:
        """
        Compute the estimated propulsion power (W) required for level cruise flight.
        """
        U_cruise = self.cruise_speed   # m/s
        rho = self.rho(300)       # kg/m³ (typical sea-level density)
        return self.get_propulsion_power(U_cruise, rho)

    @property
    def takeoff_power(self) -> float:
        """
        Estimate the propulsion power (W) needed during takeoff.
        """
        if not hasattr(self, "_cached_takeoff_power"):
            self._cached_takeoff_power = self.get_average_takeoff_power()
        return self._cached_takeoff_power

    def calculate_liftoff_energy(self):
        return self.INSTALLED_PROPULSION_POWER * 8

    def get_average_takeoff_power(self) -> float:
        """Estimate the average propulsion power (W) needed during takeoff.
        
        For this heuristic, the takeoff power is assumed to be twice the level-flight (cruise) power.
        """
        liftoff_energy = self.calculate_liftoff_energy()
        TIMESTEP_MIN = 15
        climb_energy, climb_time_s = self.climb_energy(self.cruise_speed,np.radians(2),0,300,1)
        cruise_power = self.cruise_power
        cruise_time = TIMESTEP_MIN*60 - climb_time_s
        total_energy = climb_energy + (cruise_power * cruise_time) + liftoff_energy
        average_power = total_energy / (TIMESTEP_MIN*60)
        return average_power

    def get_reynolds_number(self, U: float, altitude: float = 0.0) -> float:
        """Estimate the Reynolds number based on flight speed and altitude.
        Uses Sutherland's law for dynamic viscosity of air."""
        # Air properties
        T = 288.15 - 0.0065 * altitude  # Temperature (K)
        rho = self.rho(altitude)       # Density (kg/m^3)
        # Sutherland's law constants
        mu0 = 1.81e-5                  # Reference viscosity at T0 (Pa·s)
        T0 = 288.15                    # Reference temperature (K)
        S = 110.4                      # Sutherland's constant (K)
        # Dynamic viscosity
        mu = mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        # Reynolds number: Re = rho * U * chord / mu
        Re = rho * U * self.mean_chord / mu
        return Re

    def climb_power(self, U: float, gamma: float, rho: float = 1.2) -> dict:
        """Estimate the propulsion power and energy required for a climb maneuver.
        
        Under the constant lift-to-drag assumption, the steady-state propulsion power for climb is given by:
        
           P_climb = P_level * [cos(γ) + (L/D)·sin(γ)]
        
        where:
          • P_level is the steady-state (level) propulsion power computed at airspeed U and air density ρ.
          • γ is the climb angle (in radians; positive for a climb).
          • L/D is the lift-to-drag ratio (here assumed to be self.L_over_D).
        
        The energy required to sustain the climb for a duration of climb_time is then:
        
           E_climb = P_climb * climb_time
        
        Constant L/D ratio fomulation in Dantsker et al. (2018) [10.2514/6.2018-5009]

        Args:
            U (float): Flight speed during the climb (m/s).
            gamma (float): Climb angle (radians); positive for climb, negative for descent.
            climb_time (float): Duration of the climb maneuver (seconds).
            rho (float): Air density (kg/m³); defaults to 1.2 for sea-level conditions.
        
        Returns:
            dict: A dictionary containing:
                'climb_power' : Estimated propulsion power during the climb (W),
                'climb_energy': Energy required for the climb (J).
        """
        # TODO: Implement this so that density changes with altitude. Will likely require integrating dynamic equations of motion
        # Compute the level-flight propulsion power at speed U and air density rho.
        L_2_D_RATIO = 28
        P_level = self.get_propulsion_power(U, rho)
        # Adjust for climb: additional power is required to overcome the vertical component of weight.
        P_climb = P_level * (np.cos(gamma) + L_2_D_RATIO * np.sin(gamma))
        return P_climb

    def climb_energy(self, U: float, gamma: float, starting_altitude:float, ending_altitude:float, timestep:int) -> float:
        """Estimate the energy required for a climb maneuver.
        
        Uses the climb power and the duration of the climb to compute the total energy required.
        
        Args:
            U (float): Forward flight speed during the climb (m/s).
            gamma (float): Climb angle (radians); positive for climb, negative for descent.
            starting_altitude (float): Starting altitude (m).
            ending_altitude (float): Ending altitude (m).
            timestep (int): Time step for the climb (seconds).
        
        Returns:
            float: Energy required for the climb in Joules.
        """
        if gamma == 0.0:
            raise AssertionError("Climb angle must be non-zero.")
        altitude = starting_altitude
        E_climb = 0.0
        time = 0.0
        v_vert = U * np.sin(gamma)
        while altitude < ending_altitude:
            rho = self.rho(altitude)
            P_climb = self.climb_power(U, gamma, rho)
            E_climb += P_climb * timestep
            altitude += v_vert * timestep
            time+= timestep
        return E_climb, time

    @property    
    def landing_power(self) -> float:
        """Estimate the propulsion power (W) needed during landing.
        
        For this heuristic, the takeoff power is assumed to be twice the level-flight (cruise) power.
        """
        if not hasattr(self, "_cached_landing_power"):
            self._cached_landing_power = self.get_average_landing_power()
        return self._cached_landing_power

    def get_average_landing_power(self) -> float:
        """Estimate the average propulsion power (W) needed during takeoff.
        
        For this heuristic, the takeoff power is assumed to be twice the level-flight (cruise) power.
        """
        
        TIMESTEP_MIN = 15
        descent_energy, descent_time_s = self.descent_energy(self.cruise_speed,np.radians(-2),300,0,1)
        # cruise_power = self.cruise_power
        # cruise_time = TIMESTEP_MIN*60 - descent_time_s
        total_energy = descent_energy # + (cruise_power * cruise_time)
        average_power = total_energy / (TIMESTEP_MIN*60)
        return average_power

    def descent_power(self,U,gamma,rho):
        """Estimate the propulsion power and energy required for a descent maneuver.
        
        Under the constant lift-to-drag assumption, the steady-state propulsion power for descent is given by:
        
           P_descent = P_level * [cos(γ) - (L/D)·sin(γ)]
        
        where:
          • P_level is the steady-state (level) propulsion power computed at airspeed U and air density ρ.
          • γ is the descent angle (in radians; negative for descent).
          • L/D is the lift-to-drag ratio (here assumed to be self.L_over_D).
        
        The energy required to sustain the descent for a duration of descent_time is then:
        
           E_descent = P_descent * descent_time
        
        Args:
            U (float): Flight speed during the descent (m/s).
            gamma (float): Descent angle (radians); negative for descent, positive for climb.
            rho (float): Air density (kg/m³); defaults to 1.2 for sea-level conditions.
        
        Returns:
            'P_descent' : Estimated propulsion power during the descent (W),
        """
        L_2_D_RATIO = 28
        P_level = self.get_propulsion_power(U,rho)
        P_descent = P_level * (np.cos(gamma) - (L_2_D_RATIO)*np.sin(gamma))
        return P_descent

    def descent_energy(self, U: float, gamma: float, starting_altitude:float, ending_altitude:float, timestep:int) -> float:
        """Estimate the energy required for a climb maneuver.
        
        Uses the climb power and the duration of the climb to compute the total energy required.
        
        Args:
            U (float): Forward flight speed during the climb (m/s).
            gamma (float): Climb angle (radians); positive for climb, negative for descent.
            starting_altitude (float): Starting altitude (m).
            ending_altitude (float): Ending altitude (m).
            timestep (int): Time step for the climb (seconds).
        
        Returns:
            float: Energy required for the climb in Joules.
        """
        if gamma == 0.0:
            raise AssertionError("Climb angle must be non-zero.")
        altitude = starting_altitude
        E_descend = 0.0
        time = 0.0
        v_vert = U * np.sin(gamma)
        while altitude > ending_altitude:
            rho = self.rho(altitude)
            P_descend = self.descent_power(U, gamma, rho)
            E_descend += P_descend * timestep
            altitude += v_vert * timestep
            time+= timestep
        return E_descend, time

    def rho(self, altitude: float) -> float:
        """Dry-air density from the ISA tropospheric model (0–11 km).
        
        Args:
            altitude (float): Altitude in meters.
        
        Returns:
            float: Air density in kg/m³.
        """
        # 1. Temperature (K)
        T = 288.15 - 0.0065 * altitude
        # 2. Pressure (Pa)
        p = 101325 * (T / 288.15) ** 5.2561
        # 3. Density (kg m-3)
        return p / (287.05 * T)

    def get_mdp_power_params(self) -> dict:
        """
        Adapter method to package the power parameters for the MDP.
        """
        return {
            "idle_power": self.idle_power,
            "cruise_power": self.cruise_power,
            "takeoff_power": self.takeoff_power,
            "landing_power": self.landing_power,
        }

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    seaplane = Seaplane(lat=0.0, lon=0.0, tz="UTC", capacity=200/22.2)
    altitude = 300
    Re = seaplane.get_reynolds_number(U=20, altitude=altitude)
    print(f"Reynolds number at 20 m/s, {altitude} km: {Re:.2e}")
    power_list = []
    power_list_takeoff = []
    endurance_list = []
    capacities_wh = range(0,1000,100)
    for i in capacities_wh:
        seaplane.capacity = i/22.2
        seaplane.update_plane()
        # print(seaplane.weight/9.81)
        Lift_c = seaplane.get_lift_coefficient(1.2,20)
        print(Lift_c, seaplane.cruise_speed)
        power_landing = seaplane.landing_power
        power_list.append(power_landing)
        power_takeoff = seaplane.takeoff_power
        power_list_takeoff.append(power_takeoff)

        # endurance = seaplane.capacity * seaplane.voltage/ power
        # endurance_list.append(endurance)

    plt.plot(capacities_wh,power_list)
    plt.plot(capacities_wh,power_list_takeoff)
    # plt.plot(capacities_wh,endurance_list)
    plt.show()