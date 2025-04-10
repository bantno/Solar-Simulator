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
        n_tot=0.75,
        S=1,
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

        # Define airframe and motion parameters
        self.cd0 = cd0          # Parasite drag coefficient at zero lift
        self.n_tot = n_tot
        self.S = S              # Wing area (in m²)

        # Battery and airframe properties
        self.voltage = voltage
        self.capacity = capacity
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
        self.motor_efficiency = 0.85     # η_m
        self.propeller_efficiency = 0.85 # η_p

        # Assume a typical lift-to-drag ratio for the aircraft; this value is used in the climb power model.
        self.L_over_D = 10.0

        # Perform initial calculations
        self.calculate_pdc0()
        self.calculate_weight()
        self.required_cruise_power = self.cruise_power
        self.required_takeoff_energy = self.get_required_takeoff_energy(1)

    def get_total_mass(self, directory):
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
        self.calculate_pdc0()
        self.calculate_weight()
        self.required_cruise_power = self.cruise_power
        self.required_takeoff_energy = self.get_required_takeoff_energy(1)

    def update_location(self, lat):
        self.lat = lat
        self.location = location.Location(self.lat, self.lon, tz=self.tz)

    def calculate_pdc0(self):
        # Based on an ascent solar bare module - mid scale
        self.pdc0 = self.S * 3.3 / 0.034

    def calculate_weight(self, energy_density=150) -> float:
        """Estimates the aircraft's weight based on battery mass and fixed components.
        
        The weight (in Newtons) is updated in self.weight.
        """
        battery_mass = self.capacity * self.voltage / energy_density
        payload_mass = 1.35
        pv_mass = self.S * 0.0039 / 0.034
        fcs_mass = 0.2
        propulsion_mass = 0.0002 * 4000  # k_ps*P_ps
        k_str = 0.6

        mass = (payload_mass + pv_mass + fcs_mass + propulsion_mass) / (1 - k_str) + battery_mass
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

    @property
    def cruise_power(self) -> float:
        """Compute the estimated propulsion power (W) required for level cruise flight.
        
        Assumes a representative cruise speed (20 m/s) and sea-level air density (1.2 kg/m³).
        """
        U_cruise = 20   # m/s
        rho = 1.2       # kg/m³ (typical sea-level density)
        return self.get_propulsion_power(U_cruise, rho)

    @property
    def takeoff_power(self) -> float:
        """Estimate the propulsion power (W) needed during takeoff.
        
        For this heuristic, the takeoff power is assumed to be twice the level-flight (cruise) power.
        """
        return 2 * self.get_propulsion_power(20, 1.2)

    def get_required_takeoff_energy(self, takeoff_time_min) -> float:
        """Determine the energy cost of takeoff in Joules.
        
        Uses a fixed required power (default 4000 W) over the duration of the takeoff.
        """
        required_power_w = 4000.0
        seconds_to_takeoff = 60.0 * takeoff_time_min
        required_takeoff_energy_j = required_power_w * seconds_to_takeoff
        return required_takeoff_energy_j

    def climb(self, U: float, gamma: float, climb_time: float, rho: float = 1.2) -> dict:
        """Estimate the propulsion power and energy required for a climb maneuver.
        
        Under the constant lift-to-drag assumption, the steady-state propulsion power for climb is given by:
        
           P_climb = P_level * [cos(γ) + (L/D)·sin(γ)]
        
        where:
          • P_level is the steady-state (level) propulsion power computed at airspeed U and air density ρ.
          • γ is the climb angle (in radians; positive for a climb).
          • L/D is the lift-to-drag ratio (here assumed to be self.L_over_D).
        
        The energy required to sustain the climb for a duration of climb_time is then:
        
           E_climb = P_climb * climb_time
        
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
        # Compute the level-flight propulsion power at speed U and air density rho.
        P_level = self.get_propulsion_power(U, rho)
        # Adjust for climb: additional power is required to overcome the vertical component of weight.
        P_climb = P_level * (np.cos(gamma) + self.L_over_D * np.sin(gamma))
        climb_energy = P_climb * climb_time
        return {"climb_power": P_climb, "climb_energy": climb_energy}

    def get_mdp_power_params(self) -> dict:
        """
        Adapter method to package the power parameters for the MDP.
        Ensure that units match what your MDP expects (e.g., power vs. energy conversion).
        """
        return {
            "idle_power": self.idle_power,
            "cruise_power": self.cruise_power,
            "takeoff_power": self.takeoff_power
        }
