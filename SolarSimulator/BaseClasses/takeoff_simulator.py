import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class AmphibiousTakeoffSimulator:
    def __init__(self, params):
        """
        Initialize the simulator with aircraft and fluid properties.
        """
        self.params = params

    # Buoyancy force based on Archimedes’ principle (Eq. (1))
    def calc_buoyancy(self):
        Vp = self.params["Vp"]
        return self.params["rho_w"] * self.params["g"] * Vp

    # Modification of wetted length-beam ratio (Eqs. (5) and (6))
    def calc_kb(self):
        k = self.params["k"]
        b = self.params["b"]
        n = self.params["n"]
        # For demonstration, we do not use FrB in a complex form.
        if k <= 1:
            k1 = 1.6 * k - 0.3 * k**2
        else:
            k1 = k + 0.3
        kb = k1 / (0.8 * np.cos(b))
        return kb

    # Modification of pitch angle (Eqs. (8) and (9))
    def calc_modified_pitch_angle(self, theta):
        # Here theta is the current pitch angle
        h = theta  # using current pitch angle as "base" for h
        Ls = self.params["Ls"]
        Lb = self.params["Lb"]
        kb = self.calc_kb()
        Fr = self.params["Fr"]
        # Simple geometric correction from Eq. (8)
        hb = h + kb * ((Ls - Lb) / Ls) if Ls > Lb else h
        # Empirical correction (Eq. (9)); add a correction proportional to Fr.
        hb_corr = hb + 0.1 * Fr  
        return hb_corr

    # Hydrodynamic lift (Eq. (10))
    def calc_hydrodynamic_lift(self, V, theta):
        kb = self.calc_kb()
        B = self.params["B"]
        hb = self.calc_modified_pitch_angle(theta)
        return self.params["rho_w"] * V**2 * kb * B**2 * (0.35 * hb) / (1 + 1.4 * kb)

    # Hydrodynamic resistance (Eqs. (11)–(15))
    def calc_hydrodynamic_resistance(self, V):
        Re = self.params["Re"]
        Aas = self.params["Aas"]
        k_shape = self.params["k_shape"]
        # Frictional resistance coefficient using the ITTC formula (Eq. (12))
        Cf = 0.075 / (np.log10(Re) - 2)**2
        Rf = 0.5 * self.params["rho_w"] * V**2 * Aas * Cf
        # Viscous pressure resistance (Eq. (13))
        Cvp = k_shape * Cf
        Rvp = 0.5 * self.params["rho_w"] * V**2 * Aas * Cvp
        # Wave resistance (dummy model using Fr, Eq. (14))
        Fr = self.params["Fr"]
        Cwa = 0.1 * Fr  
        Rwa = 0.5 * self.params["rho_w"] * V**2 * Aas * Cwa
        # Spray resistance (Eq. (15)); again use Cf for friction coefficient
        Rs = 0.5 * self.params["rho_w"] * V**2 * Aas * Cf
        return Rf + Rvp + Rwa + Rs

    # Hydrodynamic damping moment (Eq. (18))
    def calc_hydrodynamic_damping_moment(self, V, pitch_rate):
        # Wetted area: assume Sw = kb * B^2, per Eq. (7) simplified.
        kb = self.calc_kb()
        B = self.params["B"]
        Sw = kb * B**2
        l = self.params["l"]
        return self.params["Cmqw"] * self.params["rho_w"] * V * Sw * l**2 * pitch_rate

    # Engine thrust model (Eq. (23))
    def single_engine_thrust(self, throttle):
        # Assume maximum thrust per engine of 40,000 N
        return throttle * 4000

    def calc_engine_thrust(self):
        nT = self.params["nT"]
        throttle = self.params["throttle"]
        uT = self.params["uT"]
        lT = self.params["lT"]
        T = self.single_engine_thrust(throttle)
        FTx = nT * T * np.cos(uT)  # x-component
        FTz = -nT * T * np.sin(uT) # z-component; negative sign indicates direction
        MT = -nT * T * lT        # Simplified engine-induced pitching moment
        return FTx, FTz, MT

    # Dynamics for the longitudinal motion (simplified Eq. (19))
    def dynamics(self, t, state):
        # Unpack state vector: [u, w, q, x, z, theta]
        # u: horizontal velocity, w: vertical velocity, 
        # q: pitch rate, x: horizontal position, 
        # z: vertical position, theta: pitch angle
        u, w, q, x, z, theta = state
        V = np.sqrt(u**2 + w**2) if (u**2 + w**2) > 0 else 1e-3  # avoid zero division
        
        # Buoyancy force (Eq. (1))
        Fbu = self.calc_buoyancy()
        
        # Aerodynamic forces (simplified from Eq. (20)):
        q_air = 0.5 * self.params["rho_air"] * V**2
        S = self.params["wing_area"]
        L_aero = q_air * S * self.params["CL"]
        D_aero = q_air * S * self.params["CD"]
        
        # Hydrodynamic forces:
        Lw = self.calc_hydrodynamic_lift(V, theta)
        Rw = self.calc_hydrodynamic_resistance(V)
        
        # Engine thrust (Eq. (23))
        FTx, FTz, MT_engine = self.calc_engine_thrust()
        
        # Sum forces (body axes)
        # x-direction: engine thrust minus aerodynamic drag and hydrodynamic resistance (projected)
        Fx = FTx - D_aero - Rw * np.cos(theta)
        # z-direction: sum of hydrodynamic and aerodynamic lift minus buoyancy and resistance (projected)
        Fz = FTz + Lw + L_aero - Fbu - Rw * np.sin(theta)
        
        # Linear accelerations:
        ax = Fx / self.params["mass"]
        az = Fz / self.params["mass"] - self.params["g"]
        
        # Rotational dynamics: Aerodynamic pitching moment (simplified Eq. (21))
        c = self.params["chord"]
        Maero = q_air * S * c * self.params["Cm"]
        # Hydrodynamic damping moment (Eq. (18))
        M_damp = self.calc_hydrodynamic_damping_moment(V, q)
        # Total moment (sum aerodynamic, engine, and damping moments)
        M_total = Maero + MT_engine + M_damp
        aq = M_total / self.params["Iyy"]
        
        # Kinematics (update positions and orientation):
        dtheta_dt = q
        dx_dt = u * np.cos(theta) - w * np.sin(theta)
        dz_dt = u * np.sin(theta) + w * np.cos(theta)
        
        return [ax, az, aq, dx_dt, dz_dt, dtheta_dt]
    
    def simulate_takeoff(self, t_span, initial_state, max_step=0.1):
        """
        Simulate the water takeoff process over the time span t_span,
        using the given initial state vector.
        """
        sol = solve_ivp(lambda t, y: self.dynamics(t, y),
                        t_span, initial_state, dense_output=True, max_step=max_step)
        return sol

    def plot_results(self, sol):
        """
        Plot key state variables from the simulation solution.
        """
        t_vals = sol.t
        state_vals = sol.y
        
        plt.figure(figsize=(10, 8))
    
        plt.subplot(3, 1, 1)
        plt.plot(t_vals, state_vals[0, :], label="u (m/s)")
        plt.plot(t_vals, state_vals[1, :], label="w (m/s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
    
        plt.subplot(3, 1, 2)
        plt.plot(t_vals, np.degrees(state_vals[5, :]), label="Pitch angle (°)")
        plt.ylabel("Pitch Angle (°)")
        plt.legend()
    
        plt.subplot(3, 1, 3)
        plt.plot(t_vals, state_vals[3, :], label="x (m)")
        plt.plot(t_vals, state_vals[4, :], label="z (m)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
    
        plt.tight_layout()
        plt.show()

# -------------------------
# Main execution: create instance and run simulation
# -------------------------
def main():
    # Define model parameters (example values; modify as needed)
    params = {
        "rho_w": 1000.0,       # Water density [kg/m^3]
        "g": 9.81,             # Gravitational acceleration [m/s^2]
        "mass": 5000.0,        # Aircraft mass [kg]
        "Vp": 20.0,            # Submerged volume [m^3]
        "rho_air": 1.225,      # Air density [kg/m^3]
        "wing_area": 30.0,     # Wing area [m^2]
        "CL": 0.5,             # Aerodynamic lift coefficient
        "CD": 0.05,            # Aerodynamic drag coefficient
        "Cm": -0.05,           # Aerodynamic moment coefficient
        "B": 10.0,             # Beam (width) of the aircraft [m]
        "k": 0.8,              # Base wetted length–beam ratio
        "b": np.radians(10),   # Deadrise angle [rad]
        "n": 4,                # Parameter for wetted ratio modification
        "FrB": 0.5,            # Dummy width-based Froude number
        "Ls": 8.0,             # Wetted length of the forebody keel [m]
        "Lb": 6.0,             # Length of straight section of forebody keel [m]
        "Fr": 0.5,             # Froude number for pitch correction (dummy)
        "Re": 1e6,             # Reynolds number
        "Aas": 50.0,           # Spray wetted area [m^2]
        "k_shape": 1.2,        # Shape coefficient for resistance
        "nT": 2,               # Number of working engines
        "throttle": 0.8,       # Throttle setting (0 to 1)
        "uT": np.radians(5),   # Engine mounting angle [rad]
        "lT": 2.0,             # Moment arm length [m]
        "Iyy": 10000.0,        # Pitch moment of inertia [kg·m^2]
        "chord": 3.0,          # Mean aerodynamic chord [m]
        "Cmqw": 0.02,          # Longitudinal hydrodynamic damping coefficient
        "l": 5.0               # Characteristic length for damping [m]
    }
    
    # Create simulator instance
    simulator = AmphibiousTakeoffSimulator(params)
    
    # Initial state vector: [u, w, q, x, z, theta]
    # Example: starting at taxiing speed 8.1 m/s, 0 vertical velocity,
    # zero pitch rate, at origin with a pitch angle of 3.7°.
    initial_state = [8.1, 0.0, 0.0, 0.0, 0.0, np.radians(3.7)]
    t_span = (0, 40)  # simulate for 40 seconds
    
    # Run simulation
    sol = simulator.simulate_takeoff(t_span, initial_state)
    
    # Plot the results
    simulator.plot_results(sol)
    
if __name__ == "__main__":
    main()
