import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimulationParams:
    def __init__(self,
                 B=5.0,          # beam, ft
                 S=375.0,        # wing area, ft^2
                 i_w=4.5,        # wing incidence angle, deg
                 W0=8000.0,      # gross weight, lbf
                 rho_water=63.5, # density of seawater, lbf/ft^3
                 rho_air=0.002378, # density of air, lbf·s^2/ft^4
                 g=32.174,       # ft/s^2
                 dt=0.5,         # time step, s
                 CV_cutoff=7.5,  # CV threshold
                 f=0.012,        # friction coefficient
                 eta_p = 0.82,   # propeller efficiency)
                 ):
        self.B = B
        self.S = S
        self.i_w = i_w
        self.W0 = W0
        self.rho_water = rho_water
        self.rho_air = rho_air
        self.g = g
        self.dt = dt
        self.CV_cutoff = CV_cutoff
        self.f = f
        self.mass = W0 / g
        self.eta = eta_p

class SeaplaneSimulator:
    def __init__(self, params: SimulationParams):
        self.p = params

    def ramp_up_factor(self, t):
        return 0.25 + 0.75 * min(t, 5) / 5 if t < 5 else 1.0

    def thrust(self, V, r):
        T_static = 0.01040 * V**2 - 10.065 * V + 3225
        return T_static * r

    def dynamic_pressure(self, V):
        return 0.5 * self.p.rho_air * V**2

    def speed_coefficient(self, V):
        return V / np.sqrt(self.p.g * self.p.B)

    def water_resistance_coeff(self, CV):
        CR = np.polyval([0.0011, -0.0221, 0.1062, -0.0149], CV)
        return max(CR, 0.0)

    def trim_angle(self, CV):
        return 7 + np.tanh(3.553 * CV - 3.891)

    def lift_drag(self, q, alpha):
        CL = 0.2 + 4.62 * alpha * np.pi / 180
        CD = 0.06 + 0.058 * CL**2
        return q * self.p.S * CL, q * self.p.S * CD

    def buoyancy_force(self, L):
        return self.p.W0 - L

    def reduced_water_resistance(self, CR, F_buoy):
        return CR * (F_buoy / self.p.W0) * self.p.W0

    def hydroplaning_resistance(self, V, CV):
        # onset at CV >= ~2.9 (~30 knots), V in ft/s -> convert to knots
        V_knots = V * 0.592484
        return self.p.f * self.p.B * V_knots**2 if CV >= 2.9 else 0.0

    def simulate(self):
        t, V, x = 0.0, 0.0, 0.0
        E_mech, E_elec = 0.0, 0.0
        rows = []
        while True:
            r = self.ramp_up_factor(t)
            T = self.thrust(V, r)
            q = self.dynamic_pressure(V)
            CV = self.speed_coefficient(V)
            CR = self.water_resistance_coeff(CV)
            alpha_trim = self.trim_angle(CV)
            alpha = alpha_trim + self.p.i_w
            L, D = self.lift_drag(q, alpha)
            F_buoy = self.buoyancy_force(L)
            R_water = self.reduced_water_resistance(CR, F_buoy) if CV <= self.p.CV_cutoff else 0.0
            R_froude = self.hydroplaning_resistance(V, CV)
            F_net = T - D - R_water - R_froude
            a = F_net / self.p.mass
            
            # mechanical power [lbf·ft/s]:
            P_mech = self.power(r)

            # electrical power [lbf·ft/s] → convert to ft·lbf/s,
            # later convert to Joules if you like by multiplying with 1.35582
            P_elec = P_mech / self.p.eta * 1.35582

            # accumulate energy
            E_mech += P_mech * self.p.dt
            E_elec += P_elec * self.p.dt

            rows.append({
                'time_s': t,
                'V_ft_s': V,
                'CV': CV,
                'Lift_lbf': L,
                'Drag_lbf': D,
                'Water_R_lbf': R_water,
                'Hydro_R_lbf': R_froude,
                'Total_R_lbf': D + R_water + R_froude,
                'Net_F_lbf': F_net,
                'Accel_ft_s2': a,
                'Distance_ft': x,
                'P_mech_lbf_ft_s': P_mech,
                'P_elec_lbf_ft_s': P_elec,
                'E_mech_lbf_ft':  E_mech,
                'E_elec_lbf_ft':  E_elec,
            })

            # Terminate when lift exceeds weight and hydroplaning started
            if L > self.p.W0 and CV >= self.p.CV_cutoff or t >= 2000:
                break

            # Integrate
            V += a * self.p.dt
            x += V * self.p.dt
            t += self.p.dt

        return pd.DataFrame(rows)

# # Usage
# params = SimulationParams()
# sim = SeaplaneSimulator(params)
# df = sim.simulate()
# # Compute total resistance
# df['Total_Resistance_lbf'] = df['Water_R_lbf'] + df['Hydro_R_lbf'] + df['Drag_lbf']

# # Convert airspeed to knots
# airspeed_knots = df['V_ft_s'] / 1.688
# # Plot forces vs airspeed
# plt.figure()
# plt.plot(airspeed_knots, df['Water_R_lbf'], label='Water Resistance')
# plt.plot(airspeed_knots, df['Hydro_R_lbf'], label='Hydroplaning Resistance')
# plt.plot(airspeed_knots, df['Drag_lbf'], label='Aerodynamic Drag')
# plt.plot(airspeed_knots, df['Total_Resistance_lbf'], linestyle='--', linewidth=2, label='Total Resistance')
# plt.xlabel('Airspeed (knots)')
# plt.ylabel('Force (lbf)')
# plt.title('Resistance Components and Total vs Airspeed')
# plt.legend()
# plt.grid(True)
# plt.show()



V_onset = 20.0 * 1.688  # ft/s (20 knots)
# Example for a 4 ft×0.5 ft wing UAV, 15 lbf weight, 1 ft beam:
uav_params = SimulationParams(
  B          = 0.5,     # ft
  S          = 5.0*0.5, # 2 ft²
  i_w        = 4.5,     # same incidence
  W0         = 8.0,    # lbf
  rho_water  = 63.5,    # lbf/ft³
  rho_air    = 0.002378,
  g          = 32.174,
  dt         = 0.01,     # you might want smaller dt for a small UAV
  CV_cutoff  = V_onset/np.sqrt(32.174*1.0),  # pick V_onset ≃20 knots → 20*1.688/√(32.174*1.0)
  f          = 0.015,
  eta_p      = 0.82,   # propeller efficiency
)


from TakeoffSimulation.motor import Motor  # wherever you defined Motor
motor = Motor()
# conversion factor from N to lbf
N_TO_LBF = 0.224808943
class UAVSimulator(SeaplaneSimulator):
    def __init__(self, params: SimulationParams, motor=None):
        super().__init__(params)
        # NACA 4414 airfoil parameters
        self.CL0 = 0.25        # Zero-lift coefficient
        self.CLa = 2 * np.pi   # Lift slope per radian (~6.283 rad⁻¹)
        self.CD0 = 0.008       # Profile drag coefficient at CL = 0
        self.k   = 0.045       # Induced drag factor (~1/(πeAR) for e~0.8, AR~6)

    def lift_drag(self, q, alpha):
        """Compute lift and drag for NACA 4414:

          CL = CL0 + CLa * alpha_rad
          CD = CD0 + k * CL^2
          L = q * S * CL
          D = q * S * CD
        alpha passed in degrees, convert to radians.
        """
        # convert alpha [deg] to radians
        alpha_rad = np.deg2rad(alpha)
        # lift and drag coefficients
        CL = self.CL0 + self.CLa * alpha_rad
        CD = self.CD0 + self.k * CL**2
        # forces
        L = q * self.p.S * CL
        D = q * self.p.S * CD
        return L, D
    def thrust(self, V, r):
        """Override to use Motor.thrust(throttle) instead of polynomial.

        V is ignored here (static test), r is ramp‑up 0–1.
        """
        # map ramp factor to throttle % (0–100)
        throttle_pct = float(np.clip(r * 100, 0, 100))
        # get thrust in Newtons from the motor map
        thrust_N = motor.thrust(throttle_pct, units='N')
        # convert to lbf for the simulator
        return thrust_N * N_TO_LBF
    
    def power(self,r):
        """Override to use Motor.power(throttle) instead of polynomial.

        V is ignored here (static test), r is ramp‑up 0–1.
        """
        # map ramp factor to throttle % (0–100)
        throttle_pct = float(np.clip(r * 100, 0, 100))
        # get power in Watts from the motor map
        power_W = motor.power(throttle_pct, units='W')
        # convert to lbf·ft/s for the simulator
        return power_W

sim = UAVSimulator(uav_params,motor)
df = sim.simulate()
df['Total_Resistance_lbf'] = df['Water_R_lbf'] + df['Hydro_R_lbf'] + df['Drag_lbf']

# Convert airspeed to knots
airspeed_knots = df['V_ft_s'] / 1.688
# Plot forces vs airspeed
fig, ax1 = plt.subplots()

# Plot forces on the left y‑axis
ax1.plot(airspeed_knots, df['Water_R_lbf'],         label='Water Resistance')
ax1.plot(airspeed_knots, df['Hydro_R_lbf'],         label='Hydroplaning Resistance')
ax1.plot(airspeed_knots, df['Drag_lbf'],            label='Aerodynamic Drag')
ax1.plot(airspeed_knots, df['Lift_lbf'],            label='Lift Force')
ax1.plot(airspeed_knots, df['Total_Resistance_lbf'],linestyle='--', label='Total Resistance')
ax1.set_xlabel('Airspeed (knots)')
ax1.set_ylabel('Force (lbf)')
ax1.grid(True)

# Create a second y‑axis on the right for power
ax2 = ax1.twinx()
ax2.plot(airspeed_knots, df['P_elec_lbf_ft_s'], color='black', label='Battery Power')
ax2.set_ylabel('Power (lbf·ft/s)')

# Merge legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Forces & Power vs Airspeed')
plt.tight_layout()
plt.show()
print(df['E_elec_lbf_ft'].values[-1], df['time_s'].values[-1])