import numpy as np
import matplotlib.pyplot as plt

# Import the MDP and solver classes.
from BaseClasses.mdp_base import DeterministicMDP
from BaseClasses.backward_induction_base import DeterministicMDPBackwardSolver

# ----------------------------
# Define the time series inputs
# ----------------------------

# Define a horizon (e.g., 24 time steps representing hours in a day)
T = 100

# Create a half sine wave for solar_rate_series (simulating daylight):
#   - The sine wave is defined over [0, π]. Outside this interval (night), the solar rate is 0.
t_day = np.linspace(0, np.pi, T)
# solar_rate_series = np.sin(t_day)
solar_rate_series = np.zeros(T)
# # For visualization, you could plot the solar_rate_series:
# plt.figure()
# plt.plot(t_day, solar_rate_series, marker='o')
# plt.title("Solar Rate Series (Half Sine Wave)")
# plt.xlabel("Time (radians)")
# plt.ylabel("Normalized Solar Rate")
# plt.show()

# Create a constant wind speed series.
constant_wind_speed = 5.0  # e.g., 5 m/s (or any appropriate units)
wind_speed_series = np.full(T, constant_wind_speed)

# Create a triangle wave for whale observation (reward) series:
#   - Peaks at noon (midpoint of the day) and is 0 at the start and end.
peak_value = 1.0  # Maximum whale reward
first_half = np.linspace(0, peak_value, T//2, endpoint=False)
second_half = np.linspace(peak_value, 0, T - T//2)
# whale_reward_series = np.concatenate([first_half, second_half])
whale_reward_series = np.full_like(solar_rate_series,1.)

# ----------------------------
# Set up other MDP parameters
# ----------------------------

battery_capacity_wh = 200*60*60*5/3600        # Battery capacity in watt-hours.
idle_power = 0                  # Power consumption (when moored) per time step.
cruise_power = 200               # Power consumption while flying.
takeoff_power = 200              # Additional power consumption for takeoff.
failure_penalty = 10           # Penalty if the vehicle becomes broken.
delta_t = 60                      # Duration of each time step (in hours).
gamma = 1.0                      # Discount factor.
transition_model_name = "nofail" # Name of the transition model (assumed to work with the factory).
soc_increment = 2.0             # SOC (State of Charge) increments.

# ----------------------------
# Create the MDP instance
# ----------------------------

mdp = DeterministicMDP(battery_capacity_wh, idle_power, cruise_power, takeoff_power,
                       solar_rate_series, wind_speed_series, whale_reward_series,
                       failure_penalty, delta_t, gamma, transition_model_name, soc_increment)

# ----------------------------
# Set up and run the backward induction solver
# ----------------------------

horizon = T  # Set the planning horizon to the number of time steps.
solver = DeterministicMDPBackwardSolver(mdp, horizon)

# Run the backward induction solver.
solver.solve()

# In a full implementation, the solver might store or return the computed value table.
# For this usage example, we assume that after calling solve(), the value table is available as:
#    solver.future_value_table
# (If not, you could modify the solver to store or return the computed table.)
try:
    value_table = solver.future_value_table
    print("Computed Value Table:")
    print(value_table[:])
except AttributeError:
    print("The solver did not store the value table. Modify the 'solve' method to return or store the table if needed.")
