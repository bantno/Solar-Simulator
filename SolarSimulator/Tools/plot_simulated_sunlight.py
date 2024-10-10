import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def time_of_day_func(stage, timestep):
    """
    Compute the time of day as a sinusoidal function based on the current stage and timestep.
    
    Args:
        stage (int): The current stage of the simulation, which is an integer representing the progression of time.
        timestep (int): The time interval in minutes between each stage.
    
    Returns:
        float: A value between 0 and 1 representing the time of day, where 0 corresponds to midnight and 1
            corresponds to the end of the day.
    """
    daily_stages = 24*60/timestep
    normalized_stage = (np.mod(stage, daily_stages) / daily_stages)
    factor = np.sin(np.pi * normalized_stage)
    if factor < 0.6:
        factor = 0
    return max(0, factor)

def expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=80):
    """
    Calculates the expected solar power output for a given stage.
    
    Parameters:
        irradiance_mean (float): Mean solar irradiance (in W/m^2) at the given stage.
        cloud_prob (float): Probability of cloudiness at the stage.
        time_of_day_factor (float): A factor (0 to 1) representing the intensity of sunlight for the time of day.
        max_solar_power (float): Maximum power output of the solar system in W (default is 80 W).
    
    Returns:
        float: Expected solar power output in W.
    """
    expected_irradiance = irradiance_mean * time_of_day_factor
    expected_power_clear = min(max_solar_power, expected_irradiance / 1000 * max_solar_power)
    expected_power = expected_power_clear * (1 - 0.5 * cloud_prob)
    return expected_power

def convert_stage_to_time(stage, timestep):
    """
    Convert a stage to actual time in hours and minutes.
    
    Args:
        stage (int): The current stage of the simulation.
        timestep (int): The time interval in minutes between each stage.
    
    Returns:
        str: Time in the format "HH:MM".
    """
    total_minutes = stage * timestep
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{int(hours):02}:{int(minutes):02}"

# Parameters
num_steps = 288         # Number of steps to plot
timestep = 10            # Timestep in minutes
irradiance_mean = 800    # Mean irradiance in W/m^2
cloud_prob = 0.3         # Probability of cloudiness

# Generate stages and compute the expected solar power for each
stages = np.arange(num_steps)
time_of_day_values = [time_of_day_func(stage, timestep) for stage in stages]
expected_solar_power_values = [
    expected_solar_power(irradiance_mean, cloud_prob, tod_factor) for tod_factor in time_of_day_values
]

# Convert stages to time
times = [convert_stage_to_time(stage, timestep) for stage in stages]

# Select every 12th label for the x-axis
x_labels = [times[i] if i % 2 == 0 else '' for i in range(len(times))]

# Plot the expected solar power values
plt.figure(figsize=(12, 6))
plt.plot(times, expected_solar_power_values, label='Expected Solar Power (W)')
plt.title('Expected Solar Power Over Time')
plt.xlabel('Time (HH:MM)')
plt.ylabel('Expected Solar Power (W)')
plt.xticks(np.arange(len(times)), x_labels, rotation=45)

# Set grid with custom spacing for the x-axis gridlines
plt.grid(True, which='both', axis='both')  # Enable gridlines for both x and y axes
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Increase spacing of the x gridlines

plt.legend()
plt.tight_layout()
# plt.show()

# Generate a timestamp and create a filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"Figures/simulated_solar_{timestamp}.png"

# Save the figure to the 'Figures' folder
plt.savefig(filename)
