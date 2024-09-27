import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt

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

# Example usage
num_timesteps = 24  # 24 hours
num_simulations = 1000
num_plots = 100  # Number of individual simulations to plot

# Weibull parameters (you should adjust these based on your specific data)
shape = 7.0  # k parameter (shape)
scale = 9.0  # c parameter (scale)

windspeed_timeseries = generate_windspeed_timeseries(num_timesteps, num_simulations, shape, scale)

# Plotting
plt.figure(figsize=(12, 6))

# Plot individual simulations
for i in range(num_plots):
    plt.scatter(range(num_timesteps), windspeed_timeseries[i], alpha=0.3, color='gray')

# Plot mean wind speed
mean_windspeed = np.mean(windspeed_timeseries, axis=0)
plt.plot(range(num_timesteps), mean_windspeed, color='red', linewidth=2, label='Mean Wind Speed')

plt.title('Wind Speed Time Series - Weibull Distribution')
plt.xlabel('Time (hours)')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
