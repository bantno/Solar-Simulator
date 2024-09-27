import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

def generate_irradiance_timeseries(num_timesteps, num_samples, alpha, beta, max_irradiance):
    timeseries = np.zeros((num_samples, num_timesteps))
    
    for t in range(num_timesteps):
        samples = beta_dist.rvs(alpha[t], beta[t], size=num_samples)
        timeseries[:, t] = samples * max_irradiance[t]
    
    return timeseries

# Example usage
num_timesteps = 24  # 24 hours
num_simulations = 10000  # Reduced for clearer visualization
num_plots = 100  # Number of individual simulations to plot

# Example parameters (you should adjust these based on your specific data)
alpha = [2 + np.sin(t/24 * 2 * np.pi) for t in range(num_timesteps)]  # Varies throughout the day
beta = [1 + 0.5 * np.cos(t/24 * 2 * np.pi) for t in range(num_timesteps)]  # Varies throughout the day
max_irradiance = [1000 * np.sin(t/24 * np.pi) if 6 <= t < 18 else 0 for t in range(num_timesteps)]  # Simple day/night cycle

irradiance_timeseries = generate_irradiance_timeseries(num_timesteps, num_simulations, alpha, beta, max_irradiance)

# Plotting
plt.figure(figsize=(12, 6))

# Plot individual simulations
for i in range(num_plots):
    plt.scatter(range(num_timesteps), irradiance_timeseries[i], alpha=0.6, color='gray')

# Plot mean irradiance
mean_irradiance = np.mean(irradiance_timeseries, axis=0)
plt.plot(range(num_timesteps), mean_irradiance, color='blue', linewidth=2, label='Mean Irradiance')

#Plot max irradiance
plt.plot(range(num_timesteps), max_irradiance, color='red', linewidth=2, label='Max Irradiance')

plt.title('Solar Irradiance Time Series - Monte Carlo Simulation')
plt.xlabel('Time (hours)')
plt.ylabel('Irradiance (W/m²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
