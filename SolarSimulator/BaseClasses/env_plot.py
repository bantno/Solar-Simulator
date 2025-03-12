import numpy as np
import matplotlib.pyplot as plt

# Define simulation horizon and time indices
horizon = 1000
t_indices = np.arange(horizon)

# Create time series used by the environment provider
x = np.linspace(0, np.pi * 48, horizon)
whale_reward_series = np.sin(x)
solar_rate_series = np.clip(np.sin(x) * 4000, 0, 4000)

# Wind distribution parameters:
# Constant wind shape and a diurnally varying wind scale parameter.
wind_shape = np.full(horizon, 2.0)  # k parameter (constant)
wind_scale = 4.0 + 3.0 * np.sin(2 * np.pi * t_indices / 24)  # λ parameter

# Prepare the figure with three subplots
plt.figure(figsize=(12, 10))

# Solar Rate Series subplot (first subplot)
plt.subplot(3, 1, 1)
plt.plot(t_indices, solar_rate_series, label="Solar Rate")
plt.title("Solar Rate Series")
plt.xlabel("Time Step")
plt.ylabel("Solar Rate")
plt.legend()
plt.grid(True)

# Wind subplot (second subplot)
# Instead of plotting a single line, we plot 5000 samples per time step.
plt.subplot(3, 1, 2)
n_samples = 1000

# Vectorized sampling:
# Since wind_shape is constant (2.0), we sample from np.random.weibull with size=(horizon, n_samples)
# and then multiply each row (each time step) by its corresponding wind_scale.
wind_samples = np.random.weibull(2.0, size=(horizon, n_samples)) * wind_scale[:, None]

# Create an x-axis array that repeats each time step n_samples times
x_vals = np.repeat(t_indices, n_samples)
y_vals = wind_samples.flatten()

# Plot the wind samples with very low alpha for transparency (s=1 for very small dots)
plt.scatter(x_vals, y_vals, color='blue', alpha=0.01, s=1)
plt.title("Wind Speed Samples over Time")
plt.xlabel("Time Step")
plt.ylabel("Wind Speed")
plt.grid(True)

# Whale Reward Series subplot (third subplot)
plt.subplot(3, 1, 3)
plt.plot(t_indices, whale_reward_series, label="Whale Reward", color='orange')
plt.title("Whale Reward Series")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
