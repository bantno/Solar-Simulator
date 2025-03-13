import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_environment(env_provider, time_steps, n=1):
    """
    Plots the environment data (solar, wind, whale observations) provided by the given environment provider.

    Parameters:
    - env_provider: instance of AbstractEnvironmentProvider (e.g., StochasticWindEnvironmentProvider)
    - time_steps: iterable of time indices to sample from
    - n: number of samples to take at each time step (default is 1)
    """
    # Prepare arrays to store mean values for each observation.
    solar_vals = []
    wind_vals = []
    whale_vals = []

    # Loop over time steps and sample data.
    for t in time_steps:
        # We take the first sample if multiple samples are generated.
        solar = env_provider.sample_sunlight(t, n)[0]
        wind = env_provider.sample_wind_speed(t, n)[0]
        whale = env_provider.sample_whale_observation(t, n)[0]

        solar_vals.append(solar)
        wind_vals.append(wind)
        whale_vals.append(whale)

    # Create the subplots for solar, wind, and whale observations.
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_steps, solar_vals, marker='o')
    plt.title('Solar Energy over Time')
    plt.ylabel('Solar Rate')

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, wind_vals, marker='o')
    plt.title('Wind Speed over Time')
    plt.ylabel('Wind Speed')

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, whale_vals, marker='o')
    plt.title('Whale Observations over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Whale Observation')

    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == '__main__':
    # Instantiate the stochastic environment provider.
    from environment_provider_base import StochasticWindEnvironmentProvider
    horizon = 100
    battery_capacity_wh = 200 * 60 * 60 * 10 / 3600
    idle_power = 0
    cruise_power = 200
    takeoff_power = 200
    failure_penalty = 15
    delta_t = 15
    gamma = 1.0
    transition_model_name = "moderate"
    soc_increment = 1.0


    # solar_rate_series = np.full(horizon, 4000)
    wind_series = np.full(horizon, 5.0)
    x = np.linspace(0, np.pi*10, horizon)
    whale_reward_series = np.sin(x)
    solar_rate_series_fake = np.clip(np.sin(x)*4000,0,4000)
    t_indices = np.arange(horizon)

    data = pd.read_pickle(rf"Data\EXPECTED_DATA\data_expected_lat0_lon-90_15min.pkl")
    # wind_shape = np.full(horizon, 2.0)  # Constant shape parameter
    wind_shape = data['weibull_k'].values[:horizon]
    # wind_scale = 4.0 + 3.0 * np.sin(2 * np.pi * t_indices / 24)  # Scale varies with time
    wind_scale = data['weibull_scale'].values[:horizon]
    wind_distributions = np.column_stack((wind_shape, wind_scale))
    solar_rate_series = data['expected_solar_rad'].values[:horizon]*0.1*delta_t*60

    # ----- Instantiate the custom environment provider -----
    env_provider = StochasticWindEnvironmentProvider(
        solar_rate_series=solar_rate_series_fake,
        wind_distributions=wind_distributions,
        whale_reward_series=whale_reward_series,
        delta_t=delta_t
    )
    # Define a range of time steps.
    time_steps = np.arange(horizon)

    # Plot the environment.
    plot_environment(env_provider, time_steps, n=1)
