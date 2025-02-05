import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_fake_weather(start_time, end_time, freq='15min', high_wind_period=24, low_wind_period=24, high_wind_speed=40, low_wind_speed=5):
    """
    Generate fake weather data with alternating high and low wind speeds and square wave solar radiation.
    
    Parameters:
        start_time (str): Start datetime (e.g., '2025-01-01 00:00:00')
        end_time (str): End datetime (e.g., '2025-01-02 00:00:00')
        freq (str): Time interval (default is '15T' for 15 minutes)
        high_wind_period (int): Number of hours for high wind period
        low_wind_period (int): Number of hours for low wind period
        high_wind_speed (float): Wind speed during high wind period
        low_wind_speed (float): Wind speed during low wind period
    """
    
    # Generate timestamp index
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq, tz="UTC-06:00")
    
    # Create alternating wind speed pattern
    total_hours = (time_index.size // 4)  # Convert 15-min intervals to hours
    wind_speeds = np.tile(
        np.concatenate((np.full(high_wind_period * 4, high_wind_speed), np.full(low_wind_period * 4, low_wind_speed))),
        total_hours // (high_wind_period + low_wind_period) + 1
    )[:time_index.size]
    
    # Generate wind direction (random for variety)
    wind_directions = np.random.uniform(0, 360, size=time_index.size)
    
    # Create square wave for solar radiation (day/night cycle, assume 8h sunlight)
    hours = time_index.hour
    shortwave_radiation = np.where((hours >= 6) & (hours < 14), 1000, 0)  # 500 represents arbitrary sunlight intensity
    
    # Create DataFrame
    df = pd.DataFrame({
        'wind_speed_10m': wind_speeds,
        'wind_direction_10m': wind_directions,
        'shortwave_radiation': shortwave_radiation
    }, index=time_index)


    beta_alpha = np.where((hours >= 6) & (hours < 14), 10, 1)
    beta_beta = np.where((hours >= 6) & (hours < 14), 3.5, 10)
    k = [10]*len(time_index)
    scale = wind_speeds

    expected_data = pd.DataFrame({
    "month": time_index.month,
    "day": time_index.day,
    "hour": time_index.hour,
    "minute": time_index.minute,
    "beta_alpha": beta_alpha,
    "beta_beta": beta_beta,
    "expected_solar_rad": shortwave_radiation,
    "weibull_k": k,
    "weibull_loc": [0]*len(time_index),
    "weibull_scale": scale,
    "expected_wind_speed": wind_speeds,
})
    
    return df,expected_data

# Example usage
data,expected_data = generate_fake_weather('2025-01-01 00:00:00', '2025-01-04 00:00:00')
data.to_pickle(r'Data\TEST_CASES\Wind\fake_weather_data.pkl')
expected_data.to_pickle(r'Data\TEST_CASES\Wind\expected_fake_weather_data.pkl')

# Plot
fig, ax1 = plt.subplots(figsize=(12, 5))

# Wind speed plot
ax1.plot(data.index, data['wind_speed_10m'], 'b-', label='Wind Speed (m/s)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Wind Speed (m/s)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Solar radiation plot
ax2 = ax1.twinx()
ax2.plot(data.index, data['shortwave_radiation'], 'r-', label='Solar Radiation (W/m²)')
ax2.set_ylabel('Solar Radiation (W/m²)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
plt.title('Wind Speed and Solar Radiation Over Time')
# plt.show()
