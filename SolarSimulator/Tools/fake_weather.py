import pandas as pd
import numpy as np

def generate_fake_weather(start_time, end_time, freq='15min', high_wind_period=24, low_wind_period=24, high_wind_speed=15, low_wind_speed=5):
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
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    
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
    shortwave_radiation = np.where((hours >= 6) & (hours < 14), 500, 0)  # 500 represents arbitrary sunlight intensity
    
    # Create DataFrame
    df = pd.DataFrame({
        'wind_speed_10m': wind_speeds,
        'wind_direction_10m': wind_directions,
        'shortwave_radiation': shortwave_radiation
    }, index=time_index)
    
    return df

# Example usage
data = generate_fake_weather('2025-01-01 00:00:00', '2025-01-02 00:00:00')
print(data.head())
