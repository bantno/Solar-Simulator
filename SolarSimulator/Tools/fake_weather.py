import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_alternating_wind_weather(
    start_time,
    end_time,
    freq="15min",
    high_wind_period=24,
    low_wind_period=24,
    high_wind_speed=40,
    low_wind_speed=5,
):
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
    total_hours = time_index.size // 4  # Convert 15-min intervals to hours
    wind_speeds = np.tile(
        np.concatenate(
            (
                np.full(high_wind_period * 4, high_wind_speed),
                np.full(low_wind_period * 4, low_wind_speed),
            )
        ),
        total_hours // (high_wind_period + low_wind_period) + 1,
    )[: time_index.size]

    # Generate wind direction (random for variety)
    wind_directions = np.random.uniform(0, 360, size=time_index.size)

    # Create square wave for solar radiation (day/night cycle, assume 8h sunlight)
    hours = time_index.hour
    shortwave_radiation = np.where(
        (hours >= 6) & (hours < 14), 1000, 0
    )  # 500 represents arbitrary sunlight intensity

    # Create DataFrame
    df = pd.DataFrame(
        {
            "wind_speed_10m": wind_speeds,
            "wind_direction_10m": wind_directions,
            "shortwave_radiation": shortwave_radiation,
        },
        index=time_index,
    )

    beta_alpha = np.where((hours >= 6) & (hours < 14), 10, 1)
    beta_beta = np.where((hours >= 6) & (hours < 14), 3.5, 10)
    k = [10] * len(time_index)
    scale = wind_speeds

    expected_data = pd.DataFrame(
        {
            "month": time_index.month,
            "day": time_index.day,
            "hour": time_index.hour,
            "minute": time_index.minute,
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "expected_solar_rad": shortwave_radiation,
            "weibull_k": k,
            "weibull_loc": [0] * len(time_index),
            "weibull_scale": scale,
            "expected_wind_speed": wind_speeds,
        }
    )

    return df, expected_data


def generate_low_wind_weather(start_time, end_time, freq="15min"):
    """
    Generate fake weather data with no wind and square wave solar radiation.
    """
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq, tz="UTC-06:00")
    hours = time_index.hour

    # No wind
    wind_speeds = np.ones(time_index.size)
    wind_directions = np.zeros(time_index.size)

    # Square wave solar radiation
    shortwave_radiation = np.where((hours >= 6) & (hours < 14), 1000, 0)

    df = pd.DataFrame(
        {
            "wind_speed_10m": wind_speeds,
            "wind_direction_10m": wind_directions,
            "shortwave_radiation": shortwave_radiation,
        },
        index=time_index,
    )

    beta_alpha = np.where((hours >= 6) & (hours < 14), 10, 1)
    beta_beta = np.where((hours >= 6) & (hours < 14), 3.5, 10)
    k = [10] * len(time_index)
    scale = wind_speeds

    expected_data = pd.DataFrame(
        {
            "month": time_index.month,
            "day": time_index.day,
            "hour": time_index.hour,
            "minute": time_index.minute,
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "expected_solar_rad": shortwave_radiation,
            "weibull_k": k,
            "weibull_loc": [0] * len(time_index),
            "weibull_scale": scale,
            "expected_wind_speed": wind_speeds,
        }
    )

    return df, expected_data


def generate_constant_wind_weather(start_time, end_time, freq="15min", wind_speed=20):
    """
    Generate fake weather data with constant wind and square wave solar radiation.
    """
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq, tz="UTC-06:00")
    hours = time_index.hour

    # Constant wind
    wind_speeds = np.full(time_index.size, wind_speed)
    wind_directions = np.random.uniform(0, 360, size=time_index.size)

    # Square wave solar radiation
    shortwave_radiation = np.where((hours >= 6) & (hours < 14), 1000, 0)

    df = pd.DataFrame(
        {
            "wind_speed_10m": wind_speeds,
            "wind_direction_10m": wind_directions,
            "shortwave_radiation": shortwave_radiation,
        },
        index=time_index,
    )

    beta_alpha = np.where((hours >= 6) & (hours < 14), 10, 1)
    beta_beta = np.where((hours >= 6) & (hours < 14), 3.5, 10)
    k = [10] * len(time_index)
    scale = wind_speeds

    expected_data = pd.DataFrame(
        {
            "month": time_index.month,
            "day": time_index.day,
            "hour": time_index.hour,
            "minute": time_index.minute,
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "expected_solar_rad": shortwave_radiation,
            "weibull_k": k,
            "weibull_loc": [0] * len(time_index),
            "weibull_scale": scale,
            "expected_wind_speed": wind_speeds,
        }
    )

    return df, expected_data


# Example usage
start = "2025-01-01 00:00:00"
end = "2025-01-05 00:00:00"
alt_wind_data, expected_alt_wind_data = generate_alternating_wind_weather(start, end)
alt_wind_data.to_pickle(r"Data\TEST_CASES\Wind\fake_weather_data_alternating.pkl")
expected_alt_wind_data.to_pickle(r"Data\TEST_CASES\Wind\expected_fake_weather_data_alternating.pkl")

constant_wind_data, expected_constant_wind_data = generate_constant_wind_weather(start, end)
constant_wind_data.to_pickle(r"Data\TEST_CASES\Wind\fake_weather_data_constant_wind.pkl")
expected_constant_wind_data.to_pickle(
    r"Data\TEST_CASES\Wind\expected_fake_weather_data_constant_wind.pkl"
)

low_wind_data, expected_low_wind_data = generate_low_wind_weather(start, end)
low_wind_data.to_pickle(r"Data\TEST_CASES\Wind\fake_weather_data_low_wind.pkl")
expected_low_wind_data.to_pickle(r"Data\TEST_CASES\Wind\expected_fake_weather_data_low_wind.pkl")
