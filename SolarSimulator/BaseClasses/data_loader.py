# data_loader.py
import os
import re
import pandas as pd

def load_actual_weather_data(dt, directory, lat=None, lon=None, index=None):
    """
    Load actual weather data from a pickle file matching the given timestep and location.
    
    Parameters:
        dt: Timestep in minutes.
        directory: Directory containing weather data files.
        lat: Optional latitude.
        lon: Optional longitude.
        index: Optional file index.
        
    Returns:
        DataFrame with actual weather data.
    """
    if index is None:
        if lat is not None and lon is not None:
            pattern = rf"data_lat{lat}_lon{lon}_{dt}min(_\d+)?\.pkl$"
        else:
            pattern = rf"data_{dt}min(_\d+)?\.pkl$"
    else:
        if lat is not None and lon is not None:
            pattern = rf"data_lat{lat}_lon{lon}_{dt}min_{index}\.pkl$"
        else:
            pattern = rf"data_{dt}min_{index}\.pkl$"
    
    actual_file = None
    for file in os.listdir(directory):
        if re.search(pattern, file):
            actual_file = os.path.join(directory, file)
            break

    if actual_file:
        return pd.read_pickle(actual_file)
    else:
        raise FileNotFoundError(f"No file matching pattern {pattern} found in {directory}")

def load_expected_weather_data(dt, lat, lon, directory):
    """
    Load expected weather data from a pickle file for a given timestep and location.
    
    Parameters:
        dt: Timestep in minutes.
        lat: Latitude.
        lon: Longitude.
        directory: Directory containing expected data.
        
    Returns:
        DataFrame with expected weather data.
    """
    pattern = rf"data_expected_lat{lat}_lon{lon}_{dt}min\.pkl$"
    expected_file = None
    for file in os.listdir(directory):
        if re.search(pattern, file):
            expected_file = os.path.join(directory, file)
            break
    if expected_file:
        return pd.read_pickle(expected_file)
    else:
        raise FileNotFoundError(f"No expected weather data file matching pattern {pattern} found in {directory}")

def get_whale_observation_probabilities(time_index, solar_radiation, threshold=0):
    """
    Compute whale observation probabilities based on solar radiation and time.
    
    Parameters:
        time_index: DatetimeIndex for simulation times.
        solar_radiation: Array of solar radiation values.
        threshold: Minimum radiation required for observation.
    
    Returns:
        Numpy array of probabilities.
    """
    import numpy as np
    # Simplified example: if solar radiation is below threshold, probability is 0.
    probabilities = (solar_radiation > threshold).astype(float)
    return probabilities
