import pandas as pd
import numpy as np

def get_whale_observation_probabilities(time_index: pd.DatetimeIndex):
    # Define the time intervals and probabilities
    time_intervals = [
        ("0600", "0800", 0.082),
        ("0800", "1000", 0.098),
        ("1000", "1200", 0.095),
        ("1200", "1400", 0.217),
        ("1400", "1600", 0.215),
        ("1600", "2000", 0.278)
    ]
    
    # Function to get the whale observation probability for a given time
    def get_probability_for_time(time):
        time_str = time.strftime("%H%M")
        for interval_start, interval_end, prob in time_intervals:
            if interval_start <= time_str < interval_end:
                return prob
        return 0.0  # Return 0.0 if the time doesn't fall into any interval

    # Apply the function to the DatetimeIndex and return the probabilities
    probabilities = np.array(time_index.map(get_probability_for_time))
    return probabilities

# Example usage with a sample DatetimeIndex
time_index = pd.date_range("2024-01-01 00:00", "2024-06-01 23:45", freq="15min", tz="UTC-06:00")
probabilities = get_whale_observation_probabilities(time_index)

# Displaying the first few probabilities
print(probabilities)
