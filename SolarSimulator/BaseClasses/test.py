import pandas as pd
import numpy as np

# Sample DataFrame with MultiIndex as described
data = {
    'expected_value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.401479, 0.344650, 0.260086, 0.151646, 0.046921]
}
index_tuples = [
    (27.75, -85.25, 1, 1, 2),  # 2 AM
    (27.75, -85.25, 1, 1, 3),  # 3 AM
    (27.75, -85.25, 1, 1, 4),  # 4 AM
    (27.75, -85.25, 1, 1, 5),  # 5 AM
    (27.75, -85.25, 1, 1, 6),  # 6 AM
    (29.5, -83.0, 1, 31, 14),  # 2 PM
    (29.5, -83.0, 1, 31, 15),  # 3 PM
    (29.5, -83.0, 1, 31, 16),  # 4 PM
    (29.5, -83.0, 1, 31, 17),  # 5 PM
    (29.5, -83.0, 1, 31, 18),  # 6 PM
]

# Create a DataFrame
expected_values_df = pd.DataFrame(data, index=index_tuples)

# Create a MultiIndex from the current index
multi_index = pd.MultiIndex.from_tuples(expected_values_df.index, names=['latitude', 'longitude', 'month', 'day', 'hour'])
expected_values_df.index = multi_index

# Function to resample to a given time step in minutes
def resample_to_timestep(df, time_step_minutes):
    # Reset index to work with columns
    df = df.reset_index()

    # Create a new MultiIndex that includes minutes
    new_index_tuples = []

    for _, row in df.iterrows():
        for minute in range(0, 60, time_step_minutes):
            new_index_tuples.append((row['latitude'], row['longitude'], row['month'], row['day'], row['hour'], minute))

    # Create a new MultiIndex
    new_multi_index = pd.MultiIndex.from_tuples(new_index_tuples, names=['latitude', 'longitude', 'month', 'day', 'hour', 'minute'])

    # Create a new DataFrame with the new MultiIndex
    new_df = pd.DataFrame(index=new_multi_index)
    new_df['expected_value'] = np.nan  # Initialize expected_value column

    # Loop through each latitude, longitude, month, and day
    for (lat, lon, month, day), group in df.groupby(['latitude', 'longitude', 'month', 'day']):
        # Get the expected values for each hour
        hour_values = group.set_index('hour')['expected_value']
        
        # Interpolate between hours
        for hour in range(hour_values.index.min(), hour_values.index.max()):
            if hour in hour_values.index and (hour + 1) in hour_values.index:
                # Current and next hour values
                current_value = hour_values[hour]
                next_value = hour_values[hour + 1]

                # Calculate the interpolated values for the new DataFrame
                for minute in range(0, 60, time_step_minutes):
                    # Calculate the proportion of the minute within the hour
                    proportion = minute / 60
                    interpolated_value = (1 - proportion) * current_value + proportion * next_value
                    new_df.loc[(lat, lon, month, day, hour, minute), 'expected_value'] = interpolated_value

    return new_df

# Resample to a 15-minute time step
resampled_df = resample_to_timestep(expected_values_df, time_step_minutes=15)

# Display the resampled DataFrame
print(resampled_df)
