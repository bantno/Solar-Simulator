import pygrib
import pandas as pd
import numpy as np
import pytz
from scipy.stats import beta
from tqdm import tqdm

class SolarRadiationProcessor:
    def __init__(self, grib_file_path):
        self.grib_file_path = grib_file_path
        self.utc_tz = pytz.utc
        self.eastern_tz = pytz.timezone('US/Eastern')

    def extract_ssrd_data(self):
        # Open the GRIB file
        grbs = pygrib.open(self.grib_file_path)

        # Initialize a list to hold data for DataFrame creation
        data_rows = []

        # Loop through all messages in the GRIB file
        for grb in tqdm(grbs.select()):
            if grb.parameterName == 'Surface solar radiation downwards' :
                lats, lons = grb.latlons()
                values = grb.values
                
                date = grb.validityDate
                hour = grb.validityTime
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)

                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        data_rows.append((lat_lon_pair[0], lat_lon_pair[1], valid_date, values[i, j] / 3600))  # Convert to W/m^2

        grbs.close()

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data_rows, columns=['latitude', 'longitude', 'datetime', 'value'])
        
        # Extract month, day, and hour from the datetime column
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour

        # Set the index to be a MultiIndex of lat-lon and datetime
        df.set_index(['latitude', 'longitude', 'datetime'], inplace=True)
        df.sort_index()
        return df

    def normalize_ssrd_data(self, df):
        normalized_values = df['value'] / 1367  # Normalize using the normalizing factor
        df['normalized_value'] = normalized_values
        return df

    def fit_beta_distributions(self, df, epsilon=1e-6):
        expected_values = {}

        # Group by latitude, longitude, month, day, and hour
        grouped = df.groupby(['latitude', 'longitude', 'month', 'day', 'hour'])

        for (lat, lon, month, day, hour), group in tqdm(grouped):
            values = group['normalized_value'].values
            
            if len(values) > 0:
                if np.any(values == 0) or np.any(values == 1) or np.ptp(values) == 0:
                    expected_values[(lat, lon, month, day, hour)] = 0
                else:
                    clipped_values = np.clip(values, epsilon, 1 - epsilon)
                    try:
                        a, b, loc, scale = beta.fit(clipped_values, floc=0, fscale=1)
                        expected_value = a / (a + b)
                        expected_values[(lat, lon, month, day, hour)] = expected_value  # Store the expected value
                    except Exception as e:
                        expected_values[(lat, lon, month, day, hour)] = np.nan
                        print(f"Fitting failed for {lat, lon, month, day, hour}: {e}")

        # Create a new DataFrame to hold expected values indexed by latitude, longitude, month, day, and hour
        expected_values_df = pd.DataFrame.from_dict(expected_values, orient='index', columns=['expected_value'])
        multi_index = pd.MultiIndex.from_tuples(expected_values_df.index, names=['latitude', 'longitude', 'month', 'day', 'hour'])
        expected_values_df.index = multi_index

        return expected_values_df

    def resample_to_timestep(self, df, time_step_minutes=15):
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
        for (lat, lon, month, day), group in tqdm(df.groupby(['latitude', 'longitude', 'month', 'day'])):
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

    

    def process_grib_file(self):
        df = self.extract_ssrd_data()  # Extract data and return DataFrame
        normalized_df = self.normalize_ssrd_data(df)  # Normalize data
        beta_fitted_df = self.fit_beta_distributions(normalized_df)  # Fit beta distributions to the data

        # Resample to 15-minute intervals
        # solar_ev_resampled = self.resample_to_timestep(beta_fitted_df)
        return beta_fitted_df

if __name__ == "__main__":
    grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\janfeb.grib'
    processor = SolarRadiationProcessor(grib_file_path)

    # Process and resample
    solar_ev_resampled = processor.process_grib_file()
    solar_ev_resampled.to_pickle("solar_ev.pkl")
    # test = pd.read_pickle(r"solar_ev_resampled.pkl")

    print("Done!")
