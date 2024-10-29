import pandas as pd
import numpy as np

import pytz
import pygrib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy.stats import weibull_min
from scipy.special import gamma

from tqdm import tqdm

class WindProcessor:
    def __init__(self, grib_file_path):
        self.grib_file_path = grib_file_path
        self.utc_tz = pytz.utc
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.data_dict = {}
    
    def extract_wind_data(self):
        # Open the GRIB file
        grbs = pygrib.open(self.grib_file_path)
        
        # Dictionary to hold U component data keyed by (lat, lon, datetime)
        u_data = {}
        
        # List to hold rows for DataFrame creation
        data_rows = []

        # Loop through all messages in the GRIB file
        for grb in tqdm(grbs.select()):
            # Extract U component of wind
            if grb.parameterName == '10 metre U wind component':
                lats, lons = grb.latlons()
                values_u = grb.values
                
                # Extract the date and time from the GRIB message
                date = grb.validityDate
                hour = grb.validityTime
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)

                # Loop through latitude and longitude arrays
                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        u_component = values_u[i, j]
                        key = (lat_lon_pair[0], lat_lon_pair[1], valid_date)

                        # Store U component in dictionary
                        u_data[key] = u_component

            # Extract V component of wind
            elif grb.parameterName == '10 metre V wind component':
                lats, lons = grb.latlons()
                values_v = grb.values
                
                # Extract the date and time from the GRIB message
                date = grb.validityDate
                hour = grb.validityTime
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)

                # Loop through latitude and longitude arrays
                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        v_component = values_v[i, j]
                        key = (lat_lon_pair[0], lat_lon_pair[1], valid_date)

                        # Check if U component exists for the given key
                        if key in u_data:
                            u_component = u_data[key]
                            # Calculate magnitude
                            magnitude = np.sqrt(u_component**2 + v_component**2)
                            data_rows.append((lat_lon_pair[0], lat_lon_pair[1], valid_date, u_component, v_component, magnitude))

        grbs.close()

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data_rows, columns=['latitude', 'longitude', 'datetime', 'u_component', 'v_component', 'magnitude'])
        
        # Extract month, day, and hour from the datetime column
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour

        # Set the index to be a MultiIndex of lat-lon and datetime
        df.set_index(['latitude', 'longitude', 'datetime'], inplace=True)
        df.sort_index()
        
        return df



    
    def calculate_wind_magnitude(self, u_data, v_data):
        wind_magnitude_data = {}
        
        for lat_lon_pair in u_data.keys():
            if lat_lon_pair in v_data:  # Ensure both U and V components exist
                for date_time in u_data[lat_lon_pair].keys():
                    if date_time in v_data[lat_lon_pair]:
                        u_values = np.array(u_data[lat_lon_pair][date_time])
                        v_values = np.array(v_data[lat_lon_pair][date_time])
                        wind_magnitudes = np.sqrt(u_values**2 + v_values**2)  # Calculate wind magnitude
                        
                        if lat_lon_pair not in wind_magnitude_data:
                            wind_magnitude_data[lat_lon_pair] = {}
                        
                        wind_magnitude_data[lat_lon_pair][date_time] = wind_magnitudes
        
        return wind_magnitude_data
    
    def fit_weibull_distributions(self, wind_magnitude_data):
        weibull_ev = {}
        
        # Group by latitude, longitude, month, day, and hour
        grouped = wind_magnitude_data.groupby(['latitude', 'longitude', 'month', 'day', 'hour'])

        for (lat, lon, month, day, hour), group in tqdm(grouped):
            values = group['magnitude'].values

            if len(values) > 0:  # Ensure there is data   
                # Fit the Weibull distribution
                try:
                    shape, loc, scale = weibull_min.fit(values, floc=0)
                    expected_value = scale * gamma(1 + 1 / shape)  # Calculate expected value
                    weibull_ev[(lat, lon, month, day, hour)] = expected_value  # Store the expected value
                except Exception as e:
                    weibull_ev[(lat, lon, month, day, hour)] = np.nan  # Store the expected value
                    print(f"Fitting failed for {lat, lon, month, day, hour}: {e}")
        
        # Create a new DataFrame to hold expected values indexed by latitude, longitude, month, day, and hour
        expected_values_df = pd.DataFrame.from_dict(weibull_ev, orient='index', columns=['expected_value'])
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
        data_df = self.extract_wind_data()
        weibull_ev = self.fit_weibull_distributions(data_df)
        weibull_resampled = self.resample_to_timestep(weibull_ev)
        return weibull_resampled

if __name__ == "__main__":
    grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\GRIB\January\wind_data.grib'
    processor = WindProcessor(grib_file_path)
    # data = processor.get_wind_magnitude_for_year(2022)
    # data.to_pickle("wind_mag_2022.pkl")
    # wind_magnitude_data = processor.process_grib_file()
    # wind_magnitude_data.to_pickle("wind_ev_data.pkl")
    test = pd.read_pickle("wind_ev_data.pkl")
    # Example lat-lon pair for plotting
    # lat_lon_pair = (29.50, -85.25)  # Replace with a specific latitude and longitude from your data
    # processor.plot_wind_magnitude(wind_magnitude_data, lat_lon_pair)
    # weibull_params.to_pickle("wind_mag_dist.pkl")
    # df = pd.read_pickle('wind_mag_dist.pkl')
    # processor.plot_expected_wind(lat_lon_pair)
    print("Done!")