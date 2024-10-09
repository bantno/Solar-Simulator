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
        
        # Initialize placeholders for U and V wind components
        u_data = {}
        v_data = {}

        # Loop through all messages in the GRIB file
        for grb in tqdm(grbs.select()):
            # Extract U component of wind
            if grb.parameterName == '10 metre U wind component' and grb.validDate.month == 1:
                lats, lons = grb.latlons()
                values_u = grb.values
                
                # Extract the date and time from the GRIB message
                date = grb.validityDate
                hour = grb.validityTime
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)
                day_month_hour = (valid_date.month, valid_date.day, valid_date.hour)

                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        
                        if lat_lon_pair not in u_data:
                            u_data[lat_lon_pair] = {}
                        
                        if day_month_hour not in u_data[lat_lon_pair]:
                            u_data[lat_lon_pair][day_month_hour] = []
                        
                        u_data[lat_lon_pair][day_month_hour].append(values_u[i, j])
            
            # Extract V component of wind
            if grb.parameterName == '10 metre V wind component' and grb.validDate.month == 1:
                lats, lons = grb.latlons()
                values_v = grb.values
                
                date = grb.validityDate
                hour = grb.validityTime
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)
                day_month_hour = (valid_date.month, valid_date.day, valid_date.hour)

                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        
                        if lat_lon_pair not in v_data:
                            v_data[lat_lon_pair] = {}
                        
                        if day_month_hour not in v_data[lat_lon_pair]:
                            v_data[lat_lon_pair][day_month_hour] = []
                        
                        v_data[lat_lon_pair][day_month_hour].append(values_v[i, j])
        
        grbs.close()
        return u_data, v_data
    
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
        weibull_params = {}
        
        for lat_lon_pair, datetime_dict in tqdm(wind_magnitude_data.items()):
            for date_time, values in datetime_dict.items():
                if len(values) > 0:  # Ensure there is data
                    values = np.array(values)
                    
                    # Fit the Weibull distribution
                    try:
                        shape, loc, scale = weibull_min.fit(values, floc=0)
                        expected_value = scale * gamma(1 + 1 / shape)  # Calculate expected value
                        
                        if lat_lon_pair not in weibull_params:
                            weibull_params[lat_lon_pair] = {}
                        
                        weibull_params[lat_lon_pair][date_time] = (shape, scale, expected_value)  # Store the shape, scale, and expected value
                    except Exception as e:
                        weibull_params[lat_lon_pair][date_time] = (np.nan, np.nan, np.nan)  # Store NaN for all
                        print(f"Fitting failed for {lat_lon_pair} at {date_time}: {e}")
        
        weibull_df = pd.DataFrame.from_dict(weibull_params)
        return weibull_df
    
    def plot_wind_magnitude(self, wind_magnitude_data, lat_lon_pair, num_days=3):
        # Extract data for the first num_days in January
        day_limit = (1, num_days)  # From day 1 to the number of days specified
        plot_data = {date_time: values for date_time, values in wind_magnitude_data[lat_lon_pair].items()
                     if 1 <= date_time[1] <= num_days}  # Filter by the first num_days
        
        raw_expected_data = pd.read_pickle('wind_mag_dist.pkl')[lat_lon_pair]
        expected_data = {date_time: values[2] for date_time, values in raw_expected_data.items()
                     if 1 <= date_time[1] <= num_days}  # Filter by the first num_days

        if not expected_data:
            print(f"No data available for {num_days} days.")
            return

        if not plot_data:
            print(f"No data available for {num_days} days.")
            return
        
        # Create a sorted list of hours and their corresponding values
        sorted_data = sorted(plot_data.items(), key=lambda x: (x[0][1], x[0][2]))  # Sort by day and hour
        data_times = [f"{date_time[1]:02}-{date_time[2]:02}" for date_time, _ in sorted_data]  # "day-hour" format
        magnitudes = [np.mean(values) for _, values in sorted_data]  # Average magnitudes for each hour

        sorted_expected_data = sorted(expected_data.items(), key=lambda x: (x[0][1], x[0][2]))  # Sort by day and hour
        expected_times = [f"{date_time[1]:02}-{date_time[2]:02}" for date_time, _ in sorted_data]  # "day-hour" format
        expected_magnitudes = [np.mean(values) for _, values in sorted_expected_data]  # Average magnitudes for each hour
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data_times, magnitudes, marker='o', color="r", label = "Average Value")
        plt.plot(expected_times, expected_magnitudes, marker='o', color="b", label = "Expected Value")
        plt.title(f'Wind Magnitude for Latitude {lat_lon_pair[0]}, Longitude {lat_lon_pair[1]} over {num_days} Days')
        plt.xlabel('Time (Day-Hour)')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=24)) 
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Wind Magnitude (m/s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_expected_wind(self, lat_lon_pair, num_days=3):
        # Extract data for the first num_days in January
        day_limit = (1, num_days)  # From day 1 to the number of days specified
        raw_expected_data = pd.read_pickle('wind_mag_dist.pkl')[lat_lon_pair]
        expected_data = {date_time: values[2] for date_time, values in raw_expected_data.items()
                     if 1 <= date_time[1] <= num_days}  # Filter by the first num_days

        if not expected_data:
            print(f"No data available for {num_days} days.")
            return
        
        # Create a sorted list of hours and their corresponding values
        sorted_data = sorted(expected_data.items(), key=lambda x: (x[0][1], x[0][2]))  # Sort by day and hour
        times = [f"{date_time[1]:02}-{date_time[2]:02}" for date_time, _ in sorted_data]  # "day-hour" format
        magnitudes = [np.mean(values) for _, values in sorted_data]  # Average magnitudes for each hour
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(times, magnitudes, marker='o')
        plt.title(f'Wind Magnitude for Latitude {lat_lon_pair[0]}, Longitude {lat_lon_pair[1]} over {num_days} Days')
        plt.xlabel('Time (Day-Hour)')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=24)) 
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Wind Magnitude (m/s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def process_grib_file(self):
        u_data, v_data = self.extract_wind_data()
        wind_magnitude_data = self.calculate_wind_magnitude(u_data, v_data)
        weibull_params = self.fit_weibull_distributions(wind_magnitude_data)
        return wind_magnitude_data,weibull_params

if __name__ == "__main__":
    grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\GRIB\January\wind_data.grib'
    processor = WindProcessor(grib_file_path)
    wind_magnitude_data,weibull_params = processor.process_grib_file()
    # print(weibull_params[(29.50,-85.25)])

    # Example lat-lon pair for plotting
    lat_lon_pair = (29.50, -85.25)  # Replace with a specific latitude and longitude from your data
    processor.plot_wind_magnitude(wind_magnitude_data, lat_lon_pair)
    # weibull_params.to_pickle("wind_mag_dist.pkl")
    # df = pd.read_pickle('wind_mag_dist.pkl')
    # processor.plot_expected_wind(lat_lon_pair)
    print("Done!")