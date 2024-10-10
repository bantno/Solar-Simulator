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
        self.data_dict = {}
        self.year_data_dict = {}
    
    def extract_ssrd_data(self):
        # Open the GRIB file
        grbs = pygrib.open(self.grib_file_path)
        
        # Loop through all messages in the GRIB file
        for grb in tqdm(grbs.select()):
            # Extract only SSRD messages for January
            if grb.parameterName == 'Surface solar radiation downwards' and grb.validDate.month == 1:
                lats, lons = grb.latlons()
                values = grb.values
                
                # Extract the date and time from the GRIB message
                date = grb.validityDate    # YYYYMMDD format
                hour = grb.validityTime    # HHMM format, may need padding
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')  # Convert to datetime
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)

                day_month_hour = (valid_date.month, valid_date.day, valid_date.hour)

                # Store the SSRD data by (latitude, longitude) and day/month/hour in January (across years)
                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        
                        if lat_lon_pair not in self.data_dict:
                            self.data_dict[lat_lon_pair] = {}
                        
                        if day_month_hour not in self.data_dict[lat_lon_pair]:
                            self.data_dict[lat_lon_pair][day_month_hour] = []
                        
                        self.data_dict[lat_lon_pair][day_month_hour].append(values[i, j]/3600)  # Add SSRD value
    
        grbs.close()

    def extract_ssrd_data_for_year(self, year):
        # Open the GRIB file
        grbs = pygrib.open(self.grib_file_path)

        # Loop through all messages in the GRIB file
        for grb in tqdm(grbs.select()):
            # Check if the message is SSRD (Surface Solar Radiation Downwards)
            if grb.parameterName == 'Surface solar radiation downwards':
                lats, lons = grb.latlons()
                values = grb.values

                # Extract the date and time from the GRIB message
                date = grb.validityDate    # YYYYMMDD format
                hour = grb.validityTime    # HHMM format, may need padding
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')  # Convert to datetime
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)

                # Filter by the specified year
                if valid_date.year == year:
                    # Extract (month, day, hour) for grouping purposes
                    day_month_hour = (valid_date.month, valid_date.day, valid_date.hour)

                    # Store the SSRD data by (latitude, longitude) and (month, day, hour) for the specified year
                    for i in range(lats.shape[0]):
                        for j in range(lons.shape[1]):
                            lat_lon_pair = (lats[i, j], lons[i, j])

                            if lat_lon_pair not in self.year_data_dict:
                                self.year_data_dict[lat_lon_pair] = {}

                            if day_month_hour not in self.year_data_dict[lat_lon_pair]:
                                self.year_data_dict[lat_lon_pair][day_month_hour] = []

                            # Add SSRD value (converted to W/m^2 by dividing by 3600)
                            self.year_data_dict[lat_lon_pair][day_month_hour].append(values[i, j] / 3600)

        grbs.close()

    
    def normalize_ssrd_data(self):
        normalized_data = {}
        normalizing_factor = 1367 # W/m^2
        
        for lat_lon_pair, datetime_dict in self.data_dict.items():
            
            for date_time, values in datetime_dict.items():
                if lat_lon_pair not in normalized_data:
                    normalized_data[lat_lon_pair] = {}
                
                normalized_values = np.array(values) / normalizing_factor  # Normalize the values by solar constant 1367 W/m^2
                normalized_data[lat_lon_pair][date_time] = normalized_values
        
        return normalized_data, normalizing_factor

    def fit_beta_distributions(self, normalized_data, normalizing_factor, epsilon=1e-6):
        expected_values = {}
        
        for lat_lon_pair, datetime_dict in normalized_data.items():
            for date_time, values in datetime_dict.items():
                if len(values) > 0:  # Ensure there is data
                    values = np.array(values)
                    
                    # Check if any of the values are zero, or if the range is too narrow
                    if np.any(values == 0) or np.any(values == 1) or np.ptp(values) == 0:
                        if lat_lon_pair not in expected_values:
                            expected_values[lat_lon_pair] = {}
                        expected_values[lat_lon_pair][date_time] = 0  # Store 0 for invalid cases
                    else:
                        clipped_values = np.clip(values, epsilon, 1 - epsilon)

                        # Fit the beta distribution
                        try:
                            a, b, loc, scale = beta.fit(clipped_values, floc=0, fscale=1)
                            expected_value = a / (a + b) * normalizing_factor  # Calculate expected value
                            if lat_lon_pair not in expected_values:
                                expected_values[lat_lon_pair] = {}
                            
                            expected_values[lat_lon_pair][date_time] = expected_value  # Store the expected value
                        except Exception as e:
                            expected_values[lat_lon_pair][date_time] = np.nan  # Store NaN if fitting fails
                            print(f"Fitting failed for {lat_lon_pair} at {date_time}: {e}")
        
        return pd.DataFrame.from_dict(expected_values)

    def process_grib_file(self):
        self.extract_ssrd_data()  # Step 1: Extract and segment
        normalized_ssrd_data, normalizing_factor = self.normalize_ssrd_data()  # Step 2: Normalize data
        solar_ev = self.fit_beta_distributions(normalized_ssrd_data,normalizing_factor)  # Step 3: Fit beta distributions
        return solar_ev

if __name__ == "__main__":
    grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\GRIB\January\solar_radiation.grib'
    processor = SolarRadiationProcessor(grib_file_path)
    # solar_ev = processor.process_grib_file()
    year = 2022
    solar_val = processor.extract_ssrd_data_for_year(year)
    pd.DataFrame.from_dict(processor.year_data_dict).to_pickle(f"{year}_solar_data.pkl")
    # beta_params.to_csv("solar_dist.csv")
    # solar_ev.to_pickle("solar_ev.pkl")
    print("Done!")
