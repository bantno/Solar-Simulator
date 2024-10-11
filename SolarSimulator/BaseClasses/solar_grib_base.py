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
            if grb.parameterName == 'Surface solar radiation downwards' and grb.validDate.month == 1:
                lats, lons = grb.latlons()
                values = grb.values
                
                date = grb.validityDate
                hour = grb.validityTime
                utc_date = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
                utc_date = utc_date.tz_localize(self.utc_tz)
                valid_date = utc_date.astimezone(self.eastern_tz)

                day_month_hour = (valid_date.month, valid_date.day, valid_date.hour)

                for i in range(lats.shape[0]):
                    for j in range(lons.shape[1]):
                        lat_lon_pair = (lats[i, j], lons[i, j])
                        
                        if lat_lon_pair not in self.data_dict:
                            self.data_dict[lat_lon_pair] = {}
                        
                        if day_month_hour not in self.data_dict[lat_lon_pair]:
                            self.data_dict[lat_lon_pair][day_month_hour] = []
                        
                        self.data_dict[lat_lon_pair][day_month_hour].append(values[i, j] / 3600)  # Convert to W/m^2
    
        grbs.close()

    def normalize_ssrd_data(self):
        normalized_data = {}
        normalizing_factor = 1367  # W/m^2
        
        for lat_lon_pair, datetime_dict in self.data_dict.items():
            for date_time, values in datetime_dict.items():
                if lat_lon_pair not in normalized_data:
                    normalized_data[lat_lon_pair] = {}
                normalized_values = np.array(values) / normalizing_factor
                normalized_data[lat_lon_pair][date_time] = normalized_values
        
        return normalized_data, normalizing_factor

    def fit_beta_distributions(self, normalized_data, normalizing_factor, epsilon=1e-6):
        expected_values = {}
        
        for lat_lon_pair, datetime_dict in normalized_data.items():
            for date_time, values in datetime_dict.items():
                if len(values) > 0:
                    values = np.array(values)
                    if np.any(values == 0) or np.any(values == 1) or np.ptp(values) == 0:
                        if lat_lon_pair not in expected_values:
                            expected_values[lat_lon_pair] = {}
                        expected_values[lat_lon_pair][date_time] = 0
                    else:
                        clipped_values = np.clip(values, epsilon, 1 - epsilon)
                        try:
                            a, b, loc, scale = beta.fit(clipped_values, floc=0, fscale=1)
                            expected_value = a / (a + b) * normalizing_factor
                            if lat_lon_pair not in expected_values:
                                expected_values[lat_lon_pair] = {}
                            expected_values[lat_lon_pair][date_time] = expected_value
                        except Exception as e:
                            expected_values[lat_lon_pair][date_time] = np.nan
                            print(f"Fitting failed for {lat_lon_pair} at {date_time}: {e}")
        
        return pd.DataFrame.from_dict(expected_values)

    def convert_to_datetime_index(self, df):
        # Convert multiindex (month, day, hour) to datetime index
        # Assume a fixed year, e.g., 2022
        df.index = pd.MultiIndex.from_tuples(df.index, names=["month", "day", "hour"])
        df = df.reset_index()

        # Create a datetime column from (month, day, hour)
        df['datetime'] = pd.to_datetime(
            dict(year=2025, month=df['month'], day=df['day'], hour=df['hour'])
        )

        # Set the datetime column as the index
        df = df.set_index('datetime')
        df = df.drop(columns=["month", "day", "hour"])

        return df

    def resample_data(self, df, interval_minutes=15):
        # Convert the multiindex to a datetime index first
        df_with_datetime = self.convert_to_datetime_index(df)

        # Resample to 15-minute intervals and interpolate
        df_resampled = df_with_datetime.resample(f'{interval_minutes}min').interpolate(method='linear')
        return df_resampled

    def process_grib_file(self):
        self.extract_ssrd_data()
        normalized_ssrd_data, normalizing_factor = self.normalize_ssrd_data()
        solar_ev = self.fit_beta_distributions(normalized_ssrd_data, normalizing_factor)

        # Resample to 15-minute intervals
        solar_ev_resampled = self.resample_data(solar_ev)
        return solar_ev_resampled

if __name__ == "__main__":
    grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\GRIB\January\solar_radiation.grib'
    processor = SolarRadiationProcessor(grib_file_path)

    # Process and resample
    # solar_ev_resampled = processor.process_grib_file()
    # solar_ev_resampled.to_pickle("solar_ev_resampled.pkl")

    year_data = processor.extract_ssrd_data_by_year(2022)
    year_solar = processor.resample_data(year_data)
    year_solar.to_pickle("2022_solar_resampled.pkl")

    print("Done!")
