import openmeteo_requests
from datetime import timezone, timedelta
import requests_cache
import pandas as pd
import numpy as np
import random
from scipy.stats import beta, weibull_min
from scipy.special import gamma
from retry_requests import retry
from tqdm import tqdm
from multiprocessing import Pool

class WeatherDataProcessor:
    def __init__(self, cache_file='.cache', retries=5, backoff_factor=0.2):
        session = requests_cache.CachedSession(cache_file, expire_after=-1)
        self.client = openmeteo_requests.Client(session=retry(session, retries=retries, backoff_factor=backoff_factor))

    def fetch_weather_data(self, latitude, longitude, start_date, end_date, hourly_vars, timezone="auto"):
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date, "end_date": end_date,
            "hourly": hourly_vars, "timezone": timezone, "cell_selection": "sea"
        }
        self.response = self.client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
        print(f"Coordinates {self.response.Latitude()}°N {self.response.Longitude()}°E")
        print(f"Elevation {self.response.Elevation()} m asl")
        print(f"Timezone {self.response.Timezone()} {self.response.TimezoneAbbreviation()}")
        print(f"UTC Offset {self.response.UtcOffsetSeconds()} s")

    def process_hourly_data(self):
        hourly = self.response.Hourly()
        offset = timezone(timedelta(seconds=self.response.UtcOffsetSeconds()))
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(offset),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert(offset),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(1).ValuesAsNumpy(),
            "shortwave_radiation": hourly.Variables(2).ValuesAsNumpy()
        }
        df = pd.DataFrame(data).set_index("date")
        self.hourly_dataframe = df[~((df.index.month == 2) & (df.index.day == 29))]
        print("Hourly data processed, leap days removed.")
        return self.hourly_dataframe

    def save_hourly_data(self, filename="data_hourly.pkl"):
        self.hourly_dataframe.to_pickle(filename)
        print(f"Hourly data saved to {filename}.")

    def resample_data(self, interval_minutes=15, filename=None):
        df_resampled = self.hourly_dataframe.resample(f"{interval_minutes}min").interpolate(method='linear')
        if filename:
            df_resampled.to_pickle(filename)
            print(f"Resampled data saved to {filename}.")
        return df_resampled

    @staticmethod
    def filter_data_by_time_step(data, month, day, hour=None, minute=None):
        data.index = pd.to_datetime(data.index)
        mask = (data.index.month == month) & (data.index.day == day)
        if hour is not None:
            mask &= (data.index.hour == hour)
        if minute is not None:
            mask &= (data.index.minute == minute)
        return data[mask]

    def fit_distributions(self, data, filename="data_expected.pkl"):
        results = []
        grouped = data.groupby([data.index.month, data.index.day, data.index.hour, data.index.minute])

        for (month, day, hour, minute), group in tqdm(grouped):
            solar_data = group['shortwave_radiation'].dropna()
            beta_params, expected_beta = self._fit_beta(solar_data)
            
            wind_data = group['wind_speed_10m'].dropna()
            weibull_params, expected_weibull = self._fit_weibull(wind_data)

            results.append({
                'month': month, 'day': day, 'hour': hour, 'minute': minute,
                'beta_alpha': beta_params[0], 'beta_beta': beta_params[1],
                'expected_solar_rad': expected_beta,
                'weibull_k': weibull_params[0], 'weibull_loc': weibull_params[1],
                'weibull_scale': weibull_params[2], 'expected_wind_speed': expected_weibull
            })

        df = pd.DataFrame(results)
        df.to_pickle(filename)
        return df

    @staticmethod
    def _fit_beta(data):
        if len(data) > 1 and np.all(data > 5):
            normalized = data / 1367
            beta_params = beta.fit(normalized, floc=0, fscale=1)
            alpha, beta_param, _, _ = beta_params
            expected_beta = alpha / (alpha + beta_param) * 1367
        else:
            beta_params, expected_beta = (1., 1000., np.nan, np.nan), 0.0
        return beta_params, expected_beta

    @staticmethod
    def _fit_weibull(data):
        if len(data) > 1:
            params = weibull_min.fit(data, floc=0)
            k, loc, scale = params
            expected = scale * gamma(1 + 1 / k)
        else:
            params, expected = (np.nan, np.nan, np.nan), np.nan
        return params, expected

def generate_single_synthetic_year(dataset_number, historical_data, years, timestep, points_per_week, save_path,latitude,longitude,seed=None):
    if seed is not None:
        random.seed(seed + dataset_number)
    synthetic_year = []
    original_timezone = historical_data.index.tz if historical_data.index.tz is not None else "UTC"

    for week_number in range(52):
        while True:
            selected_year = random.choice(years)
            weekly_data = historical_data[historical_data.index.year == selected_year]
            start, end = week_number * points_per_week, (week_number + 1) * points_per_week
            if len(weekly_data.iloc[start:end]) == points_per_week:
                synthetic_year.append(weekly_data.iloc[start:end])
                break
            print("Trying again...")

    synthetic_year_data = pd.concat(synthetic_year)
    synthetic_year_data.index = pd.date_range(
        start="2025-01-01", periods=len(synthetic_year_data), tz=original_timezone, freq=pd.Timedelta(seconds=timestep)
    )
    file_path = f"{save_path}/data_lat{latitude}_lon{longitude}_{int(timestep / 60)}min_{dataset_number}.pkl"
    synthetic_year_data.to_pickle(file_path)
    return file_path

def generate_yearly_weather_data(historical_data,N,latitude,longitude,seed=None, save_path="."):
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        raise ValueError("historical_data must have a DatetimeIndex.")

    timestep = int((historical_data.index[1] - historical_data.index[0]).total_seconds())
    points_per_week = int((7 * 24 * 3600) / timestep)
    years = historical_data.index.year.unique()

    with Pool() as pool:
        return pool.starmap(generate_single_synthetic_year, [
            (i, historical_data, years, timestep, points_per_week, save_path,latitude,longitude,seed) for i in range(N)
        ])

# Example usage
if __name__ == "__main__":
    processor = WeatherDataProcessor()
    lat, lon = 30, -90
    timestep_min = 15
    processor.fetch_weather_data(lat, lon, "1950-01-01", "2022-12-31", ["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"])
    hourly_df = processor.process_hourly_data()
    resampled_df = processor.resample_data(timestep_min)

    expected_data_filename = rf"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\EXPECTED_DATA\data_expected_lat{lat}_lon{lon}_{timestep_min}min.pkl"
    processor.fit_distributions(resampled_df, expected_data_filename)
    
    synthetic_data_directory = r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\SYNTHETIC_DATA"

    N=1000
    generate_yearly_weather_data(resampled_df,N,lat,lon,1,synthetic_data_directory)