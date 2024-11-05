import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from scipy.stats import beta, weibull_min
from scipy.special import gamma
from retry_requests import retry
from tqdm import tqdm

class WeatherDataProcessor:
    def __init__(self, cache_file='.cache', retries=5, backoff_factor=0.2):
        # Setup Open-Meteo API client with caching and retry
        cache_session = requests_cache.CachedSession(cache_file, expire_after=-1)
        retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
        self.client = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data(self, latitude, longitude, start_date, end_date, hourly_vars, timezone="auto"):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_vars,
            "timezone": timezone
        }
        self.response = self.client.weather_api(url, params=params)[0]
        print(f"Coordinates {self.response.Latitude()}°N {self.response.Longitude()}°E")
        print(f"Elevation {self.response.Elevation()} m asl")
        print(f"Timezone {self.response.Timezone()} {self.response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {self.response.UtcOffsetSeconds()} s")
    
    def process_hourly_data(self):
        # Extract variables from the response
        hourly = self.response.Hourly()
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(1).ValuesAsNumpy(),
            "shortwave_radiation": hourly.Variables(2).ValuesAsNumpy()
        }
        self.hourly_dataframe = pd.DataFrame(data=data)
        self.hourly_dataframe.set_index('date', inplace=True)
        print("Hourly data processed.")
        return self.hourly_dataframe

    def save_hourly_data(self, filename="data_hourly.pkl"):
        self.hourly_dataframe.to_pickle(filename)
        print(f"Hourly data saved to {filename}.")

    def resample_data(self, interval_minutes=15, filename=None):
        # Set date as index for resampling
        df = self.hourly_dataframe.copy()

        # Resampling and interpolation
        df_resampled = df.resample(f"{interval_minutes}min").interpolate(method='linear')
        
        # Save resampled data if filename is provided
        if filename:
            df_resampled.to_pickle(filename)
            print(f"Resampled data saved to {filename}.")
        
        return df_resampled
    
    def filter_data_by_time_step(self, data, month, day, hour=None, minute=None):
        """
        Filters and returns data from the provided DataFrame that matches a specific date and optionally time each year.
        
        Parameters:
            data (pd.DataFrame): DataFrame with a DatetimeIndex to filter.
            month (int): Month of the year to filter (1-12).
            day (int): Day of the month to filter (1-31).
            hour (int, optional): Hour of the day to filter (0-23). Defaults to None (matches all hours).
            minute (int, optional): Minute of the hour to filter (0-59). Defaults to None (matches all minutes).
        
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the matching date/time entries.
        """
        # Ensure index is datetime if not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Filter by month and day
        mask = (data.index.month == month) & (data.index.day == day)
        
        # Optionally filter by hour and minute
        if hour is not None:
            mask &= (data.index.hour == hour)
        if minute is not None:
            mask &= (data.index.minute == minute)
        
        # Return the filtered data
        filtered_data = data[mask]
        return filtered_data
    
    def fit_distributions(self, data, filename="data_expected.pkl"):
        """
        Fits solar radiation data to a beta distribution and wind speed data to a Weibull distribution
        for each unique time step in the year.
        
        Parameters:
            data (pd.DataFrame): DataFrame with a DatetimeIndex and columns for 'shortwave_radiation' and 'wind_speed_10m'.
        
        Returns:
            pd.DataFrame: A DataFrame with fitted distribution parameters and expected values for each time step.
        """
        results = []

        # Ensure index is datetime if not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Group by each unique time step (month, day, hour, minute)
        grouped = data.groupby([data.index.month, data.index.day, data.index.hour, data.index.minute])

        # Iterate over each time step in the year
        for (month, day, hour, minute), group in tqdm(grouped):
            # Fit solar radiation data to a beta distribution
            solar_data = group['shortwave_radiation'].dropna()
            if len(solar_data) > 1:
                if np.any(solar_data <= 5):
                    beta_params = (0, 0, 0, 0)  # Not enough data to fit
                    expected_beta = 0
                else:
                    # Normalize solar radiation data for beta fitting
                    solar_data_normalized = solar_data / 1367
                    beta_params = beta.fit(solar_data_normalized, floc=0, fscale=1)
                    alpha, beta_param, _, _ = beta_params
                    expected_beta = alpha / (alpha + beta_param) * 1367
            else:
                beta_params = (np.nan, np.nan, np.nan, np.nan)  # Not enough data to fit
                expected_beta = np.nan

            # Fit wind speed data to a Weibull distribution
            wind_data = group['wind_speed_10m'].dropna()
            if len(wind_data) > 1:
                weibull_params = weibull_min.fit(wind_data, floc=0)
                k, loc, scale = weibull_params
                expected_weibull = scale * gamma(1 + 1 / k)
            else:
                weibull_params = (np.nan, np.nan, np.nan)  # Not enough data to fit
                expected_weibull = np.nan  # Not enough data to fit

            # Store results in a list
            results.append({
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'beta_alpha': beta_params[0],
                'beta_beta': beta_params[1],
                'expected_solar_rad': expected_beta,
                'weibull_k': weibull_params[0],
                'weibull_loc': weibull_params[1],
                'weibull_scale': weibull_params[2],
                'expected_wind_speed': expected_weibull
            })

        # Create a DataFrame from the results list
        df = pd.DataFrame(results)
        df["datetime"] = pd.to_datetime(dict(year=2024, month=df["month"], day=df["day"], hour=df["hour"], minute=df["minute"]))

        # Step 2: Shift datetime by the desired number of hours, e.g., +3 hours
        shift_hours = -5
        df["datetime"] = df["datetime"] + pd.Timedelta(hours=shift_hours)

        # Step 3: Update month, day, hour, and minute columns from the shifted datetime
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute

        # Drop the helper datetime column if no longer needed
        results_df = df.drop(columns=["datetime"])
        results_df.to_pickle(filename)

        return results_df


# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = WeatherDataProcessor()

    # Fetch and process data
    processor.fetch_weather_data(latitude=30, longitude=-90, start_date="2000-01-01", end_date="2019-12-31", hourly_vars=["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"])
    hourly_df = processor.process_hourly_data()
    processor.save_hourly_data("data_hourly.pkl")

    # Resample and save the data to 10-minute intervals
    timestep = 10
    resampled_df = processor.resample_data(interval_minutes=timestep, filename=f"data_{timestep}min.pkl")

    # filtered_data = processor.filter_data_by_time_step(resampled_df, month=1, day=1, hour=10, minute=10)
    # print(filtered_data)
    fitted_distributions = processor.fit_distributions(hourly_df,"data_expected.pkl")
    fitted_distributions = processor.fit_distributions(resampled_df,f"data_expected_{timestep}min.pkl")
    print(fitted_distributions)

