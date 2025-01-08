import openmeteo_requests
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
            "timezone": timezone,
            "cell_selection": "sea"
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

            # # Fit wind speed data to a Weibull distribution
            # wind_data = group['wind_speed_10m'].dropna()
            # if len(wind_data) > 1:
            #     weibull_params = weibull_min.fit(wind_data, floc=0)
            #     k, loc, scale = weibull_params
            #     expected_weibull = scale * gamma(1 + 1 / k)
            # else:
            #     weibull_params = (np.nan, np.nan, np.nan)  # Not enough data to fit
            #     expected_weibull = np.nan  # Not enough data to fit

            # Store results in a list
            results.append({
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'beta_alpha': beta_params[0],
                'beta_beta': beta_params[1],
                'expected_solar_rad': expected_beta,
                # 'weibull_k': weibull_params[0],
                # 'weibull_loc': weibull_params[1],
                # 'weibull_scale': weibull_params[2],
                # 'expected_wind_speed': expected_weibull
            })

        df = pd.DataFrame(results)
        df["datetime"] = pd.to_datetime(dict(year=2024, month=df["month"], day=df["day"], hour=df["hour"], minute=df["minute"]))

        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute

        results_df = df.drop(columns=["datetime"])
        results_df.to_pickle(filename)

        return results_df

def generate_yearly_weather_data(historical_data, N=1, seed=None, save_path="synthetic_data_"):
    """
    Generates multiple synthetic years of weather data by randomly selecting weeks
    from the available historical data and saves each dataset to a file.

    Parameters:
        historical_data (pd.DataFrame): DataFrame containing historical weather data with a DatetimeIndex.
        N (int): Number of synthetic datasets to generate.
        seed (int, optional): Seed for random number generation to ensure reproducibility.
        save_path (str): Path prefix to save the generated datasets (files will be named with indices like 'synthetic_data_0.pkl').
        
    Returns:
        list: A list of file paths where the datasets are saved.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        raise ValueError("The historical_data DataFrame must have a DatetimeIndex.")

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Extract unique years from the historical data
    years = historical_data.index.year.unique()

    # Initialize a list to store the paths of the saved files
    saved_files = []

    # Generate datasets
    for i in range(N):
        # Select a random year
        selected_year = random.choice(years)
        
        # Filter data for the selected year
        year_data = historical_data[historical_data.index.year == selected_year]
        
        # Check for leap day
        if '02-29' not in year_data.index.strftime('%m-%d'):
            # Add leap day data using February 28th's data
            feb_28_data = year_data[year_data.index.strftime('%m-%d') == '02-28']
            leap_day_data = feb_28_data.copy()
            leap_day_data.index = leap_day_data.index + pd.Timedelta(days=1)  # Set index to February 29th
            year_data = pd.concat([year_data, leap_day_data]).sort_index()

        # Save the synthetic dataset
        file_path = f"{save_path}\\data_{int(timestep)}min_{i}.pkl"
        year_data.to_pickle(file_path)
        saved_files.append(file_path)

    return saved_files




def generate_single_synthetic_year(dataset_number, historical_data, years, timestep, points_per_week, save_path, seed=None):
    """
    Generates a single synthetic year's weather data by randomly selecting weeks from the available historical data.
    
    Parameters:
        dataset_number (int): Index for the synthetic dataset.
        historical_data (pd.DataFrame): DataFrame containing historical weather data.
        years (np.ndarray): List of unique years available in the historical data.
        timestep (int): Time step in seconds.
        points_per_week (int): Number of data points in a week.
        save_path (str): Path to save the generated dataset.
        seed (int, optional): Seed for reproducibility.
        
    Returns:
        str: Path where the synthetic dataset was saved.
    """
    if seed is not None:
        random.seed(seed + dataset_number)  # Ensure different seeds for different processes

    synthetic_year = []

    # Generate data for 52 weeks
    for week_number in range(52):
        valid_week = False
        
        while not valid_week:
            # Randomly select a year
            selected_year = random.choice(years)

            # Extract data for the selected year
            year_data = historical_data[historical_data.index.year == selected_year]

            # Calculate start and end indices for the selected week
            week_start = week_number * points_per_week
            week_end = (week_number + 1) * points_per_week

            # Extract the data for the selected week
            weekly_data = year_data.iloc[week_start:week_end]

            # Check if the extracted data has the correct number of points
            if len(weekly_data) == points_per_week:
                valid_week = True
                synthetic_year.append(weekly_data)

    # Concatenate all the weekly data into a single DataFrame
    synthetic_year_data = pd.concat(synthetic_year)

    # Generate a new DatetimeIndex for the synthetic year
    synthetic_year_data.index = pd.date_range(
        start="2024-01-01",
        periods=len(synthetic_year_data),
        tz="UTC",
        freq=pd.Timedelta(seconds=timestep)
    )

    # Define the file path to save the dataset
    file_path = f"{save_path}\\data_{int(timestep)}min_{dataset_number}.pkl"

    # Save the synthetic year data to a file
    synthetic_year_data.to_pickle(file_path)

    return file_path


# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = WeatherDataProcessor()

    # Fetch and process data
    lat = -30
    lon = -90
    processor.fetch_weather_data(latitude=lat, longitude=lon, start_date="2000-01-01", end_date="2023-12-31", hourly_vars=["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"])
    hourly_df = processor.process_hourly_data()
    # fitted_distributions = processor.fit_distributions(hourly_df,"data_expected.pkl")
    # hourly_df.to_pickle(r"Data\HISTORICAL_DATA")
    # fitted_distributions.to_pickle(r"Data\EXPECTED_DATA\data_expected_60min")
    
    timestep = 60
    resampled_df = processor.resample_data(interval_minutes=timestep, filename=f"Data\HISTORICAL_DATA\data_{timestep}min.pkl")
    fitted_distributions = processor.fit_distributions(resampled_df,rf"Data\EXPECTED_DATA\lat{lat}\data_expected_{timestep}min_lat{lat}.pkl")
    generate_yearly_weather_data(resampled_df,N=10,save_path=rf"Data\SYNTHETIC_DATA\lat{lat}")


