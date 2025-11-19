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

# Optional dependency for solar position
try:
    import pvlib  # pip install pvlib
except Exception:
    pvlib = None


class WeatherDataProcessor:
    """
    Fetches historical weather from Open-Meteo, resamples it, and fits
    parametric distributions for wind (Weibull) and irradiance (Beta),
    implementing the Fatemi–Kuh–Fripp (2018) normalization:
        x(n) = r(n) / (A * cos(z))  in (0, 1), with clipping of rare x>=1 to 0.99999.
    """

    # Paper constants and local safeguards
    MIN_GHI_WM2 = 5.0       # Drop/ignore values <= 5 W/m^2 (night/near-night)
    A_CLEAR = 1150.0        # Paper's scaling constant A (W/m^2)

    def __init__(self, cache_file: str = ".cache", retries: int = 5, backoff_factor: float = 0.2):
        session = requests_cache.CachedSession(cache_file, expire_after=-1)
        self.client = openmeteo_requests.Client(
            session=retry(session, retries=retries, backoff_factor=backoff_factor)
        )
        self.response = None
        self.hourly_dataframe: pd.DataFrame | None = None
        self.latitude: float | None = None
        self.longitude: float | None = None

    def fetch_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        hourly_vars: list[str],
        timezone: str = "auto",
    ):
        """
        Download historical weather data from Open-Meteo archive.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_vars,
            "timezone": timezone,
            "wind_speed_unit": "ms",
            "cell_selection": "sea",
        }
        self.response = self.client.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
        self.latitude = latitude
        self.longitude = longitude
        print(f"Coordinates {self.response.Latitude()}°N {self.response.Longitude()}°E")
        print(f"Elevation {self.response.Elevation()} m asl")
        print(f"Timezone {self.response.Timezone()} {self.response.TimezoneAbbreviation()}")
        print(f"UTC Offset {self.response.UtcOffsetSeconds()} s")

    def process_hourly_data(self) -> pd.DataFrame:
        """Construct an hourly DataFrame with leap days removed."""
        if self.response is None:
            raise RuntimeError("Call fetch_weather_data first.")
        hourly = self.response.Hourly()
        offset = timezone(timedelta(seconds=self.response.UtcOffsetSeconds()))
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(offset),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert(offset),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(1).ValuesAsNumpy(),
            "shortwave_radiation": hourly.Variables(2).ValuesAsNumpy(),
        }
        df = pd.DataFrame(data).set_index("date")
        # Drop Feb 29 to maintain 365-day alignment
        self.hourly_dataframe = df[~((df.index.month == 2) & (df.index.day == 29))]
        print("Hourly data processed, leap days removed.")
        return self.hourly_dataframe

    def save_hourly_data(self, filename: str = "data_hourly.pkl"):
        if self.hourly_dataframe is None:
            raise RuntimeError("No hourly data to save. Call process_hourly_data first.")
        self.hourly_dataframe.to_pickle(filename)
        print(f"Hourly data saved to {filename}.")

    # ---------- Normalization helpers ----------
    def _require_pvlib(self):
        if pvlib is None:
            raise ImportError("pvlib is required for solar zenith calculation. `pip install pvlib`. ")

    def _compute_cos_zenith(self, index: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.Series:
        """
        Compute cos(zenith) for each timestamp. zenith >= 90° => set to NaN (night).
        """
        self._require_pvlib()
        solpos = pvlib.solarposition.get_solarposition(index, latitude, longitude)
        zenith = solpos["zenith"]
        cos_z = np.cos(np.deg2rad(zenith))
        cos_z[zenith >= 90.0] = np.nan
        return pd.Series(cos_z.values, index=index, name="cos_zenith")

    def resample_data(self, interval_minutes: int = 15, filename: str | None = None) -> pd.DataFrame:
        """
        Resample to a finer grid, interpolate linearly, then compute cos(zenith),
        and create normalized irradiance x = r / (A * cos z) for valid daytime points
        with r > MIN_GHI_WM2 and cos z > 0. Rare x>=1 are clipped to 0.99999.
        """
        if self.hourly_dataframe is None:
            raise RuntimeError("No hourly data. Call process_hourly_data first.")

        df_resampled = self.hourly_dataframe.resample(f"{interval_minutes}min").interpolate(method="linear")

        # cos(zenith)
        if self.latitude is None or self.longitude is None:
            raise RuntimeError("Latitude/longitude are not set. Call fetch_weather_data first.")
        cos_z = self._compute_cos_zenith(df_resampled.index, self.latitude, self.longitude)
        df_resampled["cos_zenith"] = cos_z

        # Filter BEFORE normalization: avoid night and very low irradiance values
        ghi = df_resampled["shortwave_radiation"]
        good = (ghi > self.MIN_GHI_WM2) & (df_resampled["cos_zenith"] > 0)

        x = pd.Series(np.nan, index=df_resampled.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            x.loc[good] = ghi.loc[good] / (self.A_CLEAR * df_resampled.loc[good, "cos_zenith"])

        # Clip per paper for rare cloud-enhanced exceedances
        x = x.clip(lower=0.0, upper=0.99999)
        df_resampled["ghi_norm"] = x

        if filename:
            df_resampled.to_pickle(filename)
            print(f"Resampled data saved to {filename}.")
        return df_resampled

    @staticmethod
    def filter_data_by_time_step(data: pd.DataFrame, month: int, day: int, hour: int | None = None, minute: int | None = None) -> pd.DataFrame:
        data.index = pd.to_datetime(data.index)
        mask = (data.index.month == month) & (data.index.day == day)
        if hour is not None:
            mask &= data.index.hour == hour
        if minute is not None:
            mask &= data.index.minute == minute
        return data[mask]

    # ---------- Distribution fitting ----------
    def fit_distributions(self, data: pd.DataFrame, filename: str = "data_expected.pkl") -> pd.DataFrame:
        """
        Group by (month, day, hour, minute) and fit:
          • Beta to normalized irradiance (ghi_norm)
          • Weibull to 10m wind speed
        Also store an expected irradiance for the slot by mapping Beta mean back using
        A * mean(cos z) (set to 0 if A*cos_mean <= MIN_GHI_WM2).
        """
        results: list[dict] = []
        grouped = data.groupby([data.index.month, data.index.day, data.index.hour, data.index.minute])

        for (month, day, hour, minute), group in tqdm(grouped):
            # Irradiance (normalized)
            x = group.get("ghi_norm", pd.Series(dtype=float)).dropna()
            if len(x) >= 5:
                (a, b), mean_x = self._fit_beta_normalized(x)
            else:
                (a, b), mean_x = (np.nan, np.nan), np.nan

            # Map expected x back to W/m^2 for this slot using mean cos z
            cos_mean = group.get("cos_zenith", pd.Series(dtype=float)).dropna().mean()
            if np.isnan(mean_x) or np.isnan(cos_mean) or (cos_mean <= 0):
                expected_solar = np.nan
            else:
                cap = self.A_CLEAR * cos_mean
                expected_solar = (mean_x * cap) if (cap > self.MIN_GHI_WM2) else 0.0

            # Wind (unchanged)
            wind_data = group.get("wind_speed_10m", pd.Series(dtype=float)).dropna()
            weibull_params, expected_weibull = self._fit_weibull(wind_data)

            results.append(
                {
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "beta_alpha": a,
                    "beta_beta": b,
                    "expected_solar_rad": expected_solar,
                    "weibull_k": weibull_params[0],
                    "weibull_loc": weibull_params[1],
                    "weibull_scale": weibull_params[2],
                    "expected_wind_speed": expected_weibull,
                }
            )

        df = pd.DataFrame(results)
        df.to_pickle(filename)
        return df

    @staticmethod
    def _fit_beta_normalized(x: pd.Series) -> tuple[tuple[float, float], float]:
        """
        Fit Beta(alpha, beta) to normalized irradiance x in (0,1).
        Returns ((alpha, beta), mean_x).
        """
        x = x.dropna()
        if len(x) < 5:
            return (np.nan, np.nan), np.nan
        # Fit in fixed [0,1] support
        a, b, _, _ = beta.fit(x, floc=0, fscale=1)
        mean_x = a / (a + b)
        return (a, b), mean_x

    @staticmethod
    def _fit_weibull(data: pd.Series) -> tuple[tuple[float, float, float], float]:
        if len(data) > 1:
            params = weibull_min.fit(data, floc=0)
            k, loc, scale = params
            expected = scale * gamma(1 + 1 / k)
        else:
            params, expected = (np.nan, np.nan, np.nan), np.nan
        return params, expected

    # ---------- Sampling helper (optional) ----------
    def sample_irradiance(self, alpha: float, beta_param: float, target_ts: pd.Timestamp) -> float:
        """
        Sample irradiance at a target timestamp by:
          1) draw x ~ Beta(alpha, beta)
          2) scale by A * cos(zenith(target_ts))
          3) enforce MIN_GHI_WM2 near night
        """
        if np.isnan(alpha) or np.isnan(beta_param) or alpha <= 0 or beta_param <= 0:
            return np.nan
        x = np.random.beta(alpha, beta_param)
        cos_z = self._compute_cos_zenith(pd.DatetimeIndex([target_ts]), self.latitude, self.longitude).iloc[0]
        if (cos_z is None) or np.isnan(cos_z) or (cos_z <= 0):
            return 0.0
        cap = self.A_CLEAR * cos_z
        if cap <= self.MIN_GHI_WM2:
            return 0.0
        r = x * cap
        return max(0.0, float(r))


# --------- Synthetic year generation (unchanged) ---------

def generate_single_synthetic_year_worker(args):
    # Unpack the tuple into individual arguments
    dataset_number, historical_data, years, timestep, points_per_week, save_path, latitude, longitude, seed = args
    # Call the original function with the unpacked arguments
    return generate_single_synthetic_year(dataset_number, historical_data, years, timestep, points_per_week, save_path, latitude, longitude, seed)


def generate_single_synthetic_year(
    dataset_number: int,
    historical_data: pd.DataFrame,
    years: np.ndarray,
    timestep: int,
    points_per_week: int,
    save_path: str,
    latitude: float,
    longitude: float,
    seed: int | None = None,
):
    if seed is not None:
        random.seed(seed + dataset_number)
    synthetic_year = []
    original_timezone = historical_data.index.tz if historical_data.index.tz is not None else "UTC"
    for week_number in range(52):
        while True:
            selected_year = random.choice(years)
            weekly_data = historical_data[historical_data.index.year == selected_year]
            start, end = (
                week_number * points_per_week,
                (week_number + 1) * points_per_week,
            )
            if len(weekly_data.iloc[start:end]) == points_per_week:
                synthetic_year.append(weekly_data.iloc[start:end])
                break
            print("Trying again.")

    synthetic_year_data = pd.concat(synthetic_year)
    synthetic_year_data.index = pd.date_range(
        start="2025-01-01",
        periods=len(synthetic_year_data),
        tz=original_timezone,
        freq=pd.Timedelta(seconds=timestep),
    )
    file_path = f"{save_path}/data_lat{latitude}_lon{longitude}_{int(timestep / 60)}min_{dataset_number}.pkl"
    synthetic_year_data.to_pickle(file_path)
    return file_path


def generate_yearly_weather_data(historical_data: pd.DataFrame, N: int, latitude: float, longitude: float, seed: int | None = None, save_path: str = "."):
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        raise ValueError("historical_data must have a DatetimeIndex.")

    timestep = int((historical_data.index[1] - historical_data.index[0]).total_seconds())
    points_per_week = int((7 * 24 * 3600) / timestep)
    years = historical_data.index.year.unique()

    try:
        with Pool() as pool:
            results = tqdm(
                pool.imap_unordered(
                    generate_single_synthetic_year_worker,
                    [
                        (
                            i,
                            historical_data,
                            years,
                            timestep,
                            points_per_week,
                            save_path,
                            latitude,
                            longitude,
                            seed,
                        )
                        for i in range(N)
                    ],
                ),
                total=N,
                desc="Generating Synthetic Years",
            )
            return list(results)
    except KeyboardInterrupt:
        print("Process interrupted by user. Cleaning up.")
        pool.terminate()  # Immediately terminate workers
        pool.join()       # Ensure all worker processes are cleaned up
        raise  # Re-raise the exception for visibility


if __name__ == "__main__":
    # Example usage
    processor = WeatherDataProcessor()
    lat, lon = 30, -90
    timestep_min = 15
    processor.fetch_weather_data(
        lat,
        lon,
        "1950-01-01",
        "2022-12-31",
        ["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"],
    )
    hourly_df = processor.process_hourly_data()
    hourly_df.to_pickle(rf"Data\HISTORICAL_DATA\data_{lat}_{lon}.pkl")
    resampled_df = processor.resample_data(timestep_min)

    expected_data_filename = rf"Data\EXPECTED_DATA\data_expected_lat{lat}_lon{lon}_{timestep_min}min.pkl"
    processor.fit_distributions(resampled_df, expected_data_filename)
