import math
import openmeteo_requests
from datetime import timezone, timedelta
import requests_cache
import pandas as pd
import pvlib
import datetime
import numpy as np
import random
from scipy.stats import beta, weibull_min
from scipy.special import gamma
from retry_requests import retry
from tqdm import tqdm
from multiprocessing import Pool


class WeatherDataProcessor:
    def __init__(self, cache_file=".cache", retries=5, backoff_factor=0.2):
        session = requests_cache.CachedSession(cache_file, expire_after=-1)
        self.client = openmeteo_requests.Client(
            session=retry(session, retries=retries, backoff_factor=backoff_factor)
        )

    def fetch_weather_data(
        self, latitude, longitude, start_date, end_date, hourly_vars, timezone="auto"
    ):
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
        self.lat = latitude
        self.lon = longitude
        self.response = self.client.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
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
                inclusive="left",
            ),
            "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_direction_10m": hourly.Variables(1).ValuesAsNumpy(),
            "shortwave_radiation": hourly.Variables(2).ValuesAsNumpy(),
        }
        df = pd.DataFrame(data).set_index("date")
        self.hourly_dataframe = df[~((df.index.month == 2) & (df.index.day == 29))]
        print("Hourly data processed, leap days removed.")
        return self.hourly_dataframe

    def save_hourly_data(self, filename="data_hourly.pkl"):
        self.hourly_dataframe.to_pickle(filename)
        print(f"Hourly data saved to {filename}.")

    def resample_data(self, interval_minutes=15, filename=None):
        df_resampled = self.hourly_dataframe.resample(f"{interval_minutes}min").interpolate(
            method="linear"
        )
        if filename:
            df_resampled.to_pickle(filename)
            print(f"Resampled data saved to {filename}.")
        return df_resampled

    @staticmethod
    def filter_data_by_time_step(data, month, day, hour=None, minute=None):
        data.index = pd.to_datetime(data.index)
        mask = (data.index.month == month) & (data.index.day == day)
        if hour is not None:
            mask &= data.index.hour == hour
        if minute is not None:
            mask &= data.index.minute == minute
        return data[mask]

    @staticmethod
    def clearsky_ghi_fatemi(times, lat, lon, A=1150.0):
        """
        Clear-sky GHI per Fatemi–Kuh–Fripp (2018):
            GHI_cs(t) = A * cos(zenith(t))
        where 'A' is a fixed scale (paper uses ~1150 W/m^2) and
        zenith is the apparent solar zenith angle.

        Parameters
        ----------
        times : pandas.DatetimeIndex or pd.Timestamp
            TZ-aware timestamps (local or UTC). If you pass naive times,
            pvlib will treat them as naive; prefer tz-aware.
        lat, lon : float
            Latitude [deg], Longitude [deg] (east positive).
        A : float, default 1150.0
            Scaling constant so that normalized irradiance r/(A cos z) lies in (0,1).

        Returns
        -------
        pandas.Series
            Clear-sky GHI [W/m^2], clipped at 0 at night.
        """
        # Ensure we always work with a DatetimeIndex
        if isinstance(times, pd.Timestamp):
            times = pd.DatetimeIndex([times])
        elif not isinstance(times, pd.DatetimeIndex):
            times = pd.DatetimeIndex(times)

        sp = pvlib.solarposition.get_solarposition(times, lat, lon)
        zen = sp["apparent_zenith"].to_numpy()  # degrees

        # cos(zenith) with zenith in degrees; negative at night -> 0
        cosz = np.cos(np.deg2rad(zen))
        cosz = np.clip(cosz, 0.0, None)

        ghi_cs = A * cosz
        return ghi_cs[0]


    def fit_distributions(self, data, filename="data_expected.pkl"):
        results = []
        grouped = data.groupby(
            [data.index.month, data.index.day, data.index.hour, data.index.minute]
        )

        for (month, day, hour, minute), group in tqdm(grouped):
            if day == 29 and month == 2:
                continue
            solar_data = group["shortwave_radiation"].dropna()
            beta_params, expected_beta, clearsky_irradiance = self._fit_beta(solar_data)

            wind_data = group["wind_speed_10m"].dropna()
            weibull_params, expected_weibull = self._fit_weibull(wind_data)

            results.append(
                {
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "beta_alpha": beta_params[0],
                    "beta_beta": beta_params[1],
                    "expected_solar_rad": expected_beta,
                    "clearsky_irradiance": clearsky_irradiance,
                    "weibull_k": weibull_params[0],
                    "weibull_loc": weibull_params[1],
                    "weibull_scale": weibull_params[2],
                    "expected_wind_speed": expected_weibull,
                }
            )

        df = pd.DataFrame(results)
        df.to_pickle(filename)
        return df

    def _fit_beta(self, data):
        # Compute clearsky irradiance for the first timestamp
        ts0 = data.index[0]
        clearsky_irradiance = self.clearsky_ghi_fatemi(ts0, self.lat, self.lon)

        # Only fit when all measurements exceed 15.0 (same condition)
        if clearsky_irradiance > 50:
             # Check saturation BEFORE clipping
            norm_raw = np.asarray(data, dtype=float) / clearsky_irradiance
            # Consider it saturated if everything is at or above clearsky by a small tolerance
            if np.all(norm_raw >= (0.999)):
                # Near-degenerate Beta concentrated just below 1
                mu_tgt = 1.0 - 1e-6
                kappa0 = 1e6
                alpha = mu_tgt * kappa0
                beta_param = (1.0 - mu_tgt) * kappa0
                expected_beta = mu_tgt * clearsky_irradiance
                return (alpha, beta_param), expected_beta, clearsky_irradiance
            # Normalize and clip to (1e-7, 0.999999) as before
            normalized = np.clip(norm_raw, 1e-7, 0.999999)

            # 2a) Flat-bin guard: if all values equal (variance == 0), fall back
            if np.allclose(normalized, normalized[0], rtol=0.0, atol=1e-12):
                mu = float(normalized[0])
                # push off boundary if needed
                mu_tgt = min(max(mu, 1e-6), 1.0 - 1e-6)
                kappa0 = 1e6
                alpha = mu_tgt * kappa0
                beta_param = (1.0 - mu_tgt) * kappa0
                expected_beta = mu_tgt * clearsky_irradiance
                return (alpha, beta_param), expected_beta, clearsky_irradiance


            # Method-of-moments fit (unchanged API)
            alpha, beta_param = self.fit_beta_mom(normalized)

            # Guard for infinite sum (same behavior)
            if np.isinf(alpha + beta_param):
                return (1.0, 1000.0, np.nan, np.nan), 0.0, clearsky_irradiance

            # Expected value (same formula)
            expected_beta = alpha / (alpha + beta_param) * clearsky_irradiance
            return (alpha, beta_param), expected_beta, clearsky_irradiance

        # Fallback when not fitting (same constants/shape)
        return (1.0, 1000.0, np.nan, np.nan), 0.0, clearsky_irradiance
        
    
    @staticmethod
    def fit_beta_mom(data, eps: float = 1e-6):
        """
        Fit a Beta distribution to data in (0,1) using Method of Moments (MoM).

        Parameters
        ----------
        data : array-like
            1D array of observations, must lie in (0,1).
        eps : float, optional
            Small shift applied if values are exactly 0 or 1. Default 1e-6.

        Returns
        -------
        alpha : float
            Estimated alpha (shape1) parameter.
        beta : float
            Estimated beta (shape2) parameter.

        Raises
        ------
        ValueError
            If the variance is too large or if the computed parameters are invalid.
        """
        x = np.asarray(data, dtype=float)

        # Ensure within (0,1)
        x = np.clip(x, eps, 1 - eps)

        mu = np.mean(x)
        var = np.var(x, ddof=1)  # sample variance

        # Check feasibility condition
        if var >= mu * (1 - mu):
            raise ValueError(
                f"Invalid variance for Beta fit: var={var:.4f} >= mu*(1-mu)={mu*(1-mu):.4f}. TS={data.index[0]}"
            )
        if var == 0:
            print(data)
            raise ValueError(
                f"Invalid variance for Beta fit: var={var:.4f} = 0.0. TS={data.index[0]}"
            )
        kappa = mu * (1 - mu) / var - 1.0
        alpha = mu * kappa
        beta_param = (1 - mu) * kappa

        if alpha <= 0 or beta_param <= 0:
            raise ValueError(f"Invalid Beta parameters: alpha={alpha}, beta={beta}")

        return alpha, beta_param

    @staticmethod
    def _fit_weibull(data):
        if len(data) > 1:
            params = weibull_min.fit(data, floc=0)
            k, loc, scale = params
            expected = scale * gamma(1 + 1 / k)
        else:
            params, expected = (np.nan, np.nan, np.nan), np.nan
        return params, expected

# Example usage
if __name__ == "__main__":
    processor = WeatherDataProcessor()
    # latitude = [20.0,30.0,35.0,40.0,58.0]
    # longitude = [-159.0,-75.0,14.0,138.0,-161.0]

    latitude = [30.0]
    longitude = [-75.0]

    for i in range(len(latitude)):
        processor = WeatherDataProcessor()
        lat = latitude[i]
        lon = longitude[i]
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

        expected_data_filename = rf"Data\EXPECTED_DATA\data_expected_lat{lat:.1f}_lon{lon:.1f}_{timestep_min}min.pkl"
        processor.fit_distributions(resampled_df, expected_data_filename)