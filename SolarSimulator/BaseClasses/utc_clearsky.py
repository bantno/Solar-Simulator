# """
# Fast utilities for average clear-sky GHI over 15‑minute intervals using the
# pvlib Haurwitz model. Includes:

# - average_clearsky_ghi_haurwitz: dense sampling + mean (exact but slower)
# - avg_ghi_haurwitz_midpoint: midpoint approximation (fast, accurate)
# - precompute_ghi_1min_haurwitz: 1‑minute clearsky series for a span
# - average_ghi_15min_from_precomp: rolling 15‑min averages from precomp
# - HaurwitzClearskyAverager: cached, high‑throughput interface

# All functions accept *local* timestamps via tz (IANA string). If you already
# have UTC clock values, pass tz="UTC".
# """
# from __future__ import annotations

# from dataclasses import dataclass
# from functools import lru_cache
# from typing import Optional

# import numpy as np
# import pandas as pd
# import pvlib

# __all__ = [
#     "average_clearsky_ghi_haurwitz",
#     "avg_ghi_haurwitz_midpoint",
#     "precompute_ghi_1min_haurwitz",
#     "average_ghi_15min_from_precomp",
#     "HaurwitzClearskyAverager",
# ]


# def _scalar(x) -> float:
#     """Safely convert a 1‑element pandas object or numpy scalar to float.
#     Avoids FutureWarning from float(Series).
#     """
#     if hasattr(x, "item"):
#         try:
#             return float(x.item())
#         except Exception:
#             pass
#     if hasattr(x, "iloc"):
#         return float(x.iloc[0])
#     return float(x)


# def average_clearsky_ghi_haurwitz(
#     lat: float,
#     lon: float,
#     year: int,
#     month: int,
#     day: int,
#     hour: int,
#     minute: int,
#     tz: str = "America/New_York",
#     sample_every_seconds: int = 60,
# ) -> float:
#     """
#     Compute the average clear-sky GHI (W/m^2) over the 15-minute interval
#     starting at the given local time using the Haurwitz model.

#     Parameters
#     ----------
#     lat, lon : float
#         Latitude and longitude in degrees (north/east positive).
#     year, month, day, hour, minute : int
#         Start of the 15-minute interval (local clock time).
#     tz : str, default "America/New_York"
#         IANA timezone string for the *local* time you provided above.
#         (If you already have UTC, pass tz="UTC" and the UTC clock values.)
#     sample_every_seconds : int, default 60
#         Temporal resolution for the numerical average. 60 s is usually
#         sufficient. Increase to 300 for faster, or 10 for more accuracy
#         near sunrise/sunset.

#     Returns
#     -------
#     float
#         Average clear-sky GHI (W/m^2) over the interval.
#     """
#     start = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, tz=tz)

#     times = pd.date_range(
#         start=start,
#         end=start + pd.Timedelta(minutes=15),
#         freq=f"{sample_every_seconds}s",
#         inclusive="left",
#     )

#     sp = pvlib.solarposition.get_solarposition(times, lat, lon)
#     zenith = sp["apparent_zenith"]
#     ghi_cs = pvlib.clearsky.haurwitz(zenith)

#     avg = ghi_cs.mean()
#     return _scalar(avg)


# def avg_ghi_haurwitz_midpoint(
#     lat: float,
#     lon: float,
#     year: int,
#     month: int,
#     day: int,
#     hour: int,
#     minute: int,
#     tz: str = "America/New_York",
# ) -> float:
#     """Fast midpoint approximation for the 15-minute average.

#     Typically within ~1–2 W/m^2 of the 60 s average except near sunrise/sunset.
#     """
#     start = pd.Timestamp(year, month, day, hour, minute, tz=tz)
#     mid = start + pd.Timedelta(minutes=7, seconds=30)
#     sp = pvlib.solarposition.get_solarposition(mid, lat, lon)
#     ghi = pvlib.clearsky.haurwitz(sp["apparent_zenith"])
#     return _scalar(ghi)


# def precompute_ghi_1min_haurwitz(
#     lat: float,
#     lon: float,
#     start: pd.Timestamp,
#     end: pd.Timestamp,
# ) -> pd.Series:
#     """
#     Precompute Haurwitz clearsky GHI at 1-minute resolution for [start, end).

#     Parameters
#     ----------
#     start, end : tz-aware pandas Timestamps
#         Use inclusive start and exclusive end (left-closed interval).
#     """
#     if start.tz is None or end.tz is None:
#         raise ValueError("start/end must be tz-aware Timestamps.")
#     times = pd.date_range(start=start, end=end, freq="1min", inclusive="left")
#     sp = pvlib.solarposition.get_solarposition(times, lat, lon)
#     ghi_1min = pvlib.clearsky.haurwitz(sp["apparent_zenith"]).astype(float)
#     ghi_1min.name = "ghi_cs"
#     return ghi_1min


# def average_ghi_15min_from_precomp(
#     ghi_1min: pd.Series, interval_starts: pd.DatetimeIndex
# ) -> pd.Series:
#     """
#     From a 1-minute clearsky GHI series, return left-aligned 15-minute means at
#     the provided interval starts.

#     Notes
#     -----
#     Uses a left-closed rolling window "[t, t+15min)" to match discrete slots
#     that *start* at each index time.
#     """
#     if not isinstance(ghi_1min.index, pd.DatetimeIndex):
#         raise TypeError("ghi_1min must be indexed by a DatetimeIndex.")
#     rolling = ghi_1min.rolling("15min", closed="left").mean()
#     return rolling.reindex(interval_starts)


# @dataclass(frozen=True)
# class _Key:
#     lat: float
#     lon: float
#     tz: str
#     year: int


# @lru_cache(maxsize=16)
# def _cached_precomp(key: _Key) -> pd.Series:
#     tz = key.tz
#     start = pd.Timestamp(key.year, 1, 1, 0, 0, tz=tz)
#     end = pd.Timestamp(key.year + 1, 1, 1, 0, 0, tz=tz)
#     return precompute_ghi_1min_haurwitz(key.lat, key.lon, start, end)


# class HaurwitzClearskyAverager:
#     """High-throughput 15‑minute clearsky averaging with automatic caching.

#     Example
#     -------
#     averager = HaurwitzClearskyAverager(lat=25.7617, lon=-80.1918, tz="America/New_York")
#     avg = averager.get_avg_15min(2021, 6, 21, 12, 0)
#     """

#     def __init__(self, lat: float, lon: float, tz: str = "America/New_York"):
#         self.lat = float(lat)
#         self.lon = float(lon)
#         self.tz = tz

#     def get_avg_15min(self, year: int, month: int, day: int, hour: int, minute: int) -> float:
#         """Return average clearsky GHI (W/m^2) for the 15-minute slot starting at
#         the given local time (tz from constructor).

#         Falls back to midpoint if the rolling window is undefined at the very
#         start of the year.
#         """
#         key = _Key(self.lat, self.lon, self.tz, year)
#         ghi_1min = _cached_precomp(key)

#         start = pd.Timestamp(year, month, day, hour, minute, tz=self.tz)
#         # Single-value Series at the requested start time
#         avg_series = average_ghi_15min_from_precomp(ghi_1min, pd.DatetimeIndex([start]))

#         # Extract a true scalar before any checks to avoid ambiguity
#         avg = avg_series.iloc[0]
#         try:
#             avg_val = _scalar(avg)  # -> float (may be nan)
#         except Exception:
#             # ultra-defensive: fall back if anything odd happens
#             return avg_ghi_haurwitz_midpoint(self.lat, self.lon, year, month, day, hour, minute, tz=self.tz)

#         if np.isnan(avg_val):
#             # Early-year edge or empty window -> midpoint fallback
#             return avg_ghi_haurwitz_midpoint(self.lat, self.lon, year, month, day, hour, minute, tz=self.tz)

#         return float(avg_val)

# import numpy as np
# import pandas as pd
# import pvlib

# def clearsky_ghi_fatemi(times, lat, lon, A=1150.0):
#     """
#     Clear-sky GHI per Fatemi–Kuh–Fripp (2018):
#         GHI_cs(t) = A * cos(zenith(t))
#     where 'A' is a fixed scale (paper uses ~1150 W/m^2) and
#     zenith is the apparent solar zenith angle.

#     Parameters
#     ----------
#     times : pandas.DatetimeIndex or pd.Timestamp
#         TZ-aware timestamps (local or UTC). If you pass naive times,
#         pvlib will treat them as naive; prefer tz-aware.
#     lat, lon : float
#         Latitude [deg], Longitude [deg] (east positive).
#     A : float, default 1150.0
#         Scaling constant so that normalized irradiance r/(A cos z) lies in (0,1).

#     Returns
#     -------
#     pandas.Series
#         Clear-sky GHI [W/m^2], clipped at 0 at night.
#     """
#     # Ensure we always work with a DatetimeIndex
#     if isinstance(times, pd.Timestamp):
#         times = pd.DatetimeIndex([times])
#     elif not isinstance(times, pd.DatetimeIndex):
#         times = pd.DatetimeIndex(times)

#     sp = pvlib.solarposition.get_solarposition(times, lat, lon)
#     zen = sp["apparent_zenith"].to_numpy()  # degrees

#     # cos(zenith) with zenith in degrees; negative at night -> 0
#     cosz = np.cos(np.deg2rad(zen))
#     cosz = np.clip(cosz, 0.0, None)

#     ghi_cs = A * cosz
#     return pd.Series(ghi_cs, index=times, name="ghi_cs_fatemi")


# if __name__ == "__main__":
#     # Quick self-test / usage examples
#     lat, lon = 25.7617, -80.1918  # Miami, FL
#     tz = "America/New_York"

#     # # 1) One-off midpoint estimate
#     # m = avg_ghi_haurwitz_midpoint(lat, lon, 2021, 6, 21, 12, 0, tz)
#     # print(f"Midpoint clearsky GHI ~15min avg: {m:.2f} W/m^2")

#     # # 2) Dense average with 60s sampling
#     # a = average_clearsky_ghi_haurwitz(lat, lon, 2021, 6, 21, 12, 0, tz, sample_every_seconds=60)
#     # print(f"60s sampled 15min avg: {a:.2f} W/m^2")

#     # # 3) Precompute for a year and get rolling 15-min means quickly
#     # start = pd.Timestamp(2021, 1, 1, 0, 0, tz=tz)
#     # end = pd.Timestamp(2022, 1, 1, 0, 0, tz=tz)
#     # ghi_1min = precompute_ghi_1min_haurwitz(lat, lon, start, end)
#     # starts_15 = pd.date_range(start=start, end=end, freq="15min", inclusive="left")
#     # avg_15_series = average_ghi_15min_from_precomp(ghi_1min, starts_15)
#     # print(f"Yearly 15-min averages computed: {avg_15_series.notna().sum().ghi} intervals")

#     # # 4) Cached high-throughput interface
#     # averager = HaurwitzClearskyAverager(lat, lon, tz)
#     # c = averager.get_avg_15min(2021, 6, 21, 12, 0)
#     # print(f"Cached averager result: {c:.2f} W/m^2")
#     # c = averager.get_avg_15min(2021, 1, 21, 12, 0)
#     # print(f"Cached averager result: {c:.2f} W/m^2")




#     tz = "America/New_York"
#     t = pd.date_range("2018-03-09 12:00", periods=5, freq="15min", tz=tz)
#     cs = clearsky_ghi_fatemi(t, lat=25.7617, lon=-80.1918, A=1150.0)
#     print(cs.head())

import numpy as np
import pandas as pd
import pvlib

# -------------------------------
# Fatemi clear-sky: A * cos(zenith)
# -------------------------------
def clearsky_ghi_fatemi(times, lat, lon, A=1150.0):
    if isinstance(times, pd.Timestamp):
        times = pd.DatetimeIndex([times])
    sp = pvlib.solarposition.get_solarposition(times, lat, lon)
    cosz = np.cos(np.deg2rad(sp["apparent_zenith"].to_numpy()))
    cosz = np.clip(cosz, 0.0, None)
    return pd.Series(A * cosz, index=times, name="ghi_cs_fatemi")

# ---------------------------------------------
# Haurwitz rolling-average for a 15-min interval
# ---------------------------------------------
def precompute_ghi_1min_haurwitz(lat, lon, start, end):
    if start.tz is None or end.tz is None:
        raise ValueError("start/end must be tz-aware Timestamps.")
    times = pd.date_range(start=start, end=end, freq="1min", inclusive="left")
    sp = pvlib.solarposition.get_solarposition(times, lat, lon)
    ghi_1min = pvlib.clearsky.haurwitz(sp["apparent_zenith"]).astype(float)
    ghi_1min.name = "ghi_cs_haurwitz"
    return ghi_1min

def average_ghi_15min_from_precomp(ghi_1min, interval_starts):
    rolling = ghi_1min.rolling("15min", closed="left").mean()
    return rolling.reindex(interval_starts)

# -------------------------------
# Comparison at local noon
# -------------------------------
def compare_clearsky_noon(lat, lon, tz="America/New_York",
                          june_date="2021-06-21", jan_date="2021-01-15",
                          A=1150.0):
    # Build local-noon timestamps
    t_june_noon = pd.Timestamp(june_date + " 12:00", tz=tz)
    t_jan_noon  = pd.Timestamp(jan_date  + " 12:00", tz=tz)

    # Fatemi at noon (instantaneous)
    fatemi_june = float(clearsky_ghi_fatemi(t_june_noon, lat, lon, A=A).iloc[0])
    fatemi_jan  = float(clearsky_ghi_fatemi(t_jan_noon,  lat, lon, A=A).iloc[0])

    # Haurwitz 15-min average over [noon, noon+15min)
    # Precompute 1-min for each day (midnight-to-midnight)
    for t in [t_june_noon, t_jan_noon]:
        if t.hour != 12 or t.minute != 0:
            raise ValueError("This helper expects exactly 12:00 local times.")

    # June day
    start_j = t_june_noon.normalize()            # 00:00 that day
    end_j   = start_j + pd.Timedelta(days=1)
    ghi_j   = precompute_ghi_1min_haurwitz(lat, lon, start_j, end_j)
    haur_j  = float(average_ghi_15min_from_precomp(ghi_j, pd.DatetimeIndex([t_june_noon])).iloc[0])

    # January day
    start_w = t_jan_noon.normalize()
    end_w   = start_w + pd.Timedelta(days=1)
    ghi_w   = precompute_ghi_1min_haurwitz(lat, lon, start_w, end_w)
    haur_w  = float(average_ghi_15min_from_precomp(ghi_w, pd.DatetimeIndex([t_jan_noon])).iloc[0])

    # Assemble a small report
    df = pd.DataFrame({
        "When": ["June noon", "January noon"],
        "Fatemi A·cos(z) [W/m²]": [fatemi_june, fatemi_jan],
        "Haurwitz 15-min avg [W/m²]": [haur_j, haur_w],
    })
    df["Diff (Haurwitz - Fatemi) [W/m²]"] = df["Haurwitz 15-min avg [W/m²]"] - df["Fatemi A·cos(z) [W/m²]"]
    df["Percent Diff [%]"] = 100 * df["Diff (Haurwitz - Fatemi) [W/m²]"] / df["Fatemi A·cos(z) [W/m²]"].replace(0, np.nan)

    return df

# -------------------------------
# Example: Miami, FL
# -------------------------------
if __name__ == "__main__":
    # lat, lon = 25.7617, -80.1918
    # tz = "America/New_York"
    # out = compare_clearsky_noon(lat, lon, tz=tz,
    #                             june_date="2021-06-21",
    #                             jan_date="2021-01-15",
    #                             A=1150.0)
    # print(out.to_string(index=False))
