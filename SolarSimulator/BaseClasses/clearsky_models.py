"""Pluggable clear-sky GHI models for the solar pipeline.

Every stage of the solar pipeline (Beta fits, solar-chain fitting, run-time
`solar_cs`) normalizes irradiance by a clear-sky envelope G_cs(t). This module
is the single place that envelope is computed, selected by name:

    fatemi    A * max(cos(apparent zenith), 0), A = 1150 W/m^2
              (Fatemi-Kuh-Fripp 2018; the original model, and the default).
    haurwitz  pvlib.clearsky.haurwitz -- empirical GHI from zenith only.
    ineichen  pvlib Ineichen-Perez with Linke-turbidity lookup; physically the
              tightest envelope of the three (K = GHI/G_cs stays <= ~1).

All three use pvlib solar position (apparent zenith). The expected-data and
solar-chain artifacts a simulation consumes must be built with the SAME model;
`run_sim._resolve_data_path` keeps non-fatemi artifacts in separate files.
"""
import numpy as np
import pandas as pd
import pvlib

CLEARSKY_MODELS = ("fatemi", "haurwitz", "ineichen")
FATEMI_A = 1150.0


def clearsky_ghi_series(
    times,
    lat: float,
    lon: float,
    model: str = "fatemi",
    A: float = FATEMI_A,
) -> np.ndarray:
    """Clear-sky GHI [W/m^2] at each timestamp, as a plain ndarray.

    Args:
        times (pd.DatetimeIndex or pd.Timestamp): Timestamps (tz-aware preferred;
            naive is treated as such by pvlib, matching the historical record).
        lat, lon (float): Latitude / longitude in degrees (east positive).
        model (str): One of CLEARSKY_MODELS (default "fatemi").
        A (float): Fatemi scale constant (ignored by the pvlib models).

    Returns:
        np.ndarray: Clear-sky GHI, zero at night, same length as `times`.
    """
    if isinstance(times, pd.Timestamp):
        times = pd.DatetimeIndex([times])
    elif not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    if model == "fatemi":
        sp = pvlib.solarposition.get_solarposition(times, lat, lon)
        cosz = np.cos(np.deg2rad(sp["apparent_zenith"].to_numpy()))
        return A * np.clip(cosz, 0.0, None)
    if model == "haurwitz":
        sp = pvlib.solarposition.get_solarposition(times, lat, lon)
        return np.clip(
            pvlib.clearsky.haurwitz(sp["apparent_zenith"])["ghi"].to_numpy(), 0.0, None
        )
    if model == "ineichen":
        loc = pvlib.location.Location(lat, lon, altitude=0.0)
        return np.clip(
            loc.get_clearsky(times, model="ineichen")["ghi"].to_numpy(), 0.0, None
        )
    raise ValueError(f"Unknown clearsky model {model!r}; choose from {CLEARSKY_MODELS}")
