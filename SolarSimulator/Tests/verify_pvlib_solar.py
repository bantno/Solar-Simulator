"""Verification for the pluggable clear-sky models and the PVWatts panel.

Checks:
  1. Backward compatibility of the clear-sky refactor: the "fatemi" path of
     clearsky_ghi_series reproduces the original A*cos(z) formula exactly, and
     _fit_beta with the default model reproduces the clearsky_irradiance values
     stored in the existing expected-data pickle.
  2. Model sanity: haurwitz and ineichen are zero at night, positive at midday,
     and sit below the fatemi envelope at noon (fatemi's A=1150 is the loosest).
  3. PVWatts panel: defaults reproduce the "constant" panel exactly (area 0.65,
     efficiency 0.1); a hotter cell derates by 1 + gamma_pdc*(T-25); the factory
     kwargs pass-through reaches the panel.
  4. Provider equivalence: StochasticWindSolarEnvironmentProvider built with
     solar_panel_model="pvwatts" (defaults) produces bit-identical sample_sunlight
     draws to the "constant" build at the same seed.
  5. Path resolution: _resolve_data_path is identity for fatemi and suffixes
     other models before the extension.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_pvlib_solar.py
"""
import os
import sys

import numpy as np
import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.clearsky_models import clearsky_ghi_series, FATEMI_A  # noqa: E402
from BaseClasses.solar_panel_base import SolarPanelFactory  # noqa: E402
from BaseClasses.environment_provider_base import StochasticWindSolarEnvironmentProvider  # noqa: E402
from BaseClasses.run_sim import _resolve_data_path  # noqa: E402

DATA_PKL = os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                        "data_expected_lat30.0_lon-90.0_15min.pkl")
LAT, LON = 30.0, -90.0


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        sys.exit(1)


def main():
    times = pd.date_range("2020-06-21 00:00", "2020-06-21 23:45",
                          freq="15min", tz="Etc/GMT+6")

    # 1a. fatemi path == original formula
    import pvlib
    sp = pvlib.solarposition.get_solarposition(times, LAT, LON)
    expected = FATEMI_A * np.clip(np.cos(np.deg2rad(sp["apparent_zenith"].to_numpy())), 0.0, None)
    fatemi = clearsky_ghi_series(times, LAT, LON, model="fatemi")
    check("fatemi model matches original A*cos(z) formula",
          np.array_equal(fatemi, expected))

    # 1b. _fit_beta default reproduces the stored clearsky_irradiance for a slot
    if os.path.exists(DATA_PKL):
        df = pd.read_pickle(DATA_PKL)
        row = df[(df["month"] == 6) & (df["day"] == 21)
                 & (df["hour"] == 12) & (df["minute"] == 0)].iloc[0]
        # The pickle's fit used the slot group's first timestamp: 1950 (the record
        # start) at the record's fixed UTC-5 offset. Reproducing it exactly proves
        # the refactored fatemi path is the same computation.
        ts0 = pd.Timestamp("1950-06-21 12:00", tz="Etc/GMT+5")
        cs = float(clearsky_ghi_series(ts0, LAT, LON, model="fatemi")[0])
        stored = float(row["clearsky_irradiance"])
        check(f"fatemi clearsky reproduces stored pickle value exactly "
              f"({cs:.6f} vs {stored:.6f})",
              np.isclose(cs, stored, rtol=1e-12, atol=0))
    else:
        print(f"[SKIP] expected-data pickle not found at {DATA_PKL}")

    # 2. haurwitz / ineichen sanity
    noon = np.argmax(fatemi)
    night = 0  # local midnight
    for model in ("haurwitz", "ineichen"):
        cs = clearsky_ghi_series(times, LAT, LON, model=model)
        check(f"{model}: zero at night", cs[night] == 0.0)
        check(f"{model}: positive at midday", cs[noon] > 500.0)
        check(f"{model}: below fatemi envelope at noon", cs[noon] < fatemi[noon])

    # 3. PVWatts panel
    const = SolarPanelFactory.create_solar_panel("constant")
    pvw = SolarPanelFactory.create_solar_panel("pvwatts")
    check("pvwatts defaults reproduce constant panel (area)", pvw.area == const.area)
    check("pvwatts defaults reproduce constant panel (efficiency)",
          np.isclose(pvw.efficiency, const.efficiency, rtol=0, atol=1e-12))
    hot = SolarPanelFactory.create_solar_panel("pvwatts", cell_temperature=45.0)
    derate = hot.efficiency / pvw.efficiency
    check(f"pvwatts thermal derate at 45C = 1+gamma*20 (got {derate:.4f})",
          np.isclose(derate, 1.0 - 0.003 * 20.0))
    big = SolarPanelFactory.create_solar_panel("pvwatts", pdc0=130.0)
    check("pvwatts kwargs pass-through (pdc0)",
          np.isclose(big.efficiency, 2 * pvw.efficiency))

    # 4. Provider equivalence, constant vs pvwatts defaults
    H = 96
    solar_dist = np.column_stack((np.full(H, 2.0), np.full(H, 5.0), np.full(H, 800.0)))
    wind_dist = np.column_stack((np.full(H, 2.0), np.full(H, 6.0)))
    whales = np.zeros(H)

    def draws(model, kwargs=None):
        env = StochasticWindSolarEnvironmentProvider(
            solar_distributions=solar_dist, wind_distributions=wind_dist,
            whale_reward_series=whales, delta_t_min=15.0,
            solar_panel_model=model, solar_panel_kwargs=kwargs)
        env.set_seed(1234)
        return np.concatenate([env.sample_sunlight(t, 8) for t in range(H)])

    check("provider: pvwatts defaults bit-identical to constant",
          np.array_equal(draws("constant"), draws("pvwatts")))
    check("provider: hot pvwatts cell scales energy by the derate",
          np.allclose(draws("pvwatts", {"cell_temperature": 45.0}),
                      draws("constant") * (1.0 - 0.003 * 20.0)))

    # 5. Path resolution
    p = r"Data\EXPECTED_DATA\data_expected_lat30.0_lon-90.0_15min.pkl"
    check("resolve: fatemi keeps path", _resolve_data_path(p, "fatemi") == p)
    check("resolve: ineichen suffixes path",
          _resolve_data_path(p, "ineichen").endswith("_15min_ineichen.pkl"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
