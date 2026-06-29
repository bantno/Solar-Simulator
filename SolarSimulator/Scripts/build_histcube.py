"""Build historical-weather calendar cube artifacts for one or more locations.

Usage (pvlib conda env, from the SolarSimulator dir or repo root):

    conda run -n pvlib python Scripts/build_histcube.py \\
        --historical  Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --out         Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min_histcube.pkl \\
        --dt          15

The cube is saved as a pickle dict with keys:
    wind_cube   (slots_per_year, n_years)  float64
    solar_cube  (slots_per_year, n_years)  float64
    years       list[int]
    n_years     int
    slots_per_year  int
    delta_t_min     int
"""
import argparse
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from Scripts.create_weather_distributions import build_historical_cube_artifact  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build a historical-weather calendar cube artifact.")
    ap.add_argument("--historical", required=True, help="Path to hourly HISTORICAL_DATA pickle.")
    ap.add_argument("--out", required=True, help="Output path for the cube artifact pickle.")
    ap.add_argument("--dt", type=int, default=15, help="Model timestep in minutes (default 15).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build_historical_cube_artifact(
        historical_pkl=args.historical,
        out_path=args.out,
        interval_minutes=args.dt,
    )


if __name__ == "__main__":
    main()
