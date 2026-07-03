"""Build a solar (clear-sky index) Markov-chain artifact for one location.

Usage (pvlib conda env, from the SolarSimulator dir or repo root):

    python Scripts/build_solarchain.py \\
        --historical  Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --lat 30.0 --lon -90.0 \\
        --out         Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min_solarchain.pkl \\
        --n-bins      3

Unlike the wind chain, --lat/--lon are required: the clear-sky index normalization
needs the solar zenith angle (pvlib).

Bins are STAGE-RELATIVE quantile bands (bin g = the [g/n, (g+1)/n) quantile band of
each stage's own index distribution), not global cutpoints. Global edges are unusable
for solar: the hour-averaged GHI record biases the index low near sunrise/sunset, so
globally-binned dusk/dawn slots collapse into the bottom bin under every weather
regime and the day-to-day persistence channel carries nothing. There is therefore no
--bin-edges option; --n-bins is the resolution knob.

The artifact is a pickle dict with keys:
    kind                       'solar'
    n_bins                     int
    bin_mode                   'stage_quantile'
    conditioning               ('month', 'hour')
    transition_by_month_hour   np.ndarray, shape (13, 24, n_bins, n_bins)   — intra-day
    dawn_transition_by_month   np.ndarray, shape (13, n_bins, n_bins)       — dusk(d)→dawn(d+1)
    valid_threshold_wm2        float — G_cs gate defining where the index (and chain) exists
"""
import argparse
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from Scripts.create_weather_distributions import build_solar_chain_artifact  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build a solar clear-sky-index Markov-chain artifact.")
    ap.add_argument("--historical", required=True, help="Path to hourly HISTORICAL_DATA pickle.")
    ap.add_argument("--out", required=True, help="Output path for the artifact pickle.")
    ap.add_argument("--lat", type=float, required=True, help="Latitude [deg].")
    ap.add_argument("--lon", type=float, required=True, help="Longitude [deg, east positive].")
    ap.add_argument("--dt", type=int, default=15, help="Model timestep in minutes (default 15).")
    ap.add_argument(
        "--n-bins", type=int, default=3,
        help="Number of stage-relative quantile bins (default 3).",
    )
    ap.add_argument(
        "--valid-threshold", type=float, default=200.0,
        help="G_cs gate [W/m^2] defining where the index/chain exists (default 200; "
             "see build_solar_chain_artifact docstring).",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build_solar_chain_artifact(
        historical_pkl=args.historical,
        out_path=args.out,
        latitude=args.lat,
        longitude=args.lon,
        interval_minutes=args.dt,
        n_bins=args.n_bins,
        valid_threshold_wm2=args.valid_threshold,
    )


if __name__ == "__main__":
    main()
