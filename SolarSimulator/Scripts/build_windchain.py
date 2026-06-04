"""
Build a wind Markov-chain artifact for one location.

Usage (pvlib conda env, from the SolarSimulator dir or repo root):

    # Explicit aircraft-based thresholds (recommended):
    python Scripts/build_windchain.py \\
        --historical  Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --out         Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min_windchain.pkl \\
        --bin-edges   5.0 10.0

    # Equal-occupancy quantile bins (data-driven fallback):
    python Scripts/build_windchain.py \\
        --historical  Data/HISTORICAL_DATA/data_30_-90.pkl \\
        --out         Data/EXPECTED_DATA/data_expected_lat30.0_lon-90.0_15min_windchain.pkl \\
        --n-bins      3

--bin-edges takes precedence over --n-bins when both are given.
Interior cutpoints only (e.g. 5.0 10.0); 0 and inf are added automatically.

The artifact is a pickle dict with keys:
    n_bins                   int
    bin_edges                np.ndarray, shape (n_bins+1,)  — full array incl. 0 and inf
    conditioning             ('month', 'hour')
    transition_by_month_hour np.ndarray, shape (13, 24, n_bins, n_bins)
"""
import argparse
import os
import sys

import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from Scripts.create_weather_distributions import build_wind_chain_artifact  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build a wind Markov-chain artifact.")
    ap.add_argument("--historical", required=True, help="Path to hourly HISTORICAL_DATA pickle.")
    ap.add_argument("--out", required=True, help="Output path for the artifact pickle.")
    ap.add_argument("--dt", type=int, default=15, help="Model timestep in minutes (default 15).")
    ap.add_argument(
        "--bin-edges", type=float, nargs="+", metavar="M_S",
        help="Interior cutpoints in m/s (e.g. 5.0 10.0). "
             "Defines aircraft-based bins; overrides --n-bins.",
    )
    ap.add_argument(
        "--n-bins", type=int, default=3,
        help="Number of equal-occupancy quantile bins (default 3). "
             "Ignored when --bin-edges is given.",
    )
    args = ap.parse_args()

    bin_edges = None
    if args.bin_edges is not None:
        interior = np.asarray(args.bin_edges, dtype=float)
        bin_edges = np.concatenate(([0.0], interior, [np.inf]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build_wind_chain_artifact(
        historical_pkl=args.historical,
        out_path=args.out,
        interval_minutes=args.dt,
        n_bins=args.n_bins,
        bin_edges=bin_edges,
    )


if __name__ == "__main__":
    main()
