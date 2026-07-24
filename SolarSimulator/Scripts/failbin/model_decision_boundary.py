#!/usr/bin/env python3
"""model_decision_boundary.py -- derive the fly/no-fly wind boundary w* from the MODEL alone.

No threshold study, no Monte-Carlo.  We solve the bin-free IID MDP (cheap) and probe its
optimal policy: at a charged, moored state we sweep the hypothetical current wind and find
where the optimal action flips fly->idle.  Because only p_success(w) depends on wind, the
optimal policy is exactly a per-(state,stage) wind threshold; its median over the flyable
stages is a self-contained decision boundary for placing chain bins.

Usage:
    python Scripts/failbin/model_decision_boundary.py --only florida bering hawaii natlantic
"""
import argparse
import os
import sys

import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import SimulationFactory  # noqa: E402
from Scripts.failbin.failbin_experiment import LOCATIONS, FULL, INTERVAL_MIN, loc_paths, _rel  # noqa: E402


def build_iid_optimal(loc):
    """Construct and solve the bin-free IID optimal policy for a location."""
    paths = loc_paths(loc)
    config = dict(
        start_datetime=FULL["start_datetime"],
        delta_t=INTERVAL_MIN,
        transition_model="moderate",
        solar_panel_model="constant",
        whale_series="real",
        energy_increment_wh=5,
        # no wind_chain, no historical_weather -> pure IID distributional solve
    )
    loc_cfg = dict(latitude=loc["lat"], longitude=loc["lon"], data_path=os.path.abspath(paths["exp"]))
    factory = SimulationFactory(config, loc_cfg, FULL["horizon"], FULL["failure_penalties"][0])
    sim = factory.create_simulation(sim_type="optimal", cap=FULL["battery_capacities"][0])
    return sim


def decision_wind_star(sim, horizon, soc_levels=(100.0, 80.0, 60.0),
                       wgrid=np.linspace(0.0, 25.0, 126), stage_step=8):
    """Median fly/no-fly wind over charged, moored states across the mission.

    For each (SoC, stage) we find the largest wind at which the optimal action is still
    'fly' (action 1) from the moored state.  Stages where the policy never flies (no
    observation opportunity) contribute nothing.  solar=0 -> conservative (no charging credit).
    """
    boundaries = []
    for soc in soc_levels:
        for t in range(0, horizon, stage_step):
            flies = np.array([
                sim.choose_action(np.array([soc, 0]), 0.0, float(w), 1.0, t) == 1
                for w in wgrid
            ])
            if flies.any():
                boundaries.append(float(wgrid[np.where(flies)[0].max()]))
    boundaries = np.array(boundaries)
    return boundaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    locs = LOCATIONS if not args.only else [l for l in LOCATIONS if l["name"] in args.only]
    print(f"{'location':10s} {'w*_median':>9s} {'w*_p25':>7s} {'w*_p75':>7s} {'n_flyable_stages':>17s}")
    for loc in locs:
        sim = build_iid_optimal(loc)
        b = decision_wind_star(sim, FULL["horizon"])
        if len(b):
            print(f"{loc['name']:10s} {np.median(b):9.2f} {np.percentile(b,25):7.2f} "
                  f"{np.percentile(b,75):7.2f} {len(b):17d}")
        else:
            print(f"{loc['name']:10s} {'--':>9s}  (policy never flies at probed states)")


if __name__ == "__main__":
    main()
