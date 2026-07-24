#!/usr/bin/env python3
"""generate_dim_sweeps.py -- configs for the capstone comparison: chain(12 bins) vs IID vs
threshold, swept over failure penalty, battery capacity, and mission duration, at all 4 sites.

OFAT design around the validated baseline (pen=20, 300Wh, 30 days). Each sweep x arm is ONE
config carrying value lists (the runner builds the product and parallelizes solves), so the
whole study is 9 configs. All arms roll out on the same historical block-bootstrap protocol.

Arm naming for the analyzer: sweep_<dim>_<arm>.yaml, arm in {iid, chain12, thresh}.
"""
import os
import sys

import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from Scripts.failbin.failbin_experiment import LOCATIONS, loc_paths, _rel, REPO_DIR  # noqa: E402

OUT_DIR = os.path.join(REPO_DIR, "configs", "failbin_validation", "dim_sweeps")

BASE = dict(penalty=20.0, capacity=300.0, days=30)
SWEEPS = {
    "penalty":  dict(failure_penalties=[5.0, 10.0, 20.0, 40.0],
                     battery_capacities=[BASE["capacity"]], days=[BASE["days"]]),
    "battery":  dict(failure_penalties=[BASE["penalty"]],
                     battery_capacities=[150.0, 300.0, 450.0, 600.0], days=[BASE["days"]]),
    "duration": dict(failure_penalties=[BASE["penalty"]],
                     battery_capacities=[BASE["capacity"]], days=[7, 14, 30, 60]),
}
THRESH_OBS = [0.0, 0.1, 0.2, 0.3]
THRESH_WIND = [0.0, 4.0, 8.0, 12.0]
N_BINS = 12
EPISODES = 20000
DELTA_T = 15


def days_to_stages(d):
    return int(d * 24 * 60 // DELTA_T)


def base_cfg(name, sweep):
    return dict(
        experiment_name=name,
        description=f"dim sweep {name}: chain({N_BINS} bins) vs IID vs threshold, 4 sites",
        start_datetime="2025-06-10T00:00:00",
        battery_capacities=sweep["battery_capacities"],
        horizons=[days_to_stages(d) for d in sweep["days"]],
        failure_penalties=sweep["failure_penalties"],
        episodes=EPISODES,
        transition_model="moderate",
        solar_panel_model="constant",
        whale_series="real",
        energy_increment_wh=5,
        delta_t=DELTA_T,
        save_states=False,
        full_history_episodes=0,
        locations=[dict(latitude=l["lat"], longitude=l["lon"], data_path=_rel(loc_paths(l)["exp"]))
                   for l in LOCATIONS],
        historical_weather=dict(enabled=True, block_length_days=7),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dim, sweep in SWEEPS.items():
        # IID-optimal
        c = base_cfg(f"sweep_{dim}_iid", sweep)
        c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
                 wind_chain=dict(enabled=False))
        written.append(_write(c, f"sweep_{dim}_iid.yaml"))
        # Chain-optimal, 12 equal-occupancy bins (path auto-derived per location).
        c = base_cfg(f"sweep_{dim}_chain12", sweep)
        c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
                 wind_chain=dict(enabled=True, n_bins=N_BINS))
        written.append(_write(c, f"sweep_{dim}_chain12.yaml"))
        # Threshold grid (best-of reported downstream).
        c = base_cfg(f"sweep_{dim}_thresh", sweep)
        c.update(include_optimal=False, threshold_values=THRESH_OBS, wind_thresholds=THRESH_WIND,
                 wind_chain=dict(enabled=False))
        written.append(_write(c, f"sweep_{dim}_thresh.yaml"))
    for w in written:
        print("wrote", _rel(w))


def _write(cfg, fname):
    p = os.path.join(OUT_DIR, fname)
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


if __name__ == "__main__":
    main()
