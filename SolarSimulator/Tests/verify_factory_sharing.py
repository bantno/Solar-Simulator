"""Verification that sharing a SimulationFactory across a case's simulations changes nothing.

YAMLSimulationRunner._build_param_list builds one factory per (start, location, horizon,
penalty, kind) and reuses it for every capacity x threshold combo, instead of one factory
per simulation (the data load dominated sweep setup time). This check builds the same
simulations both ways, runs them serially (the strictest case: shared providers really
are shared objects), and requires bit-identical per-episode rewards, failure steps and
flight hours. It also checks that the shared build creates one factory per case.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_factory_sharing.py
"""
import os
import sys
import tempfile

import numpy as np
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import YAMLSimulationRunner, SimulationFactory  # noqa: E402

EPISODES = 300
CFG = f"""
experiment_name: factory_sharing_verify
start_datetimes: ['2020-07-01T00:00:00']
battery_capacities: [250.0, 300.0]
threshold_values: [0.1, 0.25]
wind_thresholds: [3.0, 6.0]
horizons: [96]
failure_penalties: [5.0, 20.0]
episodes: {EPISODES}
transition_model: moderate
solar_panel_model: constant
whale_series: JoA2026
energy_increment_wh: 5
delta_t: 15
include_optimal: true
holdout: false   # mechanics check on the full-record artifacts, not an evaluation
locations:
- latitude: 30.0
  longitude: -90.0
  data_path: {os.path.join(REPO_ROOT, 'Data', 'EXPECTED_DATA', 'data_expected_lat30.0_lon-90.0_15min.pkl')}
wind_chain: {{enabled: true, n_bins: 4}}
historical_weather: {{enabled: true, block_length_days: 7}}
"""


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        sys.exit(1)


def run(sims):
    out = []
    for sim in sims:
        eps = list(sim.simulate_multiple_episodes(EPISODES))
        out.append((np.array([e["total_reward"] for e in eps]),
                    np.array([e["failure_step"] for e in eps]),
                    np.array([e["flight_hrs"] for e in eps])))
    return out


def main():
    with tempfile.TemporaryDirectory(prefix="verify_factory_") as tmp:
        cfg_path = os.path.join(tmp, "factory_sharing_verify.yaml")
        with open(cfg_path, "w") as f:
            f.write(CFG)
        runner = YAMLSimulationRunner(cfg_path)
        runner.config["_run_output_dir"] = tmp
        params = runner._build_param_list()
        n_factories = len({id(p[0]) for p in params})
        # 2 penalties x {threshold, optimal} = 4 cases for 2 caps x (4 combos + 1 optimal)
        check(f"one factory per case ({n_factories} for {len(params)} sims)", n_factories == 4)

        shared = [p[0].create_simulation(sim_type=p[1], cap=p[2], threshold=p[3],
                                         wind_threshold=p[4]) for p in params]
        cfg = yaml.safe_load(CFG)
        cfg["_run_output_dir"] = tmp
        cfg["start_datetime"] = cfg["start_datetimes"][0]
        fresh = []
        for p in params:
            fac = SimulationFactory(cfg, cfg["locations"][0], p[0].horizon, p[0].failure_penalty,
                                    config_name=p[0].config_name)
            fresh.append(fac.create_simulation(sim_type=p[1], cap=p[2], threshold=p[3],
                                               wind_threshold=p[4]))

        a, b = run(shared), run(fresh)
        same = all(np.array_equal(x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
        check(f"shared and per-sim factories give identical episodes ({len(params)} sims)", same)
        # Run the shared set a second time: rollouts must not depend on earlier rollouts.
        c = run(shared)
        check("re-running shared sims reproduces them",
              all(np.array_equal(x, y) for ra, rc in zip(a, c) for x, y in zip(ra, rc)))
    print("All factory-sharing checks passed.")


if __name__ == "__main__":
    main()
