"""
Quantify mdpAnalyticalBackwardSolver.solve() speedup: vectorized vs the original
per-state scalar loop, and assert the resulting value tables are identical.

The scalar loop here is an inlined copy of the ORIGINAL solve() body (it calls the
still-present scalar `_value`), run on a separate fresh solver built from the same MDP.

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_solver_speed.py
    conda run -n pvlib python SolarSimulator/Tests/verify_solver_speed.py --horizon 1000

The speedup grows with horizon (scalar loop is O(horizon*states) of Python overhead,
vectorized is O(horizon) Python-level calls).
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import YAMLSimulationRunner, SimulationFactory  # noqa: E402
from BaseClasses.backward_induction_base import mdpAnalyticalBackwardSolver  # noqa: E402

DATA_PKL = os.path.join(
    REPO_ROOT, "Data", "EXPECTED_DATA", "data_expected_lat30.0_lon-90.0_15min.pkl"
)


def make_config(horizon):
    text = f"""\
battery_capacities:
- 400.0
failure_penalties:
- 5.0
horizons:
- {horizon}
locations:
- data_path: {DATA_PKL}
  latitude: 30.0
  longitude: -90.0
solar_panel_model: constant
start_datetime: '2020-04-01T00:00:44'
threshold_values:
- 0.25
wind_thresholds:
- 10.0
transition_model: moderate
energy_increment_wh: 5
episodes: 1
"""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="rtopt_solver_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=200)
    args = ap.parse_args()

    if not os.path.exists(DATA_PKL):
        sys.exit(f"[error] missing data file: {DATA_PKL}")

    cfg_path = make_config(args.horizon)
    try:
        runner = YAMLSimulationRunner(cfg_path)
        cfg = runner.config
        locs = runner.locations
        loc = (locs[0] if isinstance(locs, tuple) else locs)[0]
        H = runner.horizons[0]
        fp = runner.failure_penalties[0]
        cap = cfg["battery_capacities"][0]
        factory = SimulationFactory(cfg, loc, H, fp, config_name="speedtest")

        def fresh_solver():
            mdp = factory.build_mdp(cap)
            s = mdpAnalyticalBackwardSolver(mdp, H, sim_name_prefix="speedtest")
            s.set_start_date(factory.start_dt.strftime("%Y-%m-%d %H:%M:%S"))
            return s

        # New vectorized solve
        s_new = fresh_solver()
        t0 = time.time()
        s_new.solve()
        new_dt = time.time() - t0
        new_table = s_new.future_value_table.copy()

        # Original-style scalar solve (inlined copy of the old loop)
        from tqdm import tqdm
        s_old = fresh_solver()
        t0 = time.time()
        states = s_old.states
        action_list = [np.array(0)[np.newaxis], np.array(1)[np.newaxis]]
        values = np.zeros(2)
        for t in tqdm(range(H - 1, -1, -1), desc="scalar baseline"):
            s_old._vnext_cache.clear()
            for i, st in enumerate(states[:-1]):
                for a in action_list:
                    values[a] = s_old._value(st[np.newaxis], a, t, t == H - 1)
                s_old.future_value_table[i, t] = max(values)
        old_dt = time.time() - t0

        diff = float(np.max(np.abs(new_table - s_old.future_value_table)))
        print(f"horizon={H}  states={states.shape[0]}")
        print(f"  vectorized solve : {new_dt:.3f}s")
        print(f"  scalar-loop solve: {old_dt:.3f}s   speedup x{old_dt/max(new_dt,1e-9):.1f}")
        print(f"  max |new - old| table diff = {diff:.3e}")
        assert diff < 1e-8
        print("  PASS")
    finally:
        os.remove(cfg_path)
        for f in os.listdir("."):
            if f.endswith(".npy") and "Wh_" in f:
                try:
                    os.remove(f)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
