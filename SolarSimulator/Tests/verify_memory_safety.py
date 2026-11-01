"""Verification that the memory fixes leave results unchanged.

The fixes: a solved value table pickles as its saved .npy path and is mapped read-only
on arrival (one page-cache copy across processes), the unused optimal_action_table is no
longer allocated, and run_experiment builds threshold sims in the parent instead of
round-tripping each (with its own copy of the weather data) through the pool.

Checks, on the lat 30 / lon -75 hold-out files with a 16-bin chain and a short horizon:
  1. A solved chain sim pickles to a small payload; the copy's table is a read-only
     memory map equal to the solved table; optimal_action_table keeps its shape.
  2. The mapped copy's rollouts equal those of a copy carrying the full in-memory table.
  3. A copy whose saved table file was overwritten by another solve refuses to load.
  4. run_experiment with a pool (memmapped tables, parent-built threshold sims) gives
     the same per-episode results as the serial run (everything in one process).

Run (any cwd) with the pvlib conda env:
    conda run -n pvlib python SolarSimulator/Tests/verify_memory_safety.py
"""
import multiprocessing as mp
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import SimulationFactory  # noqa: E402
from harness.run_experiment import (_ensure_location_data, _resolve_paths_in_place,  # noqa: E402
                                    run_experiment)
from Scripts.compare_chain_sweep import read_all_episode_scalars  # noqa: E402

OUT = "C:/Users/bepstein8/sim_results/tmp/verify_memory_safety"
H = 192
N_EP = 200
LOC = {"latitude": 30.0, "longitude": -75.0,
       "data_path": "Data/EXPECTED_DATA/data_expected_lat30.0_lon-75.0_15min.pkl"}


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        sys.exit(1)


def base_config():
    return {
        "transition_model": "moderate", "solar_panel_model": "constant",
        "whale_series": "JoA2026", "energy_increment_wh": 5, "delta_t": 15,
        "start_datetime": "2025-06-10T00:00:00", "locations": [dict(LOC)],
        "holdout": {"fit_years": [1950, 2002], "eval_years": [2003, 2022],
                    "rollout_years": "eval"},
        "wind_chain": {"enabled": True, "n_bins": 16},
        "historical_weather": {"enabled": True, "block_length_days": 7},
    }


def rollout(sim):
    eps = list(sim.simulate_multiple_episodes(N_EP))
    return (np.array([e["total_reward"] for e in eps]),
            np.array([e.get("failure_step", -1) for e in eps]))


def check_pickling():
    cfg = base_config()
    _resolve_paths_in_place(cfg)
    _ensure_location_data(cfg, cfg["locations"][0])
    cfg["_run_output_dir"] = os.path.join(OUT, "tables")
    sim = SimulationFactory(cfg, cfg["locations"][0], H, 20.0,
                            config_name="verify_mem").create_simulation("optimal", 300.0)
    solver = sim.mdp_solver
    table = np.array(solver.future_value_table)

    print("1. solved table travels as a path")
    blob = pickle.dumps(sim)
    full = solver._table_path
    solver._table_path = None                   # the pre-fix payload: table in the pickle
    blob_full = pickle.dumps(sim)
    solver._table_path = full
    print(f"    pickle {len(blob) / 1e6:.1f} MB with the path, "
          f"{len(blob_full) / 1e6:.1f} MB with the table ({table.nbytes / 1e6:.1f} MB)")
    check("pickle drops the table", len(blob_full) - len(blob) >= 0.99 * table.nbytes)
    mapped, inmem = pickle.loads(blob), pickle.loads(blob_full)
    t = mapped.mdp_solver.future_value_table
    check("copy maps the saved file read-only",
          isinstance(t, np.memmap) and not t.flags.writeable)
    check("mapped table equals the solved table", np.array_equal(t, table))
    check("optimal_action_table keeps its shape",
          mapped.mdp_solver.optimal_action_table.shape == table.shape)

    print("2. rollouts with the mapped table")
    r_map, s_map = rollout(mapped)
    r_mem, s_mem = rollout(inmem)
    check(f"mapped and in-memory copies give identical rollouts ({N_EP} episodes)",
          np.array_equal(r_map, r_mem) and np.array_equal(s_map, s_mem))

    print("3. overwritten table file is refused")
    del mapped, t
    np.save(full, table + 1.0)
    try:
        pickle.loads(blob)
        refused = False
    except RuntimeError:
        refused = True
    check("loading after another solve overwrote the file raises", refused)


def check_run_experiment():
    print("4. run_experiment: pool vs serial")
    cfg = base_config()
    cfg.update({"experiment_name": "verify_mem_run", "episodes": N_EP,
                "include_optimal": True, "save_states": False, "full_history_episodes": 0,
                "threshold_values": [0.1, 0.25], "wind_thresholds": [3.0, 6.0],
                "battery_capacities": [250.0, 300.0], "horizons": [H, 2 * H],
                "failure_penalties": [5.0], "start_datetimes": ["2025-06-10T00:00:00"]})
    cfg.pop("start_datetime")
    path = os.path.join(OUT, "verify_mem_run.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    frames = {}
    for mode, workers, use_mp in (("pool", 4, True), ("serial", 1, False)):
        run_dir = run_experiment(path, os.path.join(OUT, mode), workers, use_mp)
        df = read_all_episode_scalars(run_dir)
        frames[mode] = df.sort_values("group", kind="stable").reset_index(drop=True)
    a, b = frames["pool"], frames["serial"]
    n_groups = a["group"].nunique()
    check(f"pool run has all 20 sims (2 horizons x (8 threshold + 2 optimal)); got {n_groups}",
          n_groups == 20 and set(a["group"]) == set(b["group"]))
    same = a.shape == b.shape and a.equals(b)
    if not same:
        print(a.compare(b).head(20) if a.shape == b.shape else (a.shape, b.shape))
    check("pool and serial runs give identical per-episode results", same)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    check_pickling()
    check_run_experiment()
    print("All memory-safety checks passed.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
