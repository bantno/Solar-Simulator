#!/usr/bin/env python3
"""Time the optimal-policy value-table solve and rollout as a function of wind-bin count.

One process, nothing else running: for the i.i.d. solver and for chain solvers with each
bin count, build the simulation (the backward-induction solve happens there) at the
wind-bin study's site and settings, then roll it out. Uses the standard year-split
hold-out files, including the per-bin-count chain files (building any that are missing). Reports seconds per solve, cost relative to i.i.d. and 4 bins,
and the fitted power law  solve time ~ n_bins^k.

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/benchmark_wind_bins_solve.py
    python Scripts/benchmark_wind_bins_solve.py --bins 4 8 12 16 24 32 --horizon 3000
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from Scripts.generate_journal_chain_sweep_configs import BASE_LOC, SWEEPS, make_config  # noqa: E402


def _config(n_bins, horizon, tmpdir):
    arm = "iid" if n_bins is None else "chain"
    overrides = {**SWEEPS["duration"][1], "horizons": [horizon]}
    cfg = make_config("duration", BASE_LOC, arm, "hist", overrides, 1, False,
                      n_bins=n_bins or 4)
    from harness.run_experiment import _ensure_location_data, _resolve_paths_in_place  # noqa: E402
    _resolve_paths_in_place(cfg)
    _ensure_location_data(cfg, cfg["locations"][0])  # builds a missing chain file
    cfg["start_datetime"] = cfg["start_datetimes"][0]
    cfg["_run_output_dir"] = tmpdir
    return cfg


def time_one(n_bins, horizon, capacity, penalty, episodes, repeats):
    from BaseClasses.run_sim import SimulationFactory  # noqa: E402
    solves = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(n_bins, horizon, tmp)
        for _ in range(repeats):
            fac = SimulationFactory(cfg, cfg["locations"][0], horizon, penalty,
                                    config_name="bench")
            t0 = time.perf_counter()
            sim = fac.create_simulation("optimal", capacity)
            solves.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in sim.simulate_multiple_episodes(episodes):
            pass
        roll = time.perf_counter() - t0
        table = max((os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".npy")),
                    key=os.path.getsize, default=None)
        mb = os.path.getsize(table) / 1e6 if table else float("nan")
    return min(solves), roll, mb


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bins", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--capacity", type=float, default=300.0)
    ap.add_argument("--penalty", type=float, default=20.0)
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=2, help="Solves per bin count (min kept).")
    ap.add_argument("--json", default=None, help="Also write the results to this file.")
    args = ap.parse_args()

    rows = []
    for n in [None] + args.bins:
        solve, roll, mb = time_one(n, args.horizon, args.capacity, args.penalty,
                                   args.episodes, args.repeats)
        rows.append({"n_bins": n or 1, "arm": "iid" if n is None else f"chain{n}",
                     "solve_s": solve, "rollout_s": roll, "table_mb": mb})
        print(f"{rows[-1]['arm']:8s} solve {solve:7.2f} s   rollout {roll:6.2f} s "
              f"({args.episodes} episodes)   table {mb:6.1f} MB", flush=True)

    base = rows[0]["solve_s"]
    four = next((r["solve_s"] for r in rows if r["n_bins"] == 4), None)
    chain = [r for r in rows if r["arm"] != "iid"]
    k = float(np.polyfit(np.log([r["n_bins"] for r in chain]),
                         np.log([r["solve_s"] for r in chain]), 1)[0]) if len(chain) > 1 else None
    print(f"\nhorizon {args.horizon}, {args.capacity:g} Wh, penalty {args.penalty:g}")
    print(f"{'arm':8s} {'solve s':>8s} {'x iid':>7s} {'x 4-bin':>8s}")
    for r in rows:
        print(f"{r['arm']:8s} {r['solve_s']:8.2f} {r['solve_s'] / base:7.1f} "
              f"{(r['solve_s'] / four if four else float('nan')):8.2f}")
    if k is not None:
        print(f"power-law fit over chain arms: solve time ~ n_bins^{k:.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"settings": vars(args), "rows": rows, "power_law_k": k}, f, indent=2)


if __name__ == "__main__":
    main()
