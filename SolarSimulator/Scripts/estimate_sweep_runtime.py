#!/usr/bin/env python3
"""Estimate the wall time of a sweep (a directory of harness configs) on this machine.

Two steps:

1. **Calibrate** (cached per machine in results/_runtime_probe_<host>.json). Small probe
   runs on an existing hold-out location measure, single-process:
     * factory construction per simulation (data + bootstrap-cube load),
     * pickling a simulation to a worker and back,
     * optimal-policy solve time per (SoC level x stage), separately for iid and chain,
     * rollout time per (episode x stage) for iid-optimal, chain-optimal and threshold sims,
   and then the per-process slowdown when `--workers` processes run at once (shared cores,
   memory bandwidth, hyper-threads).

2. **Schedule.** Each config is costed the way harness/run_experiment.py executes it:
   a serial build of every simulation's factory in the main process, a parallel build
   pool (pool.map: optimal solves), and a parallel run pool (imap_unordered, chunksize 1,
   list order: threshold sims first, then optimal sims). Each pool's makespan comes from
   greedy list scheduling on `--workers` slots, with every task stretched by the measured
   slowdown. Configs run one after another, as in run_chain_sweep.py.

Rollouts are costed at the full horizon, so episodes that end early make the estimate an
upper bound for that part (failure rates of 20-45 % put it within roughly 10-30 %).
Provisioning (Open-Meteo downloads, serial; site builds, parallel) is reported separately.

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/estimate_sweep_runtime.py --configs ../configs/journal_chain_sweep
    python Scripts/estimate_sweep_runtime.py --configs <dir> --workers 23 --recalibrate
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import platform
import sys
import tempfile
import time

import numpy as np
import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

PROBE_LOC = {"latitude": 30.0, "longitude": -90.0,
             "data_path": os.path.join(REPO_ROOT, "Data", "EXPECTED_DATA",
                                       "data_expected_lat30.0_lon-90.0_15min.pkl")}
PROBE_START = "2020-07-01T00:00:00"
# Provisioning, measured 2026-10-24 on the journal sweep (7 sites along lon -75):
# an Open-Meteo 1950-2022 download takes ~17 s, but the archive API allows only about
# three such requests per minute, then the fetcher waits ~65 s; downloads run serially.
# A site build (fit table + chain + cube) takes 7-8 min on one core; run_chain_sweep.py
# builds sites in parallel, one process per site (7 sites in 478 s on 23 workers).
PROVISION_FETCH_S = 17.0
PROVISION_FETCHES_PER_MINUTE = 3
PROVISION_RATE_WAIT_S = 65.0
PROVISION_BUILD_S = 480.0
# Fixed cost per config independent of its size: spawning two worker pools (each worker
# re-imports the package), HDF5/summary writes, figures and compaction. Measured as
# 17-23 s per config (mean 19 s) on the 8-config journal smoke sweep, 2026-10-24, 23 workers.
FIXED_CONFIG_OVERHEAD_S = 19.0


def n_soc_levels(capacity, energy_increment_wh=5.0):
    inc = (energy_increment_wh / capacity) * 100.0
    return len(np.arange(0, 100 + inc, inc))


def probe_config(chain: bool, tmpdir: str) -> dict:
    return {
        "start_datetime": PROBE_START, "transition_model": "moderate",
        "solar_panel_model": "constant", "whale_series": "JoA2026",
        "energy_increment_wh": 5, "delta_t": 15, "locations": [dict(PROBE_LOC)],
        "wind_chain": {"enabled": chain, "n_bins": 4},
        "historical_weather": {"enabled": True, "block_length_days": 7},
        "_run_output_dir": tmpdir,
    }


def _factory(chain, horizon, tmpdir, penalty=5.0):
    from BaseClasses.run_sim import SimulationFactory
    return SimulationFactory(probe_config(chain, tmpdir), dict(PROBE_LOC), horizon, penalty,
                             config_name="runtime_probe")


def _time_rollout(sim, episodes):
    t0 = time.perf_counter()
    for _ in sim.simulate_multiple_episodes(episodes):
        pass
    return time.perf_counter() - t0


def _parallel_probe(args):
    """Worker: build a threshold sim (untimed), then time its rollout."""
    horizon, episodes = args
    with tempfile.TemporaryDirectory() as tmp:
        sim = _factory(False, horizon, tmp).create_simulation("threshold", 300.0, 0.1, 6.0)
        return _time_rollout(sim, episodes)


def calibrate(workers, horizon=1000, episodes=10000):
    print(f"[calibrate] probe location lat30/lon-90, horizon {horizon}, {episodes} episodes")
    out = {"host": platform.node(), "horizon": horizon, "episodes": episodes,
           "workers": workers, "cpu_count": mp.cpu_count(),
           "date": time.strftime("%Y-%m-%d %H:%M")}
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        for _ in range(3):
            fac = _factory(False, horizon, tmp)
        out["factory_s"] = (time.perf_counter() - t0) / 3
        print(f"  factory construction        {out['factory_s']:.2f} s/sim")

        thr = fac.create_simulation("threshold", 300.0, 0.1, 6.0)
        t0 = time.perf_counter()
        pickle.loads(pickle.dumps(thr))
        out["pickle_s"] = time.perf_counter() - t0
        print(f"  pickle round trip           {out['pickle_s']:.2f} s/sim")

        for arm, chain in (("iid", False), ("chain", True)):
            per_unit = []
            for cap in (300.0, 600.0):
                f = _factory(chain, horizon, tmp)
                t0 = time.perf_counter()
                sim = f.create_simulation("optimal", cap)
                dt = time.perf_counter() - t0
                per_unit.append(dt / (n_soc_levels(cap) * horizon))
                print(f"  solve {arm:5s} {cap:5.0f} Wh         {dt:7.2f} s")
                if cap == 300.0:
                    roll = _time_rollout(sim, episodes)
                    out[f"roll_{arm}_s_per_ep_step"] = roll / (episodes * horizon)
                    print(f"  rollout {arm:5s}               {roll:7.2f} s")
            out[f"solve_{arm}_s_per_soc_step"] = float(np.mean(per_unit))
            out[f"solve_{arm}_cap_linearity"] = float(per_unit[1] / per_unit[0])
            npys = glob.glob(os.path.join(tmp, "*.npy"))
            if npys:
                biggest = max(npys, key=os.path.getsize)
                out[f"table_{arm}_bytes_per_soc_step"] = (
                    os.path.getsize(biggest) / (n_soc_levels(600.0) * horizon))
            for p in npys:
                os.remove(p)
        roll = _time_rollout(thr, episodes)
        out["roll_thresh_s_per_ep_step"] = roll / (episodes * horizon)
        print(f"  rollout threshold           {roll:7.2f} s")

        # Main-process HDF5 write per simulation (serial; scales with episode count).
        from BaseClasses.simulation_run_manager import _run_one_sim
        from BaseClasses.simulation_storage import SimulationStorageHDF5
        store_eps = 10000
        meta, eps = _run_one_sim((thr, store_eps, False, "probe", False))
        st = SimulationStorageHDF5(os.path.join(tmp, "store_probe.h5"))
        t0 = time.perf_counter()
        st.store_simulation(meta, eps, group_name="probe")
        st.h5file.flush()
        out["store_s_per_episode"] = (time.perf_counter() - t0) / store_eps
        st.close()
        print(f"  HDF5 store                  {out['store_s_per_episode'] * 1e3:.3f} ms/episode")

    print(f"[calibrate] slowdown with {workers} concurrent rollouts ...")
    single = _parallel_probe((horizon, episodes))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        times = pool.map(_parallel_probe, [(horizon, episodes)] * workers)
    out["single_rollout_s"] = single
    out["parallel_rollout_s"] = float(np.mean(times))
    out["slowdown"] = float(np.mean(times) / single)
    print(f"  1 process {single:.2f} s, {workers} processes {np.mean(times):.2f} s each "
          f"-> slowdown x{out['slowdown']:.2f}, throughput x{workers / out['slowdown']:.1f}")
    return out


def list_schedule(costs, workers):
    """Makespan of greedy list scheduling in the given order onto `workers` slots."""
    if not costs:
        return 0.0
    slots = np.zeros(workers)
    for c in costs:
        i = int(np.argmin(slots))
        slots[i] += c
    return float(slots.max())


def config_tasks(cfg):
    """(threshold sims, optimal sims) as lists of (capacity, horizon) in harness order."""
    starts = cfg.get("start_datetimes") or [cfg.get("start_datetime")]
    locs = cfg.get("locations") or [{}]
    thr, opt = [], []
    for _ in starts:
        for _loc in locs:
            for h in cfg["horizons"]:
                for _fp in cfg["failure_penalties"]:
                    for cap in cfg["battery_capacities"]:
                        for _t in cfg.get("threshold_values") or []:
                            for _w in cfg.get("wind_thresholds") or []:
                                thr.append((cap, h))
        if cfg.get("include_optimal", True):
            for _loc in locs:
                for h in cfg["horizons"]:
                    for _fp in cfg["failure_penalties"]:
                        for cap in cfg["battery_capacities"]:
                            opt.append((cap, h))
    return thr, opt


def estimate_config(cfg, cal, workers):
    eps = int(cfg.get("episodes", 3000))
    chain = bool((cfg.get("wind_chain") or {}).get("enabled"))
    arm = "chain" if chain else "iid"
    thr, opt = config_tasks(cfg)
    slow = cal["slowdown"]
    # _build_param_list builds one factory per (start, location, horizon, penalty, kind).
    cases = (len(cfg.get("start_datetimes") or [cfg.get("start_datetime")])
             * len(cfg.get("locations") or [{}]) * len(cfg["horizons"])
             * len(cfg["failure_penalties"]))
    n_factories = cases * ((1 if thr else 0) + (1 if opt else 0))
    serial = n_factories * cal["factory_s"]
    solve_costs = [cal["pickle_s"] for _ in thr] + [
        cal[f"solve_{arm}_s_per_soc_step"] * n_soc_levels(c) * h * slow + cal["pickle_s"]
        for c, h in opt]
    run_costs = ([cal["roll_thresh_s_per_ep_step"] * eps * h * slow + cal["pickle_s"]
                  for c, h in thr]
                 + [cal[f"roll_{arm}_s_per_ep_step"] * eps * h * slow + cal["pickle_s"]
                    for c, h in opt])
    build = list_schedule(solve_costs, workers)
    # The main process stores each finished sim serially while workers keep running, so
    # the run phase is bounded below by the total store time as well as by the workers.
    store = (len(thr) + len(opt)) * cal.get("store_s_per_episode", 0.0) * eps
    run = max(list_schedule(run_costs, workers), store)
    table_bytes = max((cal.get(f"table_{arm}_bytes_per_soc_step", 0) * n_soc_levels(c) * h
                       for c, h in opt), default=0)
    serial += FIXED_CONFIG_OVERHEAD_S
    return {"serial": serial, "build": build, "run": run, "total": serial + build + run,
            "n_thresh": len(thr), "n_opt": len(opt), "arm": arm,
            "max_table_gb": table_bytes / 1e9,
            "cpu_hours": (sum(solve_costs) + sum(run_costs)) / slow / 3600}


def missing_locations(configs):
    from BaseClasses.run_sim import resolve_artifact_paths
    from harness.run_experiment import (_find_historical_pkl, _hist_dir_from_data_path,
                                        _resolve_paths_in_place)
    need_fetch, need_build = set(), set()
    for cfg in configs:
        cfg = json.loads(json.dumps(cfg))  # resolve explicit artifact paths on a copy
        _resolve_paths_in_place(cfg)
        for loc in cfg.get("locations") or []:
            key = (loc["latitude"], loc["longitude"])
            paths = resolve_artifact_paths(cfg, loc)
            if not os.path.exists(paths["expected_data"]) or (
                    (cfg.get("historical_weather") or {}).get("enabled")
                    and not os.path.exists(paths["histcube"])):
                need_build.add(key)
            if _find_historical_pkl(_hist_dir_from_data_path(loc["data_path"]),
                                    loc["latitude"], loc["longitude"]) is None:
                need_fetch.add(key)
    return sorted(need_fetch), sorted(need_build)


def fmt_h(seconds):
    return f"{seconds / 3600:6.2f} h"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", required=True, help="Directory of sweep configs.")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Worker processes, as passed to run_chain_sweep.py (default CPU-1).")
    ap.add_argument("--recalibrate", action="store_true", help="Ignore the cached probe.")
    ap.add_argument("--probe-horizon", type=int, default=1000)
    ap.add_argument("--probe-episodes", type=int, default=10000,
                    help="Episodes per probe rollout; match the sweep (per-step cost falls with batch size).")
    ap.add_argument("--csv", default=None, help="Optional per-config estimate CSV.")
    args = ap.parse_args()

    cache = os.path.join(REPO_ROOT, "results", f"_runtime_probe_{platform.node()}.json")
    cal = None
    if os.path.exists(cache) and not args.recalibrate:
        with open(cache) as f:
            cal = json.load(f)
        if cal.get("workers") != args.workers or "store_s_per_episode" not in cal:
            print("[calibrate] cached probe is for other settings or older; recalibrating")
            cal = None
    if cal is None:
        cal = calibrate(args.workers, args.probe_horizon, args.probe_episodes)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"[calibrate] cached -> {cache}")
    else:
        print(f"[calibrate] using cached probe {cache} ({cal['date']})")

    paths = sorted(glob.glob(os.path.join(args.configs, "*.y*ml")))
    configs = []
    for p in paths:
        with open(p) as f:
            configs.append(yaml.safe_load(f))

    rows = []
    for p, cfg in zip(paths, configs):
        est = estimate_config(cfg, cal, args.workers)
        est["config"] = os.path.splitext(os.path.basename(p))[0]
        rows.append(est)

    print(f"\nPer-config estimate ({args.workers} workers, slowdown x{cal['slowdown']:.2f}):")
    print(f"{'config':52s} {'sims':>6s} {'setup':>9s} {'solve':>9s} {'rollout':>9s} {'total':>9s}")
    for r in rows:
        print(f"{r['config']:52s} {r['n_thresh'] + r['n_opt']:6d} {fmt_h(r['serial'])} "
              f"{fmt_h(r['build'])} {fmt_h(r['run'])} {fmt_h(r['total'])}")

    sweep_of = {}
    for mpath in glob.glob(os.path.join(args.configs, "*manifest*.json")):
        with open(mpath) as f:
            for c in json.load(f).get("cells", []):
                sweep_of[c["config_basename"]] = c["sweep"]
    by_sweep = {}
    for r in rows:
        sweep = sweep_of.get(r["config"], r["config"].split("_")[1] if "_" in r["config"]
                             else r["config"])
        by_sweep[sweep] = by_sweep.get(sweep, 0.0) + r["total"]
    total = sum(r["total"] for r in rows)
    print("\nBy sweep:")
    for k, v in sorted(by_sweep.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {fmt_h(v)}")

    fetch, build = missing_locations(configs)
    n_waits = max(0, (len(fetch) - 1) // PROVISION_FETCHES_PER_MINUTE)
    prov = (len(fetch) * PROVISION_FETCH_S + n_waits * PROVISION_RATE_WAIT_S
            + (-(-len(build) // max(1, args.workers))) * PROVISION_BUILD_S)
    print(f"\nProvisioning: {len(fetch)} location(s) to fetch, {len(build)} to build "
          f"-> about {fmt_h(prov).strip()}")
    print(f"Simulation total: {fmt_h(total).strip()} ({total / 86400:.1f} days); "
          f"CPU work {sum(r['cpu_hours'] for r in rows):.0f} core-hours single-process")
    print(f"Largest value table: {max(r['max_table_gb'] for r in rows):.2f} GB")
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
