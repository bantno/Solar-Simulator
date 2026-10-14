#!/usr/bin/env python3
"""Run the chain-vs-iid sweep: provision location data up front, then execute every config.

Thin orchestration over harness/run_experiment.py that adds:
  * front-loaded data provisioning (Open-Meteo fetch + windchain/histcube artifact builds)
    so network/build failures surface before hours of solving,
  * --resume (skip configs that already have a completed run under --out),
  * per-config wall-time logging to <out>/sweep_run_log.json.

Every run still solves its value tables fresh (the engine never reloads saved tables).

Usage (pvlib conda env, from SolarSimulator/):
    conda run -n pvlib python Scripts/run_chain_sweep.py --provision-only
    conda run -n pvlib python Scripts/run_chain_sweep.py --smoke
    conda run -n pvlib python Scripts/run_chain_sweep.py --workers 8 --resume
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../SolarSimulator
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

from harness.run_experiment import run_experiment, _ensure_location_data, _abspath  # noqa: E402


def _configs_in(config_dir):
    cfgs = sorted(glob.glob(os.path.join(config_dir, "*.y*ml")))
    if not cfgs:
        sys.exit(f"[error] no configs found in {config_dir} "
                 f"(run Scripts/generate_chain_sweep_configs.py first)")
    return cfgs


def _unique_locations(config_paths):
    seen, locs = set(), []
    for p in config_paths:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        for loc in cfg.get("locations", []) or []:
            key = (loc.get("latitude"), loc.get("longitude"))
            if key not in seen:
                seen.add(key)
                loc = dict(loc)
                loc["data_path"] = _abspath(loc["data_path"])
                locs.append(loc)
    return locs


def provision_all(config_paths):
    """Build every artifact any config will need (chain + histcube for all locations)."""
    # Take the wind-chain binning spec from the configs themselves so provisioning
    # builds (or rebuilds) exactly the artifact the runs will load.
    wind_chain = {"enabled": True, "n_bins": 3}
    for p in config_paths:
        with open(p, "r") as f:
            wc = (yaml.safe_load(f) or {}).get("wind_chain") or {}
        if wc.get("enabled", False):
            wind_chain = dict(wc)
            break
    superset_cfg = {
        "delta_t": 15,
        "wind_chain": wind_chain,
        "historical_weather": {"enabled": True},
    }
    for loc in _unique_locations(config_paths):
        print(f"[provision] checking lat{loc['latitude']}_lon{loc['longitude']} ...")
        _ensure_location_data(superset_cfg, loc)
    print("[provision] all locations ready.")


def _already_done(out_base, basename):
    for run_dir in glob.glob(os.path.join(out_base, basename, "*")):
        if os.path.isfile(os.path.join(run_dir, "summary.csv")):
            return run_dir
    return None


def _scalars_only(cfg_path):
    """True when the config retains no full-history episodes (scalars are all the HDF5 holds)."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return not cfg.get("save_states", False) and not cfg.get("full_history_episodes")


def compact_run_dir(run_dir):
    """Extract per-episode scalars to the analysis cache CSV; delete bloated HDF5s.

    Only runs for scalars-only configs (no full histories). With the columnar
    episode_scalars layout the HDF5 is already small (a 210-sim x 3000-episode
    threshold config is tens of MB), so the file is kept and only the analysis
    cache is written. Legacy group-per-episode files (~0.5 KB of HDF5 metadata
    per scalar dataset, ~1.6 GB for the same config) are still deleted once
    their content is cached.
    """
    from Scripts.compare_chain_sweep import read_all_episode_scalars
    df = read_all_episode_scalars(run_dir)  # writes <run_dir>/_episode_scalars.csv
    import pandas as pd
    n_sims = len(pd.read_csv(os.path.join(run_dir, "summary.csv")))
    n_groups = df["group"].nunique()
    h5s = glob.glob(os.path.join(run_dir, "*.h5"))
    if n_groups != n_sims:
        print(f"[compact] SKIPPED {run_dir}: scalar cache covers {n_groups} groups "
              f"but summary.csv has {n_sims} sims -- keeping HDF5")
        return
    size = sum(os.path.getsize(p) for p in h5s)
    if size < 100e6:  # columnar layout: nothing worth deleting
        print(f"[compact] {run_dir}: cached scalars for {n_groups} sims "
              f"(HDF5 is {size / 1e6:.0f} MB, keeping it)")
        return
    for p in h5s:
        os.remove(p)
    print(f"[compact] {run_dir}: cached scalars for {n_groups} sims, "
          f"freed {size / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", default=None,
                    help="Config dir (default: configs/chain_vs_iid_sweep[_smoke]).")
    ap.add_argument("--out", default=None,
                    help="Results base dir (default: results/chain_vs_iid_sweep[_smoke]).")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--no-multiproc", action="store_true", help="Run serially.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip configs that already have a run with a summary.csv.")
    ap.add_argument("--provision-only", action="store_true",
                    help="Only fetch/build location data artifacts, then exit.")
    ap.add_argument("--smoke", action="store_true",
                    help="Use the _smoke config and results directories.")
    ap.add_argument("--keep-h5", action="store_true",
                    help="Do not compact scalars-only runs (keep the raw HDF5).")
    args = ap.parse_args()

    suffix = "_smoke" if args.smoke else ""
    config_dir = args.configs or os.path.join(REPO_ROOT, "configs", f"chain_vs_iid_sweep{suffix}")
    out_base = args.out or os.path.join(REPO_ROOT, "results", f"chain_vs_iid_sweep{suffix}")
    cfgs = _configs_in(config_dir)

    provision_all(cfgs)
    if args.provision_only:
        return

    # Run hist-world configs first so the paired-weather (CRN) check can happen early.
    cfgs.sort(key=lambda p: ("_hist" not in os.path.basename(p), os.path.basename(p)))

    os.makedirs(out_base, exist_ok=True)
    log_path = os.path.join(out_base, "sweep_run_log.json")
    log = []
    if os.path.isfile(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)

    use_mp = not args.no_multiproc
    for i, cfg_path in enumerate(cfgs, 1):
        basename = os.path.splitext(os.path.basename(cfg_path))[0]
        if args.resume:
            done = _already_done(out_base, basename)
            if done:
                print(f"[skip {i}/{len(cfgs)}] {basename}: already complete ({done})")
                continue
        print(f"[sweep {i}/{len(cfgs)}] {basename}")
        t0 = time.time()
        run_dir = run_experiment(cfg_path, out_base, args.workers, use_mp)
        if not args.keep_h5 and _scalars_only(cfg_path):
            compact_run_dir(run_dir)
        elapsed = time.time() - t0
        log.append({"config": basename, "run_dir": run_dir, "seconds": round(elapsed, 1)})
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"[sweep {i}/{len(cfgs)}] {basename}: {elapsed:.1f}s")

    print(f"Sweep complete. Timing log: {log_path}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    mp.freeze_support()
    main()
