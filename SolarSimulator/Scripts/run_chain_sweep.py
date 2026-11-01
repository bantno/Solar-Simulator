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

Sharding: configs run one after another, and a config with few cells leaves most workers
idle. `--shard i/K` runs only the i-th of K shards (configs balanced by an estimate of
their solve and rollout cost), so K processes, each with `--workers` about cores / K,
keep the machine busy. Provision once first (--provision-only) so shards never build the
same artifact at the same time; shards then skip provisioning with --no-provision.
With --resume, finished configs are dropped before the split, so start all K shards
together. Keep the shards' --workers summed at or below the core count (--workers is
capped at the core count); run_thesis_runs.py plans this for the thesis runs.
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

from harness.run_experiment import run_experiment, _ensure_location_data  # noqa: E402


def _configs_in(config_dir):
    cfgs = sorted(glob.glob(os.path.join(config_dir, "*.y*ml")))
    if not cfgs:
        sys.exit(f"[error] no configs found in {config_dir} "
                 f"(run Scripts/generate_chain_sweep_configs.py first)")
    return cfgs


def _provision_site(jobs):
    """Worker: provision every (config, location, config name) job of ONE site, in order.

    One process per site, so no two processes ever write the same artifact file.
    """
    t0 = time.time()
    for cfg, loc, name in jobs:
        print(f"[provision] checking lat{loc['latitude']}_lon{loc['longitude']} ({name}) ...",
              flush=True)
        _ensure_location_data(cfg, loc)
    return jobs[0][1]["latitude"], jobs[0][1]["longitude"], time.time() - t0


def provision_all(config_paths, workers=1):
    """Build every artifact any config will need, exactly as each config will load it.

    Each config is provisioned with its own wind/solar chain, clear-sky and hold-out
    settings (the same resolver the runs use), so year-split hold-out artifacts and any
    explicit paths are honoured. Configs that resolve to the same files are checked once.

    Missing historical records are downloaded from Open-Meteo first, one at a time (the
    archive API rate-limits large requests). The artifact builds, which are single-core
    and take several minutes per site, then run with one process per site, up to
    `workers` sites at once.
    """
    from harness.run_experiment import (  # noqa: E402
        _resolve_paths_in_place, _find_historical_pkl, _fetch_historical_pkl,
        _hist_dir_from_data_path)
    from BaseClasses.run_sim import resolve_artifact_paths, wind_chain_needed  # noqa: E402
    seen = set()
    by_site = {}
    for p in config_paths:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f) or {}
        _resolve_paths_in_place(cfg)
        for loc in cfg.get("locations", []) or []:
            paths = resolve_artifact_paths(cfg, loc)
            sig = (paths["expected_data"], paths["histcube"], paths["wind_chain"],
                   wind_chain_needed(cfg),
                   (cfg.get("wind_chain") or {}).get("n_bins"),
                   str((cfg.get("wind_chain") or {}).get("bin_edges")),
                   bool((cfg.get("historical_weather") or {}).get("enabled")))
            if sig in seen:
                continue
            seen.add(sig)
            site = (loc.get("latitude"), loc.get("longitude"))
            by_site.setdefault(site, []).append((cfg, loc, os.path.basename(p)))

    # 1) Downloads, serially.
    t0 = time.time()
    n_fetched = 0
    for (lat, lon), jobs in by_site.items():
        hist_dir = _hist_dir_from_data_path(jobs[0][1]["data_path"])
        if lat is not None and lon is not None and _find_historical_pkl(hist_dir, lat, lon) is None:
            t1 = time.time()
            _fetch_historical_pkl(hist_dir, lat, lon)
            n_fetched += 1
            print(f"[provision] fetched lat{lat}_lon{lon} in {time.time() - t1:.0f}s", flush=True)
    if n_fetched:
        print(f"[provision] {n_fetched} download(s) in {time.time() - t0:.0f}s")

    # 2) Builds, one process per site.
    t0 = time.time()
    groups = list(by_site.values())
    n_proc = max(1, min(int(workers), len(groups)))
    if n_proc > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_proc) as pool:
            for lat, lon, dt in pool.imap_unordered(_provision_site, groups):
                print(f"[provision] lat{lat}_lon{lon} ready ({dt:.0f}s)", flush=True)
    else:
        for g in groups:
            _provision_site(g)
    print(f"[provision] all {len(groups)} location(s) ready; builds took "
          f"{time.time() - t0:.0f}s with {n_proc} process(es).")


def config_cost(cfg_path):
    """Rough cost of a config in worker-seconds on the 12-core workstation with every
    logical core busy. An optimal solve scales with wind bins x horizon x capacity
    cubed; a rollout with episodes x horizon (a chain policy's about twice an i.i.d.
    one's). Fitted on 15 configs of the thesis runs (2026-10-28), within about 25%;
    capacity cubed (not linear, as solve benchmarks at one capacity suggested) is what
    makes the large-battery cells dominate."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    caps = cfg.get("battery_capacities") or [cfg.get("battery_capacity", 300.0)]
    hors = cfg.get("horizons") or [cfg.get("horizon", 1000)]
    pens = cfg.get("failure_penalties") or [cfg.get("failure_penalty", 15)]
    starts = cfg.get("start_datetimes") or [cfg.get("start_datetime")]
    n_loc = len(cfg.get("locations") or [None])
    wc = cfg.get("wind_chain") or {}
    bins = int(wc.get("n_bins", 3)) if wc.get("enabled") else 1
    combos = len(cfg.get("threshold_values") or []) * len(cfg.get("wind_thresholds") or [])
    eps = cfg.get("episodes", 10000) / 10000.0
    cost = 0.0
    for c in caps:
        for h in hors:
            cell = 0.0
            if cfg.get("include_optimal", True):
                cell += 16.0 * bins * (c / 300.0) ** 3 * (h / 1000.0)
                cell += 6.0 * eps * h / 1000.0 * (2 if bins > 1 else 1)
            cell += combos * 6.0 * eps * h / 1000.0
            cost += cell * len(pens) * len(starts) * n_loc
    return cost


def shard_configs(cfgs, shard):
    """Configs of shard i of K ("i/K", 1-based): greedy longest-first balancing."""
    i, k = (int(x) for x in shard.split("/"))
    if not 1 <= i <= k:
        sys.exit(f"[error] --shard {shard}: need 1 <= i <= K")
    loads = [0.0] * k
    owned = [[] for _ in range(k)]
    for p in sorted(cfgs, key=config_cost, reverse=True):
        j = loads.index(min(loads))
        loads[j] += config_cost(p)
        owned[j].append(p)
    print(f"[shard {i}/{k}] {len(owned[i - 1])} config(s), estimated load "
          f"{loads[i - 1]:.0f} of {sum(loads):.0f}")
    return owned[i - 1]


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
    ap.add_argument("--shard", default=None, metavar="i/K",
                    help="Run only shard i of K (see the module docstring).")
    ap.add_argument("--no-provision", action="store_true",
                    help="Skip provisioning (artifacts already built, e.g. by a "
                         "--provision-only pass before sharded runs).")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="Run only these configs (basenames without .yaml), e.g. to help a "
                         "slow shard: its --resume check skips any config finished meanwhile.")
    ap.add_argument("--keep-h5", action="store_true",
                    help="Do not compact scalars-only runs (keep the raw HDF5).")
    args = ap.parse_args()

    suffix = "_smoke" if args.smoke else ""
    config_dir = args.configs or os.path.join(REPO_ROOT, "configs", f"chain_vs_iid_sweep{suffix}")
    out_base = args.out or os.path.join(REPO_ROOT, "results", f"chain_vs_iid_sweep{suffix}")
    cfgs = _configs_in(config_dir)
    args.workers = max(1, min(args.workers, os.cpu_count() or 1))
    if args.only:
        cfgs = [p for p in cfgs if os.path.splitext(os.path.basename(p))[0] in set(args.only)]
    if args.resume and args.shard:
        # Balance only the unfinished configs. Every shard of a sweep must start at the
        # same time so they all see the same finished set (run_thesis_runs.py does this).
        cfgs = [p for p in cfgs
                if not _already_done(out_base, os.path.splitext(os.path.basename(p))[0])]
    if args.shard:
        cfgs = shard_configs(cfgs, args.shard)

    # Check every config's output paths before spending time on provisioning.
    from harness.run_experiment import check_output_path_length  # noqa: E402
    for p in cfgs:
        check_output_path_length(out_base, os.path.splitext(os.path.basename(p))[0])

    if not args.no_provision:
        provision_all(cfgs, workers=1 if args.no_multiproc else args.workers)
    if args.provision_only:
        return

    # Run hist-world configs first so the paired-weather (CRN) check can happen early.
    cfgs.sort(key=lambda p: ("_hist" not in os.path.basename(p), os.path.basename(p)))

    os.makedirs(out_base, exist_ok=True)
    log_path = os.path.join(out_base, "sweep_run_log.json" if not args.shard else
                            f"sweep_run_log_shard{args.shard.replace('/', 'of')}.json")
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
