#!/usr/bin/env python3
"""Generate the journal-paper case set as a held-out chain-vs-iid-vs-threshold sweep.

Re-runs every case of the Journal of Aircraft paper (configs/final_journal_configs/) with
the Markov-chain wind formulation added, under the standard year-split hold-out (solver
fitted on 1950-2002, episodes on 2003-2022). Output is in the format of
generate_chain_sweep_configs.py, so run_chain_sweep.py and compare_chain_sweep.py run and
analyse it unchanged.

Journal sweeps reproduced (settings copied from configs/final_journal_configs/):
    capacity   battery 100-600 Wh (11), 3000 stages, penalty 5, start 06-10, lat30/lon-75.
               Its 300 Wh cell is also the paper's threshold-sweep case (same settings),
               so the threshold grid there reproduces the threshold-sweep figure.
    duration   horizon 1000-27000 stages (8, the union of the two duration configs),
               300 Wh, penalty 20, start 01-01, lat30/lon-75.
    penalty    failure penalty 0-100 (10), 300 Wh, 3000 stages, start 06-10, lat30/lon-75.
    location   lat 30 to -30 along lon -25 (7) x battery 100-600 Wh (11), 3000 stages,
               penalty 10, start 01-01. One config per location (value-table file names
               omit the location). The paper used lon -75, where lat 20 to -10 fall on
               land (Cuba, Colombia, Ecuador, Peru); lon -25 in the open Atlantic keeps
               every latitude at least 300 km offshore.
    startdate  starts Jan/Mar/May/Jul/Sep 1 (5), 300 Wh, 8500 stages, penalty 15.

Arms per sweep (all on historical bootstrap weather unless --native):
    iid, chain   optimal policies solved without / with the 4-bin quantile wind chain;
                 identical bootstrap weather and crash draws (paired comparison).
    thresh       the 7 x 8 observation x wind threshold grid on the evaluation years;
                 compare_chain_sweep.py takes each cell's best combo as the benchmark and
                 reports a split-half check of the selection bias (which favours the
                 threshold, so the comparison is conservative for the optimal arms).
    --tune       optional thresh_tune arm: the same grid rolled out on the FIT years
                 (holdout.rollout_years: fit); the combo is then picked there and scored in
                 `thresh`, removing the selection bias entirely.
    --native     adds iid/chain runs in their own synthetic world (calibration check).
    --model-worlds
                 the no-modelling-error condition: iid, chain and thresh each rolled out
                 in the i.i.d. model's synthetic weather (world `iidworld`) and in the wind
                 chain's (world `chainworld`), via the `synthetic_world` config key. Within
                 a world every policy sees identical weather. Analyse with
                 compare_model_worlds.py.

Variants for follow-up runs (each writes its own config dir and manifest):
    --bins N           wind-chain bin count (default 4; the hold-out names the chain file
                       _windchain_q<N>.pkl, so counts never share a file).
    --block-days D     historical bootstrap block length in days (default 7).
    --sweeps S ...     only these sweeps.
    --arms A ...       only emit these arms; with --reuse-results the manifest points the
                       others at an earlier sweep's runs (same episodes, so still paired).
    --prefix P         config-name prefix (default jcs).
    --site LAT LON     run the single-site sweeps at another open-ocean site.

Usage (any Python with pyyaml, from SolarSimulator/):
    python Scripts/generate_journal_chain_sweep_configs.py                  # 10,000 episodes
    python Scripts/generate_journal_chain_sweep_configs.py --episodes 3000 --native
    python Scripts/generate_journal_chain_sweep_configs.py --smoke          # tiny pipeline test
    python Scripts/generate_journal_chain_sweep_configs.py --bins 12 --arms chain thresh_tune \
        --tune --prefix jcsq12 --out ../configs/thesis_headline \
        --reuse-results C:/Users/bepstein8/sim_results/journal_chain_sweep
Then:
    python Scripts/estimate_sweep_runtime.py --configs ../configs/journal_chain_sweep
    python Scripts/run_chain_sweep.py --configs ../configs/journal_chain_sweep \
        --out ../results/journal_chain_sweep --resume
    python Scripts/compare_chain_sweep.py --results ../results/journal_chain_sweep \
        --manifest ../configs/journal_chain_sweep/chain_vs_iid_sweep_manifest.json
"""
import argparse
import json
import os
from datetime import datetime

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_WIND_BINS = 4
BLOCK_LENGTH_DAYS = 7
HOLDOUT = {"fit_years": [1950, 2002], "eval_years": [2003, 2022]}

BASE_LOC = {"latitude": 30.0, "longitude": -75.0}
CAPACITIES = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]
LOCATION_LATS = [30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0]
LOCATION_LON = -25.0  # open Atlantic: every LOCATION_LATS site is >= 300 km from land

# The paper's threshold-sweep grid (configs/final_journal_configs/threshold_sweep), with the
# wind thresholds the other paper sweeps used added (4, 5, 8 m/s), so every observation x
# wind combo any paper sweep ran is in the grid: 7 x 8 = 56 combos.
THRESHOLD_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
WIND_THRESHOLDS = [0.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0]

# name -> (locations, overrides). Every list is a swept dimension.
SWEEPS = {
    "capacity": ([BASE_LOC], {
        "battery_capacities": CAPACITIES, "horizons": [3000],
        "failure_penalties": [5.0], "start_datetimes": ["2025-06-10T00:00:00"]}),
    "duration": ([BASE_LOC], {
        "battery_capacities": [300.0],
        "horizons": [1000, 3000, 6000, 9000, 12000, 15000, 18000, 27000],
        "failure_penalties": [20.0], "start_datetimes": ["2025-01-01T00:00:00"]}),
    "penalty": ([BASE_LOC], {
        "battery_capacities": [300.0], "horizons": [3000],
        "failure_penalties": [0.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 100.0],
        "start_datetimes": ["2025-06-10T00:00:00"]}),
    "location": ([{"latitude": lat, "longitude": LOCATION_LON} for lat in LOCATION_LATS], {
        "battery_capacities": CAPACITIES, "horizons": [3000],
        "failure_penalties": [10.0], "start_datetimes": ["2025-01-01T00:00:00"]}),
    "startdate": ([BASE_LOC], {
        "battery_capacities": [300.0], "horizons": [8500], "failure_penalties": [15.0],
        "start_datetimes": [f"2025-{m:02d}-01T00:00:00" for m in (1, 3, 5, 7, 9)]}),
}

SMOKE_SWEEPS = {
    "capacity": ([BASE_LOC], {
        "battery_capacities": [250.0, 300.0], "horizons": [192],
        "failure_penalties": [5.0], "start_datetimes": ["2025-06-10T00:00:00"]}),
    "penalty": ([BASE_LOC], {
        "battery_capacities": [300.0], "horizons": [192],
        "failure_penalties": [5.0, 20.0], "start_datetimes": ["2025-06-10T00:00:00"]}),
}
SMOKE_THRESHOLD_VALUES = [0.1, 0.25]
SMOKE_WIND_THRESHOLDS = [3.0, 6.0]

# Full per-step histories feed the episode/trajectory figures; keep them only for the
# capacity sweep's optimal hist arms (histories dominate disk use elsewhere).
HISTORY_EPISODES = 32


def check_ocean(locs, min_km=100.0):
    """Exit if any site is on land or within `min_km` of a coastline (Natural Earth 10m).

    Skipped with a warning when cartopy is not installed (it is in the pvlib env).
    """
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError:
        print("[warn] cartopy not installed; sites were not checked for land")
        return
    land = unary_union(list(shpreader.Reader(
        shpreader.natural_earth("10m", "physical", "land")).geometries()))
    bad = []
    for loc in locs:
        p = Point(loc["longitude"], loc["latitude"])
        km = land.distance(p) * 111.0  # degrees -> km, adequate for a coastline margin
        if land.contains(p) or km < min_km:
            bad.append(f"{loc_tag(loc)} ({'on land' if land.contains(p) else f'{km:.0f} km offshore'})")
    if bad:
        raise SystemExit(f"[error] sites not over open ocean (< {min_km:.0f} km): " + ", ".join(bad))


def loc_tag(loc):
    return f"lat{loc['latitude']}_lon{loc['longitude']}"


def data_path_for(loc):
    return f"Data/EXPECTED_DATA/data_expected_{loc_tag(loc)}_15min.pkl"


MODEL_WORLDS = {"iidworld": "iid", "chainworld": "chain"}


def make_config(sweep, loc, arm, world, overrides, episodes, smoke, n_bins=N_WIND_BINS,
                block_days=BLOCK_LENGTH_DAYS):
    is_thresh = arm in ("thresh", "thresh_tune")
    keep_history = (sweep == "capacity" and world == "hist" and not is_thresh)
    cfg = {
        "episodes": episodes,
        "transition_model": "moderate",
        "solar_panel_model": "constant",
        "whale_series": "JoA2026",
        "energy_increment_wh": 5,
        "delta_t": 15,
        "include_optimal": not is_thresh,
        "save_states": False,
        "full_history_episodes": HISTORY_EPISODES if keep_history else 0,
        "threshold_values": [],
        "wind_thresholds": [],
        "locations": [{"latitude": loc["latitude"], "longitude": loc["longitude"],
                       "data_path": data_path_for(loc)}],
        "holdout": {**HOLDOUT, "rollout_years": "fit" if arm == "thresh_tune" else "eval"},
        "wind_chain": {"enabled": arm == "chain", "n_bins": n_bins},
    }
    cfg.update({k: list(v) for k, v in overrides.items()})
    if is_thresh:
        cfg["threshold_values"] = list(SMOKE_THRESHOLD_VALUES if smoke else THRESHOLD_VALUES)
        cfg["wind_thresholds"] = list(SMOKE_WIND_THRESHOLDS if smoke else WIND_THRESHOLDS)
    if world == "hist":
        cfg["historical_weather"] = {"enabled": True, "block_length_days": block_days}
    elif world in MODEL_WORLDS:
        cfg["synthetic_world"] = MODEL_WORLDS[world]
    return cfg


def describe(sweep, loc, arm, world, n_bins=N_WIND_BINS, block_days=BLOCK_LENGTH_DAYS):
    where = loc_tag(loc)
    years = (f"fit {HOLDOUT['fit_years'][0]}-{HOLDOUT['fit_years'][1]}, "
             f"episodes on {HOLDOUT['eval_years'][0]}-{HOLDOUT['eval_years'][1]}")
    if world in MODEL_WORLDS:
        who = {"iid": "iid-solved optimal policy",
               "chain": f"chain-solved ({n_bins}-bin quantile wind chain) optimal policy",
               "thresh": "threshold-policy grid"}[arm]
        return (f"Journal case '{sweep}' at {where}: {who} rolled out in the "
                f"{MODEL_WORLDS[world]} model's synthetic weather (fit "
                f"{HOLDOUT['fit_years'][0]}-{HOLDOUT['fit_years'][1]}); every policy in this "
                f"world sees identical weather.")
    if arm == "thresh":
        return (f"Journal case '{sweep}' at {where}: threshold-policy grid scored on the "
                f"evaluation years ({years}, {block_days:g}-day blocks); paired "
                f"episode-for-episode with the optimal arms.")
    if arm == "thresh_tune":
        return (f"Journal case '{sweep}' at {where}: threshold-policy grid rolled out on the "
                f"FIT years {HOLDOUT['fit_years'][0]}-{HOLDOUT['fit_years'][1]}, used only to "
                f"select each cell's threshold combo without seeing evaluation years.")
    solved = (f"chain-solved ({n_bins}-bin quantile wind chain)" if arm == "chain"
              else "iid-solved")
    world_txt = (f"on held-out historical bootstrap weather ({block_days:g}-day blocks)"
                 if world == "hist" else "in its own synthetic world (calibration)")
    return (f"Journal case '{sweep}' at {where}: {solved} optimal policy, episodes {world_txt} "
            f"({years}). Paired with the other optimal arm; configs differ only in "
            f"wind_chain.enabled.")


def emit(cfg, name, description, out_dir):
    ordered = {"experiment_name": name, "description": description}
    ordered.update(cfg)
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(ordered, f, sort_keys=False, default_flow_style=None)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="Config dir (default: configs/journal_chain_sweep[_smoke]).")
    ap.add_argument("--episodes", type=int, default=10000,
                    help="Episodes per simulation (default 10,000; see episodes_needed.py).")
    ap.add_argument("--native", action="store_true",
                    help="Also emit native-world iid/chain runs for the calibration check.")
    ap.add_argument("--tune", nargs="*", default=None, metavar="SWEEP",
                    help="Add the fit-years threshold-tuning arm (selection without evaluation "
                         "years); alone for every sweep, or followed by sweep names "
                         "(e.g. --tune penalty).")
    ap.add_argument("--model-worlds", action="store_true",
                    help="Emit iid/chain/thresh arms in the i.i.d. and chain models' synthetic "
                         "weather instead of historical weather (no-modelling-error "
                         "condition).")
    ap.add_argument("--bins", type=int, default=N_WIND_BINS,
                    help=f"Wind-chain bin count (default {N_WIND_BINS}).")
    ap.add_argument("--block-days", type=int, default=BLOCK_LENGTH_DAYS,
                    help=f"Bootstrap block length in days (default {BLOCK_LENGTH_DAYS}).")
    ap.add_argument("--sweeps", nargs="+", default=None, help="Only these sweeps.")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Only emit these arms (iid chain thresh thresh_tune).")
    ap.add_argument("--reuse-results", default=None,
                    help="Results dir of an earlier sweep: manifest cells for arms not "
                         "emitted point at its completed runs (named with --reuse-prefix).")
    ap.add_argument("--reuse-prefix", default="jcs",
                    help="Config-name prefix of the reused sweep (default jcs).")
    ap.add_argument("--prefix", default=None, help="Config-name prefix (default jcs).")
    ap.add_argument("--site", nargs=2, type=float, default=None, metavar=("LAT", "LON"),
                    help="Run the single-site sweeps (all but location) here instead of "
                         f"{loc_tag(BASE_LOC)}; must be open ocean.")
    ap.add_argument("--smoke", action="store_true", help="Tiny variant for pipeline testing.")
    args = ap.parse_args()
    block_days = args.block_days

    suffix = "_smoke" if args.smoke else ""
    out_dir = args.out or os.path.join(REPO_ROOT, "configs", f"journal_chain_sweep{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    episodes = 200 if args.smoke else args.episodes
    prefix = args.prefix or ("smoke_jcs" if args.smoke else "jcs")
    sweeps = SMOKE_SWEEPS if args.smoke else SWEEPS
    if args.sweeps:
        unknown = set(args.sweeps) - set(sweeps)
        if unknown:
            raise SystemExit(f"[error] --sweeps: unknown sweep(s) {sorted(unknown)}")
        sweeps = {k: v for k, v in sweeps.items() if k in args.sweeps}
    base_loc = BASE_LOC
    if args.site:
        # Move every single-site sweep (all but `location`) from the paper's base site.
        base_loc = {"latitude": args.site[0], "longitude": args.site[1]}
        sweeps = {k: ([base_loc] if locs == [BASE_LOC] else locs, ov)
                  for k, (locs, ov) in sweeps.items()}
    check_ocean({loc_tag(l): l for locs, _ in sweeps.values() for l in locs}.values())

    tune_sweeps = (set() if args.tune is None
                   else set(args.tune) if args.tune else set(sweeps))
    unknown = tune_sweeps - set(sweeps)
    if unknown:
        raise SystemExit(f"[error] --tune: unknown sweep(s) {sorted(unknown)}; "
                         f"choose from {sorted(sweeps)}")
    if args.model_worlds:
        base_arm_worlds = [(a, w) for w in MODEL_WORLDS for a in ("iid", "chain", "thresh")]
    else:
        base_arm_worlds = [("iid", "hist"), ("chain", "hist"), ("thresh", "hist")]
    if args.native:
        base_arm_worlds += [("iid", "native"), ("chain", "native")]
    if args.reuse_results:
        from compare_chain_sweep import latest_run_dir  # noqa: E402

    cells, n_solves, n_thresh = [], 0, 0
    for sweep, (locs, overrides) in sweeps.items():
        swept = {k: v for k, v in overrides.items() if len(v) > 1}
        swept_field, swept_values = next(iter(swept.items())) if swept else (None, [])
        arm_worlds = base_arm_worlds + ([("thresh_tune", "hist")] if sweep in tune_sweeps else [])
        for loc in locs:
            for arm, world in arm_worlds:
                pair_key = f"{prefix}_{sweep}_{loc_tag(loc)}_ARM_{world}"
                if args.arms and arm not in args.arms:
                    # Not re-run: point the manifest at the earlier sweep's run, if any.
                    reused = f"{args.reuse_prefix}_{sweep}_{loc_tag(loc)}_{arm}_{world}"
                    if args.reuse_results and latest_run_dir(args.reuse_results, reused):
                        cells.append({
                            "sweep": sweep, "location_id": loc_tag(loc), "arm": arm,
                            "world": world, "config_basename": reused,
                            "results_base": args.reuse_results.replace("\\", "/"),
                            "swept_field": swept_field, "swept_values": swept_values,
                            "pair_key": pair_key,
                            "world_key": f"{prefix}_{sweep}_{loc_tag(loc)}_{arm}_WORLD",
                        })
                    elif args.reuse_results:
                        print(f"[warn] no completed run to reuse for {reused}")
                    continue
                name = f"{prefix}_{sweep}_{loc_tag(loc)}_{arm}_{world}"
                cfg = make_config(sweep, loc, arm, world, overrides, episodes, args.smoke,
                                  n_bins=args.bins, block_days=block_days)
                path = emit(cfg, name, describe(sweep, loc, arm, world, args.bins, block_days),
                            out_dir)
                n_cells = 1
                for k in ("battery_capacities", "horizons", "failure_penalties",
                          "start_datetimes"):
                    n_cells *= len(cfg[k])
                if arm in ("thresh", "thresh_tune"):
                    n_thresh += n_cells * len(cfg["threshold_values"]) * len(cfg["wind_thresholds"])
                else:
                    n_solves += n_cells
                cells.append({
                    "sweep": sweep,
                    "location_id": loc_tag(loc),
                    "arm": arm,
                    "world": world,
                    "config_basename": name,
                    "config_path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
                    "swept_field": swept_field,
                    "swept_values": swept_values,
                    "pair_key": pair_key,
                    "world_key": f"{prefix}_{sweep}_{loc_tag(loc)}_{arm}_WORLD",
                })

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "configs/final_journal_configs (Journal of Aircraft paper cases)",
        "smoke": args.smoke,
        "thresholds": True,
        "threshold_values": list(SMOKE_THRESHOLD_VALUES if args.smoke else THRESHOLD_VALUES),
        "wind_thresholds": list(SMOKE_WIND_THRESHOLDS if args.smoke else WIND_THRESHOLDS),
        "threshold_selection": {s: ("fit-years tuning run" if s in tune_sweeps
                                    else "evaluation episodes") for s in sweeps},
        "wind_bins": {"mode": "quantile", "n_bins": args.bins},
        "block_length_days": block_days,
        "model_worlds": args.model_worlds,
        "holdout": HOLDOUT,
        "episodes": episodes,
        "full_history_episodes": HISTORY_EPISODES,
        "baseline": {"location_id": loc_tag(base_loc), "capacity": 300.0,
                     "start": "2025-06-10T00:00:00", "horizon": 3000},
        "cells": cells,
    }
    manifest_path = os.path.join(out_dir, "chain_vs_iid_sweep_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(cells)} configs -> {out_dir}")
    print(f"Manifest -> {manifest_path}")
    print(f"Optimal-policy solves: {n_solves}")
    print(f"Threshold-policy simulations: {n_thresh}")


if __name__ == "__main__":
    main()
