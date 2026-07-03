#!/usr/bin/env python3
"""Generate paired chain-vs-iid sweep configs for the wind-persistence evaluation.

Emits harness YAML configs (see harness/run_experiment.py) in matched pairs that differ
ONLY in ``wind_chain.enabled`` (arm) and ``historical_weather`` (eval world), so any
performance difference between the two arms of a pair is attributable to the Markov-chain
wind formulation alone. A manifest JSON records every cell and its pair key for the
downstream comparison script (Scripts/compare_chain_sweep.py).

Design (star + factorial):
    capgrid   -- full battery-capacity x location grid (one config per location so the
                 solver's value-table filenames, which omit location, never collide)
    startdate -- 1-D start-date sweep at the baseline location
    duration  -- 1-D mission-duration sweep at the baseline location

Each sweep is emitted for arm in {iid, chain} x world in {hist, native}:
    hist   -- episodes rolled out on real historical weather (block bootstrap); the FAIR
              comparison. Paired runs see identical weather episode-for-episode because
              the bootstrap provider is reset with a fixed seed per batch.
    native -- episodes rolled out in the arm's own assumed synthetic world; measures the
              solver's self-predicted performance (calibration analysis).

Usage (any Python with pyyaml):
    python Scripts/generate_chain_sweep_configs.py
    python Scripts/generate_chain_sweep_configs.py --smoke
    python Scripts/generate_chain_sweep_configs.py --full-date-grid
"""
import argparse
import json
import os
from datetime import datetime

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = ("iid", "chain")
WORLDS = ("hist", "native")
# Equal-occupancy (quantile) wind bins, edges derived per location from its historical
# record at artifact-build time. No bin_edges key in the configs -> quantile mode.
N_WIND_BINS = 4
BLOCK_LENGTH_DAYS = 7

LOCATIONS = [
    {"latitude": 30.0, "longitude": -90.0},    # Gulf of Mexico (baseline; artifacts prebuilt)
    {"latitude": 20.0, "longitude": -159.0},   # Hawaii trade winds
    {"latitude": 45.0, "longitude": -100.0},   # continental US
    {"latitude": 58.0, "longitude": -161.0},   # Bering Sea
]

BASELINE = {
    "location": LOCATIONS[0],
    "capacity": 300.0,
    "start": "2020-07-01T00:00:00",
    "horizon": 2880,               # 30 days at delta_t 15
}

CAPACITIES = [150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
START_DATES = [f"2020-{m:02d}-01T00:00:00" for m in (1, 3, 5, 7, 9, 11)]
HORIZONS = [672, 1344, 2880, 5760]  # 7 / 14 / 30 / 60 days

# Threshold-benchmark grid (same as harness/examples/journal_threshold_vs_optimal_*).
# Threshold policies use no value table and, in the hist world, roll out on the same
# seed-0 bootstrap weather as the optimal arms, so ONE arm-agnostic config per scenario
# benchmarks against both iid and chain with episode-level pairing intact.
THRESHOLD_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
WIND_THRESHOLDS = [0.0, 3.0, 6.0, 9.0, 12.0]
SMOKE_THRESHOLD_VALUES = [0.1, 0.25]
SMOKE_WIND_THRESHOLDS = [6.0, 12.0]

SMOKE = {
    "capacities": [250.0, 300.0],
    "start_dates": ["2020-05-01T00:00:00", "2020-07-01T00:00:00"],
    "horizons": [192, 300],
    "horizon": 300,
    "episodes": 200,
    "full_history_episodes": 8,
}


def loc_tag(loc):
    return f"lat{loc['latitude']}_lon{loc['longitude']}"


def data_path_for(loc):
    return f"Data/EXPECTED_DATA/data_expected_{loc_tag(loc)}_15min.pkl"


def base_config(loc, arm, world, episodes, full_history_episodes):
    """Shared skeleton; sweeps overwrite the swept list. Arms differ ONLY in the toggles."""
    cfg = {
        "battery_capacities": [BASELINE["capacity"]],
        "horizons": [BASELINE["horizon"]],
        "start_datetime": BASELINE["start"],
        "failure_penalties": [5.0],
        "threshold_values": [],      # optimal-policy arms only
        "wind_thresholds": [],
        "episodes": episodes,
        "transition_model": "moderate",
        "solar_panel_model": "constant",
        "whale_series": "real",
        "energy_increment_wh": 5,
        "delta_t": 15,
        "include_optimal": True,
        "save_states": False,
        "full_history_episodes": full_history_episodes,
        "locations": [{
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "data_path": data_path_for(loc),
        }],
        "wind_chain": {
            "enabled": arm == "chain",
            "n_bins": N_WIND_BINS,
        },
    }
    if world == "hist":
        cfg["historical_weather"] = {
            "enabled": True,
            "block_length_days": BLOCK_LENGTH_DAYS,
        }
    return cfg


def emit(cfg, name, out_dir):
    cfg = dict(cfg)
    cfg_ordered = {"experiment_name": name, "description": cfg.pop("description")}
    cfg_ordered.update(cfg)
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg_ordered, f, sort_keys=False, default_flow_style=None)
    return path


def threshold_overrides(smoke=False):
    """Config overrides turning an arm config into an arm-agnostic threshold benchmark."""
    return {
        "threshold_values": list(SMOKE_THRESHOLD_VALUES if smoke else THRESHOLD_VALUES),
        "wind_thresholds": list(SMOKE_WIND_THRESHOLDS if smoke else WIND_THRESHOLDS),
        "include_optimal": False,
        # Scalars only: 35 combos x every scenario would need ~19 GB of full histories.
        "full_history_episodes": 0,
    }


def make_cells(smoke=False, full_date_grid=False, thresholds=False):
    """Yield (sweep, loc, arm, world, overrides, swept_field, swept_values) tuples."""
    caps = SMOKE["capacities"] if smoke else CAPACITIES
    dates = SMOKE["start_dates"] if smoke else START_DATES
    horizons = SMOKE["horizons"] if smoke else HORIZONS
    h_base = SMOKE["horizon"] if smoke else BASELINE["horizon"]

    # Threshold benchmark: hist world only (arm-agnostic there -- the policy ignores the
    # wind chain and the bootstrap weather is identical across arms by CRN).
    arm_worlds = [(a, w) for a in ARMS for w in WORLDS]
    if thresholds:
        arm_worlds = arm_worlds + [("thresh", "hist")]

    cells = []
    locations = [LOCATIONS[0]] if smoke else LOCATIONS
    for loc in locations:
        for arm, world in arm_worlds:
            cells.append((
                "capgrid", loc, arm, world,
                {"battery_capacities": list(caps), "horizons": [h_base]},
                "battery_capacities", list(caps),
            ))

    date_locs = locations if full_date_grid else [LOCATIONS[0]]
    for loc in date_locs:
        for arm, world in arm_worlds:
            cells.append((
                "startdate", loc, arm, world,
                {"start_datetimes": list(dates), "horizons": [h_base]},
                "start_datetimes", list(dates),
            ))

    for arm, world in arm_worlds:
        cells.append((
            "duration", LOCATIONS[0], arm, world,
            {"horizons": list(horizons)},
            "horizons", list(horizons),
        ))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="Output config dir (default: configs/chain_vs_iid_sweep[_smoke]).")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny fast variant for end-to-end pipeline testing.")
    ap.add_argument("--full-date-grid", action="store_true",
                    help="Emit the start-date sweep for every location (location x month heatmap).")
    ap.add_argument("--thresholds", action="store_true",
                    help="Also emit arm-agnostic threshold-policy benchmark configs (hist world).")
    args = ap.parse_args()

    suffix = "_smoke" if args.smoke else ""
    out_dir = args.out or os.path.join(REPO_ROOT, "configs", f"chain_vs_iid_sweep{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    episodes = SMOKE["episodes"] if args.smoke else 3000
    fhe = SMOKE["full_history_episodes"] if args.smoke else 64
    prefix = "smoke_cvi" if args.smoke else "cvi"

    manifest_cells = []
    n_solves = 0
    n_threshold_sims = 0
    for sweep, loc, arm, world, overrides, swept_field, swept_values in make_cells(
            args.smoke, args.full_date_grid, args.thresholds):
        name = f"{prefix}_{sweep}_{loc_tag(loc)}_{arm}_{world}"
        # Full per-step histories are only consumed by the event-aligned composites and
        # trajectory figures, which read the baseline-location capgrid hist pair -- keep
        # them there and store scalars only everywhere else (histories dominate disk use).
        keep_history = (sweep == "capgrid" and world == "hist"
                        and loc_tag(loc) == loc_tag(BASELINE["location"]))
        cfg = base_config(loc, arm, world, episodes, fhe if keep_history else 0)
        cfg.update(overrides)
        if arm == "thresh":
            cfg.update(threshold_overrides(args.smoke))
        if "start_datetimes" in cfg:
            cfg.pop("start_datetime", None)
        if arm == "thresh":
            cfg["description"] = (
                f"Threshold-policy benchmark for sweep '{sweep}' at {loc_tag(loc)}: the full "
                f"observation-threshold x wind-threshold grid rolled out on historical bootstrap "
                f"weather. No value-function solve; arm-agnostic (weather draws are identical to "
                f"both optimal arms episode-for-episode, so paired comparisons hold)."
            )
        else:
            cfg["description"] = (
                f"Chain-vs-IID evaluation sweep '{sweep}' at {loc_tag(loc)}: "
                f"{'chain-solved' if arm == 'chain' else 'iid-solved'} optimal policy, episodes "
                f"{'on historical bootstrap weather' if world == 'hist' else 'in its own synthetic world'}. "
                f"Paired with the other arm; configs differ only in wind_chain.enabled"
                f"{'' if world == 'native' else ' (weather draws are identical across the pair)'}."
            )
        path = emit(cfg, name, out_dir)
        # solves = product of swept dims (single failure penalty, optimal only)
        n_caps = len(cfg["battery_capacities"])
        n_dates = len(cfg.get("start_datetimes", [None]))
        n_h = len(cfg["horizons"])
        n_cells = n_caps * n_dates * n_h
        if arm == "thresh":
            n_threshold_sims += n_cells * len(cfg["threshold_values"]) * len(cfg["wind_thresholds"])
        else:
            n_solves += n_cells
        manifest_cells.append({
            "sweep": sweep,
            "location_id": loc_tag(loc),
            "arm": arm,
            "world": world,
            "config_basename": name,
            "config_path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
            "swept_field": swept_field,
            "swept_values": swept_values,
            "pair_key": f"{prefix}_{sweep}_{loc_tag(loc)}_ARM_{world}",
            "world_key": f"{prefix}_{sweep}_{loc_tag(loc)}_{arm}_WORLD",
        })

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "smoke": args.smoke,
        "full_date_grid": args.full_date_grid,
        "thresholds": args.thresholds,
        "threshold_values": list(SMOKE_THRESHOLD_VALUES if args.smoke else THRESHOLD_VALUES)
        if args.thresholds else [],
        "wind_thresholds": list(SMOKE_WIND_THRESHOLDS if args.smoke else WIND_THRESHOLDS)
        if args.thresholds else [],
        "wind_bins": {"mode": "quantile", "n_bins": N_WIND_BINS},
        "block_length_days": BLOCK_LENGTH_DAYS,
        "episodes": episodes,
        "full_history_episodes": fhe,
        "baseline": {
            "location_id": loc_tag(BASELINE["location"]),
            "capacity": BASELINE["capacity"] if not args.smoke else SMOKE["capacities"][-1],
            "start": BASELINE["start"] if not args.smoke else SMOKE["start_dates"][-1],
            "horizon": BASELINE["horizon"] if not args.smoke else SMOKE["horizon"],
        },
        "cells": manifest_cells,
    }
    manifest_path = os.path.join(out_dir, "chain_vs_iid_sweep_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest_cells)} configs -> {out_dir}")
    print(f"Manifest -> {manifest_path}")
    print(f"Total optimal-policy solves across all configs: {n_solves}")
    if args.thresholds:
        print(f"Total threshold-policy simulations (no solve): {n_threshold_sims}")


if __name__ == "__main__":
    main()
