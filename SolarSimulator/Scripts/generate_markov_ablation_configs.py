#!/usr/bin/env python3
"""Generate the 5-arm Markov-ablation sweep configs (solar / wind / joint / iid / thresh).

Emits harness YAML configs (see harness/run_experiment.py) in matched arm sets that
differ ONLY in the chain toggles (and, for the threshold arm, ``include_optimal`` +
the threshold grids), so any performance difference between arms of a location is
attributable to the persistence formulation alone. All arms are evaluated on the same
historical block-bootstrap weather (CRN pairing holds episode-for-episode because the
provider is reset with seed 0 per batch and the bootstrap draw is its first RNG use).

Experiment architecture, checkpoint rules, and the decision log live in
docs/markov_ablation_experiment.md -- keep that document current.

Arms:
    iid    -- optimal solve, both chains off
    wind   -- optimal solve, 5-bin wind chain (explicit prebuilt _windchain_wind5.pkl;
              NEVER the derived _windchain.pkl path, which holds a 12-bin artifact)
    solar  -- optimal solve, solar chain at --solar-bins (explicit _solarchain_g{n}.pkl)
    joint  -- optimal solve, both chains (value table (5*n_g, |S|, T))
    thresh -- threshold-policy grid, no solve, SINGLE failure penalty (threshold
              behavior is penalty-invariant; analysis re-weights per penalty from
              episode scalars)

Usage (any Python with pyyaml):
    python Scripts/generate_markov_ablation_configs.py                       # Phase 4 full sweep
    python Scripts/generate_markov_ablation_configs.py --smoke-solar-bins   # Phase 2 solar bin study
    python Scripts/generate_markov_ablation_configs.py --smoke --solar-bins 3  # Phase 3 all-arm smoke
"""
import argparse
import json
import os
from datetime import datetime

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = ("iid", "wind", "solar", "joint", "thresh")
N_WIND_BINS = 5
BLOCK_LENGTH_DAYS = 7

LOCATIONS = {
    "florida":   {"latitude": 27.0, "longitude": -79.5},    # calm, variable (baseline; CRN histories kept here)
    "hawaii":    {"latitude": 21.0, "longitude": -158.0},   # steady trade winds
    "natlantic": {"latitude": 45.0, "longitude": -45.0},    # storm track
    "bering":    {"latitude": 65.0, "longitude": -169.0},   # high-latitude, windy, solar-starved winters
}
# Fifth site for the thesis sweep (historical record on disk from the Phase 7 pre-check).
GULF = {"gulf": {"latitude": 30.0, "longitude": -90.0}}     # Gulf of Mexico
BASELINE_LOC = "florida"

# Phase 8 thesis sweep grids (docs/markov_ablation_experiment.md section 8).
THESIS_CAPS = [float(c) for c in range(100, 601, 50)]                  # 11 values
THESIS_PENALTIES = [2.5, 5.0, 10.0, 20.0, 40.0, 80.0]                  # 6, log2-spaced
THESIS_DURATION_STEPS = [2880, 5760, 8640, 17280, 25920, 35040]        # 30..365 days
THESIS_DURATION_PENALTIES = [5.0, 20.0, 80.0]

# Phase 9 penalty-extension study (docs section 9): resolve the >80 regime and give the
# threshold family a fine tuning grid (the 4x4 grid saturates its conservative edge at
# pen 80). Cells pool with the thesis batgrid 300 Wh rows (same conditions/weather).
PEX_PENALTIES = [160.0, 320.0, 640.0]
PEX_THRESHOLD_VALUES = [round(0.05 * i, 2) for i in range(11)]         # 0 .. 0.5
PEX_WIND_THRESHOLDS = [float(w) for w in range(0, 15, 2)]              # 0 .. 14

CAPACITIES = [150.0, 300.0, 600.0]
PENALTIES = [5.0, 20.0, 80.0]
START_DATES = ["2025-06-10T00:00:00", "2025-12-10T00:00:00"]  # summer + winter
HORIZON = 5760            # 60 days at delta_t 15
EPISODES = 3000

# Threshold-benchmark grid (failbin convention). Behavior is penalty-invariant, so the
# thresh arm runs at THRESHOLD_PENALTY only; compare_markov_ablation.py recomputes
# rewards for the other penalties from episode scalars.
THRESHOLD_VALUES = [0.0, 0.1, 0.2, 0.3]
WIND_THRESHOLDS = [0.0, 4.0, 8.0, 12.0]
THRESHOLD_PENALTY = 5.0

# Phase 2: solar bin-resolution smoke (florida, solar arm only + iid reference).
SOLAR_BIN_CANDIDATES = [2, 3, 4, 5]
SMOKE_SOLAR = {"capacity": 300.0, "penalty": 20.0, "horizon": 2880,
               "episodes": 2000, "start": START_DATES[0]}

# Phase 3: all-arm smoke gate (florida, 5 days), plus one winter config.
SMOKE = {"capacity": 300.0, "penalty": 5.0, "horizon": 480,
         "episodes": 300, "start": START_DATES[0], "winter_start": START_DATES[1]}

FULL_HISTORY_EPISODES = 8  # baseline location only; feeds verify_crn


def data_path_for(loc, dt=15):
    return (f"Data/EXPECTED_DATA/data_expected_lat{loc['latitude']}"
            f"_lon{loc['longitude']}_{dt}min.pkl")


def wind5_path_for(loc, dt=15):
    """5-bin quantile artifact FITTED AT the model timestep (per-step transitions are
    not transferable across dt; a 60-min run needs a 60-min-fitted chain)."""
    return data_path_for(loc, dt).replace(f"_{dt}min.pkl", f"_{dt}min_windchain_wind5.pkl")


def solar_path_for(loc, n_bins, dt=15):
    """Explicit per-bin-count solar artifact so bin studies never collide/rebuild."""
    return data_path_for(loc, dt).replace(f"_{dt}min.pkl", f"_{dt}min_solarchain_g{n_bins}.pkl")


def arm_toggles(arm, loc, solar_bins):
    """The ONLY keys that differ between arms of a location (plus thresh's grids)."""
    wind_on = arm in ("wind", "joint")
    solar_on = arm in ("solar", "joint")
    toggles = {
        "wind_chain": (
            {"enabled": True, "n_bins": N_WIND_BINS, "path": wind5_path_for(loc)}
            if wind_on else {"enabled": False}
        ),
        "solar_chain": (
            {"enabled": True, "n_bins": solar_bins, "path": solar_path_for(loc, solar_bins)}
            if solar_on else {"enabled": False}
        ),
    }
    if arm == "thresh":
        toggles.update({
            "include_optimal": False,
            "threshold_values": list(THRESHOLD_VALUES),
            "wind_thresholds": list(WIND_THRESHOLDS),
            "failure_penalties": [THRESHOLD_PENALTY],
        })
    return toggles


def base_config(loc_name, loc, capacities, penalties, horizon, episodes,
                start_dates=None, start=None, dt=15):
    cfg = {
        "battery_capacities": list(capacities),
        "horizons": [horizon],
        "failure_penalties": list(penalties),
        "threshold_values": [],      # optimal-policy arms only; thresh arm overrides
        "wind_thresholds": [],
        "episodes": episodes,
        "transition_model": "moderate",
        "solar_panel_model": "constant",
        "whale_series": "real",
        "energy_increment_wh": 5,
        "delta_t": dt,
        "include_optimal": True,
        "save_states": False,
        "full_history_episodes": FULL_HISTORY_EPISODES if loc_name == BASELINE_LOC else 0,
        "locations": [{
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "data_path": data_path_for(loc, dt),
        }],
        "historical_weather": {
            "enabled": True,
            "block_length_days": BLOCK_LENGTH_DAYS,
        },
    }
    if start_dates is not None:
        cfg["start_datetimes"] = list(start_dates)
    else:
        cfg["start_datetime"] = start
    return cfg


def emit(cfg, name, description, out_dir):
    ordered = {"experiment_name": name, "description": description}
    ordered.update(cfg)
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(ordered, f, sort_keys=False, default_flow_style=None)
    return path


def solve_count(cfg):
    n_dates = len(cfg.get("start_datetimes", [cfg.get("start_datetime")]))
    n = (len(cfg["battery_capacities"]) * len(cfg["failure_penalties"])
         * len(cfg["horizons"]) * n_dates)
    if cfg.get("include_optimal", True):
        return n, 0
    return 0, n * len(cfg["threshold_values"]) * len(cfg["wind_thresholds"])


def make_full(solar_bins):
    """Phase 4: 4 locations x 5 arms, caps x penalties x dates inside each config."""
    for loc_name, loc in LOCATIONS.items():
        for arm in ARMS:
            cfg = base_config(loc_name, loc, CAPACITIES, PENALTIES, HORIZON,
                              EPISODES, start_dates=START_DATES)
            cfg.update(arm_toggles(arm, loc, solar_bins))
            desc = (f"Markov ablation, arm '{arm}' @ {loc_name}: "
                    f"{'threshold-policy benchmark (no solve, penalty-invariant, fp=5 only)' if arm == 'thresh' else 'optimal policy'} "
                    f"evaluated on historical bootstrap weather; paired with the other "
                    f"arms episode-for-episode (seed-0 CRN). See docs/markov_ablation_experiment.md.")
            yield f"mkv_{loc_name}_{arm}", cfg, desc, arm, loc_name


def make_smoke_solar():
    """Phase 2: florida iid reference + solar arm at each candidate bin count."""
    loc_name, loc = BASELINE_LOC, LOCATIONS[BASELINE_LOC]
    p = SMOKE_SOLAR
    common = dict(capacities=[p["capacity"]], penalties=[p["penalty"]],
                  horizon=p["horizon"], episodes=p["episodes"], start=p["start"])
    cfg = base_config(loc_name, loc, **common)
    cfg.update(arm_toggles("iid", loc, solar_bins=None))
    yield ("mkv2_florida_iid", cfg,
           "Solar bin-resolution smoke: IID reference arm. See docs/markov_ablation_experiment.md Phase 2.",
           "iid", loc_name)
    for g in SOLAR_BIN_CANDIDATES:
        cfg = base_config(loc_name, loc, **common)
        cfg.update(arm_toggles("solar", loc, solar_bins=g))
        yield (f"mkv2_florida_solar_g{g}", cfg,
               f"Solar bin-resolution smoke: solar chain with n_bins={g}. Checkpoint picks the "
               f"smallest n_bins within CI of the best paired dReward vs IID (knee rule).",
               "solar", loc_name)


def make_solar_res(bins_list):
    """Phase 6: solar bin-resolution study at ALL locations, full sweep conditions.

    Per location: one iid reference + one solar-only config per bin count. Same
    caps x penalties x seasons x 60-day horizon x 3000 episodes as the full sweep, so
    each solar-g pairs against the same-condition iid on identical bootstrap weather.
    """
    for loc_name, loc in LOCATIONS.items():
        cfg = base_config(loc_name, loc, CAPACITIES, PENALTIES, HORIZON, EPISODES,
                          start_dates=START_DATES)
        cfg.update(arm_toggles("iid", loc, solar_bins=None))
        yield (f"mkvsr_{loc_name}_iid", cfg,
               f"Solar-resolution study: IID reference @ {loc_name}. "
               f"See docs/markov_ablation_experiment.md Phase 6.",
               "iid", loc_name)
        for g in bins_list:
            cfg = base_config(loc_name, loc, CAPACITIES, PENALTIES, HORIZON, EPISODES,
                              start_dates=START_DATES)
            cfg.update(arm_toggles("solar", loc, solar_bins=g))
            yield (f"mkvsr_{loc_name}_solar_g{g}", cfg,
                   f"Solar-resolution study: solar-only chain n_bins={g} @ {loc_name}, "
                   f"full conditions. Paired vs the same-condition iid.",
                   f"solar_g{g}", loc_name)


def make_thesis(family):
    """Phase 8 thesis sweep (docs section 8): battery x penalty response surface at 60 d
    ('batgrid') and mission-duration curve to one year ('duration'), arms iid/wind/thresh
    at 5 locations (the 4 standard sites + Gulf of Mexico)."""
    locs = {**LOCATIONS, **GULF}
    for loc_name, loc in locs.items():
        for arm in ("iid", "wind", "thresh"):
            if family == "batgrid":
                cfg = base_config(loc_name, loc, THESIS_CAPS, THESIS_PENALTIES,
                                  5760, EPISODES, start_dates=START_DATES)
            else:
                cfg = base_config(loc_name, loc, [300.0], THESIS_DURATION_PENALTIES,
                                  2880, EPISODES, start=START_DATES[0])
                cfg["horizons"] = list(THESIS_DURATION_STEPS)
            cfg.update(arm_toggles(arm, loc, solar_bins=None))
            yield (f"ths_{loc_name}_{family}_{arm}", cfg,
                   f"Thesis sweep '{family}', arm '{arm}' @ {loc_name}: "
                   f"{'11 capacities x 6 penalties x 2 seasons at 60 days' if family == 'batgrid' else '6 mission durations (30-365 d) x 3 penalties at 300 Wh'}. "
                   f"Paired on seed-0 bootstrap weather. See docs/markov_ablation_experiment.md section 8.",
                   arm, loc_name)


def make_penalty_ext(smoke=False):
    """Phase 9: iid+wind at extreme penalties {160,320,640} plus a FINE threshold grid
    (88 combos, fp=5 only, reweighted across the full penalty ladder in analysis).
    300 Wh / 60 d / both seasons so cells pool with the thesis batgrid."""
    locs = {**LOCATIONS, **GULF}
    if smoke:
        loc_name, loc = BASELINE_LOC, LOCATIONS[BASELINE_LOC]
        for arm in ("iid", "wind"):
            cfg = base_config(loc_name, loc, [300.0], [640.0], 5760, 300,
                              start=START_DATES[0])
            cfg.update(arm_toggles(arm, loc, solar_bins=None))
            yield (f"pexs_florida_{arm}", cfg,
                   f"Penalty-ext smoke, arm '{arm}': pen 640 at full 60-d horizon "
                   f"(reward-scale sanity at extreme penalty).", arm, loc_name)
        cfg = base_config(loc_name, loc, [300.0], [5.0], 5760, 300, start=START_DATES[0])
        cfg.update(arm_toggles("thresh", loc, solar_bins=None))
        cfg["threshold_values"] = list(PEX_THRESHOLD_VALUES)
        cfg["wind_thresholds"] = list(PEX_WIND_THRESHOLDS)
        yield ("pexs_florida_thresh", cfg,
               "Penalty-ext smoke: fine 11x8 threshold grid builds and runs (88 combos).",
               "thresh", loc_name)
        return
    for loc_name, loc in locs.items():
        for arm in ("iid", "wind"):
            cfg = base_config(loc_name, loc, [300.0], PEX_PENALTIES, 5760, EPISODES,
                              start_dates=START_DATES)
            cfg.update(arm_toggles(arm, loc, solar_bins=None))
            yield (f"pex_{loc_name}_{arm}", cfg,
                   f"Penalty-extension, arm '{arm}' @ {loc_name}: penalties "
                   f"{{160, 320, 640}} at 300 Wh / 60 d / both seasons. Pools with the "
                   f"thesis batgrid 300 Wh cells. See docs section 9.", arm, loc_name)
        cfg = base_config(loc_name, loc, [300.0], [THRESHOLD_PENALTY], 5760, EPISODES,
                          start_dates=START_DATES)
        cfg.update(arm_toggles("thresh", loc, solar_bins=None))
        cfg["threshold_values"] = list(PEX_THRESHOLD_VALUES)
        cfg["wind_thresholds"] = list(PEX_WIND_THRESHOLDS)
        yield (f"pex_{loc_name}_thresh", cfg,
               f"Penalty-extension, FINE threshold grid (11 obs x 8 wind = 88 combos) @ "
               f"{loc_name}, fp=5 only; analysis reweights across 2.5-640 and compares "
               f"against the coarse 4x4 benchmark.", "thresh", loc_name)


def make_thesis_smoke():
    """Phase 8 smoke gate: florida mini batgrid (all 3 arms) + one full-length 365-day
    config (validates year wrap-around, memory, and the long-solve timing anchor)."""
    loc_name, loc = BASELINE_LOC, LOCATIONS[BASELINE_LOC]
    for arm in ("iid", "wind", "thresh"):
        cfg = base_config(loc_name, loc, [150.0, 300.0, 600.0], [5.0, 40.0],
                          5760, 300, start=START_DATES[0])
        cfg.update(arm_toggles(arm, loc, solar_bins=None))
        yield (f"thss_florida_{arm}", cfg,
               f"Thesis smoke, arm '{arm}': mini batgrid 3 caps x 2 pens @ 60 d, 300 episodes.",
               arm, loc_name)
    for arm in ("iid", "wind"):
        cfg = base_config(loc_name, loc, [150.0], [20.0], 35040, 300,
                          start=START_DATES[0])
        cfg.update(arm_toggles(arm, loc, solar_bins=None))
        yield (f"thss_florida_year_{arm}", cfg,
               f"Thesis smoke, arm '{arm}': one full 365-day mission (year wrap-around, "
               f"memory, long-solve timing anchor).",
               arm, loc_name)
    # Gulf provisioning probe: tiny config so the smoke also builds/validates Gulf artifacts.
    g_name, g = next(iter(GULF.items()))
    cfg = base_config(g_name, g, [300.0], [20.0], 480, 300, start=START_DATES[0])
    cfg.update(arm_toggles("wind", g, solar_bins=None))
    yield ("thss_gulf_wind", cfg,
           "Thesis smoke: Gulf of Mexico provisioning probe (histcube + 5-bin chain build) "
           "with a 5-day wind-arm solve.",
           "wind", g_name)


def make_dt60():
    """Phase 7: timestep-resolution check — iid + wind arms at delta_t=60.

    60-min steps use the NATIVE hourly record (resampling to 60 min is the identity,
    so no interpolation-inflated persistence), horizon 1440 steps = 60 days, and a
    wind chain fitted at the 60-min step. Same grid otherwise, so the wind-vs-iid
    chain benefit is directly comparable to the dt=15 main sweep.
    """
    for loc_name, loc in LOCATIONS.items():
        for arm in ("iid", "wind"):
            cfg = base_config(loc_name, loc, CAPACITIES, PENALTIES, 1440, EPISODES,
                              start_dates=START_DATES, dt=60)
            cfg["wind_chain"] = (
                {"enabled": True, "n_bins": N_WIND_BINS, "path": wind5_path_for(loc, dt=60)}
                if arm == "wind" else {"enabled": False})
            cfg["solar_chain"] = {"enabled": False}
            yield (f"mkvdt60_{loc_name}_{arm}", cfg,
                   f"Timestep-resolution check, arm '{arm}' @ {loc_name}: delta_t=60 min "
                   f"(native hourly weather, no interpolation), 60-day horizon (1440 steps). "
                   f"See docs/markov_ablation_experiment.md Phase 7.",
                   arm, loc_name)


def make_smoke(solar_bins):
    """Phase 3: florida, all 5 arms, 5-day horizon; plus one winter joint config."""
    loc_name, loc = BASELINE_LOC, LOCATIONS[BASELINE_LOC]
    p = SMOKE
    common = dict(capacities=[p["capacity"]], penalties=[p["penalty"]],
                  horizon=p["horizon"], episodes=p["episodes"])
    for arm in ARMS:
        cfg = base_config(loc_name, loc, start=p["start"], **common)
        cfg.update(arm_toggles(arm, loc, solar_bins))
        yield (f"mkv3_florida_{arm}", cfg,
               f"All-arm smoke gate, arm '{arm}'. Gate criteria in docs/markov_ablation_experiment.md Phase 3.",
               arm, loc_name)
    cfg = base_config(loc_name, loc, start=p["winter_start"], **common)
    cfg.update(arm_toggles("joint", loc, solar_bins))
    yield ("mkv3_florida_joint_winter", cfg,
           "All-arm smoke gate: winter-start (2025-12-10) joint config confirming winter "
           "bootstrap alignment before the full sweep's December cells.",
           "joint", loc_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solar-bins", type=int, default=3,
                    help="Solar chain n_bins for solar/joint arms (set from the Phase 2 checkpoint).")
    ap.add_argument("--smoke-solar-bins", action="store_true",
                    help="Phase 2: emit the florida solar bin-resolution study configs.")
    ap.add_argument("--smoke", action="store_true",
                    help="Phase 3: emit the florida all-arm smoke-gate configs.")
    ap.add_argument("--solar-res", action="store_true",
                    help="Phase 6: emit the all-location solar bin-resolution study (iid + solar-only "
                         "at each --solar-bins-list value, full sweep conditions).")
    ap.add_argument("--solar-bins-list", default="2,3,4,6,8",
                    help="Comma-separated solar n_bins for --solar-res (default 2,3,4,6,8).")
    ap.add_argument("--dt60", action="store_true",
                    help="Phase 7: emit the delta_t=60 timestep-resolution check (iid + wind arms).")
    ap.add_argument("--thesis", action="store_true",
                    help="Phase 8: emit the thesis sweep (batgrid + duration families, 2 config dirs).")
    ap.add_argument("--thesis-smoke", action="store_true",
                    help="Phase 8 smoke gate: florida mini batgrid + one 365-day config + Gulf probe.")
    ap.add_argument("--penalty-ext", action="store_true",
                    help="Phase 9: extreme-penalty study (iid+wind at 160/320/640 + fine threshold grid).")
    ap.add_argument("--penalty-ext-smoke", action="store_true",
                    help="Phase 9 smoke gate: florida pen-640 iid+wind + fine-grid threshold config.")
    ap.add_argument("--out", default=None, help="Output config dir override.")
    args = ap.parse_args()
    if sum([args.smoke, args.smoke_solar_bins, args.solar_res, args.dt60,
            args.thesis, args.thesis_smoke, args.penalty_ext, args.penalty_ext_smoke]) > 1:
        ap.error("phase flags are mutually exclusive; pick one.")

    if args.penalty_ext:
        emit_set("penalty_ext", "penalty_ext", list(make_penalty_ext()), args)
        return
    if args.penalty_ext_smoke:
        emit_set("penalty_ext_smoke", "penalty_ext_smoke",
                 list(make_penalty_ext(smoke=True)), args)
        return
    if args.thesis:
        # Two families -> two dirs (they run with different worker counts).
        for family in ("batgrid", "duration"):
            emit_set(f"thesis_sweep_{family}", f"thesis_{family}",
                     list(make_thesis(family)), args)
        return
    if args.thesis_smoke:
        emit_set("thesis_sweep_smoke", "thesis_smoke", list(make_thesis_smoke()), args)
        return

    if args.smoke_solar_bins:
        default_dir, mode = "markov_ablation_smoke_solar", "smoke_solar_bins"
        cells = list(make_smoke_solar())
    elif args.smoke:
        default_dir, mode = "markov_ablation_smoke", "smoke"
        cells = list(make_smoke(args.solar_bins))
    elif args.solar_res:
        default_dir, mode = "markov_solar_res", "solar_res"
        bins_list = [int(b) for b in args.solar_bins_list.split(",") if b.strip()]
        cells = list(make_solar_res(bins_list))
    elif args.dt60:
        default_dir, mode = "markov_dt60", "dt60"
        cells = list(make_dt60())
    else:
        default_dir, mode = "markov_ablation", "full"
        cells = list(make_full(args.solar_bins))

    emit_set(default_dir, mode, cells, args)


def emit_set(default_dir, mode, cells, args):
    out_dir = (args.out if args.out and not mode.startswith("thesis_")
               else os.path.join(REPO_ROOT, "configs", default_dir))
    os.makedirs(out_dir, exist_ok=True)

    manifest_cells, n_solves, n_thresh = [], 0, 0
    for name, cfg, desc, arm, loc_name in cells:
        path = emit(cfg, name, desc, out_dir)
        s, t = solve_count(cfg)
        n_solves += s
        n_thresh += t
        manifest_cells.append({
            "arm": arm,
            "location_id": loc_name,
            "config_basename": name,
            "config_path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
            "wind_chain_path": cfg["wind_chain"].get("path") if cfg["wind_chain"]["enabled"] else None,
            "solar_chain_path": cfg["solar_chain"].get("path") if cfg["solar_chain"]["enabled"] else None,
            "solar_bins": cfg["solar_chain"].get("n_bins") if cfg["solar_chain"]["enabled"] else None,
        })

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "arms": list(ARMS),
        "wind_bins": {"mode": "quantile", "n_bins": N_WIND_BINS,
                      "artifact_suffix": "_windchain_wind5.pkl"},
        "solar_bins": (SOLAR_BIN_CANDIDATES if mode == "smoke_solar_bins"
                       else [int(b) for b in args.solar_bins_list.split(",") if b.strip()]
                       if mode == "solar_res" else args.solar_bins),
        "threshold_values": list(THRESHOLD_VALUES),
        "wind_thresholds": list(WIND_THRESHOLDS),
        "threshold_penalty": THRESHOLD_PENALTY,
        "penalties": (list(THESIS_PENALTIES) if mode == "thesis_batgrid"
                      else list(THESIS_DURATION_PENALTIES) if mode == "thesis_duration"
                      else list(PENALTIES)),
        "block_length_days": BLOCK_LENGTH_DAYS,
        "baseline_location": BASELINE_LOC,
        "cells": manifest_cells,
    }
    manifest_path = os.path.join(out_dir, "markov_ablation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest_cells)} configs -> {out_dir}")
    print(f"Manifest -> {manifest_path}")
    print(f"Optimal-policy solves: {n_solves}; threshold sims (no solve): {n_thresh}")


if __name__ == "__main__":
    main()
