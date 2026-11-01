#!/usr/bin/env python3
"""Generate a wind-bin-count study: chain policies solved with different numbers of wind bins.

The journal chain sweep (generate_journal_chain_sweep_configs.py) solved the chain policy
with 4 quantile wind bins. This study re-solves it with other bin counts on chosen journal
cases, with settings identical to the journal sweep, and scores every count on the same
episodes.

Cases (--cases; the journal sweep each one comes from in brackets):
    penalty    all 10 penalties 0-100, 300 Wh, 3000 stages, start 06-10, lat 30 lon -75
               [penalty]
    duration   horizons 1000, 3000, 6000, 300 Wh, penalty 20, start 01-01 [duration]
    startdate  starts Jan/Mar/May/Jul/Sep 1, 300 Wh, 8500 stages, penalty 15 [startdate]
    lat20, latm20
               lat 20 / lat -20 along lon -25, battery 100-600 Wh, 3000 stages,
               penalty 10, start 01-01 [location]: the sites with the smallest and
               largest 4-bin chain gains. --capacities narrows the battery list.

Arms:
    chain{n}     chain-solved optimal policy with n quantile wind bins, n in --bins.
    thresh_tune  cases in --tune: the 7 x 8 threshold grid rolled out on the FIT years,
                 so the threshold combo is picked without seeing evaluation weather.
    thresh       cases in --thresh: the grid scored on the evaluation years. Normally
                 reused from the journal sweep results instead (same episodes).

Hold-out (--holdout FIT0 FIT1 EVAL0 EVAL1, default the standard 1950-2002 / 2003-2022):
the standard split pairs these runs with the journal sweep's, and a 4-bin arm reproduces
the journal chain arm episode for episode (compare_wind_bins.py checks it). An inner
split of the fit years (1950-1985 / 1986-2002) selects a bin count without touching
2003-2022 (select_wind_bins.py).

Chain files: under the hold-out the derived chain name carries the bin count
(<expected data>_fit<years>_windchain_q<n>.pkl, run_sim.resolve_artifact_paths), so the
counts never share a file and run_chain_sweep.py's provisioning builds each one from the
fit years.

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/generate_wind_bins_study_configs.py                  # penalty + duration
    python Scripts/generate_wind_bins_study_configs.py --cases lat20 latm20 startdate \\
        --bins 4 8 12 16 --tune lat20 latm20 startdate --prefix wbs2 --out ../configs/wind_bins_sites
    python Scripts/generate_wind_bins_study_configs.py --holdout 1950 1985 1986 2002 \\
        --cases penalty duration lat20 latm20 --capacities 100 200 300 400 500 600 \\
        --bins 4 6 8 12 16 24 --prefix wsel --out ../configs/wind_bins_select
    python Scripts/run_chain_sweep.py --configs ../configs/wind_bins_study \\
        --out C:/Users/bepstein8/sim_results/wind_bins_study --resume
    python Scripts/compare_wind_bins.py --results C:/Users/bepstein8/sim_results/wind_bins_study \\
        --journal-results C:/Users/bepstein8/sim_results/journal_chain_sweep
"""
import argparse
import json
import os
import sys
from datetime import datetime

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from Scripts.generate_journal_chain_sweep_configs import (  # noqa: E402
    BASE_LOC, HOLDOUT, LOCATION_LON, REPO_ROOT, SWEEPS, THRESHOLD_VALUES, WIND_THRESHOLDS,
    emit, loc_tag, make_config)

BIN_COUNTS = [4, 6, 8, 12, 16]
# case -> (journal sweep, location, overrides)
CASES = {
    "penalty": ("penalty", BASE_LOC, dict(SWEEPS["penalty"][1])),
    "duration": ("duration", BASE_LOC, {**SWEEPS["duration"][1], "horizons": [1000, 3000, 6000]}),
    "startdate": ("startdate", BASE_LOC, dict(SWEEPS["startdate"][1])),
    "lat20": ("location", {"latitude": 20.0, "longitude": LOCATION_LON},
              dict(SWEEPS["location"][1])),
    "latm20": ("location", {"latitude": -20.0, "longitude": LOCATION_LON},
               dict(SWEEPS["location"][1])),
}
SWEPT_FIELD = {"penalty": "failure_penalty", "duration": "horizon", "startdate": "start_time",
               "location": "battery_capacity"}
JOURNAL_PREFIX = "jcs"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "configs", "wind_bins_study"),
                    help="Config dir (default: configs/wind_bins_study).")
    ap.add_argument("--episodes", type=int, default=10000,
                    help="Episodes per simulation; keep the journal sweep's 10,000 so reused "
                         "arms pair episode for episode.")
    ap.add_argument("--cases", nargs="+", default=["penalty", "duration"], choices=list(CASES))
    ap.add_argument("--bins", type=int, nargs="+", default=BIN_COUNTS,
                    help=f"Wind bin counts to solve (default {BIN_COUNTS}).")
    ap.add_argument("--tune", nargs="*", default=["duration"], metavar="CASE",
                    help="Cases that get the fit-years threshold-tuning arm (default duration; "
                         "the journal sweep already has penalty's).")
    ap.add_argument("--thresh", nargs="*", default=[], metavar="CASE",
                    help="Cases that get their own evaluation-years threshold grid.")
    ap.add_argument("--capacities", type=float, nargs="+", default=None,
                    help="Battery capacities for the location cases (default all 11).")
    ap.add_argument("--holdout", type=int, nargs=4, default=None,
                    metavar=("FIT0", "FIT1", "EVAL0", "EVAL1"),
                    help="Year split (default the standard 1950 2002 2003 2022).")
    ap.add_argument("--prefix", default="wbs", help="Config-name prefix (default wbs).")
    args = ap.parse_args()

    holdout = (HOLDOUT if args.holdout is None else
               {"fit_years": args.holdout[:2], "eval_years": args.holdout[2:]})
    standard = holdout == HOLDOUT
    if standard and 4 not in args.bins:
        print("[warn] 4 bins not included: compare_wind_bins.py cannot run its "
              "reproduction check against the journal sweep's chain arm")

    os.makedirs(args.out, exist_ok=True)
    runs, n_solves, n_thresh = [], 0, 0
    cases = {}
    for case in args.cases:
        sweep, loc, overrides = CASES[case]
        overrides = dict(overrides)
        if sweep == "location" and args.capacities:
            overrides["battery_capacities"] = list(args.capacities)
        cases[case] = {"journal_sweep": sweep, "location_id": loc_tag(loc),
                       "swept_field": SWEPT_FIELD[sweep], "settings": overrides}
        arms = [(f"chain{n}", n) for n in args.bins]
        arms += [("thresh_tune", None)] if case in (args.tune or []) else []
        arms += [("thresh", None)] if case in (args.thresh or []) else []
        for arm, n in arms:
            name = f"{args.prefix}_{case}_{loc_tag(loc)}_{arm}_hist"
            base_arm = "chain" if n is not None else arm
            cfg = make_config(sweep, loc, base_arm, "hist", overrides, args.episodes, False,
                              n_bins=n or 4)
            cfg["holdout"] = {**holdout, "rollout_years": cfg["holdout"]["rollout_years"]}
            cfg["full_history_episodes"] = 0
            fit = f"{holdout['fit_years'][0]}-{holdout['fit_years'][1]}"
            ev = f"{holdout['eval_years'][0]}-{holdout['eval_years'][1]}"
            if n is not None:
                desc = (f"Wind-bin study, case '{case}' at {loc_tag(loc)}: optimal policy "
                        f"solved with a {n}-bin quantile wind chain (fit {fit}), episodes on "
                        f"held-out historical weather {ev}.")
            elif arm == "thresh_tune":
                desc = (f"Wind-bin study, case '{case}' at {loc_tag(loc)}: threshold grid "
                        f"rolled out on the FIT years {fit}, used only to pick each cell's "
                        f"threshold combo without seeing evaluation weather.")
            else:
                desc = (f"Wind-bin study, case '{case}' at {loc_tag(loc)}: threshold grid "
                        f"scored on the evaluation years {ev}.")
            path = emit(cfg, name, desc, args.out)
            n_cells = 1
            for k in ("battery_capacities", "horizons", "failure_penalties", "start_datetimes"):
                n_cells *= len(cfg[k])
            if n is None:
                n_thresh += n_cells * len(THRESHOLD_VALUES) * len(WIND_THRESHOLDS)
            else:
                n_solves += n_cells
            runs.append({"sweep": case, "location_id": loc_tag(loc), "arm": arm,
                         "n_bins": n, "config_basename": name,
                         "config_path": os.path.relpath(path, REPO_ROOT).replace("\\", "/")})

    # Journal-sweep runs on the same episodes (standard split only).
    journal = {}
    if standard:
        for case, info in cases.items():
            sweep, loc = info["journal_sweep"], info["location_id"]
            journal[case] = {a: f"{JOURNAL_PREFIX}_{sweep}_{loc}_{a}_hist"
                             for a in ("iid", "chain", "thresh")}
            if sweep == "penalty":
                journal[case]["thresh_tune"] = f"{JOURNAL_PREFIX}_penalty_{loc}_thresh_tune_hist"
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Chain policy reward by wind-bin count",
        "bin_counts": args.bins,
        "holdout": holdout,
        "episodes": args.episodes,
        "cases": cases,
        "runs": runs,
        "journal_runs": journal,
    }
    manifest_path = os.path.join(args.out, "wind_bins_study_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(runs)} configs -> {args.out}")
    print(f"Manifest -> {manifest_path}")
    print(f"Optimal-policy solves: {n_solves}")
    print(f"Threshold-policy simulations: {n_thresh}")


if __name__ == "__main__":
    main()
