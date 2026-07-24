#!/usr/bin/env python3
"""Phase 7 readout: wind-chain benefit at delta_t=60 (native hourly) vs delta_t=15.

Pairs each dt60 wind sim against its same-cell dt60 iid sim on episode_index
(CRN holds within a dt), computes the paired dReward, and converts to
PER-MISSION-DAY units so it is comparable to the dt15 main sweep (whose totals
accrue over 4x more steps). Loads the dt15 main-sweep cells the same way from
results/markov_ablation. Reports per-location and pooled means with cell-bootstrap
CIs, plus dFailure and dFlight-hrs (flight_hrs is dt-corrected in the engine).

Usage (pvlib env, from SolarSimulator/):
    python Scripts/compare_dt_resolution.py
"""
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

from Scripts.compare_chain_sweep import latest_run_dir, read_episode_scalars  # noqa: E402
from Scripts.compare_markov_ablation import _boot_mean_ci, MATCH_KEYS  # noqa: E402

LOCATIONS = ("bering", "florida", "hawaii", "natlantic")
MISSION_DAYS = 60.0


def cell_deltas(results_base, prefix, n_expect=18):
    """Per-cell paired wind-iid deltas for one dt's results. Returns a long DataFrame."""
    rows = []
    for loc in LOCATIONS:
        run_i = latest_run_dir(results_base, f"{prefix}_{loc}_iid")
        run_w = latest_run_dir(results_base, f"{prefix}_{loc}_wind")
        if run_i is None or run_w is None:
            print(f"[warn] missing runs for {loc} under {results_base}")
            continue
        sum_i = pd.read_csv(os.path.join(run_i, "summary.csv"))
        sum_w = pd.read_csv(os.path.join(run_w, "summary.csv"))
        for df in (sum_i, sum_w):
            df["start_time"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        iid_by_key = {tuple(r[k] for k in MATCH_KEYS): r
                      for _, r in sum_i.iterrows()
                      if "Optimal" in str(r["simulation_type"])}
        n_cells = 0
        for _, rw in sum_w.iterrows():
            if "Optimal" not in str(rw["simulation_type"]):
                continue
            key = tuple(rw[k] for k in MATCH_KEYS)
            ri = iid_by_key.get(key)
            if ri is None:
                print(f"[warn] {loc}: no iid match for {key}")
                continue
            e_i = read_episode_scalars(run_i, ri["group"])
            e_w = read_episode_scalars(run_w, rw["group"])
            j = e_i.join(e_w, lsuffix="_i", rsuffix="_w", how="inner")
            rows.append({
                "location_id": loc,
                "battery_capacity": rw["battery_capacity"],
                "failure_penalty": rw["failure_penalty"],
                "season": "summer" if pd.to_datetime(rw["start_time"]).month == 6 else "winter",
                "d_reward_per_day": float(
                    (j["total_reward_w"] - j["total_reward_i"]).mean()) / MISSION_DAYS,
                "d_failure_pct": 100 * float((j["failure_w"] - j["failure_i"]).mean()),
                "d_flight_hrs": float((j["flight_hrs_w"] - j["flight_hrs_i"]).mean()),
                "reward_iid_per_day": float(j["total_reward_i"].mean()) / MISSION_DAYS,
                "n_eps": len(j),
            })
            n_cells += 1
        if n_cells != n_expect:
            print(f"[warn] {loc}: {n_cells} cells (expected {n_expect})")
    return pd.DataFrame(rows)


def report(cells, label, rng, n_boot):
    print(f"\n=== {label}: wind - iid, paired ===")
    for loc in LOCATIONS:
        sub = cells[cells["location_id"] == loc]
        m, lo, hi = _boot_mean_ci(sub["d_reward_per_day"], rng, n_boot)
        print(f"  {loc:10s} dReward/day {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"dFail {sub['d_failure_pct'].mean():+.2f}pp  "
              f"dFlightHrs {sub['d_flight_hrs'].mean():+.1f}")
    m, lo, hi = _boot_mean_ci(cells["d_reward_per_day"], rng, n_boot)
    sig = "SIG+" if lo > 0 else ("SIG-" if hi < 0 else "ns")
    print(f"  {'POOLED':10s} dReward/day {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {sig}")
    return m, lo, hi


def main():
    rng = np.random.default_rng(0)
    n_boot = 10000
    dt60 = cell_deltas(os.path.join(REPO_ROOT, "results", "markov_dt60"), "mkvdt60")
    dt15 = cell_deltas(os.path.join(REPO_ROOT, "results", "markov_ablation"), "mkv")

    m60 = report(dt60, "dt=60 (native hourly, 1440 steps)", rng, n_boot)
    m15 = report(dt15, "dt=15 (interpolated, 5760 steps, main sweep)", rng, n_boot)

    out_dir = os.path.join(REPO_ROOT, "results", "markov_dt60", "_analysis")
    os.makedirs(out_dir, exist_ok=True)
    dt60.assign(dt=60).pipe(
        lambda a: pd.concat([a, dt15.assign(dt=15)], ignore_index=True)
    ).to_csv(os.path.join(out_dir, "dt_resolution_cells.csv"), index=False)

    ratio = m60[0] / m15[0] if m15[0] else float("nan")
    print(f"\nBenefit retention at dt=60: {100*ratio:.0f}% of the dt=15 per-day benefit")
    print(f"Wrote {out_dir}\\dt_resolution_cells.csv")


if __name__ == "__main__":
    main()
