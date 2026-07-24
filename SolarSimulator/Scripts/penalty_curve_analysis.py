#!/usr/bin/env python3
"""Phase 9 readout: policy performance vs failure penalty, 2.5 -> 640.

Pools three sources at 300 Wh / 60 d / 5 sites / 2 seasons on shared seed-0 weather:
  * thesis batgrid iid+wind cells at penalties 2.5-80 (results/thesis_sweep),
  * penalty_ext iid+wind cells at 160/320/640 (results/penalty_ext),
  * the penalty_ext FINE threshold grid (88 combos, run at fp=5), re-weighted per
    episode to EVERY penalty on the ladder; the legacy 4x4 grid is a subset of the
    fine grid, so the coarse-grid benchmark is recomputed from the same runs.

Per (site, season, penalty): best threshold by re-weighted mean; paired deltas
wind-iid, wind-thresh, iid-thresh with episode bootstrap. Pooled-by-penalty output
with cell-bootstrap CIs.

Usage (pvlib env, from SolarSimulator/):
    python Scripts/penalty_curve_analysis.py
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
from Scripts.compare_markov_ablation import reweighted_eps  # noqa: E402

LOCS = ["bering", "florida", "gulf", "hawaii", "natlantic"]
PENALTIES = [2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
THESIS_PENS = {2.5, 5.0, 10.0, 20.0, 40.0, 80.0}
COARSE_OBS = {0.0, 0.1, 0.2, 0.3}
COARSE_WIND = {0.0, 4.0, 8.0, 12.0}
CAP = 300.0
N_BOOT = 5000


def opt_eps(loc, arm, pen, start):
    """Episode scalars for an optimal arm at (pen, start), from thesis or penalty_ext."""
    if pen in THESIS_PENS:
        base, name = os.path.join(REPO_ROOT, "results", "thesis_sweep"), f"ths_{loc}_batgrid_{arm}"
    else:
        base, name = os.path.join(REPO_ROOT, "results", "penalty_ext"), f"pex_{loc}_{arm}"
    run = latest_run_dir(base, name)
    if run is None:
        return None
    df = pd.read_csv(os.path.join(run, "summary.csv"))
    df["start_time"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d")
    m = df[(df["battery_capacity"] == CAP) & (df["failure_penalty"] == pen)
           & (df["start_time"] == start)
           & df["simulation_type"].str.contains("Optimal", na=False)]
    if m.empty:
        return None
    return read_episode_scalars(run, m.iloc[0]["group"])


def thresh_runs(loc):
    run = latest_run_dir(os.path.join(REPO_ROOT, "results", "penalty_ext"), f"pex_{loc}_thresh")
    df = pd.read_csv(os.path.join(run, "summary.csv"))
    df["start_time"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d")
    return run, df[df["simulation_type"].str.contains("Threshold", case=False, na=False)]


def best_thresh(run, rows, pen, coarse=False):
    """(best_eps, obs, wind) at target penalty; optionally restricted to the 4x4 subset."""
    best, best_mean, combo = None, -np.inf, None
    for _, r in rows.iterrows():
        if coarse and not (r["observation_threshold"] in COARSE_OBS
                           and r["wind_threshold"] in COARSE_WIND):
            continue
        eps = reweighted_eps(read_episode_scalars(run, r["group"]), 5.0, pen)
        m = float(eps["total_reward"].mean())
        if m > best_mean:
            best, best_mean, combo = eps, m, (r["observation_threshold"], r["wind_threshold"])
    return best, combo


def main():
    rng = np.random.default_rng(0)
    cells = []
    for loc in LOCS:
        t_run, t_rows = thresh_runs(loc)
        for start in ("2025-06-10", "2025-12-10"):
            t_sub = t_rows[t_rows["start_time"] == start]
            for pen in PENALTIES:
                e_i = opt_eps(loc, "iid", pen, start)
                e_w = opt_eps(loc, "wind", pen, start)
                if e_i is None or e_w is None:
                    print(f"[warn] missing optimal cells {loc}/{start}/pen{pen}")
                    continue
                e_tf, combo_f = best_thresh(t_run, t_sub, pen, coarse=False)
                e_tc, combo_c = best_thresh(t_run, t_sub, pen, coarse=True)
                j = e_i[["total_reward", "failure"]].join(
                    e_w[["total_reward", "failure"]], lsuffix="_i", rsuffix="_w").join(
                    e_tf[["total_reward"]].rename(columns={"total_reward": "r_tf"})).join(
                    e_tc[["total_reward"]].rename(columns={"total_reward": "r_tc"}))
                cells.append({
                    "loc": loc, "start": start, "pen": pen,
                    "r_iid": float(j["total_reward_i"].mean()),
                    "r_wind": float(j["total_reward_w"].mean()),
                    "r_thresh_fine": float(j["r_tf"].mean()),
                    "r_thresh_coarse": float(j["r_tc"].mean()),
                    "f_iid": 100 * float(e_i["failure"].mean()),
                    "f_wind": 100 * float(e_w["failure"].mean()),
                    "f_thresh_fine": 100 * float(e_tf["failure"].mean()),
                    "d_wind_iid": float((j["total_reward_w"] - j["total_reward_i"]).mean()),
                    "d_wind_thresh": float((j["total_reward_w"] - j["r_tf"]).mean()),
                    "d_iid_thresh": float((j["total_reward_i"] - j["r_tf"]).mean()),
                    "d_grid": float((j["r_tf"] - j["r_tc"]).mean()),
                    "obs_f": combo_f[0], "wth_f": combo_f[1],
                })
    df = pd.DataFrame(cells)
    out_dir = os.path.join(REPO_ROOT, "results", "penalty_ext", "_analysis")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "penalty_curve_cells.csv"), index=False)

    def ci(v):
        v = np.asarray(v, float)
        s = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
        return v.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)

    print(f"\n=== penalty curve (300 Wh / 60 d, {len(df)} cells over 5 sites x 2 seasons) ===")
    print("pen     r_iid   r_wind  r_thF   r_thC  | wind-iid [CI]        wind-thF [CI]        "
          "| f_iid f_wind f_thF | best obs/wind")
    rows = []
    for pen, g in df.groupby("pen"):
        wi = ci(g["d_wind_iid"]); wt = ci(g["d_wind_thresh"])
        rows.append({"pen": pen, "d_wind_iid": wi, "d_wind_thresh": wt})
        print(f"{pen:6.1f} {g['r_iid'].mean():7.2f} {g['r_wind'].mean():7.2f} "
              f"{g['r_thresh_fine'].mean():7.2f} {g['r_thresh_coarse'].mean():7.2f} | "
              f"{wi[0]:+7.2f} [{wi[1]:+6.2f},{wi[2]:+6.2f}] "
              f"{wt[0]:+7.2f} [{wt[1]:+6.2f},{wt[2]:+6.2f}] | "
              f"{g['f_iid'].mean():5.1f} {g['f_wind'].mean():5.1f} {g['f_thresh_fine'].mean():5.1f} | "
              f"{g['obs_f'].mean():.2f}/{g['wth_f'].mean():.1f}")
    print("\nfine-vs-coarse threshold grid gain by penalty (mean):")
    print(df.groupby("pen")["d_grid"].mean().round(2).to_string())
    print(f"\nWrote {out_dir}\\penalty_curve_cells.csv")


if __name__ == "__main__":
    main()
