#!/usr/bin/env python3
"""Risk-harvest frontier representation of policy performance.

The episode reward is linear in the failure penalty:

    E[reward at penalty p] = harvest - p * P(failure),
    harvest = mean(total_reward + p_run * failure)      (exact, per episode)

so the pair (harvest, failure probability) is a SUFFICIENT STATISTIC for the entire
penalty dimension: any reward-vs-penalty result is recoverable from frontier
coordinates by arithmetic, and a swept penalty is just the slope of the tangent line
that selects an operating point. Harvest is reported as a CAPTURE RATIO -- the
fraction of the total whale-observation value available in the mission window
(sum of the diurnal series; an always-flying upper bound) -- giving a physical 0-1
scale comparable across sites, seasons, durations, and timesteps.

Families:
  * DP arms (iid / wind): one frontier point per swept penalty (the solver re-optimizes
    at each price of failure). Points read from analysis cells CSVs.
  * Threshold: the Pareto hull of all tunings, computed from a single fp=5 run of the
    threshold grid (behavior is penalty-invariant; harvest = reward + 5*failure).

Outputs (to --out): frontier_points.csv, threshold_hull.csv (+ all combos),
matched_risk.csv, frontier.png (matplotlib). See docs/markov_ablation_experiment.md
section 9 and results/penalty_ext/_analysis/frontier_report.html for the worked example.

Usage (pvlib env, from SolarSimulator/):
    python Scripts/frontier_analysis.py                      # thesis + penalty_ext defaults
    python Scripts/frontier_analysis.py --cap 300 --days 60 --out ../results/penalty_ext/_analysis
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)

from BaseClasses.whale_base import WhaleRewardSeriesFactory  # noqa: E402

LOCS = ("bering", "florida", "gulf", "hawaii", "natlantic")


def available_per_day(delta_t_min=15):
    """Expected whale-observation value per day if flying every step."""
    steps = int(1440 // delta_t_min)
    return float(WhaleRewardSeriesFactory.create_series("real", steps,
                                                        delta_t_min=delta_t_min).sum())


def dp_frontier(cells_csvs, cap, avail, extra_pens_csv=None):
    """Frontier points per arm from analysis cells CSVs (thesis-style columns).

    cells_csvs: list of markov_ablation_cells.csv paths with failure_pct_{arm} and
    nonpenalty_reward_{arm}; extra_pens_csv: penalty_curve_cells.csv-style file whose
    harvest is reconstructed as r + pen*f/100 (for penalties not in the cells CSVs).
    Returns {arm: DataFrame(pen, f, cr)} pooled over all cells, plus the per-cell rows.
    """
    rows = []
    for path in cells_csvs:
        df = pd.read_csv(path)
        if "battery_capacity" in df.columns:
            df = df[df["battery_capacity"] == cap]
        for _, r in df.iterrows():
            for arm in ("iid", "wind"):
                if f"failure_pct_{arm}" not in df.columns:
                    continue
                rows.append({"arm": arm, "pen": float(r["failure_penalty"]),
                             "loc": r["location_id"], "f": float(r[f"failure_pct_{arm}"]),
                             "h": float(r[f"nonpenalty_reward_{arm}"])})
    if extra_pens_csv and os.path.exists(extra_pens_csv):
        pex = pd.read_csv(extra_pens_csv)
        seen = {p for p in pd.DataFrame(rows)["pen"].unique()} if rows else set()
        for _, r in pex.iterrows():
            if float(r["pen"]) in seen:
                continue
            for arm in ("iid", "wind"):
                rows.append({"arm": arm, "pen": float(r["pen"]), "loc": r["loc"],
                             "f": float(r[f"f_{arm}"]),
                             "h": float(r[f"r_{arm}"]) + float(r["pen"]) * float(r[f"f_{arm}"]) / 100})
    cells = pd.DataFrame(rows)
    out = {}
    for arm, g in cells.groupby("arm"):
        pts = g.groupby("pen").agg(f=("f", "mean"), h=("h", "mean")).reset_index()
        pts["cr"] = 100 * pts["h"] / avail
        out[arm] = pts.sort_values("f").reset_index(drop=True)
    return out, cells


def threshold_family(thresh_run_glob, avail, run_penalty=5.0):
    """(all combo points, Pareto hull) pooled over the runs matching the glob."""
    combo = {}
    for pattern_loc in LOCS:
        runs = sorted(glob.glob(thresh_run_glob.format(loc=pattern_loc)))
        if not runs:
            continue
        df = pd.read_csv(runs[-1])
        df = df[df["simulation_type"].str.contains("Threshold", case=False, na=False)]
        for _, r in df.iterrows():
            key = (float(r["observation_threshold"]), float(r["wind_threshold"]))
            combo.setdefault(key, []).append(
                (100 * float(r["failure_percentage"]),
                 float(r["average_reward"]) + run_penalty * float(r["failure_percentage"])))
    pts = pd.DataFrame([{"obs": k[0], "wth": k[1],
                         "f": float(np.mean([v[0] for v in vals])),
                         "cr": 100 * float(np.mean([v[1] for v in vals])) / avail}
                        for k, vals in combo.items()])
    pts = pts.sort_values(["f", "cr"], ascending=[True, False]).reset_index(drop=True)
    hull, best = [], -np.inf
    for _, p in pts.iterrows():
        if p["cr"] > best:
            hull.append(p)
            best = p["cr"]
    return pts, pd.DataFrame(hull).reset_index(drop=True)


def matched_risk(frontiers, hull, fs=(8, 10, 15, 20, 25, 30, 40, 50)):
    """Capture ratio at fixed failure rates, interpolated along each frontier."""
    def interp(pts, f):
        pts = pts.sort_values("f")
        return float(np.interp(f, pts["f"], pts["cr"])) \
            if pts["f"].min() <= f <= pts["f"].max() else np.nan
    rows = []
    for f in fs:
        rows.append({"failure_pct": f,
                     **{arm: round(interp(p, f), 2) for arm, p in frontiers.items()},
                     "thresh_hull": round(interp(hull, f), 2)})
    return pd.DataFrame(rows)


def plot_frontier(frontiers, th_pts, hull, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.scatter(th_pts["f"], th_pts["cr"].clip(lower=0), s=12, color="#898781",
               alpha=0.25, label="all threshold tunings", zorder=1)
    ax.plot(hull["f"], hull["cr"].clip(lower=0), "--", color="#52514e", lw=1.8,
            marker="o", mfc="white", ms=6, label="threshold Pareto hull", zorder=3)
    for arm, color, lw in (("iid", "#898781", 1.8), ("wind", "#2a78d6", 2.6)):
        p = frontiers[arm]
        ax.plot(p["f"], p["cr"].clip(lower=0), "-o", color=color, lw=lw, ms=5,
                label=f"{arm} (penalty-swept)", zorder=4)
        for _, r in p.iterrows():
            if r["pen"] in (2.5, 10, 40, 160, 640):
                ax.annotate(f"p{r['pen']:g}", (r["f"], max(r["cr"], 0)),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=7.5, color="#898781")
    ax.set_xlabel("mission failure probability (%)")
    ax.set_ylabel("capture ratio (% of available observation value)")
    ax.set_title("Risk-harvest frontier (penalty = operating-point selection)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=float, default=300.0)
    ap.add_argument("--days", type=float, default=60.0)
    ap.add_argument("--delta-t", type=int, default=15)
    ap.add_argument("--cells", nargs="*", default=[os.path.join(
        REPO_ROOT, "results", "thesis_sweep", "_analysis_batgrid", "markov_ablation_cells.csv")],
        help="Analysis cells CSVs providing failure_pct_/nonpenalty_reward_ columns.")
    ap.add_argument("--extra-pens", default=os.path.join(
        REPO_ROOT, "results", "penalty_ext", "_analysis", "penalty_curve_cells.csv"))
    ap.add_argument("--thresh-glob", default=os.path.join(
        REPO_ROOT, "results", "penalty_ext", "pex_{loc}_thresh", "*", "summary.csv"),
        help="Glob (with {loc}) for the threshold-family runs at fp=5.")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "penalty_ext", "_analysis"))
    args = ap.parse_args()

    avail = available_per_day(args.delta_t) * args.days
    print(f"available value: {avail:.1f} over {args.days:g} days")

    frontiers, cells = dp_frontier(args.cells, args.cap, avail, args.extra_pens)
    th_pts, hull = threshold_family(args.thresh_glob, avail)
    mr = matched_risk(frontiers, hull)

    os.makedirs(args.out, exist_ok=True)
    pd.concat([f.assign(arm=a) for a, f in frontiers.items()]).to_csv(
        os.path.join(args.out, "frontier_points.csv"), index=False)
    th_pts.to_csv(os.path.join(args.out, "threshold_combos.csv"), index=False)
    hull.to_csv(os.path.join(args.out, "threshold_hull.csv"), index=False)
    mr.to_csv(os.path.join(args.out, "matched_risk.csv"), index=False)
    plot_frontier(frontiers, th_pts, hull, os.path.join(args.out, "frontier.png"))

    print("\nmatched-risk capture ratios (%):")
    print(mr.to_string(index=False))
    print(f"\nWrote frontier_points/threshold_combos/threshold_hull/matched_risk.csv "
          f"+ frontier.png -> {args.out}")


if __name__ == "__main__":
    main()
