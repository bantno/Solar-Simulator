#!/usr/bin/env python3
"""analyze_dim_sweeps.py -- aggregate the dimension sweeps (penalty / battery / duration)
comparing chain(12 bins) vs IID-optimal vs best-threshold, and render one figure per dimension.

Reads results/failbin_dim_sweeps/<sweep_<dim>_<arm>>/<ts>/*.h5 (latest ts per config), keys each
HDF5 group by (location, swept value), and for the threshold arm takes the best grid point per
cell by mean reward.

Outputs (to --out): dim_sweep_metrics.csv, figS1_penalty.png, figS2_battery.png,
figS3_duration.png, DIM_SWEEPS.md.
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["iid", "chain12", "thresh"]
ARM_LABEL = {"iid": "IID-optimal", "chain12": "Chain (12 bins)", "thresh": "Threshold (best)"}
ARM_COLOR = {"iid": "#4C72B0", "chain12": "#DD8452", "thresh": "#8C8C8C"}
DIMS = {"penalty": ("failure_penalty", "failure penalty"),
        "battery": ("battery_capacity", "battery capacity [Wh]"),
        "duration": ("horizon", "mission duration [days]")}
LOC_NAME = {"lat27.0_lon-79.5": "florida", "lat21.0_lon-158.0": "hawaii",
            "lat45.0_lon-45.0": "natlantic", "lat65.0_lon-169.0": "bering"}
LOC_ORDER = ["florida", "hawaii", "natlantic", "bering"]
LOC_LABEL = {"florida": "Florida", "hawaii": "Hawaii", "natlantic": "N. Atlantic", "bering": "Bering"}
DELTA_T = 15


def _attr(grp, key, default=None):
    v = grp.attrs.get(key, default)
    return v.decode() if isinstance(v, bytes) else v


def _ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def load(results_dir):
    rows = []
    for dim in DIMS:
        for arm in ARMS:
            base = os.path.join(results_dir, f"sweep_{dim}_{arm}")
            ts_dirs = sorted(glob.glob(os.path.join(base, "*")))
            if not ts_dirs:
                continue
            h5s = glob.glob(os.path.join(ts_dirs[-1], "*.h5"))
            if not h5s:
                continue
            with h5py.File(h5s[0], "r") as f:
                for gname in f.keys():
                    g = f[gname]
                    rewards = None
                    if "episode_scalars" in g and "total_reward" in g["episode_scalars"]:
                        rewards = g["episode_scalars"]["total_reward"][:]
                    rows.append(dict(
                        dim=dim, arm=arm,
                        location=LOC_NAME.get(_attr(g, "location_id"), _attr(g, "location_id")),
                        failure_penalty=float(_attr(g, "failure_penalty", np.nan)),
                        battery_capacity=float(_attr(g, "battery_capacity", np.nan)),
                        horizon=float(_attr(g, "horizon", np.nan)),
                        obs_thr=_attr(g, "observation_threshold"),
                        wind_thr=_attr(g, "wind_threshold"),
                        mean_reward=float(_attr(g, "average_reward", np.nan)),
                        failure_pct=float(_attr(g, "failure_percentage", np.nan)),
                        flight_hrs=float(_attr(g, "average_flight_hrs", np.nan)),
                        _rewards=rewards,
                    ))
    return rows


def collapse(rows):
    """One row per (dim, arm, location, swept value); threshold -> best grid point."""
    out = []
    for dim, (key, _) in DIMS.items():
        sub = [r for r in rows if r["dim"] == dim]
        cells = {}
        for r in sub:
            cells.setdefault((r["arm"], r["location"], r[key]), []).append(r)
        for (arm, loc, val), rs in cells.items():
            best = max(rs, key=lambda r: (r["mean_reward"] if np.isfinite(r["mean_reward"]) else -1e18))
            lo, hi = _ci(best["_rewards"]) if best["_rewards"] is not None else (np.nan, np.nan)
            out.append(dict(dim=dim, arm=arm, location=loc, swept=val,
                            mean_reward=best["mean_reward"], ci_lo=lo, ci_hi=hi,
                            failure_pct=best["failure_pct"], flight_hrs=best["flight_hrs"],
                            note=(f"obs={best['obs_thr']}, wind={best['wind_thr']}"
                                  if arm == "thresh" else "")))
    return pd.DataFrame(out)


def fig_dim(df, dim, out):
    key, xlabel = DIMS[dim]
    sub = df[df.dim == dim]
    locs = [l for l in LOC_ORDER if l in set(sub.location)]
    fig, axes = plt.subplots(1, len(locs), figsize=(3.5 * len(locs), 4.2), squeeze=False, sharex=True)
    for j, loc in enumerate(locs):
        ax = axes[0][j]
        for arm in ARMS:
            s = sub[(sub.location == loc) & (sub.arm == arm)].sort_values("swept")
            if not len(s):
                continue
            x = s.swept.values
            if dim == "duration":
                x = x * DELTA_T / (60 * 24)   # stages -> days
            y = s.mean_reward.values
            yerr = [y - s.ci_lo.values, s.ci_hi.values - y]
            ax.errorbar(x, y, yerr=yerr, marker="o", ms=4, lw=1.7, capsize=2.5,
                        color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        ax.set_title(LOC_LABEL.get(loc, loc))
        ax.set_xlabel(xlabel)
        if dim == "battery":
            ax.set_xticks([150, 300, 450, 600])
        if dim == "penalty":
            ax.set_xticks([5, 10, 20, 40])
        if dim == "duration":
            ax.set_xticks([7, 14, 30, 60])
        if j == 0:
            ax.set_ylabel("mean reward"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Chain(12) vs IID vs best-threshold — {dim} sweep "
                 f"(20k episodes, historical bootstrap)", y=1.03)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def write_md(df, out):
    lines = ["# Dimension sweeps — chain(12) vs IID vs best-threshold\n"]
    for dim in DIMS:
        sub = df[df.dim == dim]
        if not len(sub):
            continue
        lines.append(f"\n## {dim} sweep\n")
        lines.append("| location | swept | IID | Chain(12) | Threshold(best) | chain-thresh |")
        lines.append("|---|---|---|---|---|---|")
        for loc in LOC_ORDER:
            for val in sorted(sub[sub.location == loc].swept.unique()):
                def get(arm):
                    r = sub[(sub.location == loc) & (sub.arm == arm) & (sub.swept == val)]
                    return float(r.mean_reward.iloc[0]) if len(r) else np.nan
                i, c, t = get("iid"), get("chain12"), get("thresh")
                v = val * DELTA_T / (60 * 24) if dim == "duration" else val
                lines.append(f"| {loc} | {v:g} | {i:.2f} | {c:.2f} | {t:.2f} | {c - t:+.2f} |")
    open(out, "w", encoding="utf-8").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results = args.results or os.path.join(repo, "results", "failbin_dim_sweeps")
    outdir = args.out or os.path.join(repo, "results", "failbin_dim_analysis")
    os.makedirs(outdir, exist_ok=True)

    df = collapse(load(results))
    if df.empty:
        print("no results found under", results); return
    df = df.sort_values(["dim", "location", "arm", "swept"]).reset_index(drop=True)
    df.to_csv(os.path.join(outdir, "dim_sweep_metrics.csv"), index=False)
    print(df.drop(columns=["ci_lo", "ci_hi"]).to_string(index=False))
    for i, dim in enumerate(DIMS, start=1):
        if (df.dim == dim).any():
            fig_dim(df, dim, os.path.join(outdir, f"figS{i}_{dim}.png"))
    write_md(df, os.path.join(outdir, "DIM_SWEEPS.md"))
    print("\nwrote figures + metrics to", outdir)


if __name__ == "__main__":
    main()
