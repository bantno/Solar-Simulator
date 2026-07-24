#!/usr/bin/env python3
"""failbin_analyze.py -- aggregate the failbin validation runs and render figures.

Reads every harness run directory under --results (each holds one *.h5 + run_metadata.json),
maps each to (location, arm) via the config basename `<tag>_<location>_<arm>`, extracts
per-episode reward distributions, and computes:

  mean reward (+95% bootstrap CI), failure rate, mean flight hours, CVaR@10 (mean of worst
  10% of episodes).  For the threshold arm the best-reward grid point is taken as the
  benchmark.

Outputs (to --out):
  failbin_metrics.csv            one row per (location, arm)
  fig1_reward_by_location.png    grouped bars: mean reward, arms x locations
  fig2_failure_rate.png          grouped bars: failure rate
  fig3_bin_edges.png             per-location f(w), wind pdf, wind- vs fail-space edges
  fig4_chain_gain.png            chain - IID reward gain (wind vs fail bins)
  fig5_reward_cdf.png            per-location reward CDFs by arm
  SUMMARY.md                     text digest of the headline findings
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
from Scripts.failbin.failbin_experiment import takeoff_failure  # noqa: E402

# Full arm set (parsing + metrics + CDF). Longer names first matters for suffix parsing.
ARMS = ["iid", "chain_wind5", "chain_wind8", "chain_wind12", "chain_wind16", "chain_wind24",
        "chain_wind", "chain_fail", "chain_dec", "threshold"]
# Core arms shown in the headline grouped-bar figures.
CORE_ARMS = ["iid", "threshold", "chain_wind", "chain_fail"]
ARM_LABEL = {
    "iid": "IID-optimal",
    "chain_wind": "Chain (wind-space, 3 bins)",
    "chain_wind5": "Chain (wind-space, 5 bins)",
    "chain_wind8": "Chain (wind-space, 8 bins)",
    "chain_wind12": "Chain (wind-space, 12 bins)",
    "chain_wind16": "Chain (wind-space, 16 bins)",
    "chain_wind24": "Chain (wind-space, 24 bins)",
    "chain_fail": "Chain (failure-space bins)",
    "chain_dec": "Chain (decision-boundary bins)",
    "threshold": "Threshold (best)",
}
ARM_COLOR = {
    "iid": "#4C72B0",
    "chain_wind": "#DD8452",
    "chain_wind5": "#E8A87C",
    "chain_wind8": "#B5651D",
    "chain_wind12": "#8C5A2B",
    "chain_wind16": "#6B4423",
    "chain_wind24": "#4A2F18",
    "chain_fail": "#C44E52",
    "chain_dec": "#55A868",
    "threshold": "#8C8C8C",
}
LOC_ORDER = ["florida", "hawaii", "natlantic", "bering"]
LOC_LABEL = {
    "florida": "Florida coast", "hawaii": "Hawaii", "natlantic": "N. Atlantic", "bering": "Bering Strait",
}


def _bootstrap_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _cvar(x, q=0.10):
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.floor(q * len(x))))
    return float(x[:k].mean())


def _parse_basename(name):
    """`<tag>_<location>_<arm>` -> (location, arm). Arm may contain an underscore (chain_wind)."""
    for arm in sorted(ARMS, key=len, reverse=True):
        suffix = "_" + arm
        if name.endswith(suffix):
            head = name[: -len(suffix)]
            loc = head.split("_")[-1]
            return loc, arm
    return None, None


def _read_run(run_dir):
    meta_path = os.path.join(run_dir, "run_metadata.json")
    h5s = glob.glob(os.path.join(run_dir, "*.h5"))
    if not os.path.exists(meta_path) or not h5s:
        return None
    meta = json.load(open(meta_path))
    basename = meta.get("config_basename") or ""
    loc, arm = _parse_basename(basename)
    if arm is None:
        return None
    groups = []
    with h5py.File(h5s[0], "r") as f:
        for gname in f.keys():
            grp = f[gname]
            a = grp.attrs
            def at(k, d=None):
                v = a.get(k, d)
                return v.decode() if isinstance(v, bytes) else v
            rewards = None
            if "episode_scalars" in grp and "total_reward" in grp["episode_scalars"]:
                rewards = grp["episode_scalars"]["total_reward"][:]
            groups.append(dict(
                sim_type=at("simulation_type"),
                obs_thr=at("observation_threshold"),
                wind_thr=at("wind_threshold"),
                avg_reward=float(at("average_reward", np.nan)),
                failure_pct=float(at("failure_percentage", np.nan)),
                flight_hrs=float(at("average_flight_hrs", np.nan)),
                rewards=rewards,
            ))
    return dict(loc=loc, arm=arm, groups=groups)


def aggregate(results_dir):
    # Dedupe by (location, arm): if a config was re-run, keep the newest run dir
    # (leaf dir is a YYYYmmdd_HHMMSS timestamp, so lexical max = most recent).
    latest = {}
    for meta_path in glob.glob(os.path.join(results_dir, "**", "run_metadata.json"), recursive=True):
        run_dir = os.path.dirname(meta_path)
        r = _read_run(run_dir)
        if not r:
            continue
        key = (r["loc"], r["arm"])
        stamp = os.path.basename(run_dir)
        if key not in latest or stamp > latest[key][0]:
            latest[key] = (stamp, r)
    runs = [v[1] for v in latest.values()]
    rows = []
    for r in runs:
        groups = r["groups"]
        if r["arm"] == "threshold":
            # Best grid point by mean reward.
            g = max(groups, key=lambda d: (d["avg_reward"] if np.isfinite(d["avg_reward"]) else -1e18))
            note = f"obs_thr={g['obs_thr']}, wind_thr={g['wind_thr']}"
        else:
            g = groups[0]
            note = ""
        rw = g["rewards"]
        lo, hi = _bootstrap_ci(rw) if rw is not None else (np.nan, np.nan)
        rows.append(dict(
            location=r["loc"], arm=r["arm"], arm_label=ARM_LABEL[r["arm"]],
            mean_reward=g["avg_reward"],
            ci_lo=lo, ci_hi=hi,
            failure_pct=g["failure_pct"],
            flight_hrs=g["flight_hrs"],
            cvar10=_cvar(rw) if rw is not None else np.nan,
            n_episodes=(len(rw) if rw is not None else np.nan),
            note=note,
        ))
    df = pd.DataFrame(rows)
    return df, {(r["loc"], r["arm"]): r for r in runs}


def _locs_in(df):
    present = [l for l in LOC_ORDER if l in set(df["location"])]
    present += [l for l in sorted(set(df["location"])) if l not in present]
    return present


def fig_reward(df, out):
    locs = _locs_in(df)
    arms = [a for a in CORE_ARMS if a in set(df["arm"])]
    x = np.arange(len(locs)); w = 0.8 / max(1, len(arms))
    fig, ax = plt.subplots(figsize=(1.9 * len(locs) + 2, 5))
    for i, arm in enumerate(arms):
        vals, los, his = [], [], []
        for loc in locs:
            row = df[(df.location == loc) & (df.arm == arm)]
            v = float(row.mean_reward.iloc[0]) if len(row) else np.nan
            lo = float(row.ci_lo.iloc[0]) if len(row) else np.nan
            hi = float(row.ci_hi.iloc[0]) if len(row) else np.nan
            vals.append(v); los.append(v - lo if np.isfinite(lo) else 0); his.append(hi - v if np.isfinite(hi) else 0)
        ax.bar(x + i * w, vals, w, yerr=[los, his], capsize=3,
               label=ARM_LABEL[arm], color=ARM_COLOR[arm])
    ax.set_xticks(x + w * (len(arms) - 1) / 2)
    ax.set_xticklabels([LOC_LABEL.get(l, l) for l in locs])
    ax.set_ylabel("Mean reward (whale observations)")
    ax.set_title("Policy performance by location (30-day mission, historical weather)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_failure(df, out):
    locs = _locs_in(df)
    arms = [a for a in CORE_ARMS if a in set(df["arm"])]
    x = np.arange(len(locs)); w = 0.8 / max(1, len(arms))
    fig, ax = plt.subplots(figsize=(1.9 * len(locs) + 2, 5))
    for i, arm in enumerate(arms):
        vals = [float(df[(df.location == loc) & (df.arm == arm)].failure_pct.iloc[0])
                if len(df[(df.location == loc) & (df.arm == arm)]) else np.nan for loc in locs]
        ax.bar(x + i * w, vals, w, label=ARM_LABEL[arm], color=ARM_COLOR[arm])
    ax.set_xticks(x + w * (len(arms) - 1) / 2)
    ax.set_xticklabels([LOC_LABEL.get(l, l) for l in locs])
    ax.set_ylabel("Failure rate (% of episodes)")
    ax.set_title("Mission failure rate by location")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_edges(edges_json, out):
    data = json.load(open(edges_json))
    data = {d["name"]: d for d in data}
    locs = [l for l in LOC_ORDER if l in data] + [l for l in data if l not in LOC_ORDER]
    n = len(locs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4), squeeze=False)
    for j, loc in enumerate(locs):
        d = data[loc]; ax = axes[0][j]
        wmax = max(d["wind_p99"] * 1.15, (d["failspace_edges"][-1] if d["failspace_edges"] else 0) + 3, 20)
        ws = np.linspace(0, wmax, 400)
        # Shade where the wind actually is: [0, p90] = 90% of operating time.
        ax.axvspan(0, d["wind_p90"], color="#4C9AA8", alpha=0.12, lw=0)
        ax.plot(ws, takeoff_failure(ws), color="k", lw=1.8, label="f(w) takeoff failure")
        for p, lab in [("wind_p50", "p50"), ("wind_p90", "p90"), ("wind_p99", "p99")]:
            ax.axvline(d[p], color="0.55", ls=":", lw=1)
            ax.text(d[p], 0.94, lab, ha="center", va="top", fontsize=6.5, color="0.4", rotation=90)
        for e in d["windspace_edges"]:
            ax.axvline(e, color=ARM_COLOR["chain_wind"], ls="--", lw=1.7)
        for e in d["failspace_edges"]:
            ax.axvline(e, color=ARM_COLOR["chain_fail"], ls="-", lw=1.7)
        # Occupancy annotation: how the two schemes split the time budget.
        wo = "/".join(f"{o*100:.0f}" for o in d["windspace_occupancy"])
        fo = "/".join(f"{o*100:.0f}" for o in d["failspace_occupancy"])
        ax.text(0.97, 0.42, f"wind-space occ  {wo}%", transform=ax.transAxes, ha="right",
                fontsize=7.5, color=ARM_COLOR["chain_wind"])
        ax.text(0.97, 0.33, f"fail-space occ  {fo}%", transform=ax.transAxes, ha="right",
                fontsize=7.5, color=ARM_COLOR["chain_fail"])
        ax.set_title(f"{LOC_LABEL.get(loc, loc)}  (mean {d['mean_wind']:.1f} m/s)")
        ax.set_xlabel("wind speed [m/s]"); ax.set_ylim(0, 1); ax.set_xlim(0, wmax)
        if j == 0:
            ax.set_ylabel("takeoff failure prob")
    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="k", lw=1.8, label="f(w) takeoff failure"),
        Line2D([0], [0], color=ARM_COLOR["chain_wind"], ls="--", lw=1.6, label="wind-space edges"),
        Line2D([0], [0], color=ARM_COLOR["chain_fail"], ls="-", lw=1.6, label="failure-space edges"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Wind-bin placement: wind-space (equal occupancy) vs failure-space (equal failure mass)", y=1.12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_chain_gain(df, out):
    locs = _locs_in(df)
    fig, ax = plt.subplots(figsize=(1.9 * len(locs) + 2, 5))
    x = np.arange(len(locs)); w = 0.35
    def gain(loc, arm):
        base = df[(df.location == loc) & (df.arm == "iid")]
        cur = df[(df.location == loc) & (df.arm == arm)]
        if not len(base) or not len(cur):
            return np.nan
        return float(cur.mean_reward.iloc[0]) - float(base.mean_reward.iloc[0])
    gw = [gain(l, "chain_wind") for l in locs]
    gf = [gain(l, "chain_fail") for l in locs]
    ax.bar(x - w / 2, gw, w, label="wind-space bins", color=ARM_COLOR["chain_wind"])
    ax.bar(x + w / 2, gf, w, label="failure-space bins", color=ARM_COLOR["chain_fail"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LOC_LABEL.get(l, l) for l in locs])
    ax.set_ylabel("Reward gain over IID-optimal")
    ax.set_title("Value of modeling wind persistence (chain - IID), by bin scheme")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_reward_cdf(runs, out):
    locs = [l for l in LOC_ORDER if any(k[0] == l for k in runs)]
    locs += [l for l in sorted({k[0] for k in runs}) if l not in locs]
    n = len(locs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4), squeeze=False)
    for j, loc in enumerate(locs):
        ax = axes[0][j]
        for arm in CORE_ARMS + ["chain_dec"]:
            r = runs.get((loc, arm))
            if not r:
                continue
            if arm == "threshold":
                g = max(r["groups"], key=lambda d: (d["avg_reward"] if np.isfinite(d["avg_reward"]) else -1e18))
            else:
                g = r["groups"][0]
            rw = g["rewards"]
            if rw is None:
                continue
            xs = np.sort(rw); ys = np.linspace(0, 1, len(xs))
            ax.plot(xs, ys, label=ARM_LABEL[arm], color=ARM_COLOR[arm], lw=1.6)
        ax.set_title(LOC_LABEL.get(loc, loc)); ax.set_xlabel("episode reward")
        if j == 0:
            ax.set_ylabel("cumulative fraction")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Episode-reward CDFs by arm (left-tail = downside risk)", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_binsweep(df, out):
    """Chain reward vs wind-space bin count, per location. Tests the resolution lever."""
    order = [("chain_wind", 3), ("chain_wind5", 5), ("chain_wind8", 8),
             ("chain_wind12", 12), ("chain_wind16", 16), ("chain_wind24", 24)]
    locs = [l for l in _locs_in(df) if any((df.arm == a).any() for a, _ in order)]
    fig, ax = plt.subplots(figsize=(7, 5))
    have_any = False
    for loc in _locs_in(df):
        xs, ys = [], []
        for arm, nb in order:
            row = df[(df.location == loc) & (df.arm == arm)]
            if len(row):
                xs.append(nb); ys.append(float(row.mean_reward.iloc[0]))
        if len(xs) >= 2:
            have_any = True
            ax.plot(xs, ys, marker="o", lw=1.8, label=LOC_LABEL.get(loc, loc))
        # IID reference (dashed) for context
        iid = df[(df.location == loc) & (df.arm == "iid")]
        if len(iid) and len(xs) >= 2:
            ax.axhline(float(iid.mean_reward.iloc[0]), ls=":", lw=0.8, color="0.7")
    ax.set_xlabel("wind-space bins (equal occupancy)"); ax.set_ylabel("chain mean reward")
    ax.set_title("Finding the resolution plateau: chain reward vs bin count")
    ax.set_xticks([3, 5, 8, 12, 16, 24])
    if have_any:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_decision(df, out):
    """Decision-boundary bins vs equal-occupancy wind bins (both 3-bin), per location."""
    locs = _locs_in(df)
    x = np.arange(len(locs)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(locs) + 2, 5))
    def val(loc, arm):
        r = df[(df.location == loc) & (df.arm == arm)]
        return float(r.mean_reward.iloc[0]) if len(r) else np.nan
    vw = [val(l, "chain_wind") for l in locs]
    vd = [val(l, "chain_dec") for l in locs]
    ax.bar(x - w / 2, vw, w, label="wind-space (equal occupancy)", color=ARM_COLOR["chain_wind"])
    ax.bar(x + w / 2, vd, w, label="decision-boundary", color=ARM_COLOR["chain_dec"])
    ax.set_xticks(x); ax.set_xticklabels([LOC_LABEL.get(l, l) for l in locs])
    ax.set_ylabel("chain mean reward")
    ax.set_title("Can a decision-aware scheme beat equal-occupancy wind bins? (3 bins each)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def write_summary_md(df, out):
    lines = ["# failbin validation — headline metrics\n"]
    for loc in _locs_in(df):
        sub = df[df.location == loc].set_index("arm")
        lines.append(f"\n## {LOC_LABEL.get(loc, loc)}\n")
        lines.append("| arm | mean reward | 95% CI | failure % | flight hrs | CVaR@10 |")
        lines.append("|---|---|---|---|---|---|")
        for arm in ARMS:
            if arm not in sub.index:
                continue
            r = sub.loc[arm]
            lines.append(f"| {ARM_LABEL[arm]} | {r.mean_reward:.2f} | "
                         f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}] | {r.failure_pct:.2f} | "
                         f"{r.flight_hrs:.1f} | {r.cvar10:.2f} |")
        if "iid" in sub.index:
            base = sub.loc["iid"].mean_reward
            for arm in ("chain_wind", "chain_fail"):
                if arm in sub.index:
                    d = sub.loc[arm].mean_reward - base
                    lines.append(f"\n- {ARM_LABEL[arm]}: {d:+.2f} vs IID "
                                 f"({100*d/base:+.1f}%)" if base else f"\n- {ARM_LABEL[arm]}: {d:+.2f} vs IID")
    open(out, "w", encoding="utf-8").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Base results dir (searched recursively).")
    ap.add_argument("--edges", default=None, help="failbin_edges.json (for the bin-edge figure).")
    ap.add_argument("--out", required=True, help="Output dir for figures + metrics.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, runs = aggregate(args.results)
    if df.empty:
        print("No runs found under", args.results); return
    df = df.sort_values(["location", "arm"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.out, "failbin_metrics.csv"), index=False)
    print(df.to_string(index=False))

    fig_reward(df, os.path.join(args.out, "fig1_reward_by_location.png"))
    fig_failure(df, os.path.join(args.out, "fig2_failure_rate.png"))
    if args.edges and os.path.exists(args.edges):
        fig_edges(args.edges, os.path.join(args.out, "fig3_bin_edges.png"))
    fig_chain_gain(df, os.path.join(args.out, "fig4_chain_gain.png"))
    fig_reward_cdf(runs, os.path.join(args.out, "fig5_reward_cdf.png"))
    if any(df.arm.isin(["chain_wind5", "chain_wind8"])):
        fig_binsweep(df, os.path.join(args.out, "fig6_binsweep.png"))
    if any(df.arm == "chain_dec"):
        fig_decision(df, os.path.join(args.out, "fig7_decision.png"))
    write_summary_md(df, os.path.join(args.out, "SUMMARY.md"))
    print("\nWrote figures + metrics to", args.out)


if __name__ == "__main__":
    main()
