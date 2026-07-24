"""Script 4: Failure Penalty sweep (single combined figure, one row per metric × one column per obs threshold)

Each subplot shows metric vs failure_penalty for a given observation-threshold.
One line per wind-threshold. A common legend appears to the right.
Optimal runs are overlaid as dashed black lines.

Example:
  python results_D_vary_failure_penalty.py \
    --results simulation_results/CS/battery_sweep_config_20250910_140426.h5 \
    --config configs/test/battery_sweep_config.yaml \
    --outdir figs/04_failure_penalty \
    --formats png pdf \
    --combine-all \
    --cols 3 --figwidth 12 --figheight 12
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple, List

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from results_io import load_summary, apply_style, savefig_all_formats


# ----------------------------
# Filters
# ----------------------------

def _filter_base(
    df: pd.DataFrame,
    capacities: Optional[List[float]] = None,
    obs_thresholds: Optional[List[float]] = None,
    wind_thresholds: Optional[List[float]] = None,
    penalties: Optional[List[float]] = None,
    algorithms: Optional[List[str]] = None,
    locations: Optional[List[Tuple[float, float]]] = None,
    starts: Optional[List[pd.Timestamp]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if capacities:
        out = out[out["battery_capacity"].isin(capacities)]
    if obs_thresholds:
        out = out[out["observation_threshold"].isin(obs_thresholds)]
    if wind_thresholds:
        out = out[out["wind_threshold"].isin(wind_thresholds)]
    if penalties and "failure_penalty" in out.columns:
        out = out[out["failure_penalty"].isin(penalties)]
    if algorithms:
        mask = False
        for a in algorithms:
            mask = mask | out["sim_type"].str.contains(a, case=False, na=False)
        out = out[mask]
    if locations:
        ll_mask = False
        for (lat, lon) in locations:
            ll_mask = ll_mask | ((out["latitude"] == lat) & (out["longitude"] == lon))
        out = out[ll_mask]
    if starts and "start_time" in out.columns:
        out = out.copy()
        out["start_date"] = out["start_time"].dt.normalize()
        start_dates = pd.to_datetime(starts).normalize()
        out = out[out["start_date"].isin(start_dates)]
    return out


# ----------------------------
# Plot helper
# ----------------------------

def _plot_obs_penalty_into_ax(
    ax: plt.Axes,
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    metric: str,
    obs: float,
    percent_cap: float = 105.0,
    legend_label_mode: str = "wind_only",
) -> tuple[list, list]:
    sub = df_main[df_main["observation_threshold"] == obs].copy()
    handles: list = []
    labels: list = []

    if sub.empty:
        ax.set_visible(False)
        return handles, labels

    # Normalize percent if needed
    if metric == "failure_percentage" and not sub[metric].dropna().empty:
        if sub[metric].dropna().max() <= 1.01:
            sub[metric] = sub[metric] * 100.0

    # Aggregate duplicates
    agg = (
        sub.groupby(["failure_penalty", "wind_threshold"], as_index=False)[metric]
           .mean()
           .sort_values(["wind_threshold", "failure_penalty"])
    )

    for w in sorted(agg["wind_threshold"].dropna().unique()):
        ser = agg[agg["wind_threshold"] == w]
        lab = fr"$w_{{to}} = {w},\; O_{{th}} = {obs}$"
        ln, = ax.plot(ser["failure_penalty"], ser[metric], marker="o", label=lab)
        handles.append(ln)
        labels.append(lab)

    # Optimal overlay
    if df_opt is not None and not df_opt.empty and metric in df_opt.columns:
        opt = df_opt.copy()
        if metric == "failure_percentage" and not opt[metric].dropna().empty and opt[metric].max() <= 1.01:
            opt[metric] = opt[metric] * 100.0
        opt_pen = (
            opt.groupby("failure_penalty")[metric]
               .mean()
               .reset_index()
               .sort_values("failure_penalty")
        )
        if not opt_pen.empty:
            ln_opt, = ax.plot(opt_pen["failure_penalty"], opt_pen[metric],
                              linestyle="-", marker="o", color="black", label="Optimal")
            handles.append(ln_opt)
            labels.append("Optimal")

    ylabel = {
        "mean_reward": "Mean Total Reward",
        "failure_percentage": "Failure Percentage",
        "mean_failure_step": "Mean Failure Step",
        "average_flight_hrs": "Average Flight Hours",
    }.get(metric, metric.replace("_", " ").title())

    ax.set_xlabel("Failure Penalty")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    if metric == "failure_percentage":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0, y0), min(percent_cap, y1))

    return handles, labels


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a single figure with all metrics (rows) × obs thresholds (columns) and a common legend."
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default=None)
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--figwidth", type=float, default=12.0)
    ap.add_argument("--figheight", type=float, default=12.0)
    ap.add_argument("--percent-cap", type=float, default=105.0)

    ap.add_argument("--capacities", nargs="*", type=float)
    ap.add_argument("--obs-thresholds", nargs="*", type=float)
    ap.add_argument("--wind-thresholds", nargs="*", type=float)
    ap.add_argument("--penalties", nargs="*", type=float)
    ap.add_argument("--algorithms", nargs="*", type=str)
    ap.add_argument("--locations", nargs="*", type=str)
    ap.add_argument("--starts", nargs="*", type=str)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    rc = {
        "font.size": 11,
        "axes.labelsize": 11,
        "lines.linewidth": 2,
        "figure.dpi": args.dpi,
        "legend.fontsize": 10,
    }
    apply_style(args.style or cfg.get("style"), rc)

    df = load_summary(args.results)

    # Filters
    capacities = args.capacities or cfg.get("battery_capacities")
    obs_thresholds = args.obs_thresholds or cfg.get("threshold_values")
    wind_thresholds = args.wind_thresholds or cfg.get("wind_thresholds")
    penalties = args.penalties or cfg.get("failure_penalties")
    algorithms = args.algorithms or cfg.get("algorithms")

    # Locations
    locations = None
    if args.locations:
        locations = [tuple(map(float, s.split(":"))) for s in args.locations]
    elif cfg.get("locations"):
        locations = [(float(d["latitude"]), float(d["longitude"])) for d in cfg["locations"]]

    starts = args.starts or cfg.get("starts")

    # Split optimal vs main
    is_opt = df["sim_type"].str.contains("optimal", case=False, na=False)
    df_opt = df[is_opt]
    df_main = df[~is_opt]

    df_main = _filter_base(df_main, capacities, obs_thresholds, wind_thresholds, penalties, algorithms, locations, starts)
    df_opt = _filter_base(df_opt, capacities, None, None, penalties, None, locations, starts)

    metrics = args.metrics or ["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"]
    obs_vals = obs_thresholds or sorted(df_main["observation_threshold"].dropna().unique())
    if not obs_vals:
        print("No observation thresholds found.")
        return

    n_obs = len(obs_vals)
    n_metrics = len(metrics)
    rows = n_metrics
    cols = n_obs
    fig, axes = plt.subplots(rows, cols, figsize=(args.figwidth, args.figheight), squeeze=False, sharex=True, sharey=False)
    all_handles, all_labels = [], []

    for i_m, metric in enumerate(metrics):
        for i_o, obs in enumerate(obs_vals):
            ax = axes[i_m][i_o]
            h, l = _plot_obs_penalty_into_ax(ax, df_main, df_opt, metric, obs, percent_cap=args.percent_cap)
            # ax.text(0.02, 0.98, fr"$O_{{th}} = {obs}$", transform=ax.transAxes, va="top", ha="left")
            if not all_handles and h:
                all_handles, all_labels = h, l
            if i_m == rows - 1:
                ax.set_xlabel("Failure Penalty")
            else:
                ax.set_xlabel("")
            if i_o != 0:
                ax.set_ylabel("")

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    if all_handles:
        fig.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(0.845, 0.5),
                   frameon=True, framealpha=0.9, edgecolor="black")

    base = "combined_all_metrics_failure_penalty"
    savefig_all_formats(fig, args.outdir, base, args.formats, args.dpi)
    plt.close(fig)

    print(f"Saved single combined multi-row figure to {args.outdir}")


if __name__ == "__main__":
    main()
