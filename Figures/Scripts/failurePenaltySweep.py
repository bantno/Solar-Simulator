"""
Script 4: Failure Penalty sweep (standalone, one figure per obs threshold × metric)

For each observation-threshold value and for each metric (defaults to all),
this script creates a figure of the metric vs failure_penalty. Within each
figure, a line is drawn per wind-threshold value. Legend entries show both
thresholds as: "w_to = <wind>, O_th = <obs>". If optimal runs exist, an
aggregated optimal-vs-penalty curve is overlaid in dashed black.

Conventions:
- No figure titles; axis labels only (paper-ready).
- Legends placed outside-right of the axes.
- Failure percentage is normalized to 0–100 if provided as 0–1 and capped by --percent-cap.
- Independent of the GUI; relies only on results_io.py (same directory).

Example:
  python results_D_vary_failure_penalty.py \
    --results simulation_results/CS/battery_sweep_config_20250910_140426.h5 \
    --config configs/test/battery_sweep_config.yaml \
    --outdir figs/04_failure_penalty \
    --formats png \
    --metrics mean_reward failure_percentage
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
# Plotting (one figure per obs × metric)
# ----------------------------

def _plot_obs_penalty(
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    metric: str,
    obs: float,
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    percent_cap: float = 105.0,
) -> None:
    """Plot metric vs failure_penalty for a single observation threshold.
    One line per wind_threshold.
    """
    sub = df_main[df_main["observation_threshold"] == obs].copy()
    if sub.empty:
        return

    # Normalize percent if needed
    if metric == "failure_percentage" and not sub[metric].dropna().empty:
        if sub[metric].dropna().max() <= 1.01:
            sub[metric] = sub[metric] * 100.0

    # Aggregate duplicates to avoid duplicate-index issues: mean by (penalty, wind)
    agg = (
        sub.groupby(["failure_penalty", "wind_threshold"], as_index=False)[metric]
           .mean()
           .sort_values(["wind_threshold", "failure_penalty"]) 
    )

    wind_vals = agg["wind_threshold"].dropna().unique()

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    for w in sorted(wind_vals):
        ser = agg[agg["wind_threshold"] == w]
        if ser.empty:
            continue
        ax.plot(
            ser["failure_penalty"],
            ser[metric],
            marker="o",
            label=fr"$w_{{to}} = {w},\; O_{{th}} = {obs}$",
        )

    # Optimal overlay (aggregate mean per penalty)
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
            ax.plot(opt_pen["failure_penalty"], opt_pen[metric], linestyle="-",marker = "o", color="black", label="Optimal")

    # Labels, grid, limits
    ylabel = {
        "mean_reward": "Mean Total Reward",
        "failure_percentage": "Failure Percentage",
        "mean_failure_step": "Mean Failure Step",
        "average_flight_hrs": "Mean Flight Hours",
    }.get(metric, metric.replace("_", " ").title())
    ax.set_xlabel("Failure Penalty")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    if metric == "failure_percentage":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0, y0), min(percent_cap, y1))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            framealpha=0.9,
            edgecolor="black",
        )

    fig.tight_layout()
    base = f"obs_{obs}_failure_penalty_{metric}"
    savefig_all_formats(fig, outdir, base, formats, dpi)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate metric vs failure-penalty figures (one figure per obs × metric)"
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", nargs="+", default=["png"])  # PNG default per your preference
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default=None)
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        choices=["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"],
        help="Metrics to plot. If none provided, all metrics are plotted.",
    )

    # Filters / overrides
    ap.add_argument("--capacities", nargs="*", type=float)
    ap.add_argument("--obs-thresholds", nargs="*", type=float)
    ap.add_argument("--wind-thresholds", nargs="*", type=float)
    ap.add_argument("--penalties", nargs="*", type=float)
    ap.add_argument("--algorithms", nargs="*", type=str)
    ap.add_argument("--locations", nargs="*", type=str, help="lat:lon entries")
    ap.add_argument("--starts", nargs="*", type=str)
    ap.add_argument("--percent-cap", type=float, default=105.0)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    rc = {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "lines.linewidth": 2,
        "figure.dpi": args.dpi,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
    }
    rc = {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "figure.dpi": args.dpi,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
    }
    apply_style(args.style or cfg.get("style"), rc)

    df = load_summary(args.results)

    # Build filters
    capacities = args.capacities or cfg.get("battery_capacities")
    obs_thresholds = args.obs_thresholds or cfg.get("threshold_values")
    wind_thresholds = args.wind_thresholds or cfg.get("wind_thresholds")
    penalties = args.penalties or cfg.get("failure_penalties")
    algorithms = args.algorithms or cfg.get("algorithms")

    # Locations: from CLI (lat:lon entries) or from config (optional)
    locations: Optional[List[Tuple[float, float]]] = None
    if args.locations:
        locations = []
        for tok in args.locations:
            lat_s, lon_s = tok.split(":")
            locations.append((float(lat_s), float(lon_s)))
    elif cfg.get("locations"):
        locations = [(float(d["latitude"]), float(d["longitude"])) for d in cfg["locations"]]

    starts = args.starts or cfg.get("starts")

    # Split optimal vs main; do not filter optimal by thresholds
    is_opt = df["sim_type"].str.contains("optimal", case=False, na=False)
    df_opt_all = df[is_opt]
    df_main_all = df[~is_opt]

    df_main = _filter_base(df_main_all, capacities, obs_thresholds, wind_thresholds, penalties, algorithms, locations, starts)
    df_opt = _filter_base(df_opt_all, capacities, None, None, penalties, None, locations, starts)

    # Default to all metrics if none provided
    metrics = args.metrics or ["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"]

    # Choose which obs thresholds to render: from args/config or infer from df_main if none provided
    obs_vals = obs_thresholds or sorted(df_main["observation_threshold"].dropna().unique())

    for obs in obs_vals:
        for metric in metrics:
            _plot_obs_penalty(
                df_main, df_opt, metric, obs, args.outdir, args.formats, args.dpi, percent_cap=args.percent_cap
            )

    print(f"Saved failure-penalty figures (one per obs × metric) to {args.outdir}")


if __name__ == "__main__":
    main()
