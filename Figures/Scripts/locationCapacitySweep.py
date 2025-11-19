"""
Script 3: Location × Capacity sweep (standalone; one figure per location × metric)

For each location (lat, lon) and for each metric (defaults to all), this script
creates a figure of the metric vs battery capacity. Within each figure, a line
is drawn for every (observation_threshold, wind_threshold) pair. Legend entries
explicitly show both thresholds as: "w_to = <wind>, O_th = <obs>". If optimal
runs exist, an aggregated optimal-vs-capacity curve is overlaid in dashed black.

Other notes:
- No figure titles (paper-ready); axis labels only.
- Legends are placed outside-right of the axes (no overlap with data).
- Failure percentage is normalized to 0–100 if provided as 0–1 and capped to ≤ percent_cap.
- Independent of the GUI; relies only on results_io.py (same directory).
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
# Plotting (one figure per location × metric)
# ----------------------------

def _plot_loc_capacity(
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    metric: str,
    lat: float,
    lon: float,
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    percent_cap: float = 105.0,
) -> None:
    """Plot metric vs capacity for one location. One line per (obs, wind)."""
    sub = df_main[(df_main["latitude"] == lat) & (df_main["longitude"] == lon)].copy()
    if sub.empty:
        return

    # Normalize percent if needed
    if metric == "failure_percentage" and not sub[metric].dropna().empty:
        if sub[metric].dropna().max() <= 1.01:
            sub[metric] = sub[metric] * 100.0

    # Aggregate duplicates to avoid noisy multiple lines: mean by (cap, obs, wind)
    agg = (
        sub.groupby(["battery_capacity", "observation_threshold", "wind_threshold"], as_index=False)[metric]
           .mean()
           .sort_values(["observation_threshold", "wind_threshold", "battery_capacity"]) 
    )

    obs_vals = agg["observation_threshold"].dropna().unique()
    wind_vals = agg["wind_threshold"].dropna().unique()

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for obs in sorted(obs_vals):
        for w in sorted(wind_vals):
            ser = agg[(agg["observation_threshold"] == obs) & (agg["wind_threshold"] == w)]
            if ser.empty:
                continue
            ax.plot(
                ser["battery_capacity"],
                ser[metric],
                marker="o",
                label=f"w_to = {w}, O_th = {obs}",
            )

    # Optimal overlay (aggregate mean per capacity at this location)
    if df_opt is not None and not df_opt.empty and metric in df_opt.columns:
        opt_sub = df_opt[(df_opt["latitude"] == lat) & (df_opt["longitude"] == lon)].copy()
        if not opt_sub.empty:
            if metric == "failure_percentage" and not opt_sub[metric].dropna().empty and opt_sub[metric].max() <= 1.01:
                opt_sub[metric] = opt_sub[metric] * 100.0
            opt_cap = (
                opt_sub.groupby("battery_capacity")[metric]
                       .mean()
                       .reset_index()
                       .sort_values("battery_capacity")
            )
            if not opt_cap.empty:
                ax.plot(opt_cap["battery_capacity"], opt_cap[metric], linestyle="--", color="black", label="Optimal")

    # Labels, grid, limits
    ylabel = {
        "mean_reward": "Mean Total Reward",
        "failure_percentage": "Failure Percentage",
        "mean_failure_step": "Mean Failure Step",
        "average_flight_hrs": "Average Flight Hours",
    }.get(metric, metric.replace("_", " ").title())
    ax.set_xlabel("Battery Capacity (Wh)")
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

    # Legend outside-right
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.9,
            edgecolor="black",
        )

    fig.tight_layout()
    base = f"location_{lat}_{lon}_capacity_{metric}"
    savefig_all_formats(fig, outdir, base, formats, dpi)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate metric vs capacity figures per location (one figure per location × metric)"
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
    apply_style(args.style or cfg.get("style"), rc)

    df = load_summary(args.results)

    # Build filters
    capacities = args.capacities or cfg.get("battery_capacities")
    obs_thresholds = args.obs_thresholds or cfg.get("threshold_values")
    wind_thresholds = args.wind_thresholds or cfg.get("wind_thresholds")
    penalties = args.penalties or cfg.get("failure_penalties")
    algorithms = args.algorithms or cfg.get("algorithms")

    # Locations: from CLI (lat:lon entries) or from config; otherwise infer from data
    locations: Optional[List[Tuple[float, float]]] = None
    if args.locations:
        locations = []
        for tok in args.locations:
            lat_s, lon_s = tok.split(":")
            locations.append((float(lat_s), float(lon_s)))
    elif cfg.get("locations"):
        locations = [(float(d["latitude"]), float(d["longitude"])) for d in cfg["locations"]]
    else:
        # Fallback to all locations in the data
        loc_df = df[["latitude", "longitude"]].dropna().drop_duplicates()
        locations = [(float(r["latitude"]), float(r["longitude"])) for _, r in loc_df.iterrows()]

    starts = args.starts or cfg.get("starts")

    # Split optimal vs main; do not filter optimal by thresholds
    is_opt = df["sim_type"].str.contains("optimal", case=False, na=False)
    df_opt_all = df[is_opt]
    df_main_all = df[~is_opt]

    df_main = _filter_base(df_main_all, capacities, obs_thresholds, wind_thresholds, penalties, algorithms, locations, starts)
    df_opt = _filter_base(df_opt_all, capacities, None, None, penalties, None, locations, starts)

    metrics = args.metrics or ["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"]

    # Generate one figure per (location × metric)
    for (lat, lon) in locations:
        for metric in metrics:
            _plot_loc_capacity(df_main, df_opt, metric, lat, lon, args.outdir, args.formats, args.dpi, percent_cap=args.percent_cap)

    print(f"Saved location × capacity figures to {args.outdir}")


if __name__ == "__main__":
    main()
