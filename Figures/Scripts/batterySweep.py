"""
Script 2: Battery Capacity sweep (standalone, one figure per obs threshold)

Generates individual figures showing metrics vs battery capacity. By default,
all metrics are plotted separately. For each observation-threshold value and
each metric, one figure is saved; within each figure, lines correspond to
wind-threshold values. If optimal runs exist, a dashed black curve shows the
optimal metric vs capacity.

Requires: results_io.py in the same directory.
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from results_io import load_summary, apply_style, savefig_all_formats


def _filter_df_cap(
    df: pd.DataFrame,
    capacities: Optional[list[float]] = None,
    obs_thresholds: Optional[list[float]] = None,
    wind_thresholds: Optional[list[float]] = None,
    penalties: Optional[list[float]] = None,
    algorithms: Optional[list[str]] = None,
    locations: Optional[list[Tuple[float, float]]] = None,
    starts: Optional[list[pd.Timestamp]] = None,
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


def _plot_metric_vs_capacity_single(
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    metric: str,
    obs: float,
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    percent_cap: float = 105.0,
) -> None:
    """Render a single figure for one observation threshold and one metric."""
    main = df_main.copy()
    if metric == "failure_percentage" and not main[metric].dropna().empty:
        if main[metric].dropna().max() <= 1.01:
            main[metric] = main[metric] * 100.0

    sub = main[main["observation_threshold"] == obs]
    wind_vals = sorted(sub["wind_threshold"].dropna().unique())

    fig, ax = plt.subplots(figsize=(3.25, 2.75))

    # Lines per wind threshold
    for w in wind_vals:
        ser = sub[sub["wind_threshold"] == w].sort_values("battery_capacity")
        if ser.empty:
            continue
        ax.plot(
            ser["battery_capacity"],
            ser[metric],
            marker="o",
            label=fr"$w_{{to}} = {w},\; O_{{th}} = {obs}$",
        )

    # Optimal overlay
    if df_opt is not None and not df_opt.empty and metric in df_opt.columns:
        opt = df_opt.copy()
        if metric == "failure_percentage" and not opt[metric].dropna().empty and opt[metric].max() <= 1.01:
            opt[metric] = opt[metric] * 100.0
        opt_cap = (
            opt.groupby("battery_capacity")[metric]
            .mean()
            .reset_index()
            .sort_values("battery_capacity")
        )
        if not opt_cap.empty:
            ax.plot(
                opt_cap["battery_capacity"],
                opt_cap[metric],
                linestyle="-",
                marker="o",
                color="black",
                label="Optimal",
            )

    # Axis labels
    ylabel = {
        "mean_reward": r"Mean Total Reward $R$",
        "failure_percentage": "Failure Percentage",
        "mean_failure_step": r"Mean Failure Step $k_f$",
        "average_flight_hrs": "Average Flight Hours",
    }.get(metric, metric.replace("_", " ").title())
    ax.set_xlabel("Battery Capacity (Wh)")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    # Y limits
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    if metric == "failure_percentage":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0, y0), min(percent_cap, y1))

    # Legend outside
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.85,
            edgecolor="black",
        )

    fig.tight_layout()
    savefig_all_formats(fig, outdir, f"capacity_{metric}_obs{obs}", formats, dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate metric vs battery capacity figures (one figure per observation threshold)"
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default=None)
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        choices=["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"],
        help="Metrics to plot. If none provided, all metrics are plotted.",
    )

    # Filters
    ap.add_argument("--capacities", nargs="*", type=float)
    ap.add_argument("--obs-thresholds", nargs="*", type=float)
    ap.add_argument("--wind-thresholds", nargs="*", type=float)
    ap.add_argument("--penalties", nargs="*", type=float)
    ap.add_argument("--algorithms", nargs="*", type=str)
    ap.add_argument("--locations", nargs="*", type=str)
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

    capacities = args.capacities or cfg.get("battery_capacities")
    obs_thresholds = args.obs_thresholds or cfg.get("threshold_values")
    wind_thresholds = args.wind_thresholds or cfg.get("wind_thresholds")
    penalties = args.penalties or cfg.get("failure_penalties")
    algorithms = args.algorithms or cfg.get("algorithms")

    locations = None
    if args.locations:
        locations = []
        for tok in args.locations:
            lat_s, lon_s = tok.split(":")
            locations.append((float(lat_s), float(lon_s)))
    elif cfg.get("locations"):
        locations = [(float(d["latitude"]), float(d["longitude"])) for d in cfg["locations"]]

    starts = args.starts or cfg.get("starts")

    is_opt = df["sim_type"].str.contains("optimal", case=False, na=False)
    df_opt_all = df[is_opt]
    df_main_all = df[~is_opt]

    df_main = _filter_df_cap(
        df_main_all, capacities, obs_thresholds, wind_thresholds, penalties, algorithms, locations, starts
    )
    df_opt = _filter_df_cap(df_opt_all, capacities, None, None, penalties, None, locations, starts)

    metrics = args.metrics or ["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"]

    obs_vals = sorted(df_main["observation_threshold"].dropna().unique())
    for obs in obs_vals:
        for metric in metrics:
            _plot_metric_vs_capacity_single(
                df_main, df_opt, metric, obs, args.outdir, args.formats, args.dpi, percent_cap=args.percent_cap
            )

    print(f"Saved capacity figures (one per obs threshold and per metric) to {args.outdir}")


if __name__ == "__main__":
    main()
