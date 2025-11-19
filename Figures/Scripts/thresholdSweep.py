# -----------------------------------------------------------------------------
# results_A_compare_optimal_vs_threshold.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple
from matplotlib.ticker import PercentFormatter

from results_io import load_summary, apply_style, savefig_all_formats

import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd


def _filter_df(df: pd.DataFrame,
               obs_thresholds: Optional[list[float]] = None,
               wind_thresholds: Optional[list[float]] = None,
               algorithms: Optional[list[str]] = None,
               locations: Optional[list[tuple[float, float]]] = None,
               starts: Optional[list[pd.Timestamp]] = None) -> pd.DataFrame:
    out = df.copy()
    if obs_thresholds:
        out = out[out["observation_threshold"].isin(obs_thresholds)]
    if wind_thresholds:
        out = out[out["wind_threshold"].isin(wind_thresholds)]
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
    if starts:
        out = out.copy()
        out["start_date"] = out["start_time"].dt.normalize()
        start_dates = pd.to_datetime(starts).normalize()
        out = out[out["start_date"].isin(start_dates)]
    return out


def _plot_threshold_lines(ax, df_main: pd.DataFrame, value_col: str, ylabel: str,
                          df_opt: Optional[pd.DataFrame] = None):
    # Work on a copy so we can normalize failure percentages if needed
    main = df_main.copy()

    # Normalize failure % to 0–100 if data appears fractional
    if value_col == "failure_percentage" and not main[value_col].dropna().empty:
        if main[value_col].dropna().max() <= 1.01:
            main[value_col] = main[value_col] * 100.0

    # Build pivot for lines: x=obs threshold, columns=wind threshold
    pivot = main.pivot(index="observation_threshold", columns="wind_threshold", values=value_col)
    pivot = pivot.sort_index()

    for w in pivot.columns:
        ax.plot(pivot.index, pivot[w], marker="o", label=f"$w_{'{'}to{'}'}$ = {w}")

    # Optimal baseline (mean across matching rows); normalize like above
    opt_val = None
    if df_opt is not None and not df_opt.empty and value_col in df_opt.columns:
        opt_series = df_opt[value_col].dropna()
        if not opt_series.empty:
            if value_col == "failure_percentage" and opt_series.max() <= 1.01:
                opt_series = opt_series * 100.0
            opt_val = float(opt_series.mean())
            ax.axhline(opt_val, linestyle="--", color="black", linewidth=2, label=f"Optimal")
            # Scatter markers at each x to make it visually explicit
            for x in pivot.index:
                ax.scatter(x, opt_val, color="black", s=36, marker="x", zorder=3)

    ax.set_xlabel("Observation Threshold $O_{th}$")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    # Pad y-limits generously to avoid legend overlap
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    # If failure percentage, force percent ticks and clamp to [0,100] when appropriate
    if value_col == "failure_percentage":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        # If within a plausible range, clamp to [0,100] with a small top pad
        if y1 <= 140 and y0 >= -20:
            ax.set_ylim(max(0, y0), min(100, y1))
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, min(105, y1 + 5))

    # Place legend inside but tucked in the top-left with slight inset
    # Place legend outside to the right
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  borderaxespad=0.0, frameon=True, framealpha=0.9, edgecolor="black")



def main():
    p = argparse.ArgumentParser(description="Generate comparison figures: threshold sweeps vs optimal baseline")
    p.add_argument("--results", nargs="+", required=True, help="One or more HDF5 results files")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--outdir", required=True, help="Directory to save figures")
    p.add_argument("--formats", nargs="+", default=["png"], help="Figure formats to save")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--minutes-per-step", type=float, default=15.0)
    p.add_argument("--style", default=None)
    p.add_argument("--obs-thresholds", nargs="*", type=float)
    p.add_argument("--wind-thresholds", nargs="*", type=float)
    p.add_argument("--algorithms", nargs="*", type=str)
    p.add_argument("--locations", nargs="*", type=str)
    p.add_argument("--starts", nargs="*", type=str)

    args = p.parse_args()

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

    obs_thresholds = args.obs_thresholds or cfg.get("threshold_values")
    wind_thresholds = args.wind_thresholds or cfg.get("wind_thresholds")
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

    df_main = _filter_df(df_main_all, obs_thresholds, wind_thresholds, algorithms, locations, starts)
    # Important: optimal runs typically have no threshold fields; don't filter them by obs/wind
    df_opt = _filter_df(df_opt_all, None, None, None, locations, starts)

    fig_size = (3.25, 2.75)

    fig1, ax1 = plt.subplots(figsize=fig_size)
    _plot_threshold_lines(ax1, df_main, value_col="mean_reward", ylabel="Mean Total Reward", df_opt=df_opt)
    # ax1.set_title("Mean Total Reward vs. Thresholds")
    savefig_all_formats(fig1, args.outdir, "compare_mean_reward_by_thresholds", args.formats, args.dpi)

    fig2, ax2 = plt.subplots(figsize=fig_size)
    _plot_threshold_lines(ax2, df_main, value_col="mean_failure_step", ylabel="Mean Failure Step", df_opt=df_opt)
    # ax2.set_title("Mean Failure Step vs. Thresholds")
    savefig_all_formats(fig2, args.outdir, "compare_mean_failure_step_by_thresholds", args.formats, args.dpi)

    fig3, ax3 = plt.subplots(figsize=fig_size)
    _plot_threshold_lines(ax3, df_main, value_col="failure_percentage", ylabel="Failure Percentage", df_opt=df_opt)
    # ax3.set_title("Failure Percentage vs. Thresholds")
    savefig_all_formats(fig3, args.outdir, "compare_failure_percentage_by_thresholds", args.formats, args.dpi)

    print(f"Saved figures to {args.outdir}")


if __name__ == "__main__":
    main()
