"""
Horizon Sweep — paper subfigures (explicit; no inference)

Required columns:
sim_type, observation_threshold, wind_threshold, battery_capacity, horizon,
failure_penalty, mean_reward, failure_percentage, mean_failure_step,
average_flight_hrs, latitude, longitude, start_time, source_file

Assumptions (confirmed):
- horizon: number of TIMESTEPS (integer)
- average_flight_hrs: average cumulative flight hours per EPISODE (hours)
- failure_percentage: stored as 0–1 (fraction) → plotted as 0–100%

Outputs (each saved separately):
  (a) subfig_a_mean_reward_vs_horizon.{ext}
  (b) subfig_b_mean_reward_per_ts_vs_horizon.{ext}
  (c) subfig_c_failure_pct_vs_horizon.{ext}
  (d) subfig_d_flight_hours_vs_horizon.{ext}
  (e) subfig_e_avg_failure_step_vs_horizon.{ext}

X-axis:
- Use raw timesteps (“Horizon (timesteps)”), or
- Convert to days with --x-convert days and --dt-minutes <minutes>

Panel (d) mode (choose explicitly):
- --flight-hours-mode per_day  (DEFAULT; needs --dt-minutes)
- --flight-hours-mode episode  (plots raw average_flight_hrs)
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple, List, Set

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator, AutoMinorLocator

from results_io import load_summary, apply_style, savefig_all_formats


# ----------------------------
# Helpers (explicit only)
# ----------------------------

def _compute_reward_per_timestep(df: pd.DataFrame) -> pd.Series:
    """mean_reward_per_timestep = mean_reward / horizon (horizon = timesteps)."""
    denom = pd.to_numeric(df["horizon"], errors="coerce").astype(float).replace(0, np.nan)
    num = pd.to_numeric(df["mean_reward"], errors="coerce").astype(float)
    return num / denom

def _apply_plot_filters(
    pairs: List[Tuple[float, float]],
    plot_obs: Optional[Set[float]],
    plot_wind: Optional[Set[float]],
    plot_pairs: Optional[Set[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    out = pairs[:]
    if plot_obs:
        out = [(o, w) for (o, w) in out if o in plot_obs]
    if plot_wind:
        out = [(o, w) for (o, w) in out if w in plot_wind]
    if plot_pairs:
        out = [(o, w) for (o, w) in out if (o, w) in plot_pairs]
    return out

def _legend_label(obs: float, wind: float) -> str:
    # MathText (no LaTeX runtime needed)
    return rf"$w_{{to}}={wind},\ O_{{th}}={obs}$"


# ----------------------------
# Plotting
# ----------------------------

def _plot_panel(
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    x_vals: pd.Series,
    metric: str,
    y_label: str,
    base_name: str,
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    percent_axis: bool = False,
    percent_xmax: float = 100.0,
    markevery: int = 2,
    line_w: float = 1.1,
    opt_line_w: float = 1.6,
    msize: float = 3.0,
    opt_msize: float = 3.2,
    legend_outside: bool = True,
    plot_obs: Optional[Set[float]] = None,
    plot_wind: Optional[Set[float]] = None,
    plot_pairs: Optional[Tuple[float, float]] = None,
) -> bool:
    """Render one horizon-sweep panel with penalty-plot style. Returns True if created."""
    if metric not in df_main.columns and (df_opt is None or metric not in df_opt.columns):
        return False

    sub = df_main.copy()

    # Aggregate mean by (x, obs, wind) w/o inference
    agg = (
        pd.DataFrame({
            "x": pd.to_numeric(x_vals.loc[sub.index], errors="coerce"),
            "observation_threshold": pd.to_numeric(sub["observation_threshold"], errors="coerce"),
            "wind_threshold": pd.to_numeric(sub["wind_threshold"], errors="coerce"),
            metric: pd.to_numeric(sub[metric], errors="coerce"),
        })
        .dropna(subset=["x", "observation_threshold", "wind_threshold", metric])
        .groupby(["x", "observation_threshold", "wind_threshold"], as_index=False)[metric]
        .mean()
        .sort_values(["observation_threshold", "wind_threshold", "x"])
    )
    if agg.empty:
        return False

    obs_vals = sorted(agg["observation_threshold"].unique())
    wind_vals = sorted(agg["wind_threshold"].unique())
    pairs_all = [(o, w) for o in obs_vals for w in wind_vals]

    def _apply_filters(pairs):
        out = pairs
        if plot_obs:
            out = [(o, w) for (o, w) in out if o in plot_obs]
        if plot_wind:
            out = [(o, w) for (o, w) in out if w in plot_wind]
        if plot_pairs:
            out = [(o, w) for (o, w) in out if (o, w) in plot_pairs]
        return out

    pairs = _apply_filters(pairs_all)

    fig, ax = plt.subplots(figsize=(3.25, 2.75), dpi=dpi)

    # ----- threshold series -----
    for (obs, w) in pairs:
        ser = agg[(agg["observation_threshold"] == obs) & (agg["wind_threshold"] == w)]
        if ser.empty:
            continue
        ln, = ax.plot(
            ser["x"], ser[metric],
            marker="o", markersize=msize, linewidth=line_w,
            label=_legend_label(obs, w),
        )
        ln.set_markevery(markevery)

    # ----- optimal overlay (mean per x) -----
    if df_opt is not None and not df_opt.empty and metric in df_opt.columns:
        opt_tbl = pd.DataFrame({
            "x": pd.to_numeric(x_vals.loc[df_opt.index], errors="coerce"),
            metric: pd.to_numeric(df_opt[metric], errors="coerce"),
        }).dropna()
        if not opt_tbl.empty:
            opt_mean = opt_tbl.groupby("x", as_index=False)[metric].mean().sort_values("x")
            ln_opt, = ax.plot(
                opt_mean["x"], opt_mean[metric],
                color="black", marker="o", markersize=opt_msize,
                linewidth=opt_line_w, label="Optimal", zorder=5,
            )
            ln_opt.set_markevery(markevery)

    # Locators
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Grids: major solid, minor dotted
    ax.set_axisbelow(True)
    # ax.grid(which="major", axis="both", color="0.75", linewidth=0.8, alpha=0.6)
    # ax.grid(which="minor", axis="both", color="0.85", linewidth=0.6, alpha=0.6, linestyle=":")
    ax.grid(False)
    ax.set_xlabel("Mission Duration (days)")
    ax.set_ylabel(y_label)

    # Percent axis if requested
    if percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=percent_xmax))

    # Legend outside-right, rounded solid box
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if False:
            ax.legend(
                handles, labels,
                # --- THIS IS THE FIX ---
                loc="center left", bbox_to_anchor=(1.02, 0.5), 
                # --- END FIX ---
                frameon=True, framealpha=1.0, edgecolor="black", fancybox=True,
                borderaxespad=0.0
            )
            # Adjust the plot to make room
            fig.subplots_adjust(right=0.75) # You may need to tune 0.75 or 0.80
        else:
            # This is fine for an *internal* legend
            ax.legend(handles, labels, loc="best", frameon=True, framealpha=1.0, edgecolor="black", fancybox=True)
        # create a new figure containing only the legend
        fig_legend = plt.figure(figsize=(2, 2))
        fig_legend.legend(
            handles, labels,
            loc="center",
            frameon=True, framealpha=1.0, edgecolor="black", fancybox=True
        )

    fig_legend.savefig(f"{outdir}/{base_name}_legend.png", bbox_inches="tight")
    plt.close(fig_legend)
    fig.tight_layout()
    savefig_all_formats(fig, outdir, base_name, formats, dpi)
    plt.close(fig)
    return True




# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Horizon-sweep subfigures (explicit; no inference).")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default=None)

    # X-axis handling (explicit only)
    ap.add_argument("--x-convert", choices=["steps", "days"], default="steps",
                    help="Use horizon as raw timesteps, or convert to days with --dt-minutes.")
    ap.add_argument("--dt-minutes", type=float, default=None,
                    help="Size of each timestep in minutes (required if --x-convert days).")
    ap.add_argument("--x-label", type=str, default=None)

    # Flight hours mode for panel (d)
    ap.add_argument("--flight-hours-mode", choices=["per_day", "episode"], default="per_day",
                    help="per_day (default; needs --dt-minutes) or episode (raw average_flight_hrs).")

    # Plot-time filters
    ap.add_argument("--plot-obs-thresholds", nargs="*", type=float)
    ap.add_argument("--plot-wind-thresholds", nargs="*", type=float)
    ap.add_argument("--plot-pairs", nargs="*", type=str, help='Format "obs:wind", e.g., 0.10:4.0')
    # Multiple threshold-combination filters (obs,wind), e.g., --combo 0.10,4.0 0.25,8.0
    ap.add_argument("--combo", nargs="*", type=str,
                    help='Filter one or more (obs,wind) pairs, each given as "obs,wind" (e.g., 0.10,4.0 0.25,8.0)')

    # Appearance
    ap.add_argument("--line-width", type=float, default=1.1)
    ap.add_argument("--opt-line-width", type=float, default=1.1)
    ap.add_argument("--marker-size", type=float, default=3.0)
    ap.add_argument("--opt-marker-size", type=float, default=3.0)
    ap.add_argument("--markevery", type=int, default=1)
    ap.add_argument("--legend-outside", action="store_true", default=True)

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
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
    }
    apply_style(args.style or cfg.get("style"), rc)

    # Load summary
    df = load_summary(args.results).copy()

    # Compute mean reward per timestep (explicit arithmetic)
    df["mean_reward_per_timestep"] = _compute_reward_per_timestep(df)

    # Prepare X values and label (explicit)
    global x_label_global
    if args.x_convert == "steps":
        x_vals = pd.to_numeric(df["horizon"], errors="coerce").astype(float)
        x_label_global = args.x_label or "Horizon (timesteps)"
    else:  # days
        if args.dt_minutes is None:
            raise ValueError("--dt-minutes is required when --x-convert days is used.")
        x_vals = pd.to_numeric(df["horizon"], errors="coerce").astype(float) * (args.dt_minutes / 1440.0)
        x_label_global = args.x_label or "Mission Duration (days)"

    # Split optimal vs main (explicit detection via sim_type)
    is_opt = df["sim_type"].astype(str).str.contains("optimal", case=False, na=False)
    df_opt = df[is_opt].copy()
    df_main = df[~is_opt].copy()

    # Plot-time filters
    plot_obs: Optional[Set[float]] = set(args.plot_obs_thresholds or [])
    plot_obs = plot_obs or None
    plot_wind: Optional[Set[float]] = set(args.plot_wind_thresholds or [])
    plot_wind = plot_wind or None
    plot_pairs: Optional[Set[Tuple[float, float]]] = None
    if args.plot_pairs:
        plot_pairs = set()
        for tok in args.plot_pairs:
            o, w = tok.split(":")
            plot_pairs.add((float(o), float(w)))
    # Apply --combo if provided (adds one or more pairs to plot_pairs)
    if args.combo:
        for tok in args.combo:
            try:
                o_s, w_s = tok.split(",")
                if plot_pairs is None:
                    plot_pairs = set()
                plot_pairs.add((float(o_s), float(w_s)))
            except ValueError:
                raise ValueError('--combo entries must be formatted as "obs,wind", e.g., --combo 0.10,4.0 0.25,8.0')

    # Panel (a) Mean total reward vs horizon
    _plot_panel(
        df_main, df_opt, x_vals,
        metric="mean_reward", y_label="Mean Total Reward",
        base_name="subfig_a_mean_reward_vs_horizon",
        outdir=args.outdir, formats=args.formats, dpi=args.dpi,
        percent_axis=False, markevery=args.markevery,
        line_w=args.line_width, opt_line_w=args.opt_line_width,
        msize=args.marker_size, opt_msize=args.opt_marker_size,
        legend_outside=args.legend_outside,
        plot_obs=plot_obs, plot_wind=plot_wind, plot_pairs=plot_pairs,
    )

    # Panel (b) Average reward per timestep vs horizon
    _plot_panel(
        df_main, df_opt, x_vals,
        metric="mean_reward_per_timestep", y_label="Average Reward per Stage",
        base_name="subfig_b_mean_reward_per_ts_vs_horizon",
        outdir=args.outdir, formats=args.formats, dpi=args.dpi,
        percent_axis=False, markevery=args.markevery,
        line_w=args.line_width, opt_line_w=args.opt_line_width,
        msize=args.marker_size, opt_msize=args.opt_marker_size,
        legend_outside=args.legend_outside,
        plot_obs=plot_obs, plot_wind=plot_wind, plot_pairs=plot_pairs,
    )

    # Panel (c) Failure percentage vs horizon (0–1 → 0–100%)
    df_main_scaled = df_main.copy()
    if "failure_percentage" in df_main_scaled.columns:
        df_main_scaled["failure_percentage"] = pd.to_numeric(df_main_scaled["failure_percentage"], errors="coerce") * 100.0
    df_opt_scaled = df_opt.copy()
    if not df_opt_scaled.empty and "failure_percentage" in df_opt_scaled.columns:
        df_opt_scaled["failure_percentage"] = pd.to_numeric(df_opt_scaled["failure_percentage"], errors="coerce") * 100.0
    _plot_panel(
        df_main_scaled, df_opt_scaled, x_vals,
        metric="failure_percentage", y_label="Failure Percentage (%)",
        base_name="subfig_c_failure_pct_vs_horizon",
        outdir=args.outdir, formats=args.formats, dpi=args.dpi,
        percent_axis=True, percent_xmax=100.0, markevery=args.markevery,
        line_w=args.line_width, opt_line_w=args.opt_line_width,
        msize=args.marker_size, opt_msize=args.opt_marker_size,
        legend_outside=args.legend_outside,
        plot_obs=plot_obs, plot_wind=plot_wind, plot_pairs=plot_pairs,
    )

    args.dt_minutes = 15
    # Panel (d) Flight hours vs horizon (mode)
    if args.flight_hours_mode == "per_day":
        if args.dt_minutes is None:
            raise ValueError("--flight-hours-mode per_day requires --dt-minutes to compute days from timesteps.")
        days = pd.to_numeric(df["horizon"], errors="coerce").astype(float) * (args.dt_minutes / 1440.0)
        metric_name = "__flight_hours_per_day__"
        df_main_d = df_main.copy()
        df_main_d[metric_name] = pd.to_numeric(df_main_d["average_flight_hrs"], errors="coerce") / days.loc[df_main_d.index]
        df_opt_d = df_opt.copy()
        if not df_opt_d.empty:
            df_opt_d[metric_name] = pd.to_numeric(df_opt_d["average_flight_hrs"], errors="coerce") / days.loc[df_opt_d.index]
        ylab = "Flight Hours per Day (h/day)"
    else:
        metric_name = "average_flight_hrs"
        df_main_d = df_main
        df_opt_d = df_opt
        ylab = "Average Flight Hours per Episode (h)"

    _plot_panel(
        df_main_d, df_opt_d, x_vals,
        metric=metric_name, y_label=ylab,
        base_name="subfig_d_flight_hours_vs_horizon",
        outdir=args.outdir, formats=args.formats, dpi=args.dpi,
        percent_axis=False, markevery=args.markevery,
        line_w=args.line_width, opt_line_w=args.opt_line_width,
        msize=args.marker_size, opt_msize=args.opt_marker_size,
        legend_outside=args.legend_outside,
        plot_obs=plot_obs, plot_wind=plot_wind, plot_pairs=plot_pairs,
    )

    # Panel (e) Average failure step vs horizon
    _plot_panel(
        df_main, df_opt, x_vals,
        metric="mean_failure_step", y_label="Average Failure Stage",
        base_name="subfig_e_avg_failure_step_vs_horizon",
        outdir=args.outdir, formats=args.formats, dpi=args.dpi,
        percent_axis=False, markevery=args.markevery,
        line_w=args.line_width, opt_line_w=args.opt_line_width,
        msize=args.marker_size, opt_msize=args.opt_marker_size,
        legend_outside=args.legend_outside,
        plot_obs=plot_obs, plot_wind=plot_wind, plot_pairs=plot_pairs,
    )

    print(f"Horizon-sweep subfigures saved in {args.outdir}")

if __name__ == "__main__":
    main()
