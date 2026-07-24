"""Start-date × Metric subfigures (external, compact legend; paper-ready)

One figure per (location × metric):
  subfig_<letter>_<metric>_{lat}_{lon}.{ext}

- X-axis: Mission Start Date (from `start_time`)
- Series: one per (observation_threshold, wind_threshold)
- Legend: external, compact (uses w_{to} / O_{th})
- Optimal: solid black with circle markers (disable with --no-opt)

Data columns match your other scripts; `average_flight_hrs` is used for
"Flight Hours per Day" if `flight_hours_per_day` is absent.
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple, List, Set

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

from results_io import load_summary, apply_style, savefig_all_formats


# ----------------------------
# Data-level filtering
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
# Helpers
# ----------------------------

def _letters():
    import string
    for ch in string.ascii_lowercase:
        yield ch

def _latlon_label(lat: float, lon: float) -> str:
    def _fmt(v: float, pos: str) -> str:
        hemi = {"lat": ("S", "N"), "lon": ("W", "E")}
        h = hemi["lat"][1 if v >= 0 else 0] if pos == "lat" else hemi["lon"][1 if v >= 0 else 0]
        return f"{abs(v):.0f}°{h}"
    return f"{_fmt(lat,'lat')}, {_fmt(lon,'lon')}"

def _legend_label(obs: float, wind: float) -> str:
    return rf"$w_{{to}} = {wind},\ O_{{th}} = {obs}$"

def _collect_pairs(obs_vals: Iterable[float], wind_vals: Iterable[float]) -> List[Tuple[float, float]]:
    return [(o, w) for o in sorted(obs_vals) for w in sorted(wind_vals)]


# ----------------------------
# Core panel
# ----------------------------

def _panel_loc_startdate(
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    metric: str,
    lat: float,
    lon: float,
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    letter: str,
    percent_cap: float,
    line_width: float,
    opt_line_width: float,
    marker_size: float,
    opt_marker_size: float,
    show_opt: bool,
    # plot-time filters:
    plot_obs: Optional[Set[float]] = None,
    plot_wind: Optional[Set[float]] = None,
    plot_pairs: Optional[Set[Tuple[float, float]]] = None,
):
    sub = df_main[(df_main["latitude"] == lat) & (df_main["longitude"] == lon)].copy()
    if sub.empty:
        return

    if "start_time" not in sub.columns:
        raise KeyError("Summary must include 'start_time' for start-date sweeps.")
    sub["start_date"] = pd.to_datetime(sub["start_time"]).dt.normalize()

    # Alias: average_flight_hrs -> flight_hours_per_day
    metric_col = metric
    if metric == "flight_hours_per_day" and "flight_hours_per_day" not in sub.columns:
        if "average_flight_hrs" in sub.columns:
            metric_col = "average_flight_hrs"

    # Special metrics
    if metric == "failure_percentage" and "failure_percentage" in sub.columns:
        if sub["failure_percentage"].max() <= 1.01:
            sub["failure_percentage"] = 100.0 * sub["failure_percentage"]

    if metric == "percent_completed_before_failure":
        if "percent_completed_before_failure" not in sub.columns:
            if "mean_failure_step" in sub.columns and "horizon_steps" in sub.columns:
                sub["percent_completed_before_failure"] = (
                    100.0 * sub["mean_failure_step"] / sub["horizon_steps"]
                )
            else:
                sub["percent_completed_before_failure"] = np.nan

    need_cols = ["observation_threshold", "wind_threshold", metric_col, "start_date"]
    missing = [c for c in need_cols if c not in sub.columns]
    if missing:
        raise KeyError(f"Missing columns for plotting: {missing}")

    agg = (
        sub.groupby(["start_date", "observation_threshold", "wind_threshold"], as_index=False)[metric_col]
           .mean()
           .sort_values(["observation_threshold", "wind_threshold", "start_date"])
    )

    obs_vals = sorted(agg["observation_threshold"].dropna().unique())
    wind_vals = sorted(agg["wind_threshold"].dropna().unique())
    pairs = _collect_pairs(obs_vals, wind_vals)

    if plot_obs:
        pairs = [(o, w) for (o, w) in pairs if o in plot_obs]
    if plot_wind:
        pairs = [(o, w) for (o, w) in pairs if w in plot_wind]
    if plot_pairs:
        pairs = [(o, w) for (o, w) in pairs if (o, w) in plot_pairs]

    # Wider figure to leave space for the legend on the right
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=dpi)

    handles, labels = [], []
    for (obs, w) in pairs:
        ser = agg[(agg["observation_threshold"] == obs) & (agg["wind_threshold"] == w)]
        if ser.empty:
            continue
        ln, = ax.plot(
            ser["start_date"],
            ser[metric_col],
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
        )
        handles.append(ln)
        labels.append(_legend_label(obs, w))

    if show_opt and df_opt is not None and not df_opt.empty and metric_col in df_opt.columns:
        opt = df_opt[(df_opt["latitude"] == lat) & (df_opt["longitude"] == lon)].copy()
        if not opt.empty:
            opt["start_date"] = pd.to_datetime(opt["start_time"]).dt.normalize()
            if metric == "failure_percentage" and "failure_percentage" in opt.columns:
                if opt["failure_percentage"].max() <= 1.01:
                    opt["failure_percentage"] = 100.0 * opt["failure_percentage"]
            if metric == "percent_completed_before_failure" and "percent_completed_before_failure" not in opt.columns:
                if "mean_failure_step" in opt.columns and "horizon_steps" in opt.columns:
                    opt["percent_completed_before_failure"] = (
                        100.0 * opt["mean_failure_step"] / opt["horizon_steps"]
                    )
            use_col = metric_col if metric not in ("failure_percentage", "percent_completed_before_failure") else metric
            opt_agg = opt.groupby("start_date", as_index=False)[use_col].mean().sort_values("start_date")
            if not opt_agg.empty:
                ln_opt, = ax.plot(
                    opt_agg["start_date"],
                    opt_agg[use_col],
                    color="black",
                    marker="o",
                    markersize=opt_marker_size,
                    linewidth=opt_line_width,
                )
                handles.append(ln_opt)
                labels.append("Optimal")

    # Labels & styling
    ylabel = {
        "mean_reward": "Mean Reward",
        "failure_percentage": "Failure Percentage",
        "percent_completed_before_failure": "% Completed Before Failure",
        "mean_failure_step": "Mean Failure Step",          # << added label
        "flight_hours_per_day": "Mean Flight Hours",
        "average_flight_hrs": "Flight Hours",
    }.get(metric, metric.replace("_", " ").title())

    ax.set_xlabel("Mission Start Date")
    ax.set_ylabel(ylabel)
    # ax.grid(False, linewidth=0.5, alpha=0.45)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=9)

    # Date tick formatting: month-day, 30° rotation, ~5 ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")

    if metric in ("failure_percentage", "percent_completed_before_failure"):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0, y0), min(percent_cap, y1 if np.isfinite(y1) else 100))

    # Location subtitle
    # ax.text(0.5, 1.02, f"({_latlon_label(lat, lon)})", transform=ax.transAxes,
    #         ha="center", va="bottom", fontsize=9)

    # External compact legend
    if handles:
        ax.legend(
            handles, labels,
            loc="best",
            frameon=True, framealpha=0.85, edgecolor="black",
            fontsize=9, borderpad=0.3, labelspacing=0.3,
            handlelength=1.2, handletextpad=0.6, columnspacing=0.8,
            ncol=1,
        )
        fig.tight_layout(rect=[0, 0, 0.74, 1])
    else:
        fig.tight_layout()

    base = f"subfig_{letter}_{metric}_{lat:.2f}_{lon:.2f}"
    savefig_all_formats(fig, outdir, base, formats, dpi)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate start-date sweep subfigures (external compact legend)."
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--style", default=None)

    ap.add_argument("--metrics", nargs="*", default=None,
                    choices=[
                        "mean_reward",
                        "failure_percentage",
                        "percent_completed_before_failure",
                        "mean_failure_step",              # << added option
                        "flight_hours_per_day",
                        "average_flight_hrs",
                    ])

    # Data filters
    ap.add_argument("--capacities", nargs="*", type=float)
    ap.add_argument("--obs-thresholds", nargs="*", type=float)
    ap.add_argument("--wind-thresholds", nargs="*", type=float)
    ap.add_argument("--penalties", nargs="*", type=float)
    ap.add_argument("--algorithms", nargs="*", type=str)
    ap.add_argument("--locations", nargs="*", type=str, help="lat:lon entries")
    ap.add_argument("--starts", nargs="*", type=str)

    # Plot-time filters
    ap.add_argument("--plot-obs-thresholds", nargs="*", type=float)
    ap.add_argument("--plot-wind-thresholds", nargs="*", type=float)
    ap.add_argument("--plot-pairs", nargs="*", type=str)

    # Style knobs
    ap.add_argument("--line-width", type=float, default=1.2)
    ap.add_argument("--opt-line-width", type=float, default=1.5)
    ap.add_argument("--marker-size", type=float, default=3.0)
    ap.add_argument("--opt-marker-size", type=float, default=3.2)
    ap.add_argument("--no-opt", action="store_true", help="Do not overlay optimal series.")
    ap.add_argument("--percent-cap", type=float, default=105.0)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    rc = {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": args.dpi,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
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

    if args.locations:
        locations = []
        for tok in args.locations:
            lat_s, lon_s = tok.split(":")
            locations.append((float(lat_s), float(lon_s)))
    elif cfg.get("locations"):
        locations = [(float(d["latitude"]), float(d["longitude"])) for d in cfg["locations"]]
    else:
        loc_df = df[["latitude", "longitude"]].dropna().drop_duplicates()
        locations = [(float(r["latitude"]), float(r["longitude"])) for _, r in loc_df.iterrows()]

    starts = args.starts or cfg.get("starts")

    is_opt = df["sim_type"].str.contains("optimal", case=False, na=False)
    df_opt_all = df[is_opt]
    df_main_all = df[~is_opt]

    df_main = _filter_base(df_main_all, capacities, obs_thresholds, wind_thresholds,
                           penalties, algorithms, locations, starts)
    df_opt = _filter_base(df_opt_all, capacities, None, None, penalties, None, locations, starts)

    metrics = args.metrics or [
        "mean_reward",
        "failure_percentage",
        "percent_completed_before_failure",
        "mean_failure_step",          # << in defaults too
        "flight_hours_per_day",
    ]

    plot_obs: Set[float] = set(float(x) for x in (args.plot_obs_thresholds or []))
    plot_wind: Set[float] = set(float(x) for x in (args.plot_wind_thresholds or []))
    plot_pairs: Set[Tuple[float, float]] = set()
    if args.plot_pairs:
        for tok in args.plot_pairs:
            o, w = tok.split(":")
            plot_pairs.add((float(o), float(w)))

    letter_gen = _letters()
    for metric in metrics:
        for (lat, lon) in locations:
            letter = next(letter_gen)
            _panel_loc_startdate(
                df_main=df_main,
                df_opt=None if args.no_opt else df_opt,
                metric=metric,
                lat=lat,
                lon=lon,
                outdir=args.outdir,
                formats=args.formats,
                dpi=args.dpi,
                letter=letter,
                percent_cap=args.percent_cap,
                line_width=args.line_width,
                opt_line_width=args.opt_line_width,
                marker_size=args.marker_size,
                opt_marker_size=args.opt_marker_size,
                show_opt=(not args.no_opt),
                plot_obs=plot_obs if plot_obs else None,
                plot_wind=plot_wind if plot_wind else None,
                plot_pairs=plot_pairs if plot_pairs else None,
            )

    print(f"Saved start-date sweep figures (compact external legends) to {args.outdir}")


if __name__ == "__main__":
    main()
