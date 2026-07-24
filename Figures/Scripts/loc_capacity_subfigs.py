"""Script 3 (subfig set): Location × Capacity sweep → paper subfigures

Outputs:
  (a) subfig_a_world_map.{ext}                         # study locations on a world map
  (b..): subfig_<letter>_loc_<lat>_<lon>_{metric}.{ext}# one panel per location×metric
  (legend) subfig_legend.{ext}                         # combined legend only

Plot-time filters (do not change the data selection):
  --plot-obs-thresholds 0.10 0.25
  --plot-wind-thresholds 4.0 8.0
  --plot-pairs 0.10:4.0 0.25:8.0      (exact (obs, wind) pairs)

Legend labels use subscripts:  r"$w_{to} = 4.0,\ O_{th} = 0.10$"
Optimal overlay is a solid black line with 'o' markers.

Example:
  python script_loc_capacity_subfigs.py --results runs/*.pkl --config cfg.yml \
      --outdir figs/paper --formats pdf png --metrics mean_reward failure_percentage
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple, List, Set

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
# Helpers
# ----------------------------

def _latlon_label(lat: float, lon: float) -> str:
    def _fmt(v: float, pos: str) -> str:
        hemi = {"lat": ("S", "N"), "lon": ("W", "E")}
        if pos == "lat":
            h = hemi["lat"][1 if v >= 0 else 0]
        else:
            h = hemi["lon"][1 if v >= 0 else 0]
        return f"{abs(v):.0f}°{h}"
    return f"({_fmt(lat,'lat')}, {_fmt(lon,'lon')})"

def _letters():
    import string
    for ch in string.ascii_lowercase:
        yield ch

def _collect_pairs(obs_vals: Iterable[float], wind_vals: Iterable[float]) -> List[Tuple[float, float]]:
    pairs = []
    for o in sorted(obs_vals):
        for w in sorted(wind_vals):
            pairs.append((o, w))
    return pairs

def _legend_label(obs: float, wind: float) -> str:
    # MathText (no LaTeX engine needed)
    return rf"$w_{{to}} = {wind},\ O_{{th}} = {obs}$"


# ----------------------------
# Plotting: world map (a)
# ----------------------------

# def _plot_world_map(
#     locations: List[Tuple[float, float]],
#     outdir: str,
#     formats: Iterable[str],
#     dpi: int,
# ) -> None:
#     # Try cartopy for nice coastlines; fallback to plain axes
#     use_cartopy = False
#     try:
#         import cartopy.crs as ccrs
#         import cartopy.feature as cfeature
#         use_cartopy = True
#     except Exception:
#         use_cartopy = False

#     if use_cartopy:
#         fig = plt.figure(figsize=(7.0, 1.8), dpi=dpi)
#         ax = plt.axes(projection=ccrs.Robinson())
#         ax.set_global()
#         ax.coastlines(linewidth=0.5)
#         ax.add_feature(cfeature.LAND, facecolor="0.95")
#         ax.gridlines(draw_labels=False, linewidth=0.25, linestyle=":")
#         for (lat, lon) in locations:
#             ax.plot(lon, lat, marker="o", markersize=5, transform=ccrs.PlateCarree())
#     else:
#         fig, ax = plt.subplots(figsize=(7.0, 1.8), dpi=dpi)
#         ax.set_xlim([-180, 180])
#         ax.set_ylim([-60, 80])
#         ax.grid(True, linestyle=":", linewidth=0.25)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for (lat, lon) in locations:
#             ax.plot(lon, lat, "o", ms=5)
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#     fig.tight_layout()
#     savefig_all_formats(fig, outdir, "subfig_a_world_map", formats, dpi)
#     plt.close(fig)

def _plot_world_map(
    locations: List[Tuple[float, float]],
    outdir: str,
    formats: Iterable[str],
    dpi: int,
) -> None:
    # try cartopy
    use_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
    except Exception:
        use_cartopy = False

    from matplotlib.lines import Line2D

    handles, labels = [], []

    if use_cartopy:
        fig = plt.figure(figsize=(7.0, 1.8), dpi=dpi)
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="0.95")
        ax.gridlines(draw_labels=False, linewidth=0.25, linestyle=":")

        for (lat, lon) in locations:
            ln = ax.plot(lon, lat, "o", markersize=5, transform=ccrs.PlateCarree())[0]
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  color=ln.get_color(), markersize=6))
            labels.append(f"({lat:.2f}, {lon:.2f})")
    else:
        fig, ax = plt.subplots(figsize=(7.0, 1.8), dpi=dpi)
        ax.set_xlim([-180, 180])
        ax.set_ylim([-60, 80])
        ax.grid(True, linestyle=":", linewidth=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        for (lat, lon) in locations:
            ln = ax.plot(lon, lat, "o", ms=5)[0]
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  color=ln.get_color(), markersize=6))
            labels.append(f"({lat:.2f}, {lon:.2f})")

    # legend off to the left
    leg = ax.legend(
        handles,
        labels,
        title="Locations (lat,lon)",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
    )
    fig.subplots_adjust(left=0.35)  # leave space for legend on the left
    savefig_all_formats(fig, outdir, "subfig_a_world_map", formats, dpi)
    plt.close(fig)

# ----------------------------
# Plotting: one location×metric panel (b..)
# ----------------------------

def _panel_loc_capacity(
    df_main: pd.DataFrame,
    df_opt: Optional[pd.DataFrame],
    metric: str,
    lat: float,
    lon: float,
    outdir: str,
    formats: Iterable[str],
    dpi: int,
    letter: str,
    percent_cap: float = 105.0,
    include_xlabel: bool = True,
    include_ylabel: bool = True,
    return_handles: bool = False,
    # NEW plot-time filters:
    plot_obs: Optional[Set[float]] = None,
    plot_wind: Optional[Set[float]] = None,
    plot_pairs: Optional[Set[Tuple[float, float]]] = None,
):
    """Return handles/labels if return_handles=True (used for legend figure)."""
    sub = df_main[(df_main["latitude"] == lat) & (df_main["longitude"] == lon)].copy()
    if sub.empty:
        return []

    # Normalize percent (if stored as 0..1)
    if metric == "failure_percentage" and not sub[metric].dropna().empty:
        if sub[metric].dropna().max() <= 1.01:
            sub[metric] = sub[metric] * 100.0

    # Aggregate duplicates to mean per (cap, obs, wind)
    agg = (
        sub.groupby(["battery_capacity", "observation_threshold", "wind_threshold"], as_index=False)[metric]
           .mean()
           .sort_values(["observation_threshold", "wind_threshold", "battery_capacity"])
    )

    obs_vals = sorted(agg["observation_threshold"].dropna().unique())
    wind_vals = sorted(agg["wind_threshold"].dropna().unique())
    pairs: List[Tuple[float, float]] = _collect_pairs(obs_vals, wind_vals)

    # Apply plot-time filters
    if plot_obs:
        pairs = [(o, w) for (o, w) in pairs if o in plot_obs]
    if plot_wind:
        pairs = [(o, w) for (o, w) in pairs if w in plot_wind]
    if plot_pairs:
        pairs = [(o, w) for (o, w) in pairs if (o, w) in plot_pairs]

    fig, ax = plt.subplots(figsize=(3.6, 3.1), dpi=dpi)

    line_handles = []
    for (obs, w) in pairs:
        ser = agg[(agg["observation_threshold"] == obs) & (agg["wind_threshold"] == w)]
        if ser.empty:
            continue
        ln, = ax.plot(
            ser["battery_capacity"],
            ser[metric],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            label=_legend_label(obs, w),  # label kept (not shown), handy for debugging
        )
        line_handles.append(ln)

    # Optimal overlay (solid black with 'o' markers)
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
                ax.plot(
                    opt_cap["battery_capacity"],
                    opt_cap[metric],
                    color="black",
                    marker="o",
                    markersize=2.5,
                    linewidth=1.0,
                    label="Optimal",
                )

    # Labels, grid, limits
    ylabel = {
        "mean_reward": "Mean Reward",
        "failure_percentage": "Failure Percentage",
        "mean_failure_step": "Mean Failure Step",
        "average_flight_hrs": "Average Flight Hours",
    }.get(metric, metric.replace("_", " ").title())

    if include_xlabel:
        ax.set_xlabel("Battery Capacity (Wh)")
    if include_ylabel:
        ax.set_ylabel(ylabel)

    # ax.grid(True, linewidth=0.6, alpha=0.5)
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

    # Panel label (e.g., "(b) (30°N, 75°W)")
    ax.set_title(f"{_latlon_label(lat, lon)}", pad=6.0, fontsize=11)

    # No legend in panels
    fig.tight_layout()
    base = f"subfig_{letter}_loc_{lat:.2f}_{lon:.2f}_{metric}"
    savefig_all_formats(fig, outdir, base, formats, dpi)
    plt.close(fig)

    if return_handles:
        return line_handles
    return []


# ----------------------------
# Legend-only figure
# ----------------------------

def _legend_only_figure(
    pairs: List[Tuple[float, float]],
    outdir: str,
    formats: Iterable[str],
    dpi: int,
):
    # Build dummy lines to get consistent legend entries in Matplotlib’s color cycle
    fig, ax = plt.subplots(figsize=(7.0, 0.7), dpi=dpi)
    handles = []
    labels = []
    for (obs, w) in pairs:
        ln, = ax.plot([], [], marker="o", linewidth=2.0)
        handles.append(ln)
        labels.append(_legend_label(obs, w))
    # Optimal entry
    opt_ln, = ax.plot([], [], color="black", marker="o", linewidth=2.4)
    handles.append(opt_ln)
    labels.append("Optimal")

    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=min(6, len(labels)),
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
    )
    ax.axis("off")
    fig.tight_layout()
    savefig_all_formats(fig, outdir, "subfig_legend", formats, dpi)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate all subfigures for the Location × Capacity figure (map, panels, legend)."
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
        help="If omitted, all four metrics are generated.",
    )

    # Data-selection filters / overrides (as before)
    ap.add_argument("--capacities", nargs="*", type=float)
    ap.add_argument("--obs-thresholds", nargs="*", type=float)
    ap.add_argument("--wind-thresholds", nargs="*", type=float)
    ap.add_argument("--penalties", nargs="*", type=float)
    ap.add_argument("--algorithms", nargs="*", type=str)
    ap.add_argument("--locations", nargs="*", type=str, help="lat:lon entries")
    ap.add_argument("--starts", nargs="*", type=str)
    ap.add_argument("--percent-cap", type=float, default=105.0)

    # NEW: plot-time filters (affect which series are drawn, not which rows are loaded)
    ap.add_argument("--plot-obs-thresholds", nargs="*", type=float,
                    help="Only plot these observation thresholds (e.g., 0.10 0.25).")
    ap.add_argument("--plot-wind-thresholds", nargs="*", type=float,
                    help="Only plot these wind thresholds (e.g., 4.0 8.0).")
    ap.add_argument("--plot-pairs", nargs="*", type=str,
                    help='Only plot these (obs,wind) pairs, format "obs:wind", e.g., 0.10:4.0 0.25:8.0')

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    rc = {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "lines.linewidth": 0.1,
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

    # Locations: CLI > config > infer from data
    locations: List[Tuple[float, float]]
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

    # Split optimal vs main; do not filter optimal by thresholds
    is_opt = df["sim_type"].str.contains("optimal", case=False, na=False)
    df_opt_all = df[is_opt]
    df_main_all = df[~is_opt]

    df_main = _filter_base(df_main_all, capacities, obs_thresholds, wind_thresholds, penalties, algorithms, locations, starts)
    df_opt = _filter_base(df_opt_all, capacities, None, None, penalties, None, locations, starts)

    metrics = args.metrics or ["mean_reward", "failure_percentage", "mean_failure_step", "average_flight_hrs"]

    # Build plot-time filters
    plot_obs: Set[float] = set(float(x) for x in (args.plot_obs_thresholds or []))
    plot_wind: Set[float] = set(float(x) for x in (args.plot_wind_thresholds or []))
    plot_pairs: Set[Tuple[float, float]] = set()
    if args.plot_pairs:
        for tok in args.plot_pairs:
            o, w = tok.split(":")
            plot_pairs.add((float(o), float(w)))

    # (a) World map
    _plot_world_map(locations, args.outdir, args.formats, args.dpi)

    # Determine legend entries from the filtered data and the plot filters
    if not df_main.empty:
        agg_for_legend = df_main.groupby(["observation_threshold", "wind_threshold"], as_index=False)["battery_capacity"].count()
        obs_vals = sorted(agg_for_legend["observation_threshold"].dropna().unique())
        wind_vals = sorted(agg_for_legend["wind_threshold"].dropna().unique())
    else:
        obs_vals, wind_vals = [], []

    legend_pairs: List[Tuple[float, float]] = _collect_pairs(obs_vals, wind_vals)
    if plot_obs:
        legend_pairs = [(o, w) for (o, w) in legend_pairs if o in plot_obs]
    if plot_wind:
        legend_pairs = [(o, w) for (o, w) in legend_pairs if w in plot_wind]
    if plot_pairs:
        legend_pairs = [(o, w) for (o, w) in legend_pairs if (o, w) in plot_pairs]

    # Panels (b..)
    letter_gen = _letters()
    next(letter_gen)  # 'a' used by map
    for metric in metrics:
        for (lat, lon) in locations:
            letter = next(letter_gen)
            _panel_loc_capacity(
                df_main,
                df_opt,
                metric,
                lat,
                lon,
                args.outdir,
                args.formats,
                args.dpi,
                letter=letter,
                percent_cap=args.percent_cap,
                include_xlabel=True,
                include_ylabel=True,
                # plot-time filters:
                plot_obs=plot_obs if plot_obs else None,
                plot_wind=plot_wind if plot_wind else None,
                plot_pairs=plot_pairs if plot_pairs else None,
            )

    # Legend-only figure (only if any threshold pairs exist)
    if len(legend_pairs) > 0:
        _legend_only_figure(legend_pairs, args.outdir, args.formats, args.dpi)

    print(f"Saved subfigures (map, panels, legend) to {args.outdir}")


if __name__ == "__main__":
    main()
