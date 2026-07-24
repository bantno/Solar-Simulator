#!/usr/bin/env python3
"""Advisor-slide mean-reward figures: threshold vs i.i.d.-optimal vs wind-chain-optimal.

Absolute mean-reward overlays (no paired deltas) for every July-2026 sweep,
in the journal figure style. Unlike ``generate_chain_overlay_figures.py``,
nothing is pooled across start dates OR across sites: every figure is
per-location small multiples conditioned on a single mission start (June
primary, December where the sweep ran one), because episodes with different
start dates or locations sample different weather and are not comparable.

Sources:

* capacity + per-location capacity: ``results/thesis_sweep/_analysis_batgrid/``
  (60-day missions, failure penalty 5)
* penalty: ``results/penalty_ext/_analysis/penalty_curve_cells.csv`` (300 Wh, 60 d)
* duration + per-location duration: ``results/thesis_sweep/_analysis_duration/``
  (300 Wh, June starts, failure penalty 5; per-stage reward)
* solar bin resolution: per-run ``summary.csv`` files under
  ``results/markov_solar_res/`` (300 Wh, penalty 5)
* dt robustness: ``results/markov_dt60/_analysis/dt_resolution_cells.csv``
  (300 Wh, penalty 5; per-step reward — per-day units are not comparable
  across dt)

Usage::

    python Figures/Scripts/generate_advisor_summary_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from generate_journal_paper_figures import (  # noqa: E402
    PAPER_DPI,
    SINGLE_COL_IN,
    DOUBLE_COL_IN,
    apply_paper_style,
)

REPO_ROOT = _THIS_DIR.parent.parent
BATGRID_CSV = REPO_ROOT / "results" / "thesis_sweep" / "_analysis_batgrid" / "markov_ablation_cells.csv"
DURATION_CSV = REPO_ROOT / "results" / "thesis_sweep" / "_analysis_duration" / "markov_ablation_cells.csv"
PENALTY_CSV = REPO_ROOT / "results" / "penalty_ext" / "_analysis" / "penalty_curve_cells.csv"
SOLAR_RES_DIR = REPO_ROOT / "results" / "markov_solar_res"
DT_CSV = REPO_ROOT / "results" / "markov_dt60" / "_analysis" / "dt_resolution_cells.csv"
OUT_DIR = REPO_ROOT / "Figures" / "Advisor_Summary_Figures"

DT_MINUTES = 15
PANEL_PENALTY = 5.0
REF_CAPACITY = 300.0

STARTS = {"june": "2025-06-10", "december": "2025-12-10"}
LOC_LABELS = {
    "florida": "Florida",
    "hawaii": "Hawaii",
    "gulf": "Gulf of Mexico",
    "gom": "Gulf of Mexico",
    "natlantic": "N. Atlantic",
    "bering": "Bering Sea",
}
LOC_ORDER = ["florida", "hawaii", "gulf", "gom", "natlantic", "bering"]

ARMS = (
    ("thresh", "Best threshold", dict(color="tab:orange", linestyle="--",
                                      marker="o", mfc="white")),
    ("iid", "Optimal (i.i.d.)", dict(color="black", linestyle="-", marker="o")),
    ("wind", "Optimal (wind chain)", dict(color="tab:blue", linestyle="-",
                                          marker="o")),
)


def _save(fig: plt.Figure, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {OUT_DIR / basename}.png")


def _plot_arms(ax: plt.Axes, agg: pd.DataFrame, x_col: str,
               ms: Optional[float] = None) -> Tuple[List, List[str]]:
    handles: List = []
    labels: List[str] = []
    for arm, label, style in ARMS:
        if arm not in agg.columns:
            continue
        kw = dict(style)
        if ms is not None:
            kw["markersize"] = ms
        ser = agg.sort_values(x_col)
        h, = ax.plot(ser[x_col], ser[arm], label=label, **kw)
        handles.append(h)
        labels.append(label)
    return handles, labels


def _loc_sort(ids: Sequence[str]) -> List[str]:
    return sorted(ids, key=lambda x: LOC_ORDER.index(x) if x in LOC_ORDER else 99)


# ---------------------------------------------------------------------------
# Per-location small multiples — nothing is pooled across sites or start dates
# ---------------------------------------------------------------------------

def _batgrid(start_iso: str) -> pd.DataFrame:
    df = pd.read_csv(BATGRID_CSV)
    df = df[(df["failure_penalty"] == PANEL_PENALTY)
            & df["start_time"].str.startswith(start_iso)]
    return df

def _location_grid(n_loc: int) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, 3.6), sharex=True)
    return fig, axes.ravel()


def fig_capacity_by_location(start_key: str) -> None:
    df = _batgrid(STARTS[start_key])
    locs = _loc_sort(df["location_id"].unique())
    fig, axes = _location_grid(len(locs))
    handles: List = []
    labels: List[str] = []
    for ax, loc in zip(axes, locs):
        sub = df[df["location_id"] == loc]
        agg = (sub.groupby("battery_capacity", as_index=False)
                  .agg(thresh=("avg_reward_thresh", "mean"),
                       iid=("avg_reward_iid", "mean"),
                       wind=("avg_reward_wind", "mean")))
        handles, labels = _plot_arms(ax, agg, "battery_capacity", ms=3)
        ax.set_title(LOC_LABELS.get(loc, loc.title()), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(locs):]:
        ax.axis("off")
    axes[len(locs)].legend(handles, labels, loc="center", fontsize=8, frameon=False)
    for i in (3, 4):
        axes[i].set_xlabel("Battery Capacity (Wh)", fontsize=8)
    for i in (0, 3):
        axes[i].set_ylabel("Mean Total Reward", fontsize=8)
    fig.suptitle(f"{start_key.capitalize()} start, 60 d, penalty {PANEL_PENALTY:g}",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, f"capacity_by_location_{start_key}")


PEN_MAX = 80.0  # advisor figures stop at the thesis-grid ceiling
PEX_DIR = REPO_ROOT / "results" / "penalty_ext"


def _fixed_threshold_curve(loc: str, start_iso: str,
                           pens: Sequence[float]) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """Best combo at the nominal penalty (5), held fixed across the ladder.

    Threshold behavior never depends on the penalty, so a combo's reward at any
    penalty re-weights exactly from its fp=5 run:
    r(pen) = r(5) + (5 - pen) * failure_frac.
    """
    summ = next((PEX_DIR / f"pex_{loc}_thresh").rglob("summary.csv"))
    df = pd.read_csv(summ)
    df = df[df["start_time"].str.startswith(start_iso)]
    best = df.loc[df["average_reward"].idxmax()]
    combo = (float(best["observation_threshold"]), float(best["wind_threshold"]))
    curve = pd.DataFrame({
        "pen": list(pens),
        "fixed": [best["average_reward"] + (5.0 - p) * best["failure_percentage"]
                  for p in pens],
    })
    return curve, combo


def fig_penalty_by_location(start_key: str) -> None:
    df = pd.read_csv(PENALTY_CSV)
    df = df[df["start"].str.startswith(STARTS[start_key]) & (df["pen"] <= PEN_MAX)]
    locs = _loc_sort(df["loc"].unique())
    fig, axes = _location_grid(len(locs))
    handles: List = []
    labels: List[str] = []
    for ax, loc in zip(axes, locs):
        sub = df[df["loc"] == loc]
        agg = (sub.groupby("pen", as_index=False)
                  .agg(thresh=("r_thresh_fine", "mean"),
                       iid=("r_iid", "mean"),
                       wind=("r_wind", "mean")))
        handles, labels = _plot_arms(ax, agg, "pen", ms=3)
        ax.set_xscale("log", base=2)
        pens = sorted(agg["pen"].unique())
        ax.set_xticks(pens)
        ax.set_xticklabels([f"{p:g}" for p in pens], fontsize=6)
        ax.minorticks_off()
        ax.set_title(LOC_LABELS.get(loc, loc.title()), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(locs):]:
        ax.axis("off")
    axes[len(locs)].legend(handles, labels, loc="center", fontsize=8, frameon=False)
    for i in (3, 4):
        axes[i].set_xlabel(r"Failure Penalty $\phi$", fontsize=8)
    for i in (0, 3):
        axes[i].set_ylabel("Mean Total Reward", fontsize=8)
    fig.suptitle(f"{start_key.capitalize()} start, 300 Wh, 60 d", fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, f"penalty_by_location_{start_key}")


def fig_penalty_fixed_by_location(start_key: str) -> None:
    """Re-tuned threshold envelope vs one deployable tuning (the fp=5 winner)."""
    df = pd.read_csv(PENALTY_CSV)
    df = df[df["start"].str.startswith(STARTS[start_key]) & (df["pen"] <= PEN_MAX)]
    locs = _loc_sort(df["loc"].unique())
    fig, axes = _location_grid(len(locs))
    handles: List = []
    labels: List[str] = []
    for ax, loc in zip(axes, locs):
        sub = df[df["loc"] == loc]
        agg = (sub.groupby("pen", as_index=False)
                  .agg(thresh=("r_thresh_fine", "mean"),
                       iid=("r_iid", "mean"),
                       wind=("r_wind", "mean")))
        handles, labels = _plot_arms(ax, agg, "pen", ms=3)
        pens = sorted(agg["pen"].unique())
        fixed, combo = _fixed_threshold_curve(loc, STARTS[start_key], pens)
        h, = ax.plot(fixed["pen"], fixed["fixed"], color="saddlebrown",
                     linestyle=":", marker="s", mfc="white", markersize=3,
                     label=r"Fixed threshold ($\phi$=5 tuning)")
        if r"Fixed threshold ($\phi$=5 tuning)" not in labels:
            handles.append(h)
            labels.append(r"Fixed threshold ($\phi$=5 tuning)")
        ax.text(0.03, 0.06, f"fixed: obs {combo[0]:g}, wind {combo[1]:g}",
                transform=ax.transAxes, fontsize=6, color="saddlebrown")
        ax.set_xscale("log", base=2)
        ax.set_xticks(pens)
        ax.set_xticklabels([f"{p:g}" for p in pens], fontsize=6)
        ax.minorticks_off()
        ax.set_title(LOC_LABELS.get(loc, loc.title()), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(locs):]:
        ax.axis("off")
    axes[len(locs)].legend(handles, labels, loc="center", fontsize=7, frameon=False)
    for i in (3, 4):
        axes[i].set_xlabel(r"Failure Penalty $\phi$", fontsize=8)
    for i in (0, 3):
        axes[i].set_ylabel("Mean Total Reward", fontsize=8)
    fig.suptitle(f"{start_key.capitalize()} start, 300 Wh, 60 d", fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, f"penalty_fixed_threshold_by_location_{start_key}")


def fig_duration_by_location() -> None:
    df = pd.read_csv(DURATION_CSV)
    df = df[df["failure_penalty"] == PANEL_PENALTY].copy()
    df["days"] = df["horizon"] * DT_MINUTES / 1440.0
    for arm in ("thresh", "iid", "wind"):
        df[f"{arm}_per_stage"] = df[f"avg_reward_{arm}"] / df["horizon"]
    locs = _loc_sort(df["location_id"].unique())
    fig, axes = _location_grid(len(locs))
    handles: List = []
    labels: List[str] = []
    for ax, loc in zip(axes, locs):
        sub = df[df["location_id"] == loc]
        agg = (sub.groupby("days", as_index=False)
                  .agg(thresh=("thresh_per_stage", "mean"),
                       iid=("iid_per_stage", "mean"),
                       wind=("wind_per_stage", "mean")))
        handles, labels = _plot_arms(ax, agg, "days", ms=3)
        ax.set_title(LOC_LABELS.get(loc, loc.title()), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    for ax in axes[len(locs):]:
        ax.axis("off")
    axes[len(locs)].legend(handles, labels, loc="center", fontsize=8, frameon=False)
    for i in (3, 4):
        axes[i].set_xlabel("Mission Duration (days)", fontsize=8)
    for i in (0, 3):
        axes[i].set_ylabel("Mean Reward per Stage", fontsize=8)
    fig.suptitle(f"June start, 300 Wh, penalty {PANEL_PENALTY:g}", fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, "duration_by_location_june")


# ---------------------------------------------------------------------------
# Solar bin-resolution study (absolute rewards from per-run summary.csv)
# ---------------------------------------------------------------------------

def _solar_res_rewards() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(SOLAR_RES_DIR.glob("mkvsr_*")):
        if not run_dir.is_dir():
            continue
        name = run_dir.name.replace("mkvsr_", "")
        if name.endswith("_iid"):
            loc, arm, bins = name[:-4], "iid", None
        elif "_solar_g" in name:
            loc, g = name.split("_solar_g")
            arm, bins = "solar", int(g)
        else:
            continue
        for summ in run_dir.rglob("summary.csv"):
            df = pd.read_csv(summ)
            df = df[(df["battery_capacity"] == REF_CAPACITY)
                    & (df["failure_penalty"] == PANEL_PENALTY)
                    & df["start_time"].str.startswith(STARTS["june"])]
            for r in df.itertuples():
                rows.append(dict(loc=loc, arm=arm, bins=bins,
                                 reward=r.average_reward))
    return pd.DataFrame(rows)


def fig_solar_bins() -> None:
    df = _solar_res_rewards()
    if df.empty:
        print("[SKIP] no solar-res summary rows found", file=sys.stderr)
        return
    locs = _loc_sort(df["loc"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 3.6), sharex=True)
    axes = axes.ravel()
    for ax, loc in zip(axes, locs):
        sub = df[df["loc"] == loc]
        sol = sub[sub["arm"] == "solar"].sort_values("bins")
        iid = sub[sub["arm"] == "iid"]["reward"].mean()
        ax.axhline(iid, color="black", linestyle="-", linewidth=1.2,
                   label="Optimal (i.i.d.)")
        ax.plot(sol["bins"], sol["reward"], color="tab:green", marker="o",
                label="Optimal (solar chain)")
        ax.set_title(LOC_LABELS.get(loc, loc.title()), fontsize=8)
        ax.set_xticks(sorted(sol["bins"].unique()))
        ax.tick_params(labelsize=7)
    axes[0].legend(loc="best", fontsize=7)
    for i in (2, 3):
        axes[i].set_xlabel("Solar chain bins", fontsize=8)
    for i in (0, 2):
        axes[i].set_ylabel("Mean Total Reward", fontsize=8)
    fig.suptitle("June start, 300 Wh, 60 d, penalty 5 — solar-chain DP vs its i.i.d. reference",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, "solar_bins_mean_reward_june")


# ---------------------------------------------------------------------------
# dt robustness (per-step mean reward; per-day units differ across dt)
# ---------------------------------------------------------------------------

def fig_dt_check() -> None:
    df = pd.read_csv(DT_CSV)
    df = df[(df["season"] == "summer") & (df["failure_penalty"] == PANEL_PENALTY)
            & (df["battery_capacity"] == REF_CAPACITY)].copy()
    df["steps_per_day"] = 1440.0 / df["dt"]
    df["iid_step"] = df["reward_iid_per_day"] / df["steps_per_day"]
    df["wind_step"] = (df["reward_iid_per_day"] + df["d_reward_per_day"]) / df["steps_per_day"]
    locs = _loc_sort(df["location_id"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 3.6), sharex=True)
    axes = axes.ravel()
    w = 0.35
    for ax, loc in zip(axes, locs):
        sub = df[df["location_id"] == loc].sort_values("dt")
        x = np.arange(len(sub))
        ax.bar(x - w / 2, sub["iid_step"], w, color="black", label="Optimal (i.i.d.)")
        ax.bar(x + w / 2, sub["wind_step"], w, color="tab:blue",
               label="Optimal (wind chain)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"dt = {int(d)} min\n({'interp.' if d == 15 else 'native hourly'})"
             for d in sub["dt"]], fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        ax.set_title(LOC_LABELS.get(loc, loc.title()), fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(loc="best", fontsize=7)
    for i in (0, 2):
        axes[i].set_ylabel("Mean Reward per Step", fontsize=8)
    fig.suptitle("June start, 300 Wh, 60 d, penalty 5", fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, "dt_mean_reward_per_step_by_location_june")


FIGURES = {
    "fig_capacity_by_location_june": lambda: fig_capacity_by_location("june"),
    "fig_capacity_by_location_december": lambda: fig_capacity_by_location("december"),
    "fig_penalty_by_location_june": lambda: fig_penalty_by_location("june"),
    "fig_penalty_by_location_december": lambda: fig_penalty_by_location("december"),
    "fig_penalty_fixed_by_location_june": lambda: fig_penalty_fixed_by_location("june"),
    "fig_penalty_fixed_by_location_december": lambda: fig_penalty_fixed_by_location("december"),
    "fig_duration_by_location": fig_duration_by_location,
    "fig_solar_bins": fig_solar_bins,
    "fig_dt_check": fig_dt_check,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for name in FIGURES:
            print(name)
        return 0
    apply_paper_style()
    selected = list(FIGURES.keys()) if args.only is None else args.only
    unknown = [n for n in selected if n not in FIGURES]
    if unknown:
        print(f"Unknown figure(s): {unknown}", file=sys.stderr)
        return 2
    for name in selected:
        print(f"--- {name} ---")
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
