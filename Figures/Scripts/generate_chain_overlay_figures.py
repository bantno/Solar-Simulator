#!/usr/bin/env python3
"""Journal-style mean-reward overlays: threshold vs i.i.d.-optimal vs wind-chain-optimal.

Renders the journal paper's sweep figures (mean reward vs battery capacity,
failure penalty, and mission duration) from the thesis-sweep / penalty-extension
analysis CSVs, with the wind-chain DP arm added as a third data series. Style,
sizing, and output directory match ``generate_journal_paper_figures.py``.

Data sources (all pooled over the 5 sites; capacity/penalty panels also pool
summer + winter starts):

* capacity: ``results/thesis_sweep/_analysis_batgrid/markov_ablation_cells.csv``
  (60-day missions, failure penalty 5)
* penalty:  ``results/penalty_ext/_analysis/penalty_curve_cells.csv``
  (300 Wh, 60 days; threshold column is the fine-grid best tuning, re-weighted
  per penalty from its fp=5 run — behavior is penalty-invariant)
* duration: ``results/thesis_sweep/_analysis_duration/markov_ablation_cells.csv``
  (300 Wh, June starts, failure penalty 5; per-stage reward as in the paper's
  horizon figure)

The threshold series is the best-tuned combination per cell (winner's-curse
bounded by the split-half check in the analysis engine). "Optimal" arms are the
backward-induction DP policies solved against the i.i.d. and the 5-bin
Markov-wind weather models respectively, all evaluated on the same historical
block-bootstrap episodes (paired common random numbers).

Usage::

    python Figures/Scripts/generate_chain_overlay_figures.py            # all three
    python Figures/Scripts/generate_chain_overlay_figures.py --only fig_capacity_arms_overlay
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
    SINGLE_COL_IN,
    _panel_label,
    _save,
    apply_paper_style,
)

REPO_ROOT = _THIS_DIR.parent.parent
BATGRID_CSV = REPO_ROOT / "results" / "thesis_sweep" / "_analysis_batgrid" / "markov_ablation_cells.csv"
DURATION_CSV = REPO_ROOT / "results" / "thesis_sweep" / "_analysis_duration" / "markov_ablation_cells.csv"
PENALTY_CSV = REPO_ROOT / "results" / "penalty_ext" / "_analysis" / "penalty_curve_cells.csv"

DT_MINUTES = 15
PANEL_PENALTY = 5.0  # capacity + duration panels

# Fixed series identities across every figure: the DP-optimal arms are solid
# (i.i.d. keeps the paper's black "Optimal" role), the tuned threshold
# benchmark is dashed with open markers.
ARMS = (
    ("thresh", "Best threshold", dict(color="tab:orange", linestyle="--",
                                      marker="o", mfc="white")),
    ("iid", "Optimal (i.i.d.)", dict(color="black", linestyle="-", marker="o")),
    ("wind", "Optimal (wind chain)", dict(color="tab:blue", linestyle="-",
                                          marker="o")),
)


def _plot_arms(ax: plt.Axes, agg: pd.DataFrame, x_col: str
               ) -> Tuple[List, List[str]]:
    """Plot one line per arm from a frame with columns x_col + {arm} means."""
    handles: List = []
    labels: List[str] = []
    for arm, label, style in ARMS:
        ser = agg.sort_values(x_col)
        h, = ax.plot(ser[x_col], ser[arm], label=label, **style)
        handles.append(h)
        labels.append(label)
    return handles, labels


def fig_capacity_arms_overlay() -> None:
    df = pd.read_csv(BATGRID_CSV)
    df = df[df["failure_penalty"] == PANEL_PENALTY]
    agg = (
        df.groupby("battery_capacity", as_index=False)
          .agg(thresh=("avg_reward_thresh", "mean"),
               iid=("avg_reward_iid", "mean"),
               wind=("avg_reward_wind", "mean"))
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.5))
    _plot_arms(ax, agg, "battery_capacity")
    ax.set_xlabel("Battery Capacity (Wh)")
    ax.set_ylabel("Mean Total Reward")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, "capacity_mean_reward_arms")


def fig_failure_penalty_arms_overlay() -> None:
    df = pd.read_csv(PENALTY_CSV)
    agg = (
        df.groupby("pen", as_index=False)
          .agg(thresh=("r_thresh_fine", "mean"),
               iid=("r_iid", "mean"),
               wind=("r_wind", "mean"))
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.5))
    _plot_arms(ax, agg, "pen")
    ax.set_xscale("log", base=2)
    pens = sorted(agg["pen"].unique())
    ax.set_xticks(pens)
    ax.set_xticklabels([f"{p:g}" for p in pens])
    ax.tick_params(axis="x", labelsize=7)
    ax.minorticks_off()
    ax.set_xlabel(r"Failure Penalty $\phi$")
    ax.set_ylabel("Mean Total Reward")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, "failure_penalty_mean_reward_arms")


def fig_duration_arms_overlay() -> None:
    df = pd.read_csv(DURATION_CSV)
    df = df[df["failure_penalty"] == PANEL_PENALTY].copy()
    df["days"] = df["horizon"] * DT_MINUTES / 1440.0
    for arm in ("thresh", "iid", "wind"):
        df[f"{arm}_per_stage"] = df[f"avg_reward_{arm}"] / df["horizon"]
    agg = (
        df.groupby("days", as_index=False)
          .agg(thresh=("thresh_per_stage", "mean"),
               iid=("iid_per_stage", "mean"),
               wind=("wind_per_stage", "mean"))
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.5))
    _plot_arms(ax, agg, "days")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                        useMathText=True)
    ax.set_xlabel("Mission Duration (days)")
    ax.set_ylabel("Mean Reward per Stage")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, "duration_mean_reward_arms")


def fig_arms_overlay_combined() -> None:
    """All three sweeps as one single-column stacked figure with shared legend."""
    fig, axes = plt.subplots(
        3, 1, figsize=(SINGLE_COL_IN, 6.6),
        gridspec_kw={"hspace": 0.42},
    )

    df = pd.read_csv(BATGRID_CSV)
    df = df[df["failure_penalty"] == PANEL_PENALTY]
    agg = (
        df.groupby("battery_capacity", as_index=False)
          .agg(thresh=("avg_reward_thresh", "mean"),
               iid=("avg_reward_iid", "mean"),
               wind=("avg_reward_wind", "mean"))
    )
    handles, labels = _plot_arms(axes[0], agg, "battery_capacity")
    axes[0].set_xlabel("Battery Capacity (Wh)")
    axes[0].set_ylabel("Mean Total Reward")
    _panel_label(axes[0], "(a)")

    df = pd.read_csv(PENALTY_CSV)
    agg = (
        df.groupby("pen", as_index=False)
          .agg(thresh=("r_thresh_fine", "mean"),
               iid=("r_iid", "mean"),
               wind=("r_wind", "mean"))
    )
    _plot_arms(axes[1], agg, "pen")
    axes[1].set_xscale("log", base=2)
    pens = sorted(agg["pen"].unique())
    axes[1].set_xticks(pens)
    axes[1].set_xticklabels([f"{p:g}" for p in pens])
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].minorticks_off()
    axes[1].set_xlabel(r"Failure Penalty $\phi$")
    axes[1].set_ylabel("Mean Total Reward")
    _panel_label(axes[1], "(b)")

    df = pd.read_csv(DURATION_CSV)
    df = df[df["failure_penalty"] == PANEL_PENALTY].copy()
    df["days"] = df["horizon"] * DT_MINUTES / 1440.0
    for arm in ("thresh", "iid", "wind"):
        df[f"{arm}_per_stage"] = df[f"avg_reward_{arm}"] / df["horizon"]
    agg = (
        df.groupby("days", as_index=False)
          .agg(thresh=("thresh_per_stage", "mean"),
               iid=("iid_per_stage", "mean"),
               wind=("wind_per_stage", "mean"))
    )
    _plot_arms(axes[2], agg, "days")
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=7))
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                             useMathText=True)
    axes[2].set_xlabel("Mission Duration (days)")
    axes[2].set_ylabel("Mean Reward per Stage")
    _panel_label(axes[2], "(c)")

    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=True, framealpha=0.9, edgecolor="black",
    )
    fig.subplots_adjust(top=0.98, left=0.20, right=0.97, bottom=0.13)
    _save(fig, "arms_overlay_combined")


FIGURES = {
    "fig_capacity_arms_overlay": fig_capacity_arms_overlay,
    "fig_failure_penalty_arms_overlay": fig_failure_penalty_arms_overlay,
    "fig_duration_arms_overlay": fig_duration_arms_overlay,
    "fig_arms_overlay_combined": fig_arms_overlay_combined,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Render only these figure names (see --list).")
    ap.add_argument("--list", action="store_true",
                    help="Print the known figure names and exit.")
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
