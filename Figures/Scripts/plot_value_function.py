#!/usr/bin/env python3
"""Value-function visualisation: difference heatmap + 3D surface plot.

Usage:
    python plot_value_function.py path/to/file.npy \
        --outdir Figures/Results/test/ \
        --start-datetime "2025-06-10 00:00"
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# ── Reuse helpers from sibling scripts ──────────────────────────────
from surface_plot_alternatives import (
    split_blocks,
    infer_start_datetime_from_path,
    build_time_index,
    DT_MINUTES,
)

# ── Styling (from plot_episode.py) ──────────────────────────────────
STYLE_NAME = "seaborn-v0_8-whitegrid"
RCPARAMS = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "lines.linewidth": 0.5,
    "figure.dpi": 300,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black",
    "grid.linestyle": "-",
    "grid.alpha": 0.35,
}


def _apply_style():
    plt.style.use(STYLE_NAME)
    plt.rcParams.update(RCPARAMS)


# ── Filename metadata (adapted from plot_surfaces_plotly.py) ────────
def parse_filename(filename: str):
    """Extract (capacity, horizon_hours, penalty) from a .npy filename."""
    base = os.path.basename(filename)
    pattern = (
        r"(?P<cap>[\d\.]+)Wh_"
        r"(?P<horizon>\d+)h_"
        r"(?P<pen>[\d\.]+)p"
    )
    m = re.search(pattern, base)
    if not m:
        return None, None, None
    return float(m.group("cap")), int(m.group("horizon")), float(m.group("pen"))


# ── Figure 1: Difference heatmap ───────────────────────────────────
def plot_delta_heatmap(
    npy_path: Path,
    outdir: Path,
    start_dt: datetime,
) -> str:
    _apply_style()

    data = np.load(npy_path)
    moored, flying, soc = split_blocks(data)
    _, T = moored.shape

    delta = flying - moored

    times = build_time_index(start_dt, T)
    # Convert to days since start for the mesh
    days = (times - times[0]).total_seconds() / 86400.0

    X, Y = np.meshgrid(days, soc)

    vmax = np.nanmax(np.abs(delta))
    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(X, Y, delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(X, Y, delta, levels=[0], colors="black", linewidths=1.5)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$V_{\mathrm{fly}} - V_{\mathrm{float}}$")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SoC (%)")

    cap, horizon, pen = parse_filename(str(npy_path))
    if cap is not None:
        ax.set_title(f"{cap:.0f} Wh  |  {horizon} h  |  penalty {pen}")

    fig.tight_layout()

    stem = npy_path.stem
    out_path = outdir / f"value_delta_heatmap_{stem}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return str(out_path)


# ── Figure 2: 3D surface plot ──────────────────────────────────────
def plot_3d_surfaces(
    npy_path: Path,
    outdir: Path,
    start_dt: datetime,
) -> str:
    _apply_style()

    data = np.load(npy_path)
    moored, flying, soc = split_blocks(data)
    _, T = moored.shape

    times = build_time_index(start_dt, T)
    days = (times - times[0]).total_seconds() / 86400.0

    X, Y = np.meshgrid(days, soc)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, moored, color="steelblue", alpha=0.6, label="Floating")
    ax.plot_surface(X, Y, flying, color="orangered", alpha=0.6, label="Flying")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SoC (%)")
    ax.set_zlabel("Expected Value")

    cap, horizon, pen = parse_filename(str(npy_path))
    if cap is not None:
        ax.set_title(f"{cap:.0f} Wh  |  {horizon} h  |  penalty {pen}")

    # Matplotlib 3D surfaces don't natively support legend labels;
    # create proxy artists.
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.6, label="Floating"),
            Patch(facecolor="orangered", alpha=0.6, label="Flying"),
        ],
        loc="upper left",
    )

    fig.tight_layout()

    stem = npy_path.stem
    out_path = outdir / f"value_surfaces_{stem}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return str(out_path)


# ── Process one file ───────────────────────────────────────────────
def process_file(npy_path: Path, outdir: Path, start_dt_override: datetime | None):
    if start_dt_override:
        start_dt = start_dt_override
    else:
        start_dt = infer_start_datetime_from_path(npy_path)
        if start_dt is None:
            start_dt = datetime(2000, 1, 1)
            print("[WARN] Could not infer start datetime; using 2000-01-01.")

    outdir.mkdir(parents=True, exist_ok=True)

    plot_delta_heatmap(npy_path, outdir, start_dt)
    plot_3d_surfaces(npy_path, outdir, start_dt)


# ── CLI ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Value-function heatmap + 3D surface plots.")
    p.add_argument("path", type=str, help=".npy file or directory (batch all .npy in dir)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (default: same dir as .npy)")
    p.add_argument("--start-datetime", type=str, default=None,
                   help="Override start datetime, e.g. '2025-06-10 00:00'")
    args = p.parse_args()

    target = Path(args.path)
    start_dt = pd.to_datetime(args.start_datetime).to_pydatetime() if args.start_datetime else None

    if target.is_dir():
        npy_files = sorted(target.glob("*.npy"))
        if not npy_files:
            print(f"No .npy files found in {target}")
            return
    else:
        npy_files = [target]

    for npy in npy_files:
        outdir = Path(args.outdir) if args.outdir else npy.parent
        process_file(npy, outdir, start_dt)


if __name__ == "__main__":
    main()
