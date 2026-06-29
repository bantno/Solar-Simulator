#!/usr/bin/env python3
"""Threshold Sweep Plotter

Plots:
1. Takeoff threshold (lowest SoC where flying > moored at each timestep).
2. Landing "cliff" threshold (first large jump in flying value under 20% SoC).

QoL:
- Auto-extract mission start datetime from the filename (e.g., "..._2025-06-10 0.npy").
- Assumes 15-minute timestep (fixed).
- Boxed legend at best location.
- Saves PNG only (no PDF).
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta  # ← include timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DATE_PATTERNS = [
    r"(?P<date>\d{4}-\d{2}-\d{2})(?:[ _](?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?)?",
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?",
]

DT_MINUTES = 15  # fixed by convention


def infer_start_datetime_from_path(path: Path) -> datetime | None:
    """Try to parse a datetime from the filename."""
    s = path.as_posix()
    for pat in DATE_PATTERNS:
        m = re.search(pat, s)
        if m:
            d = m.group("date")
            hour = m.group("hour")
            minute = m.group("minute")
            y, mo, dd = map(int, d.split("-"))
            hh = int(hour) if hour is not None else 0
            mm = int(minute) if minute is not None else 0
            return datetime(y, mo, dd, hh, mm)
    return None


def split_blocks(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (moored, flying, soc%)."""
    rows, T = data.shape
    if rows % 2 == 0:
        n_soc = rows // 2
    else:
        n_soc = (rows - 1) // 2  # ignore last row if present

    moored = data[:n_soc, :]
    flying = data[n_soc:2 * n_soc, :]
    soc = np.linspace(0, 100, n_soc)
    return moored, flying, soc


def build_time_axis(start_dt: datetime, T: int):
    """Return Python datetime list with fixed 15-min increments."""
    return [start_dt + timedelta(minutes=i * DT_MINUTES) for i in range(T)]


def compute_thresholds(moored: np.ndarray, flying: np.ndarray, soc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute takeoff and landing cliff thresholds."""
    n_soc, T = moored.shape

    flying_better_threshold = np.full(T, np.nan)
    cliff_threshold = np.full(T, np.nan)

    jump_threshold = 1.0
    soc_cutoff = 20.0

    for t in range(T):
        # Takeoff threshold: first SoC where flying > moored
        diff = flying[:, t] - moored[:, t]
        idx = np.where(diff > 0)[0]
        if idx.size > 0:
            flying_better_threshold[t] = soc[idx[0]]

        # Cliff threshold: large jump in flying under cutoff
        values = flying[:, t]
        delta = np.diff(values)
        mask = soc[:-1] < soc_cutoff
        idx = np.where((delta > jump_threshold) & mask)[0]
        if idx.size > 0:
            cliff_threshold[t] = soc[idx[0] + 1]

    return flying_better_threshold, cliff_threshold


def main():
    p = argparse.ArgumentParser(description="Threshold sweep plotter.")
    p.add_argument("path", type=str, help="Path to .npy array")
    p.add_argument("--title", type=str, default="Flying vs. Mooring Thresholds")
    p.add_argument("--width", type=float, default=12, help="Figure width (inches).")
    p.add_argument("--height", type=float, default=4, help="Figure height (inches).")
    p.add_argument("--line-width", type=float, default=1.5)
    p.add_argument("--no-save", action="store_true", help="Do not save figure.")
    p.add_argument("--show", action="store_true", help="Show figure window.")
    p.add_argument("--out-suffix", type=str, default="_thresholds", help="Suffix for saved PNG.")
    args = p.parse_args()

    npy_path = Path(args.path)
    data = np.load(npy_path)
    moored, flying, soc = split_blocks(data)
    _, T = moored.shape

    # Start datetime
    inferred = infer_start_datetime_from_path(npy_path)
    if inferred is None:
        print("[WARN] Could not infer start date from filename; using 2000-01-01 00:00.")
        start_dt = datetime(2000, 1, 1)
    else:
        start_dt = inferred

    times = build_time_axis(start_dt, T)

    # Compute thresholds
    flying_better, cliff = compute_thresholds(moored, flying, soc)

    # Plot
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    ax.plot(times, flying_better, label="Takeoff Threshold", color="tab:blue", linewidth=args.line_width)
    ax.plot(times, cliff, label="Landing Threshold", color="tab:red", linewidth=args.line_width)

    # Highlight missing takeoffs
    no_takeoff = np.isnan(flying_better)
    if np.any(no_takeoff):
        ax.scatter(np.array(times, dtype="object")[no_takeoff],  # ensure array indexing aligns
                   [100] * int(np.sum(no_takeoff)),
                   color="gray", s=12, label="No Takeoff", zorder=3)

    # Format
    ax.set_xlabel("Time")
    ax.set_ylabel("SoC (%)")
    # ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="best")  # boxed legend
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()

    if not args.no_save:
        out_base = npy_path.with_suffix("")
        out_png = out_base.as_posix() + args.out_suffix + ".png"
        fig.savefig(out_png, dpi=200)
        print(f"[OK] Saved: {out_png}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
