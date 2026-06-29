#!/usr/bin/env python3
"""Option 4 only: plot V_moored and V_flying over time for selected SoC slices.

QoL:
- Auto-extract mission start datetime from the filename (e.g., "..._2025-01-01 0.npy").
- CLI args for SoC slices.
- Robust handling of odd (2*n_soc + 1) vs even (2*n_soc) row layouts.
- Saves figure as PNG next to the .npy by default; can disable with --no-save.

Assumptions:
- Timestep is always 15 minutes.
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DATE_PATTERNS = [
    r"(?P<date>\d{4}-\d{2}-\d{2})(?:[ _](?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?)?",
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?",
]

DT_MINUTES = 15  # fixed timestep


def infer_start_datetime_from_path(path: Path) -> datetime | None:
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


def build_time_index(start_dt: datetime, T: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_dt, periods=T, freq=f"{DT_MINUTES}min")


def main():
    p = argparse.ArgumentParser(description="Plot V*(SoC,t) for fixed SoC levels (Option 4 only).")
    p.add_argument("path", type=str, help="Path to .npy array")
    p.add_argument("--soc", type=float, nargs="+", default=[100],
                   help="SoC slice(s) in percent to plot (e.g., --soc 100 80 50). Default: 100.")
    p.add_argument("--start-datetime", type=str, default=None,
                   help="Override start datetime (e.g., '2025-07-01 00:00'). If omitted, inferred from filename.")
    p.add_argument("--no-save", action="store_true", help="Do not save figure to file.")
    p.add_argument("--show", action="store_true", help="Show the plot in a window.")
    p.add_argument("--title", type=str, default=None, help="Optional figure title.")
    p.add_argument("--width", type=float, default=10, help="Figure width (inches).")
    p.add_argument("--height", type=float, default=4, help="Figure height (inches).")
    p.add_argument("--line-width", type=float, default=1.2, help="Line width for curves.")
    p.add_argument("--out-suffix", type=str, default="_soc_slices",
                   help="Suffix for saved filename (default: '_soc_slices').")

    args = p.parse_args()
    npy_path = Path(args.path)

    data = np.load(npy_path)
    moored, flying, soc = split_blocks(data)
    _, T = moored.shape

    # Infer start datetime
    if args.start_datetime:
        start_dt = pd.to_datetime(args.start_datetime).to_pydatetime()
    else:
        inferred = infer_start_datetime_from_path(npy_path)
        if inferred is None:
            start_dt = datetime(2000, 1, 1, 0, 0)
            print("[WARN] Could not infer start date from filename; using 2000-01-01 00:00. "
                  "Override with --start-datetime.")
        else:
            start_dt = inferred

    # Build time axis
    times = build_time_index(start_dt, T)

    # Slice by requested SoC values
    soc_vals = np.array(args.soc, dtype=float)
    slice_indices = [int(np.argmin(np.abs(soc - val))) for val in soc_vals]

    # Plot
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for idx in slice_indices:
        s_disp = soc[idx]
        ax.plot(times, moored[idx, :], linestyle="--", linewidth=args.line_width,
                label=f"Floating {s_disp:.0f}%")
        ax.plot(times, flying[idx, :], linestyle="-", linewidth=args.line_width,
                label=f"Flying {s_disp:.0f}%")

    if args.title:
        ax.set_title(args.title)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, frameon=True, loc="best")  # box around legend, auto placement
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()

    if not args.no_save:
        out_base = npy_path.with_suffix("")  # drop .npy
        out_png = out_base.as_posix() + f"{args.out_suffix}.png"
        fig.savefig(out_png, dpi=200)
        print(f"[OK] Saved: {out_png}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
