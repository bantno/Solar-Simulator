#!/usr/bin/env python3
"""
Histogram of episode total rewards by failure penalty (one figure per sim).

Assumptions / notes
-------------------
- The HDF5 file stores results as *per-episode* groups or datasets.
- Each episode group (or its ancestors) must contain:
    - a simulation type (e.g., 'Optimal', 'Threshold') discoverable via:
        * group.attrs['sim_type'] OR any ancestor's attrs, OR
        * a token in the full path like 'sim_type=Optimal' or 'algorithm=Optimal'
    - a failure penalty (float) discoverable via:
        * group.attrs['failure_penalty'] OR any ancestor's attrs, OR
        * a token in the path like 'failure_penalty=5.0', 'penalty=5.0', or 'fp=5.0'
- Each episode must contain ONE of these datasets with a scalar float:
    'total_reward' (preferred), 'reward', or 'episode_reward'.

What it does
------------
- Builds a dictionary: rewards[sim_type][penalty] = list of episode rewards.
- For a user-provided list of penalties, plots stacked subplots of histograms.
- Creates ONE figure per sim type (no mixing). Saves PNG/PDF/SVG.

Example
-------
python hist_episode_rewards_by_penalty.py \
  --h5 /path/to/results.h5 \
  --penalties 0 5 20 40 \
  --bins 50 \
  --outdir ./figs \
  --figwidth 6 --figheight 10 \
  --dpi 200

Tips
----
- If your HDF5 field names differ, adjust DATASET_CANDIDATES or the
  key-parsing regex in `_extract_meta_from_path`.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Iterable

import h5py
import numpy as np
import matplotlib.pyplot as plt


# Dataset names we try (first match wins)
DATASET_CANDIDATES: Tuple[str, ...] = ("total_reward", "reward", "episode_reward")

# Regex patterns for parsing metadata out of full HDF5 paths if attrs are absent.
# We accept a few common spellings.
SIM_PATTERNS = [
    re.compile(r"(?:^|/)(?:sim_type|algorithm)=(?P<sim>[^/]+)(?:/|$)"),
]
PENALTY_PATTERNS = [
    re.compile(r"(?:^|/)(?:failure_penalty|penalty|fp)=(?P<pen>[-+]?\d*\.?\d+)(?:/|$)"),
]


def _find_first_dataset(group: h5py.Group) -> Optional[h5py.Dataset]:
    """Return the first dataset among DATASET_CANDIDATES directly under the group,
    or None if not found."""
    for name in DATASET_CANDIDATES:
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def _walk_ancestors_for_attr(h5obj: h5py.Group, key: str) -> Optional[object]:
    """Walk up ancestors (including self) to find attribute `key`."""
    cur = h5obj
    while cur is not None:
        if isinstance(cur, h5py.Group) and key in cur.attrs:
            return cur.attrs[key]
        # climb to parent (None if at root)
        parent_name = cur.parent.name if hasattr(cur, "parent") else None
        if not parent_name:
            break
        cur = cur.parent
    return None


def _extract_meta_from_path(path: str) -> Tuple[Optional[str], Optional[float]]:
    """Try to parse sim_type and failure_penalty from path tokens."""
    sim_val = None
    pen_val = None
    for rp in SIM_PATTERNS:
        m = rp.search(path)
        if m:
            sim_val = m.group("sim")
            break
    for rp in PENALTY_PATTERNS:
        m = rp.search(path)
        if m:
            try:
                pen_val = float(m.group("pen"))
            except ValueError:
                pen_val = None
            break
    return sim_val, pen_val


def _get_episode_meta(group: h5py.Group) -> Tuple[Optional[str], Optional[float]]:
    """Obtain (sim_type, failure_penalty) for an episode group."""
    # Try attributes first (on group or ancestors)
    sim_attr = _walk_ancestors_for_attr(group, "sim_type")
    pen_attr = _walk_ancestors_for_attr(group, "failure_penalty")
    if sim_attr is None:
        # Some files use 'algorithm'
        sim_attr = _walk_ancestors_for_attr(group, "algorithm")
    if pen_attr is None:
        # Fallbacks some people use
        pen_attr = _walk_ancestors_for_attr(group, "penalty") or _walk_ancestors_for_attr(group, "fp")

    sim = sim_attr.decode() if isinstance(sim_attr, (bytes, bytearray)) else sim_attr
    pen = float(pen_attr) if pen_attr is not None else None

    # If still missing, parse from path
    if sim is None or pen is None:
        sim2, pen2 = _extract_meta_from_path(group.name)
        sim = sim if sim is not None else sim2
        pen = pen if pen is not None else pen2

    return sim, pen


def _gather_rewards(h5path: str) -> Dict[str, Dict[float, List[float]]]:
    """Traverse the HDF5 and collect rewards keyed by sim_type and failure_penalty."""
    rewards: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    with h5py.File(h5path, "r") as f:
        # Traverse all groups; treat any group that contains one of the dataset candidates as an episode
        def visit_fn(name: str, obj):
            if not isinstance(obj, h5py.Group):
                return
            ds = _find_first_dataset(obj)
            if ds is None:
                return
            # Must be scalar or single value
            try:
                val = ds[()]  # read
            except Exception:
                return
            # Convert to float if array-like
            if isinstance(val, (np.ndarray, list, tuple)):
                if np.size(val) == 1:
                    val = float(np.ravel(val)[0])
                else:
                    # Not a scalar reward; skip
                    return
            else:
                val = float(val)

            sim, pen = _get_episode_meta(obj)
            if sim is None or pen is None:
                # Cannot categorize—skip this episode
                return
            rewards[sim][pen].append(val)

        f.visititems(visit_fn)
    return rewards


def _nice_bins(data: Iterable[float], bins: int) -> np.ndarray:
    """Compute common bin edges spanning the full data range."""
    data = np.asarray(list(data))
    if data.size == 0:
        return np.linspace(0, 1, bins + 1)
    lo, hi = np.nanmin(data), np.nanmax(data)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, bins + 1)


def _plot_for_sim(sim: str,
                  pen_order: List[float],
                  rewards_by_pen: Dict[float, List[float]],
                  bins: int,
                  figwidth: float,
                  figheight: float,
                  dpi: int,
                  outdir: str):
    """Create stacked histograms for a single sim."""
    # Gather all rewards across selected penalties to make a shared x-range
    all_vals = []
    for p in pen_order:
        all_vals.extend(rewards_by_pen.get(p, []))
    bin_edges = _nice_bins(all_vals, bins)

    n = len(pen_order)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figwidth, figheight), sharex=True, constrained_layout=False)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, pen_order):
        vals = rewards_by_pen.get(p, [])
        ax.hist(vals, bins=bin_edges, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Penalty {p}", fontsize=12, pad=8)
        ax.set_ylabel("Episode Count", fontsize=10)
        ax.grid(True, linestyle="-", alpha=0.3)
        # Clean look
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Total Reward", fontsize=11)

    # Overall figure title
    fig.suptitle(f"{sim}", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"hist_rewards_by_penalty__{sim.replace(' ', '_')}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot histograms of episode total rewards by failure penalty (stacked), one figure per sim type.")
    ap.add_argument("--h5", required=True, help="Path to results .h5 file.")
    ap.add_argument("--penalties", type=float, nargs="+", required=True,
                    help="List of failure penalties to include (order is preserved).")
    ap.add_argument("--bins", type=int, default=50, help="Number of bins to use for all subplots.")
    ap.add_argument("--figwidth", type=float, default=6.0, help="Figure width in inches.")
    ap.add_argument("--figheight", type=float, default=10.0, help="Figure height in inches.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    ap.add_argument("--outdir", default="figs", help="Directory to save figures.")
    args = ap.parse_args()

    rewards = _gather_rewards(args.h5)

    if not rewards:
        raise RuntimeError("No rewards found. Check dataset names and metadata extraction in the script header.")

    # One figure per sim_type; never mix sims
    for sim, by_pen in rewards.items():
        _plot_for_sim(
            sim=sim,
            pen_order=args.penalties,
            rewards_by_pen=by_pen,
            bins=args.bins,
            figwidth=args.figwidth,
            figheight=args.figheight,
            dpi=args.dpi,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
