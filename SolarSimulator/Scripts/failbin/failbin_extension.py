#!/usr/bin/env python3
"""failbin_extension.py -- follow-up arms for the failbin study.

Two questions the baseline (3-bin wind vs failure) raised:
  (A) Is operating-regime RESOLUTION the lever?  -> wind-space bin-count sweep (5, 8 bins).
  (B) Can anything beat equal-occupancy wind bins? -> a DECISION-BOUNDARY scheme that
      concentrates the 3 bins around the marginal fly/no-fly wind (from the best threshold
      policy's wind_threshold), weighted by how often winds occur.

Modes:
  binsweep   : build wind-space quantile chains at n_bins in {5,8} and emit configs.
  decision   : read best threshold wind_threshold per location from a baseline metrics CSV,
               build decision-boundary chains, emit configs.

New arm tags: chain_wind5, chain_wind8, chain_dec.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from Scripts.create_weather_distributions import build_wind_chain_artifact  # noqa: E402
from Scripts.failbin.failbin_experiment import (  # noqa: E402
    LOCATIONS, FULL, N_BINS, INTERVAL_MIN, CONFIG_DIR, REPO_DIR,
    loc_paths, resampled_wind, _rel, _base_cfg, _dump,
)


def _chain_path(loc, suffix):
    base, ext = os.path.splitext(loc_paths(loc)["exp"])
    return f"{base}_windchain_{suffix}{ext}"


# ----------------------------------------------------------------------------------
# (A) Wind-space bin-count sweep
# ----------------------------------------------------------------------------------
def build_binsweep(loc, nbins_list=(5, 8)):
    paths = loc_paths(loc)
    made = {}
    for n in nbins_list:
        out = _chain_path(loc, f"wind{n}")
        if not os.path.exists(out):
            print(f"[{loc['name']}] building wind-space n_bins={n} -> {os.path.basename(out)}")
            build_wind_chain_artifact(paths["hist"], out, interval_minutes=INTERVAL_MIN, n_bins=n)
        else:
            print(f"[{loc['name']}] wind{n} cached")
        made[n] = out
    return made


def write_binsweep_configs(loc, made, out_dir):
    paths = loc_paths(loc)
    os.makedirs(out_dir, exist_ok=True)
    for n, path in made.items():
        c = _base_cfg(loc, paths, FULL, f"full_{loc['name']}_chain_wind{n}")
        c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
                 wind_chain=dict(enabled=True, path=_rel(path), n_bins=n))
        _dump(c, out_dir, f"full_{loc['name']}_chain_wind{n}.yaml")


# ----------------------------------------------------------------------------------
# (B) Decision-boundary bins
# ----------------------------------------------------------------------------------
def decision_boundary_edges(wind_vals, w_star, n_bins=N_BINS, sigma=None):
    """Interior edges that equalize DECISION mass g(w)=p(w)*exp(-((w-w*)/sigma)^2/2).

    Concentrates resolution around the marginal fly/no-fly wind w*, weighted by how
    often winds occur.  sigma defaults to half the wind IQR (climate-scaled).
    """
    w = np.asarray(wind_vals, float)
    w = w[np.isfinite(w)]
    if sigma is None:
        iqr = np.percentile(w, 75) - np.percentile(w, 25)
        sigma = max(1.0, 0.5 * iqr)
    order = np.argsort(w)
    w_sorted = w[order]
    s = np.exp(-0.5 * ((w_sorted - w_star) / sigma) ** 2)
    cum = np.cumsum(s)
    total = cum[-1]
    edges = []
    for k in range(1, n_bins):
        i = int(np.searchsorted(cum, total * k / n_bins))
        i = min(i, len(w_sorted) - 1)
        edges.append(float(w_sorted[i]))
    for k in range(1, len(edges)):
        if edges[k] <= edges[k - 1]:
            edges[k] = np.nextafter(edges[k - 1], np.inf)
    return edges, float(sigma)


def _best_wind_threshold(metrics_csv, loc_name):
    """Read the best threshold policy's wind_threshold for a location from the note column."""
    df = pd.read_csv(metrics_csv)
    row = df[(df.location == loc_name) & (df.arm == "threshold")]
    if not len(row):
        return None
    note = str(row.iloc[0].get("note", ""))
    # note like "obs_thr=0.1, wind_thr=4.0"
    for part in note.split(","):
        if "wind_thr" in part:
            try:
                return float(part.split("=")[1])
            except Exception:
                return None
    return None


def build_decision(loc, w_star):
    paths = loc_paths(loc)
    wind = resampled_wind(paths["hist"]).values
    edges, sigma = decision_boundary_edges(wind, w_star)
    full_edges = np.concatenate(([0.0], np.asarray(edges, float), [np.inf]))
    out = _chain_path(loc, "decision")
    print(f"[{loc['name']}] decision-boundary w*={w_star} sigma={sigma:.2f} "
          f"edges={np.round(edges,2)} -> {os.path.basename(out)}")
    build_wind_chain_artifact(paths["hist"], out, interval_minutes=INTERVAL_MIN, bin_edges=full_edges)
    # occupancy for reporting
    idx = np.digitize(wind[np.isfinite(wind)], edges)
    occ = [float(np.mean(idx == b)) for b in range(len(edges) + 1)]
    return dict(name=loc["name"], w_star=w_star, sigma=sigma, edges=edges, occupancy=occ), out


def write_decision_config(loc, path, out_dir):
    paths = loc_paths(loc)
    os.makedirs(out_dir, exist_ok=True)
    c = _base_cfg(loc, paths, FULL, f"full_{loc['name']}_chain_dec")
    c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
             wind_chain=dict(enabled=True, path=_rel(path), n_bins=N_BINS))
    _dump(c, out_dir, f"full_{loc['name']}_chain_dec.yaml")


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["binsweep", "decision"])
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--bins", type=int, nargs="+", default=[5, 8],
                    help="Equal-occupancy bin counts to build (binsweep mode).")
    ap.add_argument("--metrics", help="baseline failbin_metrics.csv (decision mode).")
    ap.add_argument("--out", default=os.path.join(CONFIG_DIR, "full"))
    args = ap.parse_args()

    locs = LOCATIONS if not args.only else [l for l in LOCATIONS if l["name"] in args.only]
    summ = []
    for loc in locs:
        if args.mode == "binsweep":
            made = build_binsweep(loc, nbins_list=tuple(args.bins))
            write_binsweep_configs(loc, made, args.out)
        else:
            w_star = _best_wind_threshold(args.metrics, loc["name"]) if args.metrics else None
            if w_star is None:
                print(f"[{loc['name']}] no w* found; skipping"); continue
            s, path = build_decision(loc, w_star)
            write_decision_config(loc, path, args.out)
            summ.append(s)
    if summ:
        p = os.path.join(CONFIG_DIR, "decision_edges.json")
        json.dump(summ, open(p, "w"), indent=2)
        print("decision edges ->", _rel(p))


if __name__ == "__main__":
    main()
