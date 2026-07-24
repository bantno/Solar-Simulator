#!/usr/bin/env python3
"""persistence_bins.py -- choose wind-bin edges by MAXIMIZING captured persistence.

A weather-only bin selector: pick the interior edges that maximize the discrete chain's
first-order captured persistence, stratified by (month, hour) exactly as the chain conditions:

    I(B_{t+1} ; B_t | month, hour) = H(B_{t+1}|m,h) - H(B_{t+1}|B_t, m, h)   [bits]

computed directly from the historical 15-min wind series -- NO MDP solve.  This optimizes exactly
what the chain exists to exploit (one-step predictability) and answers "can bins be selected by
preprocessing the weather, without solving the value function twice?".

Reuses the information machinery in Scripts/wind_persistence_precheck.py (make_bins) and the failbin
provisioning idiom (resampled_wind, build_wind_chain_artifact, config templates).

Modes:
  select   : print selected edges + MI vs equal-occupancy for a location (sanity, no solve).
  provision: build persistence chain artifacts + configs for n in {3,4,5}, plus the matched-count
             equal-occupancy baseline chain_wind4 (3 and 5 already exist).  Writes persist_edges.json.
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

from Scripts.wind_persistence_precheck import make_bins  # noqa: E402
from Scripts.create_weather_distributions import build_wind_chain_artifact  # noqa: E402
from Scripts.failbin.failbin_experiment import (  # noqa: E402
    LOCATIONS, FULL, INTERVAL_MIN, CONFIG_DIR, loc_paths, resampled_wind,
    _rel, _base_cfg, _dump,
)

LN2 = np.log(2.0)
MIN_OCC = 0.05          # reject any bin with <5% occupancy (guards against MI bias degeneracy)
N_LIST = (3, 4, 5)


# ----------------------------------------------------------------------------------
# First-order captured persistence I(next; curr | month, hour), from 15-min pairs
# ----------------------------------------------------------------------------------
def consecutive_pairs(bins, idx, step_min, n_bins):
    """(curr, nxt, month, hour) for adjacent samples exactly step_min apart, both bins valid.
    month/hour taken at the current step i (source of the curr->next transition)."""
    step = np.diff(idx.values).astype("timedelta64[m]").astype(int)   # gap i -> i+1
    n = bins.size
    i = np.arange(0, n - 1)
    curr, nxt = bins[i], bins[i + 1]
    ok = (step == step_min) & (curr >= 0) & (curr < n_bins) & (nxt >= 0) & (nxt < n_bins)
    sel = i[ok]
    return curr[ok], nxt[ok], idx.month.values[sel], idx.hour.values[sel]


def mi_first_order_bits(curr, nxt, month, hour, n_bins):
    """Occupancy-weighted stratified first-order MI in bits (sparsity rule: skip strata < nb^2)."""
    nb = n_bins
    key = month.astype(np.int64) * 24 + hour.astype(np.int64)
    uniq = np.unique(key)
    skey = np.searchsorted(uniq, key)
    n_strata = uniq.size
    n_total = curr.size
    flat = (skey * nb + curr) * nb + nxt
    c3 = np.bincount(flat, minlength=n_strata * nb * nb).astype(np.float64).reshape(n_strata, nb, nb)
    stratum_tot = c3.sum(axis=(1, 2))                       # (s,)
    c_c = c3.sum(axis=2)                                    # (s, curr)
    c_n = c3.sum(axis=1)                                    # (s, next)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_cn = c3 / c_c[:, :, None]                         # P(next | curr, s)
        p_n = c_n / stratum_tot[:, None]                   # P(next | s)
        ratio = p_cn / p_n[:, None, :]
        term = c3 * np.log(ratio)
    term = np.where(c3 > 0, term, 0.0)
    incl = stratum_tot >= (nb * nb)
    total = float(np.where(incl[:, None, None], term, 0.0).sum())
    return total / (LN2 * n_total) if n_total else 0.0


def _precompute(series):
    """Precompute per-pair invariants once (independent of the edges): step-gap mask and the
    month/hour at the *current* step.  Only the binning changes across candidate edges."""
    vals = series.values.astype(float)
    idx = pd.DatetimeIndex(series.index)
    step = np.diff(idx.values).astype("timedelta64[m]").astype(int)   # gap i -> i+1, len N-1
    return dict(vals=vals,
                step_ok=(step == INTERVAL_MIN),
                month=idx.month.values[:-1].astype(np.int64),          # at curr = i
                hour=idx.hour.values[:-1].astype(np.int64))


def _score(pc, interior, n_bins):
    """MI + occupancy for candidate interior edges, using precomputed invariants."""
    vals = pc["vals"]
    bins = np.digitize(vals, np.asarray(interior, float))            # 0..n_bins-1
    bins = np.where(np.isfinite(vals), bins, -1)
    vb = bins[bins >= 0]
    occ = np.bincount(vb, minlength=n_bins)[:n_bins] / max(vb.size, 1)
    if occ.min() < MIN_OCC:
        return -np.inf, occ
    curr, nxt = bins[:-1], bins[1:]
    ok = pc["step_ok"] & (curr >= 0) & (curr < n_bins) & (nxt >= 0) & (nxt < n_bins)
    mi = mi_first_order_bits(curr[ok], nxt[ok], pc["month"][ok], pc["hour"][ok], n_bins)
    return mi, occ


# ----------------------------------------------------------------------------------
# Edge optimizer (no MDP solve)
# ----------------------------------------------------------------------------------
def _candidates(vals):
    """Interior-edge candidate wind speeds: percentiles 5..95 of the wind distribution."""
    v = vals[np.isfinite(vals)]
    return np.unique(np.percentile(v, np.arange(5.0, 95.01, 2.5)))


def select_edges(series, n_bins, pc=None):
    """Return (interior_edges, mi, occupancy) maximizing stratified first-order MI."""
    if pc is None:
        pc = _precompute(series)
    vals = pc["vals"]
    cand = _candidates(vals)

    # Seed at equal-occupancy edges.
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = list(np.quantile(vals[np.isfinite(vals)], qs))
    best_mi, best_occ = _score(pc, interior, n_bins)

    if n_bins == 3:
        # 2-D grid over ordered candidate pairs.
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                mi, occ = _score(pc, [cand[a], cand[b]], n_bins)
                if mi > best_mi:
                    best_mi, best_occ, interior = mi, occ, [cand[a], cand[b]]
    else:
        # Greedy coordinate ascent: line-search each interior edge, keep order, ~3 passes.
        for _ in range(3):
            improved = False
            for k in range(len(interior)):
                lo = interior[k - 1] if k > 0 else 0.0
                hi = interior[k + 1] if k + 1 < len(interior) else np.inf
                for c in cand:
                    if not (lo < c < hi):
                        continue
                    trial = list(interior); trial[k] = float(c)
                    mi, occ = _score(pc, trial, n_bins)
                    if mi > best_mi:
                        best_mi, best_occ, interior, improved = mi, occ, trial, True
            if not improved:
                break

    return [float(x) for x in interior], float(best_mi), best_occ


def equal_occ_edges(series, n_bins, pc=None):
    if pc is None:
        pc = _precompute(series)
    vals = pc["vals"]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = list(np.quantile(vals[np.isfinite(vals)], qs))
    mi, occ = _score(pc, interior, n_bins)
    return [float(x) for x in interior], float(mi), occ


# ----------------------------------------------------------------------------------
# Provisioning
# ----------------------------------------------------------------------------------
def _chain_path(loc, suffix):
    base, ext = os.path.splitext(loc_paths(loc)["exp"])
    return f"{base}_windchain_{suffix}{ext}"


def _write_cfg(loc, path, arm, n_bins, out_dir):
    paths = loc_paths(loc)
    c = _base_cfg(loc, paths, FULL, f"full_{loc['name']}_{arm}")
    c.update(include_optimal=True, threshold_values=[], wind_thresholds=[],
             wind_chain=dict(enabled=True, path=_rel(path), n_bins=n_bins))
    _dump(c, out_dir, f"full_{loc['name']}_{arm}.yaml")


def provision(loc, out_dir):
    paths = loc_paths(loc)
    series = resampled_wind(paths["hist"])
    pc = _precompute(series)
    summ = {"name": loc["name"], "per_n": {}}
    for n in N_LIST:
        p_int, p_mi, p_occ = select_edges(series, n, pc=pc)
        e_int, e_mi, e_occ = equal_occ_edges(series, n, pc=pc)
        print(f"[{loc['name']}] n={n}  persist edges={np.round(p_int,2)} MI={p_mi:.4f} occ={np.round(p_occ,2)}"
              f"  | equal-occ MI={e_mi:.4f}  (gain {100*(p_mi-e_mi)/max(e_mi,1e-9):+.1f}%)")
        pth = _chain_path(loc, f"persist{n}")
        build_wind_chain_artifact(paths["hist"], pth, interval_minutes=INTERVAL_MIN,
                                  bin_edges=np.concatenate(([0.0], np.asarray(p_int), [np.inf])))
        _write_cfg(loc, pth, f"chain_persist{n}", n, out_dir)
        summ["per_n"][n] = dict(persist_edges=p_int, persist_mi=p_mi, persist_occ=list(p_occ),
                                equalocc_edges=e_int, equalocc_mi=e_mi, equalocc_occ=list(e_occ))
    # Matched-count equal-occupancy baseline missing today: chain_wind4.
    p4 = _chain_path(loc, "wind4")
    build_wind_chain_artifact(paths["hist"], p4, interval_minutes=INTERVAL_MIN, n_bins=4)
    _write_cfg(loc, p4, "chain_wind4", 4, out_dir)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "provision"])
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--out", default=os.path.join(CONFIG_DIR, "full"))
    args = ap.parse_args()
    locs = LOCATIONS if not args.only else [l for l in LOCATIONS if l["name"] in args.only]

    if args.mode == "select":
        for loc in locs:
            series = resampled_wind(loc_paths(loc)["hist"])
            pc = _precompute(series)
            for n in N_LIST:
                p_int, p_mi, p_occ = select_edges(series, n, pc=pc)
                e_int, e_mi, e_occ = equal_occ_edges(series, n, pc=pc)
                print(f"{loc['name']:10s} n={n} persist={np.round(p_int,2)} MI={p_mi:.4f} occ={np.round(p_occ,2)}"
                      f" | eqocc={np.round(e_int,2)} MI={e_mi:.4f} occ={np.round(e_occ,2)}"
                      f" | gain {100*(p_mi-e_mi)/max(e_mi,1e-9):+.1f}%")
        return

    summaries = [provision(loc, args.out) for loc in locs]
    p = os.path.join(CONFIG_DIR, "persist_edges.json")
    json.dump(summaries, open(p, "w"), indent=2)
    print("persist edges ->", _rel(p))


if __name__ == "__main__":
    main()
