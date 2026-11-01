#!/usr/bin/env python3
"""Pick the wind-bin count on the fit years only (thesis run 1).

Reads a wind-bin study run on an inner split of the fit years (fit 1950-1985, episodes
1986-2002; generate_wind_bins_study_configs.py --holdout 1950 1985 1986 2002), so the
choice never sees the 2003-2022 evaluation weather. The rule was fixed before any result
was seen (2026-10-27):

    score(n)   per episode k, the chain policy's reward with n bins averaged over the
               cells of each case, then averaged over the cases with equal weight
               (penalty, duration and each location site count once, however many
               cells they have). Every run uses the same episode seeds, so episode k is
               comparable across bin counts.
    best       n* = the bin count with the highest mean score.
    chosen     the smallest n whose mean score is within one standard error of n*'s,
               where the standard error is that of the episode-paired difference
               score(n) - score(n*).

No threshold arm is needed: within a cell the threshold benchmark is the same for every
bin count, so ranking bin counts by chain reward ranks them by margin over the threshold.

Outputs (default <results>/_analysis/):
    bin_selection.json   chosen count, rule, per-count scores
    bin_selection.md     tables per case and overall

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/select_wind_bins.py --results C:/Users/bepstein8/sim_results/wind_bins_select \\
        --manifest ../configs/wind_bins_select/wind_bins_study_manifest.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from Scripts import compare_chain_sweep as ccs  # noqa: E402

def episode_rewards(manifest, results):
    """{case: {n_bins: (rewards, failures)}}, each an (episodes,) array averaged over the
    case's cells."""
    runs = [r for r in manifest["runs"] if r["n_bins"] is not None]
    cells = [{"config_basename": r["config_basename"], "sweep": r["sweep"], "arm": r["arm"],
              "world": "hist", "pair_key": r["sweep"]} for r in runs]
    df = ccs.load_summaries({"cells": cells}, results)
    n_of = {r["arm"]: r["n_bins"] for r in runs}
    out, n_cells = {}, {}
    for (case, arm), grp in df.groupby(["sweep", "arm"]):
        rew, fail = [], []
        for _, row in grp.iterrows():
            eps = ccs.read_episode_scalars(row["run_dir"], row["group"]).sort_index()
            rew.append(eps["total_reward"].to_numpy(float))
            fail.append(eps["failure"].to_numpy(float))
        lengths = {len(r) for r in rew}
        if len(lengths) != 1:
            sys.exit(f"[error] {case}/{arm}: cells have different episode counts {lengths}")
        out.setdefault(case, {})[n_of[arm]] = (np.mean(rew, axis=0), np.mean(fail, axis=0))
        n_cells.setdefault(case, set()).add(len(grp))
    for case, d in out.items():
        if len(n_cells[case]) != 1:
            sys.exit(f"[error] case {case}: bin counts have different cell counts "
                     f"{n_cells[case]} (a run is incomplete)")
    return out, {c: n.pop() for c, n in n_cells.items()}


def select(per_case):
    counts = sorted(set.intersection(*[set(d) for d in per_case.values()]))
    missing = {c: sorted(set().union(*[set(d) for d in per_case.values()]) - set(d))
               for c, d in per_case.items()}
    if any(missing.values()):
        print(f"[warn] bin counts missing in some cases (excluded): {missing}")
    score = {n: np.mean([per_case[c][n][0] for c in per_case], axis=0) for n in counts}
    best = max(counts, key=lambda n: score[n].mean())
    rows = []
    for n in counts:
        d = score[n] - score[best]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if n != best else 0.0
        rows.append({"n_bins": n, "score": float(score[n].mean()),
                     "diff_vs_best": float(d.mean()), "se_diff": se,
                     "within_1se": bool(d.mean() >= -se)})
    chosen = min(r["n_bins"] for r in rows if r["within_1se"])
    return chosen, best, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None, help="Output dir (default <results>/_analysis).")
    args = ap.parse_args()

    manifest = ccs.load_manifest(args.manifest)
    per_case, n_cells = episode_rewards(manifest, args.results)
    chosen, best, rows = select(per_case)

    out_dir = args.out or os.path.join(args.results, "_analysis")
    os.makedirs(out_dir, exist_ok=True)
    ho = manifest["holdout"]
    result = {
        "chosen_n_bins": chosen, "best_n_bins": best,
        "rule": "smallest bin count whose equal-case-weighted mean chain reward is within one "
                "episode-paired standard error of the best count's",
        "holdout": ho, "cases": {c: {"cells": n_cells[c]} for c in per_case},
        "scores": rows,
        "per_case": {c: {str(n): {"reward": float(v[0].mean()), "failure_pct": 100 * float(v[1].mean())}
                         for n, v in sorted(d.items())} for c, d in per_case.items()},
    }
    with open(os.path.join(out_dir, "bin_selection.json"), "w") as f:
        json.dump(result, f, indent=2)

    counts = [r["n_bins"] for r in rows]
    lines = ["# Wind-bin count selected on the fit years", "",
             f"Inner split: chains fitted on {ho['fit_years'][0]}-{ho['fit_years'][1]}, "
             f"episodes on {ho['eval_years'][0]}-{ho['eval_years'][1]}; the 2003-2022 "
             "evaluation years are not used.", "",
             f"Rule (fixed in advance): {result['rule']}.", "",
             f"**Chosen: {chosen} bins** (best: {best}).", "",
             "| bins | score | minus best | SE | within 1 SE |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['n_bins']} | {r['score']:.3f} | {r['diff_vs_best']:+.3f} | "
                     f"{r['se_diff']:.3f} | {'yes' if r['within_1se'] else 'no'} |")
    lines += ["", "Mean chain reward (failure %) per case, averaged over its cells:", "",
              "| case | cells | " + " | ".join(f"{n} bins" for n in counts) + " |",
              "|---|---|" + "---|" * len(counts)]
    for c, d in per_case.items():
        lines.append(f"| {c} | {n_cells[c]} | " + " | ".join(
            f"{d[n][0].mean():.2f} ({100 * d[n][1].mean():.1f}%)" for n in counts) + " |")
    with open(os.path.join(out_dir, "bin_selection.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
