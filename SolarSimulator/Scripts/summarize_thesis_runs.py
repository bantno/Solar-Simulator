#!/usr/bin/env python3
"""One report over thesis runs 1-5, with the analysis plan's corrections applied.

Inputs are the analysis outputs of each run (run_thesis_runs.py --steps analyse, and
select_wind_bins.py for run 1). Primary comparison, fixed in advance: the chain policy at
the run-1 bin count against the fit-years-tuned threshold, every cell of run 2, with a
Holm correction across all cells (family-wise error 5%). p-values are two-sided normal
approximations from the paired bootstrap interval (SE = CI width / 3.92).

Output: <out>/thesis_summary.md and thesis_holm_cells.csv

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/summarize_thesis_runs.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = "C:/Users/bepstein8/sim_results"
SWEEP_ORDER = ["capacity", "duration", "penalty", "location", "startdate"]


def holm(p, alpha=0.05):
    """Holm step-down: boolean rejections and adjusted p-values, in input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj <= alpha, adj


def pvalue(d, lo, hi):
    se = (hi - lo) / (2 * 1.959964)
    return 2 * norm.sf(abs(d) / se) if se > 0 else 0.0


def verdict(d, lo, hi):
    return "win" if lo > 0 else "loss" if hi < 0 else "tie"


def fmt(d, lo, hi):
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def counts(df, col):
    v = [verdict(r[col], r[f"{col}_lo"], r[f"{col}_hi"]) for _, r in df.iterrows()]
    return f"{v.count('win')} / {v.count('tie')} / {v.count('loss')}"


def headline(lines, path):
    c = pd.read_csv(path)
    c = c[c["world"] == "hist"].copy()
    col = "d_total_reward_chain_vs_thresh"
    c["p"] = [pvalue(r[col], r[f"{col}_lo"], r[f"{col}_hi"]) for _, r in c.iterrows()]
    c["holm_reject"], c["p_holm"] = holm(c["p"])
    c["holm_verdict"] = np.where(~c["holm_reject"], "not significant",
                                 np.where(c[col] > 0, "win", "loss"))
    lines += ["## Run 2: headline sweep", "",
              f"{len(c)} cells. Wins / ties / losses by 95% CI; Holm column: cells "
              "significant after Holm correction across all cells (wins / losses).", "",
              "| sweep | cells | chain − thresh | range | Holm wins / losses | chain − iid | "
              "iid − thresh | threshold picked on |", "|---|---|---|---|---|---|---|---|"]
    for sweep in [s for s in SWEEP_ORDER if s in set(c["sweep"])]:
        s = c[c["sweep"] == sweep]
        on = ", ".join(sorted(set(s["thresh_selected_on"])))
        lines.append(
            f"| {sweep} | {len(s)} | {counts(s, col)} | {s[col].min():+.2f} to {s[col].max():+.2f} | "
            f"{(s['holm_verdict'] == 'win').sum()} / {(s['holm_verdict'] == 'loss').sum()} | "
            f"{counts(s, 'd_total_reward')} | {counts(s, 'd_total_reward_iid_vs_thresh')} | {on} |")
    lines += ["", f"All cells: chain − thresh {counts(c, col)}; Holm-significant wins "
              f"{(c['holm_verdict'] == 'win').sum()}, losses {(c['holm_verdict'] == 'loss').sum()}.",
              ""]
    weak = c[c["holm_verdict"] != "win"].sort_values(col)
    if len(weak):
        lines += ["Cells that are not Holm-significant wins:", "",
                  "| sweep | location | cap | horizon | penalty | start | chain − thresh | Holm p |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in weak.iterrows():
            lines.append(f"| {r['sweep']} | {r['location_id']} | {r['battery_capacity']:g} | "
                         f"{r['horizon']} | {r['failure_penalty']:g} | {str(r['start_time'])[:10]} | "
                         f"{fmt(r[col], r[col + '_lo'], r[col + '_hi'])} | {r['p_holm']:.3g} |")
        lines.append("")
    return c


def worlds(lines, path):
    w = pd.read_csv(path)
    lines += ["## Run 3: each model's own weather (no modelling error)", "",
              "Wins / ties / losses by 95% CI, threshold picked on that world's episodes.", "",
              "| sweep | world | iid − thresh | chain − thresh | chain − iid |", "|---|---|---|---|---|"]
    for (sweep, world), g in w.groupby(["sweep", "world"], sort=False):
        lines.append(f"| {sweep} | {world} | {counts(g, 'd_reward_iid_vs_thresh')} | "
                     f"{counts(g, 'd_reward_chain_vs_thresh')} | {counts(g, 'd_reward_chain_vs_iid')} |")
    lines.append("")


def sites(lines, path):
    s = pd.read_csv(path)
    bins = sorted(s["n_bins"].unique())
    lines += ["## Run 4: wind-bin count at more sites", "",
              "Mean over each case's cells of chain − fit-years-tuned threshold; wins / ties / "
              "losses by 95% CI in brackets.", "",
              "| case | cells | " + " | ".join(f"{n} bins" for n in bins) + " |",
              "|---|---|" + "---|" * len(bins)]
    for case, g in s.groupby("sweep", sort=False):
        cols = []
        for n in bins:
            h = g[g["n_bins"] == n]
            v = [verdict(r["d_reward_vs_thresh"], r["d_reward_vs_thresh_lo"],
                         r["d_reward_vs_thresh_hi"]) for _, r in h.iterrows()]
            cols.append(f"{h['d_reward_vs_thresh'].mean():+.2f} ({v.count('win')}/"
                        f"{v.count('tie')}/{v.count('loss')})" if len(h) else "-")
        lines.append(f"| {case} ({g['location_id'].iloc[0]}) | {g['battery_capacity'].count() // len(bins)} | "
                     + " | ".join(cols) + " |")
    lines.append("")


def block30(lines, path, head):
    b = pd.read_csv(path)
    b = b[b["world"] == "hist"]
    keys = ["sweep", "location_id", "battery_capacity", "start_time", "horizon", "failure_penalty"]
    for df in (b, head):
        df["start_time"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    j = b.merge(head, on=keys, suffixes=("_30", "_7"))
    col = "d_total_reward_chain_vs_thresh"
    lines += ["## Run 5: 30-day bootstrap blocks", "",
              "Same cells with 7-day blocks (run 2) and 30-day blocks; threshold tuned on the "
              "fit years with the same block length.", "",
              "| sweep | cells | chain − thresh, 7-day | chain − thresh, 30-day | chain − iid, 7-day | "
              "chain − iid, 30-day | mean change in chain − thresh |", "|---|---|---|---|---|---|---|"]
    for sweep, g in j.groupby("sweep", sort=False):
        g7 = g.rename(columns=lambda x: x[:-2] if x.endswith("_7") else x)
        g30 = g.rename(columns=lambda x: x[:-3] if x.endswith("_30") else x)
        lines.append(f"| {sweep} | {len(g)} | {counts(g7, col)} | {counts(g30, col)} | "
                     f"{counts(g7, 'd_total_reward')} | {counts(g30, 'd_total_reward')} | "
                     f"{(g[col + '_30'] - g[col + '_7']).mean():+.2f} |")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    sys.stdout.reconfigure(encoding="utf-8")
    ap.add_argument("--out", default=None, help="Output dir (default <base>/thesis_summary).")
    args = ap.parse_args()
    out = args.out or f"{args.base}/thesis_summary"
    os.makedirs(out, exist_ok=True)

    sel = json.load(open(f"{args.base}/wind_bins_select/_analysis/bin_selection.json"))
    lines = ["# Thesis runs 1-5: summary", "",
             "## Run 1: bin count chosen on the fit years", "",
             f"Inner split {sel['holdout']['fit_years']} / {sel['holdout']['eval_years']}. "
             f"Rule: {sel['rule']}. **Chosen {sel['chosen_n_bins']} bins** "
             f"(best {sel['best_n_bins']}).", "",
             "| bins | score | minus best | SE |", "|---|---|---|---|"]
    for r in sel["scores"]:
        lines.append(f"| {r['n_bins']} | {r['score']:.3f} | {r['diff_vs_best']:+.3f} | {r['se_diff']:.3f} |")
    lines.append("")
    head = headline(lines, f"{args.base}/thesis_headline/_analysis/comparison_cells.csv")
    head.to_csv(f"{out}/thesis_holm_cells.csv", index=False)
    worlds(lines, f"{args.base}/thesis_worlds/_analysis/model_worlds_cells.csv")
    sites(lines, f"{args.base}/thesis_sites/_analysis/wind_bins_cells.csv")
    block30(lines, f"{args.base}/thesis_block30/_analysis/comparison_cells.csv", head.copy())
    with open(f"{out}/thesis_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
