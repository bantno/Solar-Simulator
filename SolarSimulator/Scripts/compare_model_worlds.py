#!/usr/bin/env python3
"""Compare policies in each model's own synthetic weather and on historical weather.

The no-modelling-error condition of the Journal of Aircraft paper, re-run with the
thesis settings (thesis run 3). Reads a sweep generated with
generate_journal_chain_sweep_configs.py --model-worlds: the i.i.d.-solved policy, the
chain-solved policy and the threshold grid, each rolled out in the i.i.d. model's
synthetic weather (world `iidworld`) and in the wind chain's (world `chainworld`).
Within a world every policy sees identical weather, so deltas are episode-paired.

Per cell and world:
    reward and failure rate of iid, chain and the best threshold combo (picked on that
    world's episodes, the paper's practice; the split-half check bounds the pick's luck),
    paired deltas iid - thresh, chain - thresh, chain - iid with 95% bootstrap CIs.
With --hist-cells (compare_chain_sweep.py's comparison_cells.csv for the same cells on
held-out historical weather) the report sets each policy's own-world reward beside its
historical reward: the gap is what modelling error costs that policy.

Outputs (default <results>/_analysis/):
    model_worlds_cells.csv   one row per (cell, world)
    model_worlds_report.md

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/compare_model_worlds.py --results C:/Users/bepstein8/sim_results/thesis_model_worlds \\
        --manifest ../configs/thesis_model_worlds/chain_vs_iid_sweep_manifest.json \\
        --hist-cells C:/Users/bepstein8/sim_results/thesis_headline/_analysis/comparison_cells.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from Scripts import compare_chain_sweep as ccs  # noqa: E402

CELL = ["sweep", "location_id", "battery_capacity", "start_time", "horizon", "failure_penalty"]
WORLDS = ("iidworld", "chainworld")
DELTAS = (("iid", "thresh"), ("chain", "thresh"), ("chain", "iid"))


def build(df, n_boot, rng):
    rows = []
    for key, grp in df.groupby(CELL + ["world"]):
        cell = dict(zip(CELL + ["world"], key))
        arms = {r["arm"]: r for _, r in grp[grp["arm"] != "thresh"].iterrows()}
        thresh_rows = grp[grp["arm"] == "thresh"]
        if set(arms) != {"iid", "chain"} or not len(thresh_rows):
            print(f"[warn] incomplete cell {cell}: arms {sorted(arms)}, "
                  f"{len(thresh_rows)} threshold combos")
            continue
        best_row, best_eps, extras = ccs.best_threshold_for_cell(thresh_rows)
        eps = {a: ccs.read_episode_scalars(r["run_dir"], r["group"]) for a, r in arms.items()}
        eps["thresh"] = best_eps
        cell.update({"thresh_obs": extras["thresh_best_obs"],
                     "thresh_wind": extras["thresh_best_wind"],
                     "avg_reward_thresh_split_half": extras["avg_reward_thresh_holdout"]})
        for a, row in list(arms.items()) + [("thresh", best_row)]:
            cell[f"avg_reward_{a}"] = float(row["average_reward"])
            cell[f"failure_pct_{a}"] = 100 * float(row["failure_percentage"])
        for a, b in DELTAS:
            d = ccs.paired_delta(eps[b], eps[a], n_boot, rng)
            for k in ("", "_lo", "_hi"):
                cell[f"d_reward_{a}_vs_{b}{k}"] = d[f"d_total_reward{k}"]
        rows.append(cell)
    return pd.DataFrame(rows)


def _ci(r, base):
    v, lo, hi = r[base], r[f"{base}_lo"], r[f"{base}_hi"]
    star = "" if lo <= 0 <= hi else " *"
    return f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def _cell_label(r):
    return {"capacity": f"{r['battery_capacity']:g} Wh", "penalty": f"penalty {r['failure_penalty']:g}",
            "duration": f"{r['horizon']} stages", "startdate": str(r["start_time"])[:10]
            }.get(r["sweep"], f"{r['location_id']} {r['battery_capacity']:g} Wh")


def write_report(cells, hist, path, manifest):
    lines = ["# Policies in their own model's weather and on historical weather", "",
             f"Chain: {manifest['wind_bins']['n_bins']}-bin quantile wind chain. Synthetic "
             "worlds are the solver models fitted on "
             f"{manifest['holdout']['fit_years'][0]}-{manifest['holdout']['fit_years'][1]}; "
             f"{manifest['episodes']:,} episodes per policy and cell. Within a world every "
             "policy sees identical weather; deltas are episode-paired, mean with 95% "
             "bootstrap CI, * where the CI excludes 0. The threshold combo is the best on "
             "that world's episodes (the paper's practice).", ""]
    names = {"iidworld": "i.i.d. model's weather", "chainworld": "chain model's weather"}
    for sweep, s in cells.groupby("sweep", sort=False):
        for world in WORLDS:
            w = s[s["world"] == world].sort_values(CELL)
            if not len(w):
                continue
            lines += [f"## {sweep}: {names[world]}", "",
                      "| cell | reward iid / chain / thresh | fail % iid / chain / thresh | "
                      "iid − thresh | chain − thresh | chain − iid |",
                      "|---|---|---|---|---|---|"]
            for _, r in w.iterrows():
                lines.append(
                    f"| {_cell_label(r)} | {r['avg_reward_iid']:.2f} / {r['avg_reward_chain']:.2f}"
                    f" / {r['avg_reward_thresh']:.2f} | {r['failure_pct_iid']:.1f} / "
                    f"{r['failure_pct_chain']:.1f} / {r['failure_pct_thresh']:.1f} | "
                    f"{_ci(r, 'd_reward_iid_vs_thresh')} | {_ci(r, 'd_reward_chain_vs_thresh')} | "
                    f"{_ci(r, 'd_reward_chain_vs_iid')} |")
            lines.append("")
        if hist is not None:
            h = hist[(hist["sweep"] == sweep) & (hist["world"] == "hist")]
            own = {"iid": s[s["world"] == "iidworld"], "chain": s[s["world"] == "chainworld"]}
            keys = [k for k in CELL if k != "sweep"]
            j = h.set_index(keys)
            lines += [f"## {sweep}: own-world reward vs historical reward", "",
                      "Each optimal policy's reward in its own model's weather (what its "
                      "solver expects) against its reward on held-out 2003-2022 weather.", "",
                      "| cell | iid own world | iid historical | chain own world | "
                      "chain historical | threshold historical |", "|---|---|---|---|---|---|"]
            for _, r in own["iid"].sort_values(CELL).iterrows():
                k = tuple(r[c] for c in keys)
                c = own["chain"]
                c = c[np.logical_and.reduce([c[x] == r[x] for x in keys])]
                if k not in j.index or not len(c):
                    continue
                hr = j.loc[k]
                hr = hr.iloc[0] if isinstance(hr, pd.DataFrame) else hr
                lines.append(f"| {_cell_label(r)} | {r['avg_reward_iid']:.2f} | "
                             f"{hr['avg_reward_iid']:.2f} | {c.iloc[0]['avg_reward_chain']:.2f} | "
                             f"{hr['avg_reward_chain']:.2f} | {hr['avg_reward_thresh']:.2f} |")
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hist-cells", default=None,
                    help="comparison_cells.csv of the matching historical-weather sweep.")
    ap.add_argument("--out", default=None, help="Output dir (default <results>/_analysis).")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    manifest = ccs.load_manifest(args.manifest)
    df = ccs.load_summaries(manifest, args.results)
    cells = build(df, args.n_boot, np.random.default_rng(12345))
    hist = None
    if args.hist_cells:
        hist = pd.read_csv(args.hist_cells)
        hist["start_time"] = pd.to_datetime(hist["start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = args.out or os.path.join(args.results, "_analysis")
    os.makedirs(out_dir, exist_ok=True)
    cells.to_csv(os.path.join(out_dir, "model_worlds_cells.csv"), index=False)
    write_report(cells, hist, os.path.join(out_dir, "model_worlds_report.md"), manifest)
    print(f"Wrote {len(cells)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
