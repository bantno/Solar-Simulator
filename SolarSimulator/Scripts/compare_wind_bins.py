#!/usr/bin/env python3
"""Compare chain policies with different wind-bin counts against the threshold benchmark.

Reads a wind-bin study (Scripts/generate_wind_bins_study_configs.py) and the journal
chain sweep it extends. For every cell of every case (a penalty value, horizon, start
date or battery capacity at one site) it scores each chain bin count against the cell's
threshold combo, episode-paired with bootstrap CIs, using compare_chain_sweep.py's
selection and statistics unchanged:
    threshold pick   best combo on the fit-years tuning run (the study's thresh_tune run,
                     or the journal sweep's for penalty), scored on the evaluation
                     episodes; falls back to the evaluation-episode best when no tuning
                     run is found.
    chain{n} - thresh, chain{n} - chain (journal 4-bin)   paired reward deltas.
Before any comparison it requires the study's 4-bin arm to reproduce the journal sweep's
chain arm episode for episode: same chain fit, same files, same random numbers.

Outputs (default <results>/_analysis/):
    wind_bins_cells.csv    one row per (cell, bin count)
    wind_bins_report.md    reward tables

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/compare_wind_bins.py --results C:/Users/bepstein8/sim_results/wind_bins_study \
        --journal-results C:/Users/bepstein8/sim_results/journal_chain_sweep
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from Scripts import compare_chain_sweep as ccs  # noqa: E402

REPO_ROOT = os.path.dirname(PKG_DIR)
CELL = ["location_id", "battery_capacity", "start_time", "horizon", "failure_penalty"]
# Manifests from before the multi-case generator (2026-10-25) have no per-case field.
LEGACY_SWEPT = {"penalty": "failure_penalty", "duration": "horizon"}


def swept_fields(manifest):
    """{case: swept summary column}."""
    cases = manifest.get("cases") or {}
    if all(isinstance(v, dict) and "swept_field" in v for v in cases.values()) and cases:
        return {c: v["swept_field"] for c, v in cases.items()}
    return dict(LEGACY_SWEPT)


def load_runs(manifest, results, journal_results):
    """Summary rows of every study run and the journal runs it reuses, tagged by arm."""
    study = [{"config_basename": r["config_basename"], "sweep": r["sweep"], "arm": r["arm"],
              "world": "hist", "pair_key": r["sweep"]} for r in manifest["runs"]]
    journal = [{"config_basename": name, "sweep": sweep,
                "arm": arm if arm.startswith("thresh") else f"journal_{arm}",
                "world": "hist", "pair_key": sweep}
               for sweep, arms in manifest["journal_runs"].items() for arm, name in arms.items()]
    frames = [ccs.load_summaries({"cells": study}, results),
              ccs.load_summaries({"cells": journal}, journal_results)]
    return pd.concat(frames, ignore_index=True)


def check_reproduction(df, sweep, field):
    """The study's chain4 must reproduce the journal chain arm episode for episode."""
    rows = df[(df["sweep"] == sweep) & df["arm"].isin(["chain4", "journal_chain"])]
    msgs, ok, n = [], True, 0
    for key, grp in rows.groupby(CELL):
        arms = {r["arm"]: r for _, r in grp.iterrows()}
        if len(arms) < 2:
            continue
        a = ccs.read_episode_scalars(arms["chain4"]["run_dir"], arms["chain4"]["group"])
        b = ccs.read_episode_scalars(arms["journal_chain"]["run_dir"],
                                     arms["journal_chain"]["group"])
        same = (a.index.equals(b.index)
                and np.array_equal(a["total_reward"].to_numpy(), b["total_reward"].to_numpy())
                and np.array_equal(a["failure_step"].to_numpy(), b["failure_step"].to_numpy()))
        ok &= same
        n += 1
        if not same:
            msgs.append(f"{field}={key[CELL.index(field)]}: differs")
    if n == 0:
        return False, ["no cell has both a 4-bin study arm and a journal chain arm"]
    return ok, msgs


def build(df, swept, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = []
    bin_arms = sorted({a for a in df["arm"] if a.startswith("chain") and a[5:].isdigit()},
                      key=lambda a: int(a[5:]))
    for sweep in swept:
        sdf = df[df["sweep"] == sweep]
        study_cells = sdf[sdf["arm"].isin(bin_arms)][CELL].drop_duplicates()
        for _, cell in study_cells.iterrows():
            grp = sdf[np.logical_and.reduce([sdf[k] == cell[k] for k in CELL])]
            thresh_rows = grp[grp["arm"] == "thresh"]
            tune_rows = grp[grp["arm"] == "thresh_tune"]
            best_row, best_eps, extras = ccs.best_threshold_for_cell(thresh_rows, tune_rows)
            j4 = grp[grp["arm"] == "journal_chain"]
            j4_eps = (ccs.read_episode_scalars(j4.iloc[0]["run_dir"], j4.iloc[0]["group"])
                      if len(j4) else None)
            base = {"sweep": sweep, **{k: cell[k] for k in CELL},
                    "thresh_selected_on": extras["thresh_selected_on"],
                    "thresh_obs": extras["thresh_best_obs"],
                    "thresh_wind": extras["thresh_best_wind"],
                    "avg_reward_thresh": float(best_row["average_reward"]),
                    "failure_pct_thresh": 100 * float(best_row["failure_percentage"])}
            for arm in bin_arms:
                r = grp[grp["arm"] == arm]
                if not len(r):
                    continue
                r = r.iloc[0]
                eps = ccs.read_episode_scalars(r["run_dir"], r["group"])
                row = {**base, "n_bins": int(arm[5:]),
                       "avg_reward_chain": float(r["average_reward"]),
                       "failure_pct_chain": 100 * float(r["failure_percentage"]),
                       "avg_flight_hrs_chain": float(r["average_flight_hrs"])}
                d = ccs.paired_delta(best_eps, eps, n_boot, rng)
                row.update({"d_reward_vs_thresh": d["d_total_reward"],
                            "d_reward_vs_thresh_lo": d["d_total_reward_lo"],
                            "d_reward_vs_thresh_hi": d["d_total_reward_hi"]})
                if j4_eps is not None:
                    d = ccs.paired_delta(j4_eps, eps, n_boot, rng)
                    row.update({"d_reward_vs_chain4": d["d_total_reward"],
                                "d_reward_vs_chain4_lo": d["d_total_reward_lo"],
                                "d_reward_vs_chain4_hi": d["d_total_reward_hi"]})
                out.append(row)
    return pd.DataFrame(out)


def _ci(v, lo, hi):
    star = "" if lo <= 0 <= hi else " *"
    return f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def _val(v):
    return f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)[:10]


def write_report(cells, repro, path, manifest, swept):
    lines = ["# Wind-bin study: chain policy vs threshold by wind-bin count", "",
             f"Bin counts {manifest['bin_counts']}; {manifest['episodes']:,} episodes per arm "
             "on held-out historical weather (fit "
             f"{manifest['holdout']['fit_years'][0]}-{manifest['holdout']['fit_years'][1]}, "
             f"episodes {manifest['holdout']['eval_years'][0]}-"
             f"{manifest['holdout']['eval_years'][1]}), identical episodes across arms.", ""]
    for sweep, (ok, msgs) in repro.items():
        lines.append(f"- Reproduction check ({sweep}): 4-bin study arm vs journal chain arm "
                     f"episode for episode: **{'PASS' if ok else 'FAIL'}**"
                     + ("" if ok else " (" + "; ".join(msgs) + ")"))
    lines += ["", "Cells: chain reward minus threshold reward, mean with 95% paired bootstrap "
              "CI; * marks a CI that excludes 0.", ""]
    for sweep, field in swept.items():
        s = cells[cells["sweep"] == sweep] if len(cells) else cells
        if not len(s):
            continue
        bins = sorted(s["n_bins"].unique())
        loc = s["location_id"].iloc[0]
        lines += [f"## {sweep} ({loc})", "",
                  "| " + field + " | threshold pick (obs, wind) | threshold reward | "
                  + " | ".join(f"{n} bins" for n in bins) + " |",
                  "|---|---|---|" + "---|" * len(bins)]
        for val, g in s.groupby(field):
            g = g.set_index("n_bins")
            first = g.iloc[0]
            pick = (f"({first['thresh_obs']:g}, {first['thresh_wind']:g}) "
                    f"{'fit years' if first['thresh_selected_on'].startswith('fit') else 'eval years'}")
            cols = [_ci(g.loc[n, "d_reward_vs_thresh"], g.loc[n, "d_reward_vs_thresh_lo"],
                        g.loc[n, "d_reward_vs_thresh_hi"]) if n in g.index else "-"
                    for n in bins]
            lines.append(f"| {_val(val)} | {pick} | {first['avg_reward_thresh']:.2f} | "
                         + " | ".join(cols) + " |")
        lines += ["", "Chain reward and failure rate by bin count:", "",
                  "| " + field + " | " + " | ".join(f"{n} bins" for n in bins) + " |",
                  "|---|" + "---|" * len(bins)]
        for val, g in s.groupby(field):
            g = g.set_index("n_bins")
            cols = [f"{g.loc[n, 'avg_reward_chain']:.2f} ({g.loc[n, 'failure_pct_chain']:.1f}%)"
                    if n in g.index else "-" for n in bins]
            lines.append(f"| {_val(val)} | " + " | ".join(cols) + " |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="Study results dir.")
    ap.add_argument("--journal-results", required=True,
                    help="Journal chain sweep results dir (threshold and 4-bin chain runs).")
    ap.add_argument("--manifest", default=os.path.join(
        REPO_ROOT, "configs", "wind_bins_study", "wind_bins_study_manifest.json"))
    ap.add_argument("--out", default=None, help="Output dir (default <results>/_analysis).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = ccs.load_manifest(args.manifest)
    df = load_runs(manifest, args.results, args.journal_results)
    swept = swept_fields(manifest)
    repro = {s: check_reproduction(df, s, f) for s, f in swept.items()}
    for s, (ok, msgs) in repro.items():
        print(f"[verify] {s}: 4-bin study arm reproduces the journal chain arm: "
              f"{'PASS' if ok else 'FAIL ' + '; '.join(msgs)}")
    cells = build(df, swept, args.n_boot, args.seed)

    out_dir = args.out or os.path.join(args.results, "_analysis")
    os.makedirs(out_dir, exist_ok=True)
    cells.to_csv(os.path.join(out_dir, "wind_bins_cells.csv"), index=False)
    write_report(cells, repro, os.path.join(out_dir, "wind_bins_report.md"), manifest, swept)
    print(f"Wrote {len(cells)} rows -> {out_dir}")
    if not all(ok for ok, _ in repro.values()):
        sys.exit("[error] reproduction check failed; the study arms are not comparable")


if __name__ == "__main__":
    main()
