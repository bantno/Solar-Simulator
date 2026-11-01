#!/usr/bin/env python3
"""Compare thesis runs 2 and 3 at the default site with their repeat at another site.

Reads <base>/thesis_headline[_<tag>]/_analysis/comparison_cells.csv and
<base>/thesis_worlds[_<tag>]/_analysis/model_worlds_cells.csv for the default site and for
each --tags site (written by run_thesis_runs.py --site), and writes, under
<base>/thesis_summary/sites/:
    sites_summary.md      per sweep and site: wins / ties / losses by 95% CI and the range
                          of chain - threshold, chain - iid and iid - threshold; the same
                          for each model world (run 3)
    sites_cells.csv       every historical-weather cell of both sites, side by side

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/compare_thesis_sites.py --tags latm20_lonm75
"""
import argparse
import os

import pandas as pd

BASE = "C:/Users/bepstein8/sim_results"
SWEEPS = ["capacity", "duration", "penalty", "startdate"]
PAIRS = {"chain - threshold": "d_total_reward_chain_vs_thresh",
         "chain - iid": "d_total_reward",
         "iid - threshold": "d_total_reward_iid_vs_thresh"}
WORLD_PAIRS = {"iid - threshold": ("iid", "thresh"), "chain - threshold": ("chain", "thresh"),
               "chain - iid": ("chain", "iid")}


def wtl(d, lo, hi):
    return f"{(lo > 0).sum()} / {((lo <= 0) & (hi >= 0)).sum()} / {(hi < 0).sum()}"


def rng(d):
    return f"{d.min():+.2f} to {d.max():+.2f}"


def load(base, tag):
    sfx = f"_{tag}" if tag else ""
    h = pd.read_csv(f"{base}/thesis_headline{sfx}/_analysis/comparison_cells.csv")
    h = h[(h.world == "hist") & h.sweep.isin(SWEEPS)]
    if not tag:  # the default site's run also holds the location sweep's sites
        h = h[h.location_id == "lat30.0_lon-75.0"]
    w = pd.read_csv(f"{base}/thesis_worlds{sfx}/_analysis/model_worlds_cells.csv")
    return h, w


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--tags", nargs="+", required=True,
                    help="Site tags of the repeated runs (e.g. latm20_lonm75).")
    args = ap.parse_args()
    out = f"{args.base}/thesis_summary/sites"
    os.makedirs(out, exist_ok=True)

    sites = {"": load(args.base, "")}
    sites.update({t: load(args.base, t) for t in args.tags})
    names = {t: (h.location_id.iloc[0]) for t, (h, _) in sites.items()}

    lines = ["# Thesis runs 2 and 3 by site", "",
             "Wins / ties / losses by 95% CI, then the range over the sweep's cells. "
             "Threshold tuned on the fit years (historical) or picked on the world's "
             "episodes (model worlds).", "", "## Historical weather 2003-2022 (run 2)", "",
             "| sweep | site | cells | " + " | ".join(PAIRS) + " |",
             "|---|---|---|" + "---|" * len(PAIRS)]
    for s in SWEEPS:
        for t, (h, _) in sites.items():
            g = h[h.sweep == s]
            cols = [f"{wtl(g[c], g[c + '_lo'], g[c + '_hi'])} ({rng(g[c])})"
                    for c in PAIRS.values()]
            lines.append(f"| {s} | {names[t]} | {len(g)} | " + " | ".join(cols) + " |")

    lines += ["", "## Each model's own weather (run 3)", "",
              "| sweep | world | site | " + " | ".join(WORLD_PAIRS) + " |",
              "|---|---|---|" + "---|" * len(WORLD_PAIRS)]
    for s in ("capacity", "penalty"):
        for world in ("iidworld", "chainworld"):
            for t, (_, w) in sites.items():
                g = w[(w.sweep == s) & (w.world == world)]
                cols = [f"{wtl(g[c], g[c + '_lo'], g[c + '_hi'])} ({rng(g[c])})"
                        for c in (f"d_reward_{a}_vs_{b}" for a, b in WORLD_PAIRS.values())]
                lines.append(f"| {s} | {world} | {names[t]} | " + " | ".join(cols) + " |")
    with open(f"{out}/sites_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    keep = ["sweep", "location_id", "battery_capacity", "horizon", "failure_penalty",
            "start_time", "avg_reward_iid", "avg_reward_chain", "avg_reward_thresh",
            "failure_pct_iid", "failure_pct_chain", "failure_pct_thresh"]
    keep += [c + sfx for c in PAIRS.values() for sfx in ("", "_lo", "_hi")]
    pd.concat([h[keep] for h, _ in sites.values()]).round(3).to_csv(
        f"{out}/sites_cells.csv", index=False)
    print("\n".join(lines))
    print(f"[out] {out}/sites_summary.md, {out}/sites_cells.csv")


if __name__ == "__main__":
    main()
