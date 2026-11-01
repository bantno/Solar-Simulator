#!/usr/bin/env python3
"""Pack the wind-chain thesis experiment into one zip for writing up.

Always included (a few tens of MB):
    sim_results/thesis_summary/**          summary tables, Holm table, figures 1-5, dashboard
                                           CSVs, five-site comparison
    sim_results/<run>/_analysis*/**        every run's analysis (comparison cells, reports)
    sim_results/<run>/<config>/<ts>/       config.yaml, run_metadata.json, summary.csv of the
                                           run dir the analyses use (newest with a summary.csv)
    sim_results/<run>/*.json, *.log        sweep run logs, driver/shard/analyse logs
    repo/configs/<run>/                    the configs and manifests of every packed run
    repo/SolarSimulator/**.py, docs/**     the code and docs as they stand, plus git HEAD,
                                           status and a diff of uncommitted tracked changes
    README.md, MANIFEST.csv                layout, what each run is, how to rerun the analyses
--weather adds every site's inputs (about 0.5 GB zipped):
    repo/Data/HISTORICAL_DATA/data_<lat>_<lon>.pkl      hourly record 1950-2022
    repo/Data/EXPECTED_DATA/data_expected_<site>_*      climatology, fitted wind chains,
                                                        hold-out weather cubes
--episodes adds the per-episode results (about 3-4 GB zipped):
    _episode_scalars.csv and the *.h5 sample histories of each packed run dir
Never included: solved value tables (solver_tables/, ~50 GB; rebuilt from the configs).

Runs packed: thesis_* (runs 2-5 and the site repeats), wind_bins_select (run 1), and the
journal_chain_sweep and wind_bins_study runs they reuse or cite.

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/pack_thesis_results.py --dry-run --weather --episodes
    python Scripts/pack_thesis_results.py --weather --episodes
"""
import argparse
import datetime
import glob
import os
import subprocess
import sys
import time
import zipfile

import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from Scripts.compare_chain_sweep import latest_run_dir  # noqa: E402

BASE = "C:/Users/bepstein8/sim_results"
EXTRA_RUNS = ["journal_chain_sweep", "wind_bins_select", "wind_bins_study"]
RUN_FILES = ("config.yaml", "run_metadata.json", "summary.csv")
EPISODE_PATTERNS = ("_episode_scalars.csv", "*.h5")
DOC_LINKS = {
    "Wind-chain thesis findings: writing brief": "https://claude.ai/artifact/NNMZ5kGDqp4f2tZ6puf5RA",
    "Wind-chain policies on held-out weather: results and thesis run plan":
        "https://claude.ai/artifact/9ETbT27k4Nsd8ngDSXpdLP",
    "Results dashboard": "https://claude.ai/artifact/X9c9L1jMSuutohBmWovq3X",
}


def run_names(base):
    names = sorted(d for d in os.listdir(base) if d.startswith("thesis_") and d != "thesis_summary"
                   and os.path.isdir(os.path.join(base, d)))
    return names + [d for d in EXTRA_RUNS if os.path.isdir(os.path.join(base, d))]


def walk(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            yield os.path.join(dirpath, f)


def collect(base, weather, episodes):
    """[(source path, archive path)] for everything to pack."""
    items = []
    add = lambda src, arc: items.append((src, arc.replace("\\", "/")))  # noqa: E731
    for p in walk(os.path.join(base, "thesis_summary")):
        add(p, "sim_results/" + os.path.relpath(p, base))
    for p in glob.glob(os.path.join(base, "thesis_*.log")) + glob.glob(os.path.join(base, "*_driver.log")):
        add(p, "sim_results/logs/" + os.path.basename(p))

    runs = run_names(base)
    for run in runs:
        rdir = os.path.join(base, run)
        for entry in os.listdir(rdir):
            path = os.path.join(rdir, entry)
            if entry.startswith("_analysis") and os.path.isdir(path):
                for p in walk(path):
                    add(p, f"sim_results/{run}/" + os.path.relpath(p, rdir))
            elif os.path.isfile(path) and entry.endswith((".json", ".md", ".log")):
                add(path, f"sim_results/{run}/{entry}")
            elif os.path.isdir(path):
                ts = latest_run_dir(rdir, entry)
                if ts is None:
                    continue
                names = list(RUN_FILES)
                if episodes:
                    names += [os.path.basename(p) for pat in EPISODE_PATTERNS
                              for p in glob.glob(os.path.join(ts, pat))]
                for n in names:
                    p = os.path.join(ts, n)
                    if os.path.isfile(p):
                        add(p, f"sim_results/{run}/" + os.path.relpath(p, rdir))

    sites = set()
    for run in runs:
        cdir = os.path.join(REPO_ROOT, "configs", run)
        if not os.path.isdir(cdir):
            continue
        for p in walk(cdir):
            add(p, "repo/" + os.path.relpath(p, REPO_ROOT))
            if p.endswith(".yaml"):
                with open(p) as f:
                    cfg = yaml.safe_load(f) or {}
                for loc in cfg.get("locations") or []:
                    sites.add((float(loc["latitude"]), float(loc["longitude"])))
    for p in walk(PKG_DIR):
        rel = os.path.relpath(p, REPO_ROOT)
        if p.endswith((".py", ".md")) or (p.endswith(".yaml") and "examples" in rel):
            add(p, "repo/" + rel)
    for p in walk(os.path.join(REPO_ROOT, "docs")):
        add(p, "repo/" + os.path.relpath(p, REPO_ROOT))

    if weather:
        data = os.path.join(REPO_ROOT, "Data")
        for lat, lon in sorted(sites):
            hist = os.path.join(data, "HISTORICAL_DATA", f"data_{lat}_{lon}.pkl")
            if os.path.isfile(hist):
                add(hist, "repo/" + os.path.relpath(hist, REPO_ROOT))
            for p in glob.glob(os.path.join(data, "EXPECTED_DATA", f"data_expected_lat{lat}_lon{lon}_*")):
                add(p, "repo/" + os.path.relpath(p, REPO_ROOT))
    return items, runs, sorted(sites)


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=True).stdout
    except Exception as e:  # noqa: BLE001
        return f"(git {' '.join(args)} failed: {e})\n"


def readme(runs, sites, weather, episodes, head):
    site_list = ", ".join(f"lat {a:g} lon {b:g}" for a, b in sites)
    docs = "\n".join(f"- {k}: {v}" for k, v in DOC_LINKS.items())
    return f"""# Wind-chain thesis experiment pack

Built {datetime.date.today()} from branch wind-persistence, HEAD {head.strip()}, with uncommitted
changes (see repo/GIT_STATUS.txt and repo/GIT_DIFF.patch; new scripts are included as files).
Weather inputs: {'included' if weather else 'not included'}. Per-episode results:
{'included' if episodes else 'not included'}. Solved value tables are never included (about 50 GB);
any one can be rebuilt by rerunning its config.

## Where to start

- sim_results/thesis_summary/thesis_summary.md: run-by-run summary tables (runs 1-5, default site).
- sim_results/thesis_summary/thesis_holm_cells.csv: run 2, all 111 cells, with Holm p-values.
- sim_results/thesis_summary/sites/: five-site comparison (sites_summary.md, sites_cells.csv,
  wind_persistence_by_site.csv).
- sim_results/thesis_summary/figures/: figures 1-5, PNG and PDF.
- Each run's _analysis*/ folder: per-cell results with 95% bootstrap CIs and a report.

The two write-ups are Claude docs and are not in this zip. Export them from each doc's menu
(Markdown, PDF or Word) if you want copies here:
{docs}

## Runs

| Folder (sim_results/) | Run |
|---|---|
| wind_bins_select | Run 1: bin-count selection on the fit years (16 bins chosen) |
| thesis_headline | Run 2: headline sweep, 111 cells, lat 30 lon -75 plus location sweep at lon -25 |
| thesis_worlds | Run 3: each policy in the i.i.d. and chain models' weather |
| thesis_sites | Run 4: bin counts 4-16 at more sites |
| thesis_block30 | Run 5: 30-day bootstrap blocks |
| thesis_headline_<site>, thesis_worlds_<site> | Runs 2 and 3 repeated at another site |
| journal_chain_sweep | Journal sweep (4-bin chain); its i.i.d. and threshold runs are reused by run 2 |
| wind_bins_study | Earlier bin study and the solve-time benchmark (figure 4c) |

Sites in the packed configs: {site_list}.
Fit years 1950-2002, evaluation years 2003-2022, 10,000 episodes per policy and cell, paired
across policies (common random numbers).

## Layout

- sim_results/<run>/<config>/<timestamp>/: config.yaml, run_metadata.json, summary.csv
  (and _episode_scalars.csv, *.h5 with per-episode results) of the run each analysis used.
- sim_results/logs/: driver, shard, provision and analyse logs.
- repo/: configs, SolarSimulator code (*.py), docs, and (with weather) Data/ inputs, at their
  repository paths.

## Rerunning an analysis from this pack

Needs the per-episode results. From repo/SolarSimulator/ in the pvlib environment, pointing at the
unpacked sim_results folder, e.g.:

    python Scripts/compare_chain_sweep.py --results <pack>/sim_results/thesis_headline_latm30_lonm25 --manifest ../configs/thesis_headline_latm30_lonm25/chain_vs_iid_sweep_manifest.json
    python Scripts/compare_thesis_sites.py --base <pack>/sim_results --tags latm20_lonm75 latm30_lonm75 lat0_lonm25 latm30_lonm25

Run 2 at lat 30 reads its reused i.i.d. and threshold runs from the results_base its manifest
names (C:/Users/bepstein8/sim_results/journal_chain_sweep); edit that path if unpacked elsewhere.
The solver-predicted V0 columns need the value tables and come out empty without them.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=None, help="Default <base>/thesis_pack_<date>.zip.")
    ap.add_argument("--weather", action="store_true", help="Add every site's weather inputs.")
    ap.add_argument("--episodes", action="store_true", help="Add per-episode results.")
    ap.add_argument("--dry-run", action="store_true", help="List sizes only.")
    args = ap.parse_args()

    items, runs, sites = collect(args.base, args.weather, args.episodes)
    sizes = [os.path.getsize(s) for s, _ in items]
    total = sum(sizes)
    by_top = {}
    for (_, arc), n in zip(items, sizes):
        key = "episodes" if arc.endswith(EPISODE_PATTERNS[0]) or arc.endswith(".h5") else \
              "weather" if "/Data/" in arc else arc.split("/")[0]
        by_top[key] = by_top.get(key, 0) + n
    print(f"[pack] {len(items)} files, {total / 1e9:.2f} GB raw, {len(runs)} runs, {len(sites)} sites")
    for k, v in sorted(by_top.items(), key=lambda kv: -kv[1]):
        print(f"    {v / 1e9:8.3f} GB  {k}")
    if args.dry_run:
        return

    out = args.out or f"{args.base}/thesis_pack_{datetime.date.today():%Y%m%d}.zip"
    head = git("rev-parse", "HEAD")
    t0, done, next_report = time.time(), 0, 2e9
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.writestr("thesis_pack/README.md", readme(runs, sites, args.weather, args.episodes, head))
        z.writestr("thesis_pack/repo/GIT_STATUS.txt", head + git("status", "--short"))
        z.writestr("thesis_pack/repo/GIT_DIFF.patch", git("diff"))
        z.writestr("thesis_pack/MANIFEST.csv", "path,bytes\n" + "".join(
            f"{arc},{n}\n" for (_, arc), n in zip(items, sizes)))
        for (src, arc), n in zip(items, sizes):
            z.write(src, "thesis_pack/" + arc)
            done += n
            if done >= next_report:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[pack] {done / 1e9:.1f} / {total / 1e9:.1f} GB raw, "
                      f"{(total - done) / rate / 60:.0f} min left", flush=True)
                next_report += 2e9
    print(f"[pack] wrote {out}: {os.path.getsize(out) / 1e9:.2f} GB in "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
