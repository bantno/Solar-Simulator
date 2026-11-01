#!/usr/bin/env python3
"""Generate, run and analyse thesis runs 2-5 at the bin count chosen by run 1.

The plan (doc "Wind-chain policies on held-out weather: results and thesis run plan"):
    1  bin-count selection on the fit years      select_wind_bins.py (run separately first)
    2  headline     all 5 journal sweeps, chain arms at the chosen count, fit-years
                    threshold tuning in every sweep; iid and threshold runs reused from
                    the journal sweep (same episodes)
    3  worlds       capacity + penalty sweeps: iid, chain and threshold in the i.i.d.
                    and chain models' synthetic weather (no-modelling-error condition)
    4  sites        wind-bin study at lat 20 and lat -20 (lon -25) and on the start-date
                    sweep: bins 4, 8, 12, 16 (+ the chosen count), fit-years tuning
    5  block30      capacity + penalty sweeps, all arms, 30-day bootstrap blocks

Steps (--steps, default all, in order):
    generate   write each run's configs and manifest under configs/thesis_<run>/
    provision  build every artifact the configs need (run_chain_sweep.py --provision-only),
               one run at a time so no two processes build the same file
    run        launch every run's shards at once (run_chain_sweep.py --shard i/K
               --no-provision --resume), logs in <out-base>/thesis_<run>_shard<i>.log.
               --max-workers (default cores - 1) workers in all, split over the runs by
               the estimated cost of their unfinished configs; waits for free RAM first
               and logs the lowest free RAM seen
    analyse    compare_chain_sweep.py (runs 2, 5), compare_model_worlds.py (3, beside
               run 2's historical results), compare_wind_bins.py (4)

Usage (pvlib conda env, from SolarSimulator/):
    python Scripts/run_thesis_runs.py --selection C:/Users/bepstein8/sim_results/wind_bins_select/_analysis/bin_selection.json
    python Scripts/run_thesis_runs.py --bins 12 --steps analyse
    python Scripts/run_thesis_runs.py --bins 16 --site -20 -75   # runs 2-3 at another site
"""
import argparse
import json
import os
import subprocess
import sys
import time

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)

from Scripts.run_chain_sweep import _already_done, _configs_in, config_cost  # noqa: E402

JOURNAL_RESULTS = "C:/Users/bepstein8/sim_results/journal_chain_sweep"
OUT_BASE = "C:/Users/bepstein8/sim_results"
RUNS = ("headline", "worlds", "sites", "block30")
SITE_RUNS = ("headline", "worlds")  # the runs --site repeats at another site
SITE_SWEEPS = ("capacity", "duration", "penalty", "startdate")
WORKERS_PER_SHARD = 4   # a shard's pool also has to cover the cells inside one config
MIN_FREE_GB = 8.0       # wait for this much free RAM before launching the shards
TAG = ""                # "_<site>" with --site, so another site's runs never share a dir


def site_tag(lat, lon):
    return f"lat{lat:g}_lon{lon:g}".replace("-", "m")


def cfg_dir(run):
    return os.path.join(REPO_ROOT, "configs", f"thesis_{run}{TAG}")


def out_dir(run, base):
    return f"{base}/thesis_{run}{TAG}"


def py(script, *args):
    return [sys.executable, os.path.join(PKG_DIR, "Scripts", script), *map(str, args)]


def generate_cmds(n, site=None):
    gen = "generate_journal_chain_sweep_configs.py"
    if site:
        # Nothing to reuse at a new site: every arm runs, with fit-years tuning.
        where = ["--site", *site]
        return {
            "headline": py(gen, "--bins", n, "--sweeps", *SITE_SWEEPS, "--tune", *where,
                           "--prefix", f"jcsq{n}", "--out", cfg_dir("headline")),
            "worlds": py(gen, "--bins", n, "--sweeps", "capacity", "penalty",
                         "--model-worlds", *where, "--prefix", f"jcsw{n}",
                         "--out", cfg_dir("worlds")),
        }
    site_bins = sorted({4, 8, 12, 16, n})
    return {
        "headline": py(gen, "--bins", n, "--arms", "chain", "thresh_tune", "--tune",
                       "--prefix", f"jcsq{n}", "--out", cfg_dir("headline"),
                       "--reuse-results", JOURNAL_RESULTS),
        "worlds": py(gen, "--bins", n, "--sweeps", "capacity", "penalty", "--model-worlds",
                     "--prefix", f"jcsw{n}", "--out", cfg_dir("worlds")),
        "sites": py("generate_wind_bins_study_configs.py", "--cases", "lat20", "latm20",
                    "startdate", "--bins", *site_bins, "--tune", "lat20", "latm20",
                    "startdate", "--prefix", "wbs2", "--out", cfg_dir("sites")),
        "block30": py(gen, "--bins", n, "--sweeps", "capacity", "penalty", "--block-days", 30,
                      "--tune", "--prefix", f"jcsb30q{n}", "--out", cfg_dir("block30")),
    }


def analyse_cmds(base):
    ccs = "compare_chain_sweep.py"
    man = "chain_vs_iid_sweep_manifest.json"
    return {
        "headline": py(ccs, "--results", out_dir("headline", base),
                       "--manifest", os.path.join(cfg_dir("headline"), man), "--verify"),
        "worlds": py("compare_model_worlds.py", "--results", out_dir("worlds", base),
                     "--manifest", os.path.join(cfg_dir("worlds"), man), "--hist-cells",
                     f"{out_dir('headline', base)}/_analysis/comparison_cells.csv"),
        "sites": py("compare_wind_bins.py", "--results", out_dir("sites", base),
                    "--journal-results", JOURNAL_RESULTS, "--manifest",
                    os.path.join(cfg_dir("sites"), "wind_bins_study_manifest.json")),
        "block30": py(ccs, "--results", out_dir("block30", base),
                      "--manifest", os.path.join(cfg_dir("block30"), man), "--verify"),
    }


def free_ram_gb():
    """Available physical memory in GB (Windows API; sysconf elsewhere)."""
    if os.name == "nt":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "totalPF", "availPF",
                                                  "totalV", "availV", "availExtV")]
        m = MemStatus()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 1e9
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9


def plan_shards(runs, base, budget):
    """{run: [workers of each shard]}: `budget` workers split over the runs in proportion
    to the estimated cost of their unfinished configs, about WORKERS_PER_SHARD per shard.
    The total never exceeds the budget, so the machine is never oversubscribed."""
    cost = {}
    for r in runs:
        todo = [p for p in _configs_in(cfg_dir(r))
                if not _already_done(out_dir(r, base), os.path.splitext(os.path.basename(p))[0])]
        if todo:
            cost[r] = (sum(config_cost(p) for p in todo), len(todo))
    if not cost:
        return {}
    if budget < len(cost):
        sys.exit(f"[error] {budget} workers cannot cover {len(cost)} runs; use fewer --runs")
    total = sum(c for c, _ in cost.values())
    workers = {r: max(1, int(budget * c / total)) for r, (c, _) in cost.items()}
    while sum(workers.values()) > budget:  # the max(1, ...) floors can overshoot
        workers[max(workers, key=workers.get)] -= 1
    for r in sorted(cost, key=lambda r: -cost[r][0])[:budget - sum(workers.values())]:
        workers[r] += 1  # hand the rounding remainder to the costliest runs
    plan = {}
    for r, w in workers.items():
        k = max(1, min(cost[r][1], round(w / WORKERS_PER_SHARD)))
        plan[r] = [w // k + (i < w % k) for i in range(k)]
    return plan


def run(cmd, log=None):
    print("$ " + " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, cwd=PKG_DIR)
        return
    with open(log, "w") as f:
        subprocess.run(cmd, check=True, cwd=PKG_DIR, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selection", default=None, help="Run 1's bin_selection.json.")
    ap.add_argument("--bins", type=int, default=None, help="Bin count (overrides --selection).")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=RUNS)
    ap.add_argument("--steps", nargs="+", default=["generate", "provision", "run", "analyse"],
                    choices=["generate", "provision", "run", "analyse"])
    ap.add_argument("--out-base", default=OUT_BASE)
    ap.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Pool workers across all shards (default cores - 1; capped at cores).")
    ap.add_argument("--site", nargs=2, type=float, default=None, metavar=("LAT", "LON"),
                    help="Repeat runs 2 and 3 (single-site sweeps) at this open-ocean site; "
                         "configs and results go in thesis_<run>_<site>.")
    args = ap.parse_args()
    args.max_workers = max(1, min(args.max_workers, os.cpu_count() or 1))
    global TAG
    if args.site:
        TAG = "_" + site_tag(*args.site)
        if "--runs" not in sys.argv:
            args.runs = list(SITE_RUNS)
        if set(args.runs) - set(SITE_RUNS):
            sys.exit(f"[error] --site repeats only {list(SITE_RUNS)}")

    n = args.bins
    if n is None:
        if not args.selection:
            sys.exit("[error] give --bins or --selection")
        with open(args.selection) as f:
            n = int(json.load(f)["chosen_n_bins"])
    print(f"[thesis] wind bins: {n}; runs: {args.runs}; steps: {args.steps}"
          + (f"; site {args.site}" if args.site else ""), flush=True)
    base = args.out_base

    if "generate" in args.steps:
        cmds = generate_cmds(n, args.site)
        for r in args.runs:
            # A rerun with another bin count must not leave the old count's configs behind.
            if os.path.isdir(cfg_dir(r)):
                for fn in os.listdir(cfg_dir(r)):
                    if fn.endswith((".yaml", ".json")):
                        os.remove(os.path.join(cfg_dir(r), fn))
            run(cmds[r])
    if "provision" in args.steps:
        for r in args.runs:
            t0 = time.time()
            run(py("run_chain_sweep.py", "--configs", cfg_dir(r), "--out", out_dir(r, base),
                   "--provision-only"), log=f"{base}/thesis_{r}{TAG}_provision.log")
            print(f"[thesis] {r} provisioned in {time.time() - t0:.0f}s", flush=True)
    if "run" in args.steps:
        plan = plan_shards(args.runs, base, args.max_workers)
        print(f"[thesis] worker plan (budget {args.max_workers} of {os.cpu_count()} cores): "
              + ", ".join(f"{r} {ws}" for r, ws in plan.items()), flush=True)
        while free_ram_gb() < MIN_FREE_GB:
            print(f"[thesis] {free_ram_gb():.1f} GB free; waiting for {MIN_FREE_GB:.0f} GB",
                  flush=True)
            time.sleep(60)
        t0 = time.time()
        procs = []
        # All shards of a run start together: with --resume each shard drops finished
        # configs before splitting, so they must all see the same finished set.
        for r, ws in plan.items():
            for i, w in enumerate(ws, 1):
                cmd = py("run_chain_sweep.py", "--configs", cfg_dir(r), "--out", out_dir(r, base),
                         "--resume", "--no-provision", "--shard", f"{i}/{len(ws)}", "--workers", w)
                log = open(f"{base}/thesis_{r}{TAG}_shard{i}.log", "w")
                print("$ " + " ".join(cmd), flush=True)
                procs.append((r, i, subprocess.Popen(cmd, cwd=PKG_DIR, stdout=log,
                                                     stderr=subprocess.STDOUT), log))
        failed, low = [], free_ram_gb()
        while any(p.poll() is None for _, _, p, _ in procs):
            time.sleep(30)
            low = min(low, free_ram_gb())
            if (time.time() - t0) % 1800 < 30:
                print(f"[thesis] {(time.time() - t0) / 60:.0f} min: {free_ram_gb():.1f} GB "
                      f"free (lowest {low:.1f})", flush=True)
        for r, i, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"[thesis] {r} shard {i} exited {rc}", flush=True)
            if rc:
                failed.append(f"{r}#{i}")
        print(f"[thesis] run step took {(time.time() - t0) / 60:.0f} min; lowest free RAM "
              f"{low:.1f} GB", flush=True)
        if failed:
            sys.exit(f"[error] shards failed: {failed}; see the shard logs, then rerun with "
                     "--steps run analyse (--resume skips finished configs)")
    if "analyse" in args.steps:
        cmds = analyse_cmds(base)
        for r in [r for r in RUNS if r in args.runs]:  # headline before worlds
            run(cmds[r], log=f"{base}/thesis_{r}{TAG}_analyse.log")
            print(f"[thesis] {r} analysed", flush=True)


if __name__ == "__main__":
    main()
