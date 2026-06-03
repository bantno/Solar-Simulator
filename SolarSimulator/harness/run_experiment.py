#!/usr/bin/env python3
"""
run_experiment.py -- declarative, self-describing simulation validation harness.

One YAML fully describes an experiment (the sweep matrix + all behavior flags). This CLI
expands and executes it, reusing the existing SimulationFactory / SimulationRunManager / HDF5
stack, and collects everything for that experiment into a single timestamped run directory:

    <storage_dir>/<config_basename>/<YYYYmmdd_HHMMSS>/
        config.yaml            # copy of the resolved spec
        <config_basename>_*.h5 # raw episode data (one HDF5 group per simulation)
        summary.csv            # tidy metrics, one row per simulation
        run_metadata.json      # git SHA, timestamp, command
        solver_tables/         # redirected value-function .npy files
        figures/               # sweep-summary plots + a trajectory replay

Modes are selectable subcommands. Only `experiment` (sweep) is implemented; `regression` and
`perf` are reserved seams for a later task.

Usage (pvlib conda env):
    python harness/run_experiment.py experiment harness/examples/iid_small.yaml
    python harness/run_experiment.py experiment ./my_configs --workers 12
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

# Make BaseClasses importable from any cwd (mirrors the verify_*.py scripts).
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../SolarSimulator
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from BaseClasses.run_sim import YAMLSimulationRunner          # noqa: E402
from BaseClasses.simulation_run_manager import SimulationRunManager  # noqa: E402
from harness import HARNESS_VERSION                            # noqa: E402
from harness.summarize import write_summary_csv                # noqa: E402
from harness.figures import plot_sweep_summary, plot_trajectory_replay, render_paper_figure  # noqa: E402


# --------------------------------------------------------------------------------------
# Picklable sim-creation worker (must be top-level for spawn). Reads behavior from config,
# NOT from CLI flags, so the YAML alone determines the run. (Avoids importing gui -> PyQt5.)
# --------------------------------------------------------------------------------------
def create_simulation_wrapper(args):
    factory, sim_type, cap, threshold, wind_threshold, save_history, full_history_episodes = args
    return factory.create_simulation(
        sim_type=sim_type,
        cap=cap,
        threshold=threshold,
        wind_threshold=wind_threshold,
        save_states=save_history,
        full_history_episodes=full_history_episodes,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PKG_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _abspath(p: str) -> str:
    """Resolve a config path to absolute, relative to the repo root when not already absolute."""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(PKG_DIR), p))


def _resolve_paths_in_place(config: dict) -> None:
    """Make every data_path / wind_chain.path absolute so spawned workers don't depend on cwd."""
    for loc in config.get("locations", []) or []:
        if isinstance(loc, dict) and loc.get("data_path"):
            loc["data_path"] = _abspath(loc["data_path"])
    if config.get("data_path"):
        config["data_path"] = _abspath(config["data_path"])
    wc = config.get("wind_chain")
    if isinstance(wc, dict) and wc.get("path"):
        wc["path"] = _abspath(wc["path"])


def _representative_optimal_sim(sims):
    """Pick a sim for the trajectory figure: first Optimal sim, else the first sim."""
    for s in sims:
        if "Optimal" in s.__class__.__name__:
            return s
    return sims[0] if sims else None


def run_experiment(config_path, storage_base, workers, use_multiproc):
    runner = YAMLSimulationRunner(config_path)
    config = runner.config
    config_basename = runner.config_basename

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = storage_base or config.get("storage_dir") or os.path.join(os.path.dirname(PKG_DIR), "results")
    run_dir = os.path.join(base, config_basename, timestamp)
    solver_tables = os.path.join(run_dir, "solver_tables")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(solver_tables, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Make the config self-contained for spawned workers + downstream summarize/figures.
    _resolve_paths_in_place(config)
    config["_run_output_dir"] = solver_tables       # solver writes value-table .npy here
    config["_config_basename"] = config_basename     # carried into summary.csv

    # Copy the resolved spec (drop harness-internal underscore keys for a clean, re-runnable file).
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)

    # Behavior flags come from the config (self-describing), not CLI flags.
    save_history = bool(config.get("save_states", False))
    full_history_eps = int(config.get("full_history_episodes") or 0)

    # Build sims (solve() for optimal runs here, in the build pool).
    param_list = runner._build_param_list()
    job_args = [(*p, save_history, full_history_eps) for p in param_list]
    print(f"[build] {config_basename}: creating {len(job_args)} simulations "
          f"(workers={workers if use_multiproc else 1})")
    if use_multiproc and workers > 1:
        with mp.Pool(processes=workers) as pool:
            sims = pool.map(create_simulation_wrapper, job_args)
    else:
        sims = [create_simulation_wrapper(a) for a in job_args]
    sims = [s for s in sims if s is not None]
    if not sims:
        print(f"[warn] no simulations produced for {config_basename}; skipping.")
        return run_dir

    # Run episodes and store one HDF5 in the run dir.
    episodes = int(config.get("episodes", 3000))
    manager = SimulationRunManager(
        episodes_per_simulation=episodes,
        storage_dir=run_dir,
        sim_name_prefix=config_basename,
    )
    t0 = time.time()
    manager.run_simulations(
        simulation_list=sims,
        use_multiprocessing=use_multiproc,
        num_workers=workers,
    )
    print(f"[run]  {config_basename}: {time.time() - t0:.1f}s")

    # Locate the HDF5 the manager just wrote (exactly one per run dir).
    h5_matches = glob.glob(os.path.join(run_dir, "*.h5"))
    h5_path = h5_matches[0] if h5_matches else None

    # Summary CSV (the stable metrics seam) + figures.
    if h5_path:
        df = write_summary_csv(h5_path, config, os.path.join(run_dir, "summary.csv"))
        # If the config names a journal-paper figure, reproduce it exactly from this run's HDF5
        # (matches Figures/Scripts/generate_journal_paper_figures.py). Otherwise draw the generic
        # family-of-curves summary.
        paper_figure = config.get("paper_figure")
        if paper_figure:
            render_paper_figure(paper_figure, h5_path, figures_dir)
        else:
            plot_sweep_summary(df, figures_dir)
    rep = _representative_optimal_sim(sims)
    if rep is not None:
        plot_trajectory_replay(
            rep, config,
            location=getattr(rep, "location", {}),
            start_iso=getattr(rep, "start_datetime", None),
            out_dir=figures_dir,
            interval_min=int(config.get("delta_t", 15)),
        )

    # Light reproducibility capture (no seeding machinery -- deferred by design).
    meta = {
        "harness_version": HARNESS_VERSION,
        "git_sha": _git_sha(),
        "timestamp": timestamp,
        "command": " ".join(sys.argv),
        "config_basename": config_basename,
        "experiment_name": config.get("experiment_name"),
        "description": config.get("description"),
        "n_simulations": len(sims),
        "episodes_per_simulation": episodes,
        "hdf5": os.path.basename(h5_path) if h5_path else None,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] {config_basename}: results in {run_dir}")
    return run_dir


def _cmd_experiment(args):
    if os.path.isdir(args.path):
        cfgs = sorted(glob.glob(os.path.join(args.path, args.pattern)))
        if not cfgs:
            sys.exit(f"[error] no configs matched {args.pattern} in {args.path}")
    else:
        cfgs = [args.path]
    use_mp = not args.no_multiproc
    workers = max(1, args.workers)
    for cfg in cfgs:
        run_experiment(cfg, args.out, workers, use_mp)
    print("All done.")


def _cmd_not_implemented(args):
    sys.exit(f"[info] '{args._mode}' mode is not implemented yet (reserved seam). "
             f"Use 'experiment'. See harness/README.md.")


def build_parser():
    ap = argparse.ArgumentParser(description="Declarative simulation validation harness.")
    sub = ap.add_subparsers(dest="mode", required=True)

    ex = sub.add_parser("experiment", help="Run a sweep described by a YAML config (or dir of configs).")
    ex.add_argument("path", help="YAML config file or directory of configs.")
    ex.add_argument("--pattern", default="*.y*ml", help="Glob when path is a directory.")
    ex.add_argument("--out", default=None,
                    help="Storage base dir (default: config['storage_dir'] or <repo>/results).")
    ex.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Worker processes for sim creation + execution (default: CPU-1).")
    ex.add_argument("--no-multiproc", action="store_true", help="Run serially.")
    ex.set_defaults(func=_cmd_experiment)

    for mode in ("regression", "perf"):
        p = sub.add_parser(mode, help=f"[reserved] {mode} mode -- not implemented yet.")
        p.set_defaults(func=_cmd_not_implemented, _mode=mode)

    return ap


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    mp.freeze_support()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
