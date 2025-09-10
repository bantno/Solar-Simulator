#!/usr/bin/env python3
"""
run_batch.py — CLI runner that mirrors the GUI flow and scales well.

Behavior:
- Builds simulations exactly like the GUI:
    YAMLSimulationRunner._build_param_list() -> create_simulation_wrapper(job_args)
- Runs SIMULATIONS in parallel (processes), EPISODES serially inside each sim
- Writes ONE HDF5 per CONFIG (timestamped), with each simulation as its own group
- Dynamic scheduling when supported by SimulationRunManager (chunksize=1 keeps cores busy)
- Optional verbose per-worker START/DONE messages when SimulationRunManager supports 'verbose'

Usage examples:
  python run_batch.py ./configs                         # run all *.y*ml in a dir
  python run_batch.py ./configs --pattern "case*.yaml"  # filter
  python run_batch.py ./configs/case1.yaml              # single config
  python run_batch.py ./configs --workers 20 --chunksize 1 --maxtasksperchild 1 --verbose-workers
"""

import argparse
import glob
import inspect
import multiprocessing as mp
import os
import sys
import time
import traceback

# --- match the GUI wiring ---
from gui import create_simulation_wrapper
from BaseClasses.run_sim import YAMLSimulationRunner
from BaseClasses.simulation_run_manager import SimulationRunManager


def _build_sims_like_gui(
    runner: YAMLSimulationRunner,
    build_workers: int,
    save_history: bool,
    full_history_eps: int,
):
    """
    Replicates the GUI's path:
      param_list = runner._build_param_list()
      job_args   = [(*args, save_history, full_history_eps)]
      sims       = map(create_simulation_wrapper, job_args)
    If build_workers > 1, use a process pool to parallelize SIM CREATION
    (this pushes optimal value-function solves into worker processes).
    """
    # (factory, sim_type, cap, th, wth)
    param_list = runner._build_param_list()

    # Append flags expected by create_simulation_wrapper(...)
    job_args = [(*args, save_history, full_history_eps) for args in param_list]

    if build_workers > 1:
        print(f"[info] creating simulations with multiprocessing (workers={build_workers})")
        with mp.Pool(processes=build_workers) as pool:
            sims = pool.map(create_simulation_wrapper, job_args)
    else:
        sims = [create_simulation_wrapper(arg) for arg in job_args]

    # Filter out any failed creations (None) defensively
    sims = [s for s in sims if s is not None]
    return sims


def _call_manager_run(
    mgr: SimulationRunManager,
    sims: list,
    use_multiproc: bool,
    run_workers: int,
    chunk_size: int | None,
    maxtasksperchild: int | None,
    verbose: bool,
):
    """
    Call SimulationRunManager.run_simulations with best-available signature.
    Older versions may not support chunk_size / maxtasksperchild / verbose: fall back cleanly.
    """
    sig = inspect.signature(mgr.run_simulations)
    kwargs = dict(
        simulation_list=sims,
        use_multiprocessing=use_multiproc,
        num_workers=max(1, run_workers),
    )
    if "chunk_size" in sig.parameters and chunk_size is not None:
        kwargs["chunk_size"] = max(1, int(chunk_size))
    if "maxtasksperchild" in sig.parameters and maxtasksperchild is not None:
        kwargs["maxtasksperchild"] = int(maxtasksperchild)
    if "verbose" in sig.parameters:
        kwargs["verbose"] = bool(verbose)

    return mgr.run_simulations(**kwargs)


def _run_one_config(
    config_path: str,
    out_dir: str | None,
    include_optimal: str,
    build_workers: int,
    run_workers: int,
    use_multiproc: bool,
    chunk_size: int | None,
    maxtasksperchild: int | None,
    save_history: bool,
    full_history_eps: int,
    verbose_workers: bool,
):
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    print(f"[run] config: {cfg_base}")

    try:
        runner = YAMLSimulationRunner(config_path)
        config = runner.config
    except Exception as e:
        print(f"[WARN] skip '{cfg_base}': failed to load config: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    # Mirror GUI checkbox behavior for including optimal sims
    if include_optimal != "auto":
        config["include_optimal"] = (include_optimal == "yes")

    # Build sims (optionally in parallel) exactly like the GUI path
    try:
        sims = _build_sims_like_gui(
            runner=runner,
            build_workers=max(1, build_workers),
            save_history=save_history,
            full_history_eps=full_history_eps,
        )
    except Exception as e:
        print(f"[WARN] skip '{cfg_base}': simulation creation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    if not sims:
        print(f"[info] no simulations to run for '{cfg_base}'")
        return

    # Episodes per simulation from config; fallback to full-history value (GUI spirit)
    episodes = int(config.get("episodes", full_history_eps))
    storage_dir = out_dir or config.get("storage_dir") or "simulation_results"
    os.makedirs(storage_dir, exist_ok=True)

    # One HDF5 per CONFIG; groups per simulation inside
    manager = SimulationRunManager(
        episodes_per_simulation=episodes,
        storage_dir=storage_dir,
        sim_name_prefix=runner.config_basename,  # recognizable, timestamped by manager
    )

    t0 = time.time()
    _call_manager_run(
        mgr=manager,
        sims=sims,
        use_multiproc=use_multiproc,
        run_workers=run_workers,
        chunk_size=chunk_size,
        maxtasksperchild=maxtasksperchild,
        verbose=verbose_workers,
    )
    dt = time.time() - t0
    print(f"[done] {cfg_base}: {dt:.2f}s ({dt/3600:.2f}h)")


def main():
    ap = argparse.ArgumentParser(
        description="Batch-run simulations like the GUI: SIMS parallel, EPISODES serial."
    )
    ap.add_argument("path", help="YAML file or directory of YAMLs")
    ap.add_argument("--pattern", default="*.y*ml",
                    help="Glob pattern when 'path' is a directory (default: *.y*ml)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: config['storage_dir'] or ./simulation_results)")

    # Parallelism knobs
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Worker processes for running SIMULATIONS (default: CPU-1)")
    ap.add_argument("--build-workers", type=int, default=1,
                    help="Worker processes for SIM CREATION (default: equals --workers)")
    ap.add_argument("--no-multiproc", action="store_true",
                    help="Disable multiprocessing during run (serial across sims)")

    # Load-balancing / hygiene (used if your SimulationRunManager supports them)
    ap.add_argument("--chunksize", type=int, default=1,
                    help="Tasks per worker pull for run phase (1 = best load balancing).")
    ap.add_argument("--maxtasksperchild", type=int, default=None,
                    help="Recycle a worker after N sims (optional hygiene).")

    # GUI-like toggles
    ap.add_argument("--include-optimal", choices=["auto", "yes", "no"], default="auto",
                    help="Force inclusion of optimal-policy sims (default: auto -> use config).")
    ap.add_argument("--save-history", action="store_true",
                    help="Pass save_history=True to simulations.")
    ap.add_argument("--full-history-eps", type=int, default=20,
                    help="Episodes to record full history for; also used as fallback for 'episodes' if not set.")

    # Worker logging
    ap.add_argument("--verbose-workers", action="store_true",
                    help="Workers print START/DONE for each simulation (requires manager support).")

    args = ap.parse_args()

    # Windows-friendly multiprocessing setup
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    mp.freeze_support()

    build_workers = args.build_workers if args.build_workers is not None else args.workers
    use_multiproc = not args.no_multiproc

    # Resolve input paths
    if os.path.isdir(args.path):
        cfg_files = sorted(glob.glob(os.path.join(args.path, args.pattern)))
        if not cfg_files:
            print(f"[error] no configs matched {args.pattern} in {args.path}", file=sys.stderr)
            sys.exit(1)
    else:
        cfg_files = [args.path]

    for cfg in cfg_files:
        _run_one_config(
            config_path=cfg,
            out_dir=args.out,
            include_optimal=args.include_optimal,
            build_workers=max(1, build_workers),
            run_workers=max(1, args.workers),
            use_multiproc=use_multiproc,
            chunk_size=args.chunksize,
            maxtasksperchild=args.maxtasksperchild,
            save_history=args.save_history,
            full_history_eps=args.full_history_eps,
            verbose_workers=args.verbose_workers,
        )

    print("All done.")


if __name__ == "__main__":
    main()
