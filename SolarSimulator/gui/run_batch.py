#!/usr/bin/env python3
"""
CLI runner that behaves like the GUI:
- Builds simulations via YAMLSimulationRunner._build_param_list() and create_simulation_wrapper(...)
- Runs SIMULATIONS in parallel (process pool), EPISODES serially inside each sim
- Stores one HDF5 per config (timestamped), with each sim as its own group

Usage:
  python run_batch.py path/to/config.yaml
  python run_batch.py path/to/configs_dir --pattern "*.yaml" --workers 8
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
import traceback

# --- match the GUI wiring ---
from gui import create_simulation_wrapper                   # :contentReference[oaicite:0]{index=0}
from BaseClasses.run_sim import YAMLSimulationRunner                    # :contentReference[oaicite:1]{index=1}
from BaseClasses.simulation_run_manager import SimulationRunManager     # :contentReference[oaicite:2]{index=2}


def _build_sims_like_gui(runner: YAMLSimulationRunner,
                         use_multiproc: bool,
                         workers: int,
                         save_history: bool,
                         full_history_eps: int):
    """
    Replicates the GUI's path:
      param_list = runner._build_param_list()
      job_args   = [(*args, save_history, full_history_eps)]
      sims       = pool.map(create_simulation_wrapper, job_args)  (if use_multiproc)
    """
    # Build parameter list including all locations/horizons/etc.  (GUI behavior)
    param_list = runner._build_param_list()  # (factory, sim_type, cap, th, wth) :contentReference[oaicite:3]{index=3}

    # Append flags for create_simulation_wrapper(...) (GUI behavior)
    job_args = [(*args, save_history, full_history_eps) for args in param_list]  # :contentReference[oaicite:4]{index=4}

    if use_multiproc:
        # Parallelize SIM CREATION like the GUI (this also parallelizes optimal value-function builds)
        num_cores = max(1, workers)
        print(f"[info] creating simulations with multiprocessing (workers={num_cores})")
        with mp.Pool(processes=num_cores) as pool:
            sims = pool.map(create_simulation_wrapper, job_args)
    else:
        # Serial creation
        sims = [create_simulation_wrapper(arg) for arg in job_args]

    return sims


def _run_one_config(config_path: str,
                    out_dir: str | None,
                    use_multiproc: bool,
                    workers: int,
                    include_optimal: str,
                    save_history: bool,
                    full_history_eps: int):
    cfg_base = os.path.splitext(os.path.basename(config_path))[0]
    print(f"[run] config: {cfg_base}")

    try:
        runner = YAMLSimulationRunner(config_path)  # loads YAML and preps fields  :contentReference[oaicite:5]{index=5}
        config = runner.config
    except Exception as e:
        print(f"[WARN] skip '{cfg_base}': failed to load config: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    # Teach the runner whether to generate optimal-policy jobs.
    # (GUI sets this from a checkbox; default True.)
    if include_optimal != "auto":
        config["include_optimal"] = (include_optimal == "yes")

    # Build sims exactly like the GUI path
    sims = _build_sims_like_gui(
        runner=runner,
        use_multiproc=use_multiproc,
        workers=workers,
        save_history=save_history,
        full_history_eps=full_history_eps,
    )
    if not sims:
        print(f"[info] no simulations to run for '{cfg_base}'")
        return

    # Episodes per simulation: GUI uses config['episodes'] with a fallback to the full-history spin value
    episodes = int(config.get("episodes", full_history_eps))
    storage_dir = out_dir or "simulation_results"  # GUI hard-codes this directory  :contentReference[oaicite:6]{index=6}

    os.makedirs(storage_dir, exist_ok=True)
    manager = SimulationRunManager(
        episodes_per_simulation=episodes,
        storage_dir=storage_dir,
        sim_name_prefix=runner.config_basename,  # prefix for the timestamped HDF5 filename  :contentReference[oaicite:7]{index=7}
    )

    # Run SIMULATIONS in parallel (processes) or serial; EPISODES remain serial inside _run_one_sim  :contentReference[oaicite:8]{index=8}
    start = time.time()
    manager.run_simulations(
        sims,
        use_multiprocessing=use_multiproc,
        num_workers=max(1, workers),
    )
    elapsed = time.time() - start
    print(f"[done] {cfg_base}: {elapsed:.2f}s ({elapsed/3600:.2f}h)")


def main():
    ap = argparse.ArgumentParser(description="Batch-run simulations like the GUI (sims parallel, episodes serial).")
    ap.add_argument("path", help="YAML file or directory containing YAML configs")
    ap.add_argument("--pattern", default="*.y*ml", help="Glob pattern when 'path' is a directory (default: *.y*ml)")
    ap.add_argument("--out", default=None, help="Output directory (default: simulation_results)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Worker processes for parallel SIMULATIONS (default: CPU-1)")
    ap.add_argument("--no-multiproc", action="store_true",
                    help="Disable multiprocessing (forces serial across sims)")
    ap.add_argument("--include-optimal", choices=["auto", "yes", "no"], default="auto",
                    help="Force inclusion of optimal-policy sims (default: auto -> use config/default True)")
    ap.add_argument("--save-history", action="store_true",
                    help="Save full state info for episodes (passed through to sim creation)")
    ap.add_argument("--full-history-eps", type=int, default=1,
                    help="Number of episodes to save full history for; also used as fallback for 'episodes' if not in config (GUI behavior)")
    args = ap.parse_args()

    # Windows-friendly multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    mp.freeze_support()

    use_multiproc = not args.no_multiproc
    paths: list[str]
    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, args.pattern)))
        if not paths:
            print(f"[error] no configs matched {args.pattern} in {args.path}", file=sys.stderr)
            sys.exit(1)
    else:
        paths = [args.path]

    for cfg in paths:
        _run_one_config(
            config_path=cfg,
            out_dir=args.out,
            use_multiproc=use_multiproc,
            workers=args.workers,
            include_optimal=args.include_optimal,
            save_history=args.save_history,
            full_history_eps=args.full_history_eps,
        )

    print("All done.")


if __name__ == "__main__":
    main()
