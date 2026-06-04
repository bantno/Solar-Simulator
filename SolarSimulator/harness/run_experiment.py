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
from BaseClasses.run_sim import _derive_chain_path, _derive_histcube_path  # noqa: E402
from harness import HARNESS_VERSION                            # noqa: E402
from harness.summarize import write_summary_csv                # noqa: E402
from harness.figures import plot_sweep_summary, plot_trajectory_replay, render_paper_figure  # noqa: E402


# --------------------------------------------------------------------------------------
# Data provisioning: fetch historical weather and build missing artifacts on demand.
# --------------------------------------------------------------------------------------

def _hist_dir_from_data_path(data_path: str) -> str:
    """Return the HISTORICAL_DATA sibling directory for a given EXPECTED_DATA path."""
    return os.path.join(os.path.dirname(os.path.dirname(data_path)), "HISTORICAL_DATA")


def _find_historical_pkl(hist_dir: str, lat: float, lon: float) -> str:
    """Return the path of an existing historical pkl for this location, or None."""
    candidates = [
        os.path.join(hist_dir, f"data_{lat}_{lon}.pkl"),
        os.path.join(hist_dir, f"data_{int(lat)}_{int(lon)}.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Broader glob: any pkl whose filename contains both coordinate substrings.
    lat_str, lon_str = f"{lat:g}", f"{lon:g}"
    for p in glob.glob(os.path.join(hist_dir, "*.pkl")):
        name = os.path.basename(p)
        if lat_str in name and lon_str in name:
            return p
    return None


def _configured_bin_edges(wc_cfg: dict):
    """
    Return the full bin-edge array from wind_chain.bin_edges (interior cutpoints in the YAML),
    or None if the key is absent (falls back to quantile-based derivation at build time).
    """
    import numpy as np  # noqa: E402 (local import avoids top-level dep for non-chain runs)
    edges = wc_cfg.get("bin_edges")
    if edges is None:
        return None
    interior = np.asarray(edges, dtype=float)
    return np.concatenate(([0.0], interior, [np.inf]))


def _fetch_historical_pkl(hist_dir: str, lat: float, lon: float) -> str:
    """Fetch historical weather from Open-Meteo (1950-2022) and save to hist_dir."""
    from Scripts.create_weather_distributions import WeatherDataProcessor  # noqa: E402
    os.makedirs(hist_dir, exist_ok=True)
    out_path = os.path.join(hist_dir, f"data_{lat}_{lon}.pkl")
    print(f"[provision] Fetching historical weather for lat={lat}, lon={lon} "
          f"(1950-01-01 to 2022-12-31) - this may take a few minutes ...")
    proc = WeatherDataProcessor()
    proc.fetch_weather_data(
        lat, lon, "1950-01-01", "2022-12-31",
        ["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"],
    )
    proc.process_hourly_data()
    proc.hourly_dataframe.to_pickle(out_path)
    print(f"[provision] Historical data saved to {out_path}")
    return out_path


def _ensure_location_data(config: dict, location: dict) -> None:
    """
    Check that all artifact files required by config exist for this location.
    Builds missing windchain / histcube from historical data, fetching from
    Open-Meteo first if the historical pkl itself is absent.
    """
    from Scripts.create_weather_distributions import (  # noqa: E402
        build_wind_chain_artifact,
        build_historical_cube_artifact,
    )

    data_path = location["data_path"]
    lat = location.get("latitude")
    lon = location.get("longitude")
    interval_min = int(config.get("delta_t", 15))

    wc_cfg = config.get("wind_chain") or {}
    hw_cfg = config.get("historical_weather") or {}

    import numpy as np    # noqa: E402
    import pandas as pd   # noqa: E402

    missing_base = not os.path.exists(data_path)
    chain_path = wc_cfg.get("path") or _derive_chain_path(data_path)
    cube_path = hw_cfg.get("path") or _derive_histcube_path(data_path)

    # Chain: missing file OR configured edges have changed since last build.
    missing_chain = False
    if wc_cfg.get("enabled", False):
        configured_edges = _configured_bin_edges(wc_cfg)
        explicit_path = bool(wc_cfg.get("path"))
        if not os.path.exists(chain_path):
            missing_chain = True
        elif not explicit_path and configured_edges is not None:
            existing = pd.read_pickle(chain_path)
            if not np.allclose(existing["bin_edges"], configured_edges):
                print(f"[provision] wind_chain bin_edges changed - rebuilding {chain_path}")
                missing_chain = True

    missing_cube = hw_cfg.get("enabled", False) and not os.path.exists(cube_path)

    if not missing_base and not missing_chain and not missing_cube:
        return

    if lat is None or lon is None:
        raise RuntimeError(
            f"Cannot provision data for {data_path!r}: "
            "latitude and longitude must be set in the location config."
        )

    hist_dir = _hist_dir_from_data_path(data_path)
    hist_pkl = _find_historical_pkl(hist_dir, lat, lon)
    if hist_pkl is None:
        hist_pkl = _fetch_historical_pkl(hist_dir, lat, lon)

    if missing_base:
        from BaseClasses.weather_processor_cs_normalization import build_expected_data_artifact  # noqa: E402
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"[provision] Building expected data -> {data_path}")
        build_expected_data_artifact(hist_pkl, data_path, lat, lon, interval_min)

    if missing_chain:
        configured_edges = _configured_bin_edges(wc_cfg)
        n_bins = wc_cfg.get("n_bins", 3)
        os.makedirs(os.path.dirname(chain_path), exist_ok=True)
        print(f"[provision] Building wind-chain artifact -> {chain_path}")
        build_wind_chain_artifact(
            hist_pkl, chain_path,
            interval_minutes=interval_min,
            n_bins=n_bins,
            bin_edges=configured_edges,
        )

    if missing_cube:
        os.makedirs(os.path.dirname(cube_path), exist_ok=True)
        print(f"[provision] Building histcube artifact -> {cube_path}")
        build_historical_cube_artifact(hist_pkl, cube_path, interval_minutes=interval_min)


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
    """Make every data_path / artifact path absolute so spawned workers don't depend on cwd."""
    for loc in config.get("locations", []) or []:
        if isinstance(loc, dict) and loc.get("data_path"):
            loc["data_path"] = _abspath(loc["data_path"])
    if config.get("data_path"):
        config["data_path"] = _abspath(config["data_path"])
    wc = config.get("wind_chain")
    if isinstance(wc, dict) and wc.get("path"):
        wc["path"] = _abspath(wc["path"])
    hw = config.get("historical_weather")
    if isinstance(hw, dict) and hw.get("path"):
        hw["path"] = _abspath(hw["path"])


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

    # Build any missing windchain / histcube artifacts before spawning workers.
    for loc in config.get("locations", []) or []:
        _ensure_location_data(config, loc)

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
