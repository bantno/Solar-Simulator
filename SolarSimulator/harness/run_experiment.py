#!/usr/bin/env python3
"""run_experiment.py -- declarative, self-describing simulation validation harness.

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
from BaseClasses.run_sim import (  # noqa: E402
    _derive_chain_path, _derive_histcube_path, _derive_solar_chain_path,
    _resolve_data_path, holdout_spec, resolve_artifact_paths, wind_chain_needed,
)
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
    # Broader search: any data_<lat>_<lon>.pkl whose parsed coordinates match numerically
    # (e.g. data_30_-90.pkl for 30.0/-90.0). Not a substring test: "0" and "-75" both
    # occur in data_30.0_-75.0.pkl, which once resolved lat 0 to the lat 30 record.
    import re  # noqa: E402
    pat = re.compile(r"^data_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.pkl$")
    for p in sorted(glob.glob(os.path.join(hist_dir, "*.pkl"))):
        m = pat.match(os.path.basename(p))
        if m and float(m.group(1)) == float(lat) and float(m.group(2)) == float(lon):
            return p
    return None


def _configured_bin_edges(wc_cfg: dict):
    """Return the full bin-edge array from wind_chain.bin_edges (interior cutpoints in the YAML),
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
    # A 73-year hourly request is weighted heavily by Open-Meteo's rate limiter; after
    # about three in a minute it answers "Minutely API request limit exceeded". Wait out
    # the window and retry rather than failing the whole provisioning step.
    for attempt in range(1, 7):
        try:
            proc.fetch_weather_data(
                lat, lon, "1950-01-01", "2022-12-31",
                ["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"],
            )
            break
        except Exception as e:  # openmeteo_requests.OpenMeteoRequestsError
            msg = str(e).lower()
            if "limit exceeded" not in msg or attempt == 6:
                raise
            wait = 65 if "minutely" in msg else 600
            print(f"[provision] Open-Meteo rate limit ({e}); retry {attempt}/5 in {wait}s",
                  flush=True)
            time.sleep(wait)
    proc.process_hourly_data()
    proc.hourly_dataframe.to_pickle(out_path)
    print(f"[provision] Historical data saved to {out_path}")
    return out_path


def _year_slice_pkl(hist_pkl: str, years, out_dir: str) -> str:
    """Write the rows of an hourly historical pkl whose (local) year lies in `years`
    (inclusive [first, last]) to a temporary pkl in `out_dir` and return its path.
    Every year in the range must be present in the record."""
    import pandas as pd  # noqa: E402
    hist = pd.read_pickle(hist_pkl)
    yrs = hist.index.year
    sliced = hist[(yrs >= years[0]) & (yrs <= years[1])]
    present = set(int(y) for y in sliced.index.year.unique())
    missing = [y for y in range(years[0], years[1] + 1) if y not in present]
    if missing:
        raise ValueError(
            f"{hist_pkl} has no data for year(s) {missing} requested by the hold-out "
            f"range {list(years)} (record spans {int(yrs.min())}-{int(yrs.max())})."
        )
    out = os.path.join(out_dir, f"hist_{years[0]}-{years[1]}.pkl")
    if not os.path.exists(out):
        sliced.to_pickle(out)
        print(f"[provision] hold-out slice {years[0]}-{years[1]}: {len(sliced)} hourly rows")
    return out


def _ensure_location_data(config: dict, location: dict) -> dict:
    """Check that all artifact files required by config exist for this location.

    Builds missing expected-data / windchain / solarchain / histcube artifacts from
    historical data, fetching from Open-Meteo first if the historical pkl itself is
    absent. Under the year-split hold-out (run_sim.holdout_spec, on by default) the
    expected data and chains are built from the fit years only and the histcube from
    the evaluation years only. Returns run_sim.resolve_artifact_paths for the location.
    """
    from Scripts.create_weather_distributions import (  # noqa: E402
        build_wind_chain_artifact,
        build_solar_chain_artifact,
        build_historical_cube_artifact,
    )

    cs_model = config.get("clearsky_model", "fatemi")
    # Clear-sky suffix + hold-out year tags (see run_sim.resolve_artifact_paths); every
    # artifact path below comes from the same resolver the simulation factory uses.
    paths = resolve_artifact_paths(config, location)
    data_path = paths["expected_data"]
    lat = location.get("latitude")
    lon = location.get("longitude")
    interval_min = int(config.get("delta_t", 15))

    wc_cfg = config.get("wind_chain") or {}
    sc_cfg = config.get("solar_chain") or {}
    hw_cfg = config.get("historical_weather") or {}

    import numpy as np    # noqa: E402
    import pandas as pd   # noqa: E402

    missing_base = not os.path.exists(data_path)
    chain_path = paths["wind_chain"]
    solar_chain_path = paths["solar_chain"]
    cube_path = paths["histcube"]

    # Explicitly named artifacts are hand-built (e.g. the 2026-10-20 APL hold-out files).
    # Building one from the full record under that name would mislabel it, so refuse.
    for blk_cfg, label, p in ((wc_cfg, "wind_chain", chain_path),
                              (sc_cfg, "solar_chain", solar_chain_path),
                              (hw_cfg, "historical_weather", cube_path)):
        needed = (wind_chain_needed(config) if label == "wind_chain"
                  else blk_cfg.get("enabled", False))
        if needed and blk_cfg.get("path") and not os.path.exists(p):
            raise FileNotFoundError(
                f"{label}.path is set explicitly but the file is missing: {p}. Explicit "
                "artifacts are never auto-built; restore the file, or remove the path "
                "to let provisioning derive and build it."
            )

    # Chain: missing file OR the configured binning has changed since last build.
    missing_chain = False
    if wind_chain_needed(config):
        configured_edges = _configured_bin_edges(wc_cfg)
        explicit_path = bool(wc_cfg.get("path"))
        if not os.path.exists(chain_path):
            missing_chain = True
        elif not explicit_path:
            existing = pd.read_pickle(chain_path)
            if configured_edges is not None:
                stale = (len(existing["bin_edges"]) != len(configured_edges)
                         or not np.allclose(existing["bin_edges"], configured_edges))
                reason = "bin_edges changed"
            else:
                # Quantile mode: rebuild if the bin count differs or the existing
                # artifact was built from explicit (non-quantile) edges.
                n_bins = int(wc_cfg.get("n_bins", 3))
                stale = (int(existing["n_bins"]) != n_bins
                         or not existing.get("quantile_derived", False))
                reason = f"quantile bins (n_bins={n_bins}) requested"
            if stale:
                print(f"[provision] wind_chain {reason} - rebuilding {chain_path}")
                missing_chain = True

    # Solar chain: missing file OR a different bin count requested (stage-quantile
    # bins have no edge array; n_bins is the only binning knob).
    missing_solar_chain = False
    if sc_cfg.get("enabled", False):
        explicit_path = bool(sc_cfg.get("path"))
        if not os.path.exists(solar_chain_path):
            missing_solar_chain = True
        elif not explicit_path:
            existing = pd.read_pickle(solar_chain_path)
            n_bins = int(sc_cfg.get("n_bins", 3))
            if int(existing["n_bins"]) != n_bins:
                print(f"[provision] solar_chain n_bins={n_bins} requested - "
                      f"rebuilding {solar_chain_path}")
                missing_solar_chain = True
            elif existing.get("clearsky_model", "fatemi") != cs_model:
                print(f"[provision] solar_chain clearsky_model={cs_model} requested - "
                      f"rebuilding {solar_chain_path}")
                missing_solar_chain = True

    missing_cube = hw_cfg.get("enabled", False) and not os.path.exists(cube_path)

    if not missing_base and not missing_chain and not missing_solar_chain and not missing_cube:
        return paths

    if lat is None or lon is None:
        raise RuntimeError(
            f"Cannot provision data for {data_path!r}: "
            "latitude and longitude must be set in the location config."
        )

    hist_dir = _hist_dir_from_data_path(_resolve_data_path(location["data_path"], cs_model))
    hist_pkl = _find_historical_pkl(hist_dir, lat, lon)
    if hist_pkl is None:
        hist_pkl = _fetch_historical_pkl(hist_dir, lat, lon)

    import tempfile  # noqa: E402
    with tempfile.TemporaryDirectory(prefix="holdout_slices_") as slice_dir:
        # Hold-out: fit artifacts from the fit years, the cube from its own years.
        # In-sample (years None): everything from the full record, as before.
        def _source(years):
            return hist_pkl if years is None else _year_slice_pkl(hist_pkl, years, slice_dir)

        if missing_base:
            from BaseClasses.weather_processor_cs_normalization import build_expected_data_artifact  # noqa: E402
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            print(f"[provision] Building expected data -> {data_path}")
            build_expected_data_artifact(_source(paths["fit_years"]), data_path, lat, lon,
                                         interval_min, clearsky_model=cs_model)

        if missing_chain:
            configured_edges = _configured_bin_edges(wc_cfg)
            n_bins = wc_cfg.get("n_bins", 3)
            os.makedirs(os.path.dirname(chain_path), exist_ok=True)
            print(f"[provision] Building wind-chain artifact -> {chain_path}")
            build_wind_chain_artifact(
                _source(paths["fit_years"]), chain_path,
                interval_minutes=interval_min,
                n_bins=n_bins,
                bin_edges=configured_edges,
            )

        if missing_solar_chain:
            os.makedirs(os.path.dirname(solar_chain_path), exist_ok=True)
            print(f"[provision] Building solar-chain artifact -> {solar_chain_path}")
            build_solar_chain_artifact(
                _source(paths["fit_years"]), solar_chain_path,
                latitude=lat, longitude=lon,
                interval_minutes=interval_min,
                n_bins=int(sc_cfg.get("n_bins", 3)),
                clearsky_model=cs_model,
            )

        if missing_cube:
            os.makedirs(os.path.dirname(cube_path), exist_ok=True)
            print(f"[provision] Building histcube artifact -> {cube_path}")
            build_historical_cube_artifact(_source(paths["cube_years"]), cube_path,
                                           interval_minutes=interval_min)
    return paths


def provision_config(config: dict, label: str = "") -> dict:
    """Pin the hold-out decision into `config` and build every missing artifact.

    Writes the normalized spec (or False for in-sample) to config["holdout"], so worker
    processes, the copied config.yaml and downstream analysis all see one explicit
    decision, then runs _ensure_location_data for each location. Every entry point that
    builds simulations (harness, GUI, run_batch) calls this first. Returns the resolved
    artifact paths keyed by location id.
    """
    spec = holdout_spec(config)
    config["holdout"] = spec if spec is not None else False
    tag = f" {label}" if label else ""
    if spec is None:
        print(f"[holdout]{tag}: in-sample (full record for fit and rollouts)")
    else:
        roll = spec["eval_years"] if spec["rollout_years"] == "eval" else spec["fit_years"]
        print(f"[holdout]{tag}: fit {spec['fit_years'][0]}-{spec['fit_years'][1]}, "
              f"rollouts on {spec['rollout_years']} years {roll[0]}-{roll[1]}")
    artifact_paths = {}
    for loc in config.get("locations", []) or []:
        p = _ensure_location_data(config, loc)
        artifact_paths[f"lat{loc.get('latitude')}_lon{loc.get('longitude')}"] = {
            k: v for k, v in p.items() if k != "holdout"}
    return artifact_paths


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
    sc = config.get("solar_chain")
    if isinstance(sc, dict) and sc.get("path"):
        sc["path"] = _abspath(sc["path"])
    hw = config.get("historical_weather")
    if isinstance(hw, dict) and hw.get("path"):
        hw["path"] = _abspath(hw["path"])


# Windows refuses paths over MAX_PATH (260) unless long paths are enabled system-wide.
# Value tables are the longest files a run writes:
#   <base>/<config>/<YYYYmmdd_HHMMSS>/solver_tables/<config>_<cap>Wh_<H>h_<fp>p_<start> 0.npy
_WINDOWS_MAX_PATH = 259
_SOLVER_TABLE_SUFFIX = "_600.0Wh_27000h_100.0p_2025-06-10 0.npy"   # a long worst case


def _windows_long_paths_enabled() -> bool:
    try:
        import winreg  # noqa: E402
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except Exception:
        return False


def check_output_path_length(storage_base: str, config_basename: str) -> None:
    """Fail fast when a run's value-table paths would exceed the Windows path limit.

    Without this the solver dies on its first np.save, after provisioning and the build
    phase. No-op off Windows or when long paths are enabled.
    """
    if os.name != "nt" or _windows_long_paths_enabled():
        return
    longest = os.path.join(os.path.abspath(storage_base), config_basename, "20261231_235959",
                           "solver_tables", config_basename + _SOLVER_TABLE_SUFFIX)
    if len(longest) > _WINDOWS_MAX_PATH:
        raise OSError(
            f"Output paths for '{config_basename}' would be about {len(longest)} characters, "
            f"over the Windows limit of {_WINDOWS_MAX_PATH + 1}. Pass a shorter --out "
            "(for example C:\\Users\\<you>\\sim_results\\<sweep>), shorten the config names, "
            "or enable Windows long paths (LongPathsEnabled, needs admin)."
        )


def _representative_optimal_sim(sims):
    """Pick a sim for the trajectory figure: first Optimal sim, else the first sim."""
    for s in sims:
        if "Optimal" in s.__class__.__name__:
            return s
    return sims[0] if sims else None


def run_experiment(config_path, storage_base, workers, use_multiproc):
    workers = max(1, min(int(workers), os.cpu_count() or 1))  # never more processes than cores
    runner = YAMLSimulationRunner(config_path)
    config = runner.config
    config_basename = runner.config_basename

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = storage_base or config.get("storage_dir") or os.path.join(os.path.dirname(PKG_DIR), "results")
    check_output_path_length(base, config_basename)
    run_dir = os.path.join(base, config_basename, timestamp)
    solver_tables = os.path.join(run_dir, "solver_tables")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(solver_tables, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Make the config self-contained for spawned workers + downstream summarize/figures.
    _resolve_paths_in_place(config)
    config["_run_output_dir"] = solver_tables       # solver writes value-table .npy here
    config["_config_basename"] = config_basename     # carried into summary.csv

    # Pin the hold-out decision and build any missing artifacts before spawning workers.
    artifact_paths = provision_config(config, label=config_basename)

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
    # Threshold sims only wrap their case's shared factory (no solve), so they are built
    # here. From the pool each would come back with its own copy of the weather data
    # (~30 MB): 18 GB for a 616-combo tuning config. Optimal sims solve in the pool,
    # longest horizon first so the slowest solve starts first.
    opt = sorted((i for i, a in enumerate(job_args) if a[1] == "optimal"),
                 key=lambda i: (-job_args[i][0].horizon, -float(job_args[i][2])))
    sims = [None if a[1] == "optimal" else create_simulation_wrapper(a) for a in job_args]
    if use_multiproc and workers > 1 and len(opt) > 1:
        with mp.Pool(processes=min(workers, len(opt))) as pool:
            solved = pool.map(create_simulation_wrapper, [job_args[i] for i in opt], chunksize=1)
    else:
        solved = [create_simulation_wrapper(job_args[i]) for i in opt]
    for i, s in zip(opt, solved):
        sims[i] = s
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
    try:
        rep = _representative_optimal_sim(sims)
        if rep is not None and h5_path:
            out = plot_trajectory_replay(
                rep, h5_path,
                out_dir=figures_dir,
                interval_min=int(config.get("delta_t", 15)),
            )
            if out:
                print(f"[fig]  trajectory replay -> {os.path.basename(out)}")
    except Exception as e:
        print(f"[warn] trajectory replay figure skipped: {e}")

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
        "holdout": config.get("holdout"),
        "artifacts": artifact_paths,
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
             f"Use 'experiment'. See docs/experiments/index.md.")


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
