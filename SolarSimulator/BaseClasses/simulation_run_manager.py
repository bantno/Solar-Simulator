import os
import time
import multiprocessing
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from BaseClasses.simulation_storage import SimulationStorageHDF5


# ----------------------------
# Helpers (top-level & picklable)
# ----------------------------
def _label_for_sim(sim) -> str:
    """
    Build a compact, human-friendly label for logging.
    Example: 'OptimalContinuousAnalyticalPolicySimulation|c200Wh|H30h|t5|w3mps|f5p|lat30.0_lon-90.0'
    """
    parts = [sim.__class__.__name__]
    # capacity
    try:
        cap = getattr(sim.mdp, "battery_capacity_wh", None)
        if cap is not None:
            parts.append(f"c{int(cap)}Wh")
    except Exception:
        pass
    # horizon, thresholds, penalty
    for attr, key, fmt in [
        ("horizon", "H", "{}h"),
        ("observation_threshold", "t", "{}"),
        ("wind_threshold", "w", "{}mps"),
        ("failure_penalty", "f", "{}p"),
    ]:
        try:
            v = getattr(sim, attr, None)
            if v is not None:
                if isinstance(v, (int, float)) and float(v).is_integer():
                    v = int(v)
                parts.append(f"{key}{fmt.format(v)}")
        except Exception:
            pass
    # location
    try:
        loc = getattr(sim, "location", None)
        if isinstance(loc, dict) and "latitude" in loc and "longitude" in loc:
            parts.append(f"lat{loc['latitude']}_lon{loc['longitude']}")
    except Exception:
        pass
    return "|".join(parts)


def _run_one_sim(args: Tuple[Any, int, bool, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Worker function for a single simulation object.
    Args:
        args = (sim, episodes_per_simulation, verbose, label)
    Returns:
        (simulation_metadata, episodes)
    """
    sim, episodes_per_simulation, verbose, label = args
    pid = os.getpid()
    t0 = time.time()
    if verbose:
        print(f"[PID {pid}] ▶ START {label}", flush=True)

    episodes: List[Dict[str, Any]] = []
    total_reward = 0.0
    failure_step = 0.0
    failure = 0
    flight_hrs = 0.0

    # Episodes run SERIALLY inside this worker
    for episode in sim.simulate_multiple_episodes(episodes_per_simulation):
        flight_hrs   += episode["flight_hrs"]
        total_reward += episode["total_reward"]
        failure      += episode["failure"]
        failure_step += episode["failure_step"]
        episodes.append(episode)

    # Simulation-level metadata
    simulation_metadata: Dict[str, Any] = {
        "simulation_type": sim.__class__.__name__,
        "episodes_count": len(episodes),
        "battery_capacity": getattr(sim.mdp, "battery_capacity_wh", None),
        "horizon": getattr(sim, "horizon", None),
        "initial_state": getattr(sim, "initial_state", []),
        "start_time": getattr(sim, "start_datetime", None),
        "failure_penalty": getattr(sim, "failure_penalty", None),
        "average_failure_step": failure_step / episodes_per_simulation if episodes_per_simulation else 0.0,
        "failure_percentage":   failure / episodes_per_simulation      if episodes_per_simulation else 0.0,
        "average_reward":       total_reward / episodes_per_simulation if episodes_per_simulation else 0.0,
        "average_flight_hrs":   flight_hrs / episodes_per_simulation   if episodes_per_simulation else 0.0,
    }
    if hasattr(sim, "observation_threshold"):
        simulation_metadata["observation_threshold"] = sim.observation_threshold
    if hasattr(sim, "wind_threshold"):
        simulation_metadata["wind_threshold"] = sim.wind_threshold
    if hasattr(sim, "location"):
        loc = sim.location
        simulation_metadata["location_id"] = f"lat{loc['latitude']}_lon{loc['longitude']}"

    if verbose:
        dt = time.time() - t0
        print(f"[PID {pid}] ✓ DONE  {label} in {dt:.1f}s", flush=True)

    return simulation_metadata, episodes


# ----------------------------
# Manager
# ----------------------------
class SimulationRunManager:
    """
    Runs multiple simulations and stores each simulation's episodes as a group
    in ONE HDF5 (per manager instance). Simulations can be parallelized across
    worker processes; episodes remain SERIAL within each simulation.
    """
    def __init__(
        self,
        episodes_per_simulation: int,
        storage_dir: str,
        sim_name_prefix: Optional[str] = None
    ):
        """
        Parameters:
            episodes_per_simulation: Number of episodes to run per simulation.
            storage_dir: Directory to place the batch HDF5 file.
            sim_name_prefix: Prefix for the timestamped HDF5 filename.
        """
        self.sim_name_prefix = sim_name_prefix
        self.episodes_per_simulation = episodes_per_simulation

        # Timestamped output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.sim_name_prefix or f"sim_{episodes_per_simulation}_eps"
        unique_filename = f"{prefix}_{timestamp}.h5"
        batch_path = os.path.join(storage_dir, unique_filename)

        self.storage = SimulationStorageHDF5(batch_path)

    def run_simulations(
        self,
        simulation_list: list,
        use_multiprocessing: bool = False,
        num_workers: Optional[int] = None,
        chunk_size: int = 1,
        maxtasksperchild: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Execute simulations and incrementally store results by simulation group.

        Args:
            simulation_list: list of simulation objects.
            use_multiprocessing: if True, run sims in parallel across processes.
            num_workers: number of worker processes (defaults to CPU-1 if None).
            chunk_size: tasks per worker pull (1 = best load balancing).
            maxtasksperchild: recycle a worker after N sims (optional hygiene).
            verbose: per-worker START/DONE prints.
        """
        # Determine worker count
        if use_multiprocessing and (num_workers is None):
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        # --- SERIAL execution ---
        if not use_multiprocessing:
            for sim in simulation_list:
                label = _label_for_sim(sim)
                sim_meta, episodes = _run_one_sim(
                    (sim, self.episodes_per_simulation, verbose, label)
                )
                group = self._make_group_name(sim_meta)
                self.storage.store_simulation(sim_meta, episodes, group_name=group)
                episodes = None
                # Optional extra durability: flush after each sim
                try:
                    self.storage.h5file.flush()
                except Exception:
                    pass
                # print(f"→ Stored group '{group}' with {len(episodes)} episodes")
                print(f"→ Stored group '{group}' with {sim_meta.get('episodes_count', 'n/a')} episodes")

        # --- PARALLEL execution ---
        else:
            tasks = [
                (sim, self.episodes_per_simulation, verbose, _label_for_sim(sim))
                for sim in simulation_list
            ]
            pool_kwargs = {"processes": num_workers}
            if maxtasksperchild is not None:
                pool_kwargs["maxtasksperchild"] = int(maxtasksperchild)

            with multiprocessing.Pool(**pool_kwargs) as pool:
                # dynamic scheduling; workers pull one sim at a time
                for sim_meta, episodes in pool.imap_unordered(_run_one_sim, tasks, chunksize=max(1, int(chunk_size))):
                    group = self._make_group_name(sim_meta)
                    self.storage.store_simulation(sim_meta, episodes, group_name=group)
                    episodes = None
                    # Optional extra durability: flush after each sim
                    try:
                        self.storage.h5file.flush()
                    except Exception:
                        pass
                    # print(f"→ Stored group '{group}' with {len(episodes)} episodes")
                    print(f"→ Stored group '{group}' with {sim_meta.get('episodes_count', 'n/a')} episodes")

        # Close the HDF5 file when all writes are done
        self.storage.close()
        print("All simulations completed and stored in one file.")

    def _make_group_name(self, meta: dict) -> str:
        """
        Build a group name mirroring previous filenames, e.g.:
        'threshold_c100_f5_h30_lat30.0_lon-90.0_t0.5_w2.0_d162'
        """
        parts = [str(meta.get("simulation_type", "")).lower()]
        # capacity (int)
        cap = meta.get("battery_capacity")
        if cap is not None:
            try:
                parts.append(f"c{int(cap)}")
            except Exception:
                parts.append(f"c{cap}")
        # penalty, horizon
        if "failure_penalty" in meta and meta["failure_penalty"] is not None:
            try:
                parts.append(f"f{int(meta['failure_penalty'])}")
            except Exception:
                parts.append(f"f{meta['failure_penalty']}")
        if "horizon" in meta and meta["horizon"] is not None:
            try:
                parts.append(f"h{int(meta['horizon'])}")
            except Exception:
                parts.append(f"h{meta['horizon']}")
        # location
        if "location_id" in meta and meta["location_id"]:
            parts.append(meta["location_id"])
        # thresholds
        if "observation_threshold" in meta and meta["observation_threshold"] is not None:
            parts.append(f"t{meta['observation_threshold']}")
        if "wind_threshold" in meta and meta["wind_threshold"] is not None:
            parts.append(f"w{meta['wind_threshold']}")
        # day-of-year from ISO start_time
        start_time = meta.get("start_time")
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time)
                parts.append(f"d{dt.timetuple().tm_yday}")
            except Exception:
                pass
        return "_".join(parts)
