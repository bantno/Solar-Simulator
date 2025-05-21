import os
import multiprocessing
from datetime import datetime
from BaseClasses.simulation_storage import SimulationStorageHDF5

def _run_one_sim(args):
    """
    Worker function for a single simulation object.
    Returns (simulation_metadata, episodes).
    """
    sim, episodes_per_simulation = args
    episodes = []
    for episode in sim.simulate_multiple_episodes(episodes_per_simulation):
        # Ensure total_reward is set
        episode["total_reward"] = episode.get(
            "total_reward", sum(episode.get("rewards", []))
        )

        episodes.append(episode)

    # Create overall simulation-level metadata
    simulation_metadata = {
        "simulation_type": sim.__class__.__name__,
        "episodes_count": len(episodes),
        "battery_capacity": sim.mdp.battery_capacity_wh,
        "horizon": sim.horizon,
        "initial_state": sim.initial_state.tolist(),
        "start_time": sim.start_datetime,
        "failure_penalty": sim.failure_penalty,
    }
    if hasattr(sim, "observation_threshold"):
        simulation_metadata["observation_threshold"] = sim.observation_threshold
    if hasattr(sim, "wind_threshold"):
        simulation_metadata["wind_threshold"] = sim.wind_threshold
    if hasattr(sim, "location"):
        # standardize into a short string, e.g. “lat30.0_lon-90.0”
        loc = sim.location
        simulation_metadata["location_id"] = (
        f"lat{loc['latitude']}_lon{loc['longitude']}"
        )

    return (simulation_metadata, episodes)

class SimulationRunManager:
    """
    A generic manager for running multiple simulations and storing each simulation run's
    episodes together as a batch in one file (all_simulations.h5).
    """
    def __init__(self, episodes_per_simulation: int, storage_dir: str):
        """
        Parameters:
            episodes_per_simulation (int): Number of episodes to run for each simulation instance.
            storage_dir (str): Directory where the batch HDF5 file will be stored.
        """
        self.episodes_per_simulation = episodes_per_simulation
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Example metadata – you can substitute or extend this
        sim_name = f"sim_{episodes_per_simulation}_eps"

        # Create unique filename
        unique_filename = f"{sim_name}_{timestamp}.h5"

        # Join with the storage directory
        batch_path = os.path.join(storage_dir, unique_filename)
        self.storage = SimulationStorageHDF5(batch_path)

    def run_simulations(self, simulation_list: list, use_multiprocessing=False, num_workers=None):
        """
        Runs each simulation in the provided list, collects its episodes, enriches metadata,
        and stores all episodes for that simulation into one HDF5 file.
        """
        # Determine worker count
        if use_multiprocessing and (num_workers is None):
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        # --- SERIAL execution ---
        if not use_multiprocessing:
            for sim in simulation_list:
                sim_meta, episodes = _run_one_sim((sim, self.episodes_per_simulation))
                group = self._make_group_name(sim_meta)
                self.storage.store_simulation(sim_meta, episodes, group_name=group)
                print(f"→ Stored group '{group}' with {len(episodes)} episodes")

        # --- PARALLEL execution ---
        else:
            tasks = [(sim, self.episodes_per_simulation) for sim in simulation_list]
            with multiprocessing.Pool(processes=num_workers) as pool:
                for sim_meta, episodes in pool.imap_unordered(_run_one_sim, tasks):
                    group = self._make_group_name(sim_meta)
                    self.storage.store_simulation(sim_meta, episodes, group_name=group)
                    print(f"→ Stored group '{group}' with {len(episodes)} episodes")

        # Close the HDF5 file when all writes are done
        self.storage.close()
        print("All simulations completed and stored in one file.")

    def _make_group_name(self, meta: dict) -> str:
        """
        Build a group name mirroring previous filenames, e.g. 'threshold_c100_t0.5_w2.0'.
        """
        parts = [meta["simulation_type"].lower()]
        parts.append(f"c{int(meta['battery_capacity'])}")
        if "failure_penalty" in meta:
            parts.append(f"f{int(meta['failure_penalty'])}")
        if "horizon" in meta:
            parts.append(f"h{int(meta['horizon'])}")
        if "location_id" in meta:
            parts.append(meta["location_id"])
        if "observation_threshold" in meta:
            parts.append(f"t{meta['observation_threshold']}")
        if "wind_threshold" in meta:
            parts.append(f"w{meta['wind_threshold']}")
        return "_".join(parts)
