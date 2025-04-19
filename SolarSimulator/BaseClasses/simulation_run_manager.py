from BaseClasses.simulation_storage import SimulationStorage, SimulationStorageHDF5
import multiprocessing

def _run_one_sim(args):
    """
    Worker function for a single simulation object.
    Returns (simulation_metadata, episodes).
    """
    sim, episodes_per_simulation = args
    episodes = []
    for episode in sim.simulate_multiple_episodes(episodes_per_simulation):
        # Enrich each episode’s metadata
        metadata = {"simulation_type": sim.__class__.__name__}
        # Optionally add simulation-specific parameters if they exist
        if hasattr(sim, "observation_threshold"):
            metadata["observation_threshold"] = sim.observation_threshold
        if hasattr(sim, "wind_threshold"):
            metadata["wind_threshold"] = sim.wind_threshold

        episode["metadata"].update(metadata)

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
    }
    if hasattr(sim, "observation_threshold"):
        simulation_metadata["observation_threshold"] = sim.observation_threshold
    if hasattr(sim, "wind_threshold"):
        simulation_metadata["wind_threshold"] = sim.wind_threshold

    return (simulation_metadata, episodes)

class SimulationRunManager:
    """
    A generic manager for running multiple simulations and storing each simulation run's
    episodes together as a batch in one file.
    
    The manager accepts a list of simulation instances (each must implement
    simulate_multiple_episodes(num_episodes) as a generator) and stores the results.
    """
    
    def __init__(self, episodes_per_simulation: int, storage_dir: str):
        """
        Parameters:
            episodes_per_simulation (int): Number of episodes to run for each simulation instance.
            storage_dir (str): Directory where the simulation results will be stored.
        """
        self.episodes_per_simulation = episodes_per_simulation
        self.storage = SimulationStorageHDF5(storage_dir)
    
    def run_simulations(self, simulation_list: list, use_multiprocessing=False, num_workers=None):
        """
        Runs each simulation in the provided list, collects its episodes, enriches metadata,
        and stores all episodes for that simulation. Optionally parallelized.
        """
        if use_multiprocessing and (num_workers is None):
            num_workers = multiprocessing.cpu_count()-1

        # --- SERIAL path (no multiprocessing) ---
        if not use_multiprocessing:
            for sim in simulation_list:
                # Reuse the same worker logic but just call it directly
                simulation_metadata, episodes = _run_one_sim((sim, self.episodes_per_simulation))
                self.storage.store_simulation(simulation_metadata, episodes)
                self.storage.close()
                print(f"Stored simulation {simulation_metadata} with {len(episodes)} episodes.")
                
        # --- MULTIPROCESSING path ---
        else:
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Build tasks as (sim, episodes_per_simulation) pairs
                tasks = [(sim, self.episodes_per_simulation) for sim in simulation_list]

                # Use imap_unordered so results arrive as soon as they're ready
                for (sim_metadata, episodes) in pool.imap_unordered(_run_one_sim, tasks):
                    self.storage.store_simulation(sim_metadata, episodes)
                    self.storage.close()
                    print(f"Stored simulation {sim_metadata} with {len(episodes)} episodes.")
                


        print("All simulations completed and stored.")
        return
