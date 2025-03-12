from BaseClasses.simulation_storage import SimulationStorage

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
        self.storage = SimulationStorage(storage_dir)
    
    def run_simulations(self, simulation_list: list):
        """
        Runs each simulation in the provided list, collects its episodes, enriches metadata,
        and stores all episodes for that simulation run together in one file.
        
        Parameters:
            simulation_list (list): A list of simulation objects that implement
                                    simulate_multiple_episodes(num_episodes).
        """
        for sim in simulation_list:
            episodes = []
            for episode in sim.simulate_multiple_episodes(self.episodes_per_simulation):
                # Enrich each episode's metadata with generic simulation information.
                metadata = {
                    "simulation_type": sim.__class__.__name__,
                }
                # Optionally add simulation-specific parameters if they exist.
                if hasattr(sim, "observation_threshold"):
                    metadata["observation_threshold"] = sim.observation_threshold
                if hasattr(sim, "wind_threshold"):
                    metadata["wind_threshold"] = sim.wind_threshold
                # Update the episode's metadata (the generator already set an episode_index).
                episode["metadata"].update(metadata)
                # Compute and add the total reward.
                episode["total_reward"] = episode.get("total_reward",sum(episode.get("rewards", [])))
                episodes.append(episode)
            
            # Create simulation-level metadata.
            simulation_metadata = {
                "simulation_type": sim.__class__.__name__,
                "episodes_count": len(episodes)
            }
            if hasattr(sim, "observation_threshold"):
                simulation_metadata["observation_threshold"] = sim.observation_threshold
            if hasattr(sim, "wind_threshold"):
                simulation_metadata["wind_threshold"] = sim.wind_threshold
            
            # Store all episodes from this simulation run together as a batch.
            self.storage.store_simulation(simulation_metadata, episodes)
            print(f"Stored simulation {simulation_metadata} with {len(episodes)} episodes.")
