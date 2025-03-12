import os
import numpy as np
from datetime import datetime

class SimulationStorage:
    """
    Manages simulation result storage by grouping all episodes from a single simulation run
    into one compressed .npz file. A unique timestamp is added so that each file is unique.
    """
    
    def __init__(self, storage_dir: str):
        """
        Parameters:
            storage_dir (str): Directory where simulation files will be stored.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        # Create a unique timestamp for this storage instance.
        self.run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.simulation_counter = 0
    
    def _get_simulation_filename(self, simulation_id: int) -> str:
        """
        Generate a filename that includes the unique timestamp and simulation counter.
        """
        return os.path.join(self.storage_dir, f"simulation_{self.run_timestamp}_{simulation_id:04d}.npz")
    
    def store_simulation(self, simulation_metadata: dict, episodes: list):
        """
        Stores all episodes from a single simulation run in one file.
        
        Parameters:
            simulation_metadata (dict): Metadata describing the simulation run.
            episodes (list): A list of episode dictionaries.
        """
        filename = self._get_simulation_filename(self.simulation_counter)
        print("Storing Simulation Data")
        np.savez_compressed(filename, simulation_metadata=simulation_metadata, episodes=episodes)
        self.simulation_counter += 1
    
    def load_simulation(self, simulation_id: int) -> dict:
        """
        Loads a single simulation file by simulation id.
        
        Returns:
            dict: A dictionary with keys 'simulation_metadata' and 'episodes'.
        """
        filename = self._get_simulation_filename(simulation_id)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Simulation file {filename} not found.")
        with np.load(filename, allow_pickle=True) as data:
            simulation_metadata = data['simulation_metadata'].item() if isinstance(data['simulation_metadata'], np.ndarray) else data['simulation_metadata']
            episodes = data['episodes'].tolist() if isinstance(data['episodes'], np.ndarray) else data['episodes']
            return {'simulation_metadata': simulation_metadata, 'episodes': episodes}
    
    def load_simulation_by_id(self,storage_dir, simulation_id):
        simulation_files = sorted([f for f in os.listdir(storage_dir) if f.startswith("simulation_") and f.endswith(".npz")])
        if simulation_id < 0 or simulation_id >= len(simulation_files):
            raise FileNotFoundError(f"No simulation file with id {simulation_id} exists.")
        filename = os.path.join(storage_dir, simulation_files[simulation_id])
        with np.load(filename, allow_pickle=True) as data:
            simulation_metadata = data['simulation_metadata'].item() if isinstance(data['simulation_metadata'], np.ndarray) else data['simulation_metadata']
            episodes = data['episodes'].tolist() if isinstance(data['episodes'], np.ndarray) else data['episodes']
            return {'simulation_metadata': simulation_metadata, 'episodes': episodes}
    
    def load_all_simulations(self) -> list:
        """
        Loads all simulation files from the storage directory.
        
        Returns:
            list: A list of dictionaries, each containing simulation metadata and episodes.
        """
        simulation_files = sorted([f for f in os.listdir(self.storage_dir) if f.startswith("simulation_") and f.endswith(".npz")])
        all_simulations = []
        for filename in simulation_files:
            full_path = os.path.join(self.storage_dir, filename)
            with np.load(full_path, allow_pickle=True) as data:
                simulation_metadata = data['simulation_metadata'].item() if isinstance(data['simulation_metadata'], np.ndarray) else data['simulation_metadata']
                episodes = data['episodes'].tolist() if isinstance(data['episodes'], np.ndarray) else data['episodes']
                all_simulations.append({'simulation_metadata': simulation_metadata, 'episodes': episodes})
        return all_simulations
