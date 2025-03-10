import os
import numpy as np
from glob import glob

class SimulationStorage:
    """
    Manages simulation results storage using batched writes.
    
    Episodes are buffered in memory until the batch size is reached, then all are saved together 
    in a compressed npz file. This reduces the frequency of disk writes and improves overall efficiency.
    """
    
    def __init__(self, storage_dir: str, batch_size: int = 10):
        """
        Parameters:
            storage_dir (str): Directory where batch files will be stored.
            batch_size (int): Number of episodes to buffer before writing to disk.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.batch_size = batch_size
        self.buffer = []  # Buffer to accumulate episodes.
        self.batch_counter = 0  # Tracks the number of batch files created.
        self.total_episode_counter = 0  # Total episodes stored across all batches.
    
    def store_episode(self, episode_data: dict):
        """
        Buffer an episode and flush if the batch size is reached.
        
        Parameters:
            episode_data (dict): Dictionary containing simulation results 
                                 (e.g., trajectory, actions, rewards).
        """
        self.buffer.append(episode_data)
        if len(self.buffer) >= self.batch_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """
        Write all buffered episodes to a single compressed npz file.
        
        The file is named using the current batch counter, and each episode is stored
        as a separate entry in the npz archive.
        """
        if not self.buffer:
            return  # Nothing to flush.
        filename = os.path.join(self.storage_dir, f"batch_{self.batch_counter:04d}.npz")
        # Create a dict with keys as episode identifiers.
        episodes_to_save = {
            f"episode_{self.total_episode_counter + i:04d}": ep 
            for i, ep in enumerate(self.buffer)
        }
        np.savez_compressed(filename, **episodes_to_save)
        self.total_episode_counter += len(self.buffer)
        self.batch_counter += 1
        self.buffer = []  # Clear the buffer.
    
    def load_batch(self, batch_number: int) -> dict:
        """
        Load a specific batch file and return its stored episodes.
        
        Parameters:
            batch_number (int): The batch number to load.
        
        Returns:
            dict: A dictionary mapping episode keys to episode data.
        """
        filename = os.path.join(self.storage_dir, f"batch_{batch_number:04d}.npz")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Batch file {filename} not found.")
        data = np.load(filename, allow_pickle=True)
        # Each stored episode was a dictionary; retrieve the original object.
        return {key: data[key].item() for key in data.files}
    
    def load_all_episodes(self) -> list:
        """
        Flush any remaining buffered episodes, then load all episodes from storage.
        
        Returns:
            list: A list of all stored episode data dictionaries.
        """
        # Flush remaining episodes before loading.
        self.flush_buffer()
        batch_files = sorted(glob(os.path.join(self.storage_dir, "batch_*.npz")))
        episodes = []
        for file in batch_files:
            data = np.load(file, allow_pickle=True)
            for key in sorted(data.files):
                episodes.append(data[key].item())
        return episodes
