import os
import numpy as np
import h5py
import json
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



import os
import h5py
import numpy as np
from datetime import datetime


class SimulationStorageHDF5:
    """
    Stores simulation runs in individual HDF5 files. Filenames are auto-generated
    based on provided metadata and a timestamp for uniqueness.

    Usage:
        storage = SimulationStorageHDF5('path/to/storage')
        storage.store_simulation(sim_meta, episodes)
        storage.close()
    """

    def __init__(self, storage_dir: str):
        """
        Initialize storage directory and internal counters.

        Args:
            storage_dir (str): Directory where HDF5 files will be saved.
        """
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.simulation_counter = 0
        # Base timestamp for this run (will vary per file if multiple store calls)
        self.run_base_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_file = None

    def _generate_filename(self, simulation_metadata: dict) -> str:
        """
        Create a filename using metadata fields and timestamp.
        Example: ObservationThresholdContinuousSimulation_10ep_400Ah_0.0obs_10.0wind_20250418-153000_0001.hdf5
        """
        mtype = simulation_metadata.get('simulation_type', 'sim')
        epc = simulation_metadata.get('episodes_count', self.simulation_counter)
        cap = simulation_metadata.get('battery_capacity', 'cap')
        obs = simulation_metadata.get('observation_threshold', 'obs')
        wind = simulation_metadata.get('wind_threshold', 'wind')
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        idx = f"{self.simulation_counter:04d}"
        fname = f"{mtype}_{epc}ep_{cap}Ah_{obs}obs_{wind}wind_{timestamp}_{idx}.hdf5"
        return os.path.join(self.storage_dir, fname)

    def _open_file(self, filepath: str):
        """
        Opens (or reopens) the HDF5 file for writing.
        """
        # Close previous if open
        if self.data_file:
            self.data_file.close()
        # Remove existing and open new
        if os.path.exists(filepath):
            os.remove(filepath)
        self.data_file = h5py.File(filepath, 'w', libver='earliest')

    def close(self):
        """
        Closes the HDF5 file if open.
        """
        if self.data_file:
            self.data_file.close()
            self.data_file = None

    def save_dataset(self, grp, data_name: str, data_values: np.ndarray):
        """
        Saves or appends data to a dataset under a given group.
        """
        if data_name in grp:
            ds = grp[data_name]
            old_shape = ds.shape
            new_len = old_shape[0] + data_values.shape[0]
            ds.resize((new_len,) + old_shape[1:])
            ds[old_shape[0]:] = data_values
        else:
            maxshape = (None,) + data_values.shape[1:]
            grp.create_dataset(
                data_name,
                data=data_values,
                maxshape=maxshape,
                chunks=True,
                compression='gzip',
                compression_opts=4
            )

    def store_simulation(self, simulation_metadata: dict, episodes: list):
        """
        Writes a simulation run to its own HDF5 file:

        - Filename auto-generated from metadata and timestamp
        - Root attributes store simulation_metadata keys
        - '/episodes' group contains one subgroup per episode
        - Episode metadata stored as group attributes, data as datasets
        """
        # Prepare file
        filepath = self._generate_filename(simulation_metadata)
        self._open_file(filepath)
        # Write global metadata
        for key, val in simulation_metadata.items():
            self.data_file.attrs[key] = val
        # Episodes container
        ep_root = self.data_file.require_group('episodes')
        # Write each episode
        for idx, ep in enumerate(episodes):
            grp = ep_root.require_group(f'episode_{idx:04d}')
            # Episode metadata
            meta = ep.get('metadata', {})
            for mkey, mval in meta.items():
                grp.attrs[mkey] = mval
            if 'total_reward' in ep:
                grp.attrs['total_reward'] = ep['total_reward']
            # Time-series and arrays
            for key, vals in ep.items():
                if key in ('metadata', 'total_reward'):
                    continue
                arr = np.array(vals)
                if arr.ndim == 0:
                    grp.attrs[key] = arr.item()
                else:
                    self.save_dataset(grp, key, arr)
        for key, val in simulation_metadata.items():
            ep_root.attrs[key] = val
        
        # Flush and increment
        self.data_file.flush()
        self.simulation_counter += 1

# Example:
# storage = SimulationStorageHDF5('Data/EXPECTED_DATA')
# storage.store_simulation(sim_meta, episodes)
# storage.close()
