import os
import numpy as np
import h5py
import json
from datetime import datetime
import h5py

class SimulationStorage:
    """Manages simulation result storage by grouping all episodes from a single simulation run
    into one compressed .npz file. A unique timestamp is added so that each file is unique.
    """
    
    def __init__(self, storage_dir: str):
        """Args:
            storage_dir (str): Directory where simulation files will be stored.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        # Create a unique timestamp for this storage instance.
        self.run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.simulation_counter = 0
    
    def _get_simulation_filename(self, simulation_id: int) -> str:
        """Generate a filename that includes the unique timestamp and simulation counter."""
        return os.path.join(self.storage_dir, f"simulation_{self.run_timestamp}_{simulation_id:04d}.npz")
    
    def store_simulation(self, simulation_metadata: dict, episodes: list):
        """Stores all episodes from a single simulation run in one file.
        
        Args:
            simulation_metadata (dict): Metadata describing the simulation run.
            episodes (list): A list of episode dictionaries.
        """
        filename = self._get_simulation_filename(self.simulation_counter)
        print("Storing Simulation Data")
        np.savez_compressed(filename, simulation_metadata=simulation_metadata, episodes=episodes)
        self.simulation_counter += 1
    
    def load_simulation(self, simulation_id: int) -> dict:
        """Loads a single simulation file by simulation id.
        
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
        """Loads all simulation files from the storage directory.
        
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


class SimulationStorageHDF5:
    """HDF5-based storage for batches of simulations.
    Each simulation is stored as a top-level group within one file:

        <group>/attrs                    simulation metadata
        <group>/episode_scalars/<field>  per-episode scalars as one 1-D
                                         (episodes,) dataset per field
        <group>/episodes/episode <i>     one group per full-history episode,
                                         holding its array fields (wind_series,
                                         trajectory, ...)

    Per-episode scalars are stored as columns rather than one dataset per
    episode: every HDF5 object costs ~0.5 KB of file metadata, so the old
    group-per-episode layout made a scalars-only 210-sim x 3000-episode run
    ~1.6 GB for ~40 MB of actual content. Episodes without array fields get
    no group at all.
    """
    def __init__(self, file_path: str):
        """Open the HDF5 file in append mode (creates if not exists)."""
        self.h5file = h5py.File(file_path, "a")

    def store_simulation(self,
                        sim_metadata: dict,
                        episodes: list,
                        group_name: str = None):
        """Store a batch of episodes under a unique top-level group,
        using lzf compression for array datasets.

        Args:
            sim_metadata: dict of simple metadata entries for the simulation
            episodes: list of dicts, each dict represents one episode
            group_name: mandatory name for the group identifying this simulation

        Raises:
            ValueError: if group_name missing or already exists.
        """
        if not group_name:
            raise ValueError("group_name must be provided to store_simulation")
        if group_name in self.h5file:
            raise ValueError(f"Simulation group '{group_name}' already exists.")

        # Create simulation group + write metadata as attributes
        grp = self.h5file.create_group(group_name)
        for key, val in sim_metadata.items():
            try:
                grp.attrs[key] = val
            except TypeError:
                grp.attrs[key] = str(val)

        n_eps = len(episodes)
        scalar_cols = {}
        eps_grp = grp.create_group("episodes")

        for pos, ep in enumerate(episodes, start=1):
            meta = ep.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            scalars = {}
            arrays = {}
            for field, data in ep.items():
                if field == "metadata":
                    continue
                arr = np.asarray(data)
                if arr.ndim == 0 and arr.dtype != object:
                    scalars[field] = arr.item()
                else:
                    arrays[field] = arr
            for mkey, mval in meta.items():
                if isinstance(mval, (bool, int, float, np.generic)):
                    scalars.setdefault(mkey, mval)
            scalars.setdefault("episode_index", pos - 1)

            for field, val in scalars.items():
                scalar_cols.setdefault(field, [None] * n_eps)[pos - 1] = val

            if not arrays:
                continue

            # Full-history episode: keep the per-episode group for its arrays.
            ep_grp = eps_grp.create_group(f"episode {pos}")
            for mkey, mval in meta.items():
                try:
                    ep_grp.attrs[mkey] = mval
                except TypeError:
                    ep_grp.attrs[mkey] = str(mval)

            for field, arr in arrays.items():
                # Object-dtype: JSON-encode to bytes
                if arr.dtype == object:
                    str_data = np.array([json.dumps(x) for x in arr.ravel()], dtype="S")
                    # reshape back to original shape
                    str_data = str_data.reshape(arr.shape)
                    ep_grp.create_dataset(
                        field,
                        data=str_data,
                        compression="lzf",
                        shuffle=True,
                        # a simple chunk along first axis
                        chunks=(min(arr.shape[0], 1024),) + arr.shape[1:]
                    )
                    continue

                # Numeric arrays: choose a chunk size on the first axis
                # to keep each chunk ≲1 MB
                item_bytes = arr.dtype.itemsize
                max_elems = max(1, (1024 * 1024) // item_bytes)
                chunk_len = min(arr.shape[0], max_elems)
                chunk_shape = (chunk_len,) + arr.shape[1:] if arr.ndim > 1 else (chunk_len,)

                ep_grp.create_dataset(
                    field,
                    data=arr,
                    compression="lzf",
                    shuffle=True,
                    chunks=chunk_shape
                )

        sc_grp = grp.create_group("episode_scalars")
        for field, vals in scalar_cols.items():
            if any(v is None for v in vals):
                col = np.asarray([np.nan if v is None else v for v in vals],
                                 dtype=float)
            else:
                col = np.asarray(vals)
            if col.dtype == object:
                # Non-numeric scalar field: not representable as a column.
                continue
            sc_grp.create_dataset(field, data=col)

    def close(self):
        """Close the underlying HDF5 file."""
        self.h5file.close()