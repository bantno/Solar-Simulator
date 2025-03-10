import matplotlib.pyplot as plt
import numpy as np
from BaseClasses.simulation_storage import SimulationStorage

def plot_episode(episode):
    """
    Plot an episode's trajectory, actions, and rewards on separate subplots.
    
    Parameters:
        episode (dict): An episode dictionary containing keys:
                        - 'trajectory': list/array of states (each a 1D array)
                        - 'actions': list of actions taken
                        - 'rewards': list of rewards received
                        - 'metadata': metadata dictionary (for display purposes)
                        - 'total_reward': total reward (optional)
    """
    # Convert trajectory to a NumPy array for easier indexing.
    trajectory = np.array(episode['trajectory'])
    actions = episode['actions']
    rewards = episode['rewards']
    total_reward = episode.get('total_reward', sum(rewards))
    metadata = episode.get('metadata', {})

    # Create a figure with three subplots.
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # --- Trajectory Plot ---
    # Plot each dimension of the state.
    num_dims = trajectory.shape[1]
    for dim in range(num_dims):
        axs[0].plot(trajectory[:, dim], label=f"State dim {dim}")
    axs[0].set_title("Trajectory")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("State Value")
    axs[0].legend()
    
    # --- Actions Plot ---
    # Using a step plot to better show discrete actions.
    axs[1].step(range(len(actions)), actions, where='mid')
    axs[1].set_title("Actions")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Action")
    
    # --- Rewards Plot ---
    axs[2].plot(range(len(rewards)), rewards, marker='o')
    axs[2].set_title("Rewards")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Reward")
    
    # Add an overall title using the total reward and metadata.
    sim_type = metadata.get('simulation_type', 'Unknown')
    obs_thr = metadata.get('observation_threshold', 'N/A')
    wind_thr = metadata.get('wind_threshold', 'N/A')
    
    fig.suptitle(
        f"Episode Plot: {sim_type} | Obs Thresh: {obs_thr} | Wind Thresh: {wind_thr}\n"
        f"Total Reward: {total_reward}",
        fontsize=16
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_specific_episode(storage_dir, simulation_id, episode_index):
    """
    Loads a simulation run from a directory and plots the specified episode.

    Parameters:
        storage_dir (str): Directory where simulation files are stored.
        simulation_id (int): The simulation counter (ID) used in the filename.
        episode_index (int): The index of the episode within that simulation to plot.
    """
    storage = SimulationStorage(storage_dir)
    # Using a method name consistent with your existing pattern:
    sim_data = storage.load_simulation_by_id(storage_dir, simulation_id)
    episodes = sim_data['episodes']
    
    if episode_index < 0 or episode_index >= len(episodes):
        print("Episode index out of range!")
        return
    
    episode = episodes[episode_index]
    plot_episode(episode)


# --------------------------- NEW CODE FOR MULTI-EPISODE PLOTS ---------------------------

def plot_multiple_episodes(episodes, episode_indices=None):
    """
    Plot multiple episodes (potentially from different simulations) on the same figure.
    """
    if episode_indices is None:
        # If not specified, we default to all episodes in the list
        episode_indices = range(len(episodes))

    fig, axs = plt.subplots(6, 1, figsize=(10, 12))
    axs[0].set_title("Trajectory")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("State Value")
    
    axs[1].set_title("Actions")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Action")
    
    axs[2].set_title("Rewards")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Reward")

    axs[3].set_title("Solar")
    axs[3].set_xlabel("Time Step")
    axs[3].set_ylabel("Unit")

    axs[4].set_title("Wind")
    axs[4].set_xlabel("Time Step")
    axs[4].set_ylabel("Unit")

    axs[5].set_title("Whale")
    axs[5].set_xlabel("Time Step")
    axs[5].set_ylabel("Unit")

    
    for i in episode_indices:
        if i < 0 or i >= len(episodes):
            print(f"Skipping invalid episode index {i}.")
            continue
        
        ep = episodes[i]
        traj = np.array(ep['trajectory'])
        actions = ep['actions']
        rewards = ep['rewards']
        total_reward = ep.get('total_reward', sum(rewards))
        solar = ep['solar_series']
        wind = ep['wind_series']
        whale = ep['whale_series']


        meta = ep.get('metadata', {})
        sim_type = meta.get('simulation_type', 'Unknown')
        obs_thr = meta.get('observation_threshold', 'N/A')
        wind_thr = meta.get('wind_threshold', 'N/A')
        
        # Build a label to distinguish each episode
        label = (f"SimType={sim_type}, Obs={obs_thr}, Wind={wind_thr}, "
                 f"EpIndex={meta.get('episode_index', 'N/A')}, "
                 f"TotReward={total_reward}")
        
        # Plot trajectory
        if traj.ndim == 2:  # shape (time, dimension)
            num_dims = traj.shape[1]
            for dim in range(1):
                axs[0].plot(traj[:, dim], label=f"{label}, dim {dim}")
        else:
            axs[0].plot(traj, label=label)
        
        # Plot actions
        axs[1].step(range(len(actions)), actions, where='mid', label=label)
        
        # Plot rewards
        axs[2].plot(range(len(rewards)), rewards, marker='o', label=label)

        # Plot solar
        axs[3].plot(range(len(solar)), solar, marker='o', label=label)

        # Plot wind
        axs[4].plot(range(len(wind)), wind, marker='o', label=label)

        # Plot whale
        axs[5].plot(range(len(whale)), whale, marker='o', label=label)

    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.suptitle("Comparing Episodes Across Simulations", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_multiple_episodes_from_simulation(storage_dir, simulation_id, episode_indices=None):
    """
    Load a simulation run by ID, then plot multiple episodes (by index) on the same figure.
    
    Parameters:
        storage_dir (str): Directory with simulation files
        simulation_id (int): The simulation counter (ID) in the filename
        episode_indices (list of int): Which episodes to plot. If None, plots all.
    """
    storage = SimulationStorage(storage_dir)
    sim_data = storage.load_simulation_by_id(storage_dir, simulation_id)
    episodes = sim_data['episodes']

    if not episodes:
        print("No episodes found in this simulation.")
        return

    # If no specific indices given, plot all episodes
    if episode_indices is None:
        episode_indices = list(range(len(episodes)))
    
    plot_multiple_episodes(episodes, episode_indices)

def compare_episodes_across_simulations(storage_dir, sims_and_episodes=None):
    """
    Loads multiple simulations from the same directory and plots selected episodes
    from each of them on the same figure.

    Parameters:
        storage_dir (str): Directory where simulation files are stored.
        sims_and_episodes (dict or list of tuples):
            A structure indicating which simulation IDs to load, and which episodes.
            For example:
                {
                    0: [0, 1],     # load simulation ID=0, plot episodes #0 and #1
                    1: [2],        # load simulation ID=1, plot episode #2 only
                    2: None        # load simulation ID=2, plot ALL episodes
                }
            or a list of tuples like:
                [(0, [0,1]), (1, [2]), (2, None)]
            If an entry is None or missing, it means "plot all episodes."
    """
    if sims_and_episodes is None:
        print("No simulations/episodes specified.")
        return

    # Convert dict to list of (sim_id, ep_list) pairs if needed.
    if isinstance(sims_and_episodes, dict):
        sims_and_episodes = list(sims_and_episodes.items())

    storage = SimulationStorage(storage_dir)
    combined_episodes = []  # We'll store episodes from all sims here.

    for sim_id, ep_indices in sims_and_episodes:
        sim_data = storage.load_simulation_by_id(storage_dir, sim_id)
        episodes = sim_data['episodes']
        if not episodes:
            print(f"No episodes found for simulation {sim_id}. Skipping.")
            continue
        
        if ep_indices is None:
            # Means "use all episodes in this simulation"
            ep_indices = list(range(len(episodes)))
        
        # Collect requested episodes
        for ei in ep_indices:
            if ei < 0 or ei >= len(episodes):
                print(f"Invalid episode index {ei} for simulation {sim_id}. Skipping.")
                continue
            combined_episodes.append(episodes[ei])

    if not combined_episodes:
        print("No valid episodes found across the specified simulations.")
        return

    # Now we have a single combined list of episodes, each from potentially different simulations.
    # Plot them using the multi-episode routine.
    plot_multiple_episodes(combined_episodes)

# --------------------------- EXAMPLE USAGE ---------------------------
# 1) Plot a single episode from simulation run #0
# plot_specific_episode(storage_dir="simulation_results", simulation_id=0, episode_index=0)

# 2) Plot multiple episodes (indices 0, 1, and 2) from simulation run #0
# plot_multiple_episodes_from_simulation(storage_dir="simulation_results", simulation_id=0, episode_indices=[0,1,2])

# --------------------------- EXAMPLE USAGE ---------------------------
# Suppose you want to compare:
#   - Simulation #0, episodes #0 and #1
#   - Simulation #1, episode #2
#   - Simulation #2, all episodes (None)
usage_example = {
    5: [0],
    10: [0]
}
compare_episodes_across_simulations(storage_dir="simulation_results", sims_and_episodes=usage_example)