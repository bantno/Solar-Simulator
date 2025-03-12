import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from BaseClasses.simulation_storage import SimulationStorage

class SimulationPlotter:
    def __init__(self, storage_dir=None):
        """
        Initialize the plotter.
        
        Parameters:
            storage_dir (str): Optional default directory where simulation files are stored.
        """
        self.storage_dir = storage_dir

    def plot_episode(self, episode):
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
        num_dims = trajectory.shape[1]
        for dim in range(num_dims):
            axs[0].plot(trajectory[:, dim], label=f"State dim {dim}")
        axs[0].set_title("Trajectory")
        axs[0].set_xlabel("Time Step")
        axs[0].set_ylabel("State Value")
        axs[0].legend()
        
        # --- Actions Plot ---
        axs[1].step(range(len(actions)), actions, where='mid')
        axs[1].set_title("Actions")
        axs[1].set_xlabel("Time Step")
        axs[1].set_ylabel("Action")
        
        # --- Rewards Plot ---
        axs[2].plot(range(len(rewards)), rewards, marker='o')
        axs[2].set_title("Rewards")
        axs[2].set_xlabel("Time Step")
        axs[2].set_ylabel("Reward")
        
        # Add an overall title using metadata.
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

    def plot_specific_episode(self, simulation_id, episode_index, storage_dir=None):
        """
        Loads a simulation run from a directory and plots the specified episode.
        
        Parameters:
            simulation_id (int): The simulation counter (ID) used in the filename.
            episode_index (int): The index of the episode within that simulation to plot.
            storage_dir (str): Directory where simulation files are stored. If not provided, self.storage_dir is used.
        """
        if storage_dir is None:
            if self.storage_dir is None:
                print("Storage directory must be provided either in constructor or as argument.")
                return
            else:
                storage_dir = self.storage_dir
        storage = SimulationStorage(storage_dir)
        sim_data = storage.load_simulation_by_id(storage_dir, simulation_id)
        episodes = sim_data['episodes']
        
        if episode_index < 0 or episode_index >= len(episodes):
            print("Episode index out of range!")
            return
        
        episode = episodes[episode_index]
        self.plot_episode(episode)

    def plot_multiple_episodes(self, episodes, episode_indices=None):
        """
        Plot multiple episodes (potentially from different simulations) on the same figure.
        
        Parameters:
            episodes (list): List of episode dictionaries.
            episode_indices (list): List of indices to plot. If None, all episodes are plotted.
        """
        if episode_indices is None:
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
            
            # Build a label to distinguish each episode.
            label = (f"SimType={sim_type}, Obs={obs_thr}, Wind={wind_thr}, "
                     f"EpIndex={meta.get('episode_index', 'N/A')}, TotReward={total_reward}")
            
            # Plot trajectory (for the first state dimension only).
            if traj.ndim == 2:
                axs[0].plot(traj[:, 0], label=f"{label}, dim 0")
            else:
                axs[0].plot(traj, label=label)
            
            axs[1].step(range(len(actions)), actions, where='mid', label=label)
            axs[2].plot(range(len(rewards)), rewards, marker='o', label=label)
            axs[3].plot(range(len(solar)), solar, marker='o', label=label)
            axs[4].plot(range(len(wind)), wind, marker='o', label=label)
            axs[5].plot(range(len(whale)), whale, marker='o', label=label)
        
        fig.suptitle("Comparing Episodes Across Simulations", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_multiple_episodes_from_simulation(self, simulation_id, episode_indices=None, storage_dir=None):
        """
        Load a simulation run by ID, then plot multiple episodes (by index) on the same figure.
        
        Parameters:
            simulation_id (int): The simulation counter (ID) in the filename.
            episode_indices (list): Which episodes to plot. If None, plots all.
            storage_dir (str): Directory where simulation files are stored. If not provided, self.storage_dir is used.
        """
        if storage_dir is None:
            if self.storage_dir is None:
                print("Storage directory must be provided either in constructor or as argument.")
                return
            else:
                storage_dir = self.storage_dir
        storage = SimulationStorage(storage_dir)
        sim_data = storage.load_simulation_by_id(storage_dir, simulation_id)
        episodes = sim_data['episodes']

        if not episodes:
            print("No episodes found in this simulation.")
            return

        if episode_indices is None:
            episode_indices = list(range(len(episodes)))
        
        self.plot_multiple_episodes(episodes, episode_indices)

    def compare_episodes_across_simulations(self, sims_and_episodes=None, storage_dir=None):
        """
        Loads multiple simulations from the same directory and plots selected episodes
        from each of them on the same figure.
        
        Parameters:
            sims_and_episodes (dict or list of tuples):
                A structure indicating which simulation IDs to load, and which episodes.
                For example:
                    {
                        0: [0, 1],     # load simulation ID=0, plot episodes #0 and #1
                        1: [2],        # load simulation ID=1, plot episode #2 only,
                        2: None        # load simulation ID=2, plot ALL episodes
                    }
            storage_dir (str): Directory where simulation files are stored. If not provided, self.storage_dir is used.
        """
        if sims_and_episodes is None:
            print("No simulations/episodes specified.")
            return

        if storage_dir is None:
            if self.storage_dir is None:
                print("Storage directory must be provided either in constructor or as argument.")
                return
            else:
                storage_dir = self.storage_dir

        if isinstance(sims_and_episodes, dict):
            sims_and_episodes = list(sims_and_episodes.items())

        storage = SimulationStorage(storage_dir)
        combined_episodes = []  # Collect episodes from all specified simulations.

        for sim_id, ep_indices in sims_and_episodes:
            sim_data = storage.load_simulation_by_id(storage_dir, sim_id)
            episodes = sim_data['episodes']
            if not episodes:
                print(f"No episodes found for simulation {sim_id}. Skipping.")
                continue
            
            if ep_indices is None:
                ep_indices = list(range(len(episodes)))
            
            for ei in ep_indices:
                if ei < 0 or ei >= len(episodes):
                    print(f"Invalid episode index {ei} for simulation {sim_id}. Skipping.")
                    continue
                combined_episodes.append(episodes[ei])

        if not combined_episodes:
            print("No valid episodes found across the specified simulations.")
            return

        self.plot_multiple_episodes(combined_episodes)

    def plot_reward_distribution(self, sim_results, alpha=0.5, jitter=0.1):
        """
        Create a scatter plot of the total reward for each episode for each simulation.
        
        Parameters:
            sim_results (dict): A dictionary where keys are simulation IDs (or names) and values are lists of episodes.
                                Each episode is expected to be a dictionary containing a 'total_reward' key.
            alpha (float): The opacity of the scatter points (default 0.5, less than 1 for transparency).
            jitter (float): Horizontal jitter added to each simulation's x position to reduce overlap.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for sim_id, episodes in sim_results.items():
            total_rewards = []
            for ep in episodes:
                if 'total_reward' in ep:
                    total_rewards.append(ep['total_reward'])
                else:
                    total_rewards.append(sum(ep.get('rewards', [])))
            
            # Create x positions for these episodes (using the simulation id) with jitter.
            x_positions = np.ones(len(total_rewards)) * sim_id + np.random.uniform(-jitter, jitter, len(total_rewards))
            ax.scatter(x_positions, total_rewards, alpha=alpha, label=f"Sim {sim_id}")
        
        ax.set_xlabel("Simulation ID")
        ax.set_ylabel("Total Reward")
        ax.set_title("Distribution of Total Rewards per Episode for Each Simulation")
        ax.legend()
        plt.show()

    def load_and_plot_reward_distribution(self, simulation_ids, alpha=0.5, jitter=0.1, storage_dir=None):
        """
        Loads episodes from the specified simulation IDs (from a directory) and plots the reward distribution.
        
        Parameters:
            simulation_ids (list): List of simulation IDs to load.
            alpha (float): Opacity for scatter points (default 0.5).
            jitter (float): Horizontal jitter (default 0.1).
            storage_dir (str): Directory where simulation files are stored. If not provided, self.storage_dir is used.
        """
        if storage_dir is None:
            if self.storage_dir is None:
                print("Storage directory must be provided either in constructor or as argument.")
                return
            else:
                storage_dir = self.storage_dir

        storage = SimulationStorage(storage_dir)
        sim_results = {}
        for sim_id in simulation_ids:
            sim_data = storage.load_simulation_by_id(storage_dir, sim_id)
            episodes = sim_data.get('episodes', [])
            if episodes:
                sim_results[sim_id] = episodes
            else:
                print(f"No episodes found for simulation {sim_id}.")
        
        if not sim_results:
            print("No valid simulation results loaded.")
            return

        self.plot_reward_distribution(sim_results, alpha=alpha, jitter=jitter)

    def plot_reward_histogram_subplots(self, sim_results, alpha=0.5, bins=100):
        """
        Create a histogram of the total rewards for each simulation on separate subplots.

        Parameters:
            sim_results (dict): A dictionary where keys are simulation IDs (or names) and values are lists of episodes.
                                Each episode should be a dictionary containing a 'total_reward' key (or a 'rewards' list).
            alpha (float): Opacity of the histogram bars (default 0.5).
            bins (int): Number of bins in the histogram (default 20).
        """
        sim_ids = sorted(sim_results.keys())
        num_sims = len(sim_ids)
        fig, axs = plt.subplots(num_sims, 1, figsize=(10, 4 * num_sims), sharex=True)
        # Ensure axs is iterable even for one simulation.
        if num_sims == 1:
            axs = [axs]

        for ax, sim_id in zip(axs, sim_ids):
            total_rewards = []
            for ep in sim_results[sim_id]:
                if 'total_reward' in ep:
                    total_rewards.append(ep['total_reward'])
                else:
                    total_rewards.append(sum(ep.get('rewards', [])))
            ax.hist(total_rewards, bins=bins, alpha=alpha, label=f"Sim {sim_id}")
            ax.set_ylim([0,5000])
            ax.set_title(f"Histogram of Total Rewards - Simulation {sim_id}")
            ax.set_ylabel("Frequency")
            ax.legend()
        axs[-1].set_xlabel("Total Reward")
        plt.tight_layout()
        plt.show()


    def load_and_plot_reward_histogram_subplots(self, simulation_ids, alpha=0.5, bins=20, storage_dir=None):
        """
        Loads episodes from the specified simulation IDs (from a directory) and plots a histogram
        of the reward distribution for each simulation on separate subplots.

        Parameters:
            simulation_ids (list): List of simulation IDs to load.
            alpha (float): Opacity for histogram bars (default 0.5).
            bins (int): Number of bins in the histogram (default 20).
            storage_dir (str): Directory where simulation files are stored. If not provided, self.storage_dir is used.
        """
        if storage_dir is None:
            if self.storage_dir is None:
                print("Storage directory must be provided either in constructor or as argument.")
                return
            else:
                storage_dir = self.storage_dir

        storage = SimulationStorage(storage_dir)
        sim_results = {}
        for sim_id in simulation_ids:
            sim_data = storage.load_simulation_by_id(storage_dir, sim_id)
            episodes = sim_data.get('episodes', [])
            if episodes:
                sim_results[sim_id] = episodes
            else:
                print(f"No episodes found for simulation {sim_id}.")
        
        if not sim_results:
            print("No valid simulation results loaded.")
            return

        self.plot_reward_histogram_subplots(sim_results, alpha=alpha, bins=bins)

    
    # def plot_reward_violin(self, sim_results):
    #     """
    #     Create a violin plot of the total rewards for each simulation using Plotly.
    #     """
    #     import plotly.graph_objects as go
        
    #     # Sort simulation ids for consistent x-axis ordering
    #     sim_ids = sorted(sim_results.keys())
        
    #     # Create the figure
    #     fig = go.Figure()
        
    #     # Iterate over simulation IDs and add a violin trace for each
    #     for sim_id in sim_ids:
    #         total_rewards = []
    #         for ep in sim_results[sim_id]:
    #             if 'total_reward' in ep:
    #                 total_rewards.append(ep['total_reward'])
    #             else:
    #                 total_rewards.append(sum(ep.get('rewards', [])))
            
    #         # Add the trace for this simulation
    #         fig.add_trace(go.Violin(
    #             y=total_rewards,
    #             name=str(sim_id),
    #             box_visible=True,        # Show box plot inside the violin
    #             meanline_visible=True,   # Show mean line inside the violin
    #             points="all",            # Display all individual data points
    #             opacity=0.7              # Set transparency
    #         ))
        
    #     # Update layout with titles and labels
    #     fig.update_layout(
    #         title="Violin Plot of Total Rewards per Simulation",
    #         xaxis_title="Simulation ID",
    #         yaxis_title="Total Reward",
    #         violingap=0.2,     # Gap between violins
    #         violinmode='overlay'  # Overlay violins; you can also use "group" for side-by-side
    #     )
        
    #     fig.show()
    def plot_reward_violin(self, sim_results):
        """
        Create a violin plot of the total rewards for each simulation using Plotly.
        For each run, summary statistics (n, mean, median, std) are shown as an annotation,
        and the legend entry is set to the run's metadata (if available).
        """
        import plotly.graph_objects as go

        # Sort simulation ids for consistent x-axis ordering
        sim_ids = sorted(sim_results.keys())

        # Create the figure
        fig = go.Figure()

        # Iterate over simulation IDs and add a violin trace for each
        for sim_id in sim_ids:
            total_rewards = []
            for ep in sim_results[sim_id]:
                if 'total_reward' in ep:
                    total_rewards.append(ep['total_reward'])
                else:
                    total_rewards.append(sum(ep.get('rewards', [])))

            # Compute summary stats if there is data
            if total_rewards:
                n = len(total_rewards)
                mean_val = np.mean(total_rewards)
                median_val = np.median(total_rewards)
                std_val = np.std(total_rewards)
                min_val = np.min(total_rewards)
                max_val = np.max(total_rewards)
            else:
                n = 0
                mean_val = median_val = std_val = min_val = max_val = 0

            # Use metadata from the first episode if available, else fall back to sim_id
            metadata = sim_results[sim_id][0].get('metadata', str(sim_id)) if sim_results[sim_id] else str(sim_id)

            if metadata['simulation_type'] == 'ObservationThresholdSimulation':
                name = f"Threshold (obs,wind): {metadata['observation_threshold']},{metadata['wind_threshold']}"
            elif metadata['simulation_type'] == 'OptimalSimulation':
                name = "OptimalSimulation"
            # Add the violin trace using the metadata as the legend entry
            fig.add_trace(go.Violin(
                y=total_rewards,
                name=name,
                box_visible=True,        # Show box plot inside the violin
                meanline_visible=True,   # Show mean line inside the violin
                points="all",            # Display all individual data points
                opacity=0.7,
                hovertemplate=
                    'Reward: %{y}<br>'+
                    f'n: {n}<br>'+
                    f'Mean: {mean_val:.2f}<br>'+
                    f'Median: {median_val:.2f}<br>'+
                    f'Std: {std_val:.2f}<br>'+
                    f'Min: {min_val:.2f}<br>'+
                    f'Max: {max_val:.2f}<extra></extra>'
            ))

        # Update layout with titles and labels
        fig.update_layout(
            title="Violin Plot of Total Rewards per Simulation",
            xaxis_title="Simulation Metadata",
            yaxis_title="Total Reward",
            violingap=0.2,         # Gap between violins
            violinmode='overlay'   # Overlay violins; use "group" for side-by-side
        )

        fig.show()


    def load_and_plot_reward_violin(self, simulation_ids, storage_dir=None):
        """
        Loads episodes from the specified simulation IDs (from a directory) and plots a violin chart 
        of the reward distribution.

        Parameters:
            simulation_ids (list): List of simulation IDs to load.
            storage_dir (str): Directory where simulation files are stored. If not provided, self.storage_dir is used.
        """
        if storage_dir is None:
            if self.storage_dir is None:
                print("Storage directory must be provided either in constructor or as argument.")
                return
            else:
                storage_dir = self.storage_dir

        storage = SimulationStorage(storage_dir)
        sim_results = {}
        for sim_id in simulation_ids:
            sim_data = storage.load_simulation_by_id(storage_dir, sim_id)
            episodes = sim_data.get('episodes', [])
            if episodes:
                sim_results[sim_id] = episodes
            else:
                print(f"No episodes found for simulation {sim_id}.")
        
        if not sim_results:
            print("No valid simulation results loaded.")
            return

        self.plot_reward_violin(sim_results)


# --------------------- EXAMPLE USAGE ---------------------
if __name__ == '__main__':
    # For demonstration, here is an example sim_results dictionary.

    plotter = SimulationPlotter(storage_dir="simulation_results")
    plotter.load_and_plot_reward_violin(simulation_ids=[0,1,2,3])
    # plotter.load_and_plot_reward_histogram_subplots(simulation_ids=[0,2,3])
    
    # Other methods can be used similarly:
    # plotter.plot_specific_episode(simulation_id=0, episode_index=0)
    # plotter.plot_multiple_episodes_from_simulation(simulation_id=0, episode_indices=[0, 1, 2])
    # plotter.compare_episodes_across_simulations({0: [0], 1: [2], 2: None})
