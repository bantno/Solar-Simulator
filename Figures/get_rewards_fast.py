import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_reward_analysis(file_path, output_dir="analysis_plots"):
    """Generates a dual-pane plot (Scatter + Histogram) for each simulation run
    in the HDF5 file to analyze reward distribution.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        sim_keys = sorted(f.keys())
        
        for sim_name in sim_keys:
            sim_grp = f[sim_name]
            if "episodes" not in sim_grp:
                continue
            
            rewards = []
            eps_grp = sim_grp["episodes"]
            # Sort episodes numerically: "episode 1", "episode 2", etc.
            ep_keys = sorted(eps_grp.keys(), key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else x)
            
            for ep_key in ep_keys:
                ep_grp = eps_grp[ep_key]
                # Target the scalar dataset format from SimulationStorageHDF5
                if "total_reward" in ep_grp:
                    rewards.append(float(ep_grp["total_reward"][()]))
                elif "total_reward" in ep_grp.attrs:
                    rewards.append(float(ep_grp.attrs["total_reward"]))
            
            if rewards:
                # Create a figure with two subplots side-by-side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                indices = np.arange(len(rewards))
                mean_val = np.mean(rewards)
                std_val = np.std(rewards)

                # --- Subplot 1: Scatter Plot ---
                ax1.scatter(indices, rewards, color='tab:blue', alpha=0.6, s=30, label='Episode Reward')
                ax1.axhline(y=mean_val, color='tab:red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax1.set_title(f"Scatter: {sim_name}")
                ax1.set_xlabel("Episode Index")
                ax1.set_ylabel("Total Reward")
                ax1.grid(True, linestyle=':', alpha=0.5)
                ax1.legend()

                # --- Subplot 2: Histogram ---
                # Using 'auto' bins to adapt to the number of episodes
                ax2.hist(rewards, bins='auto', color='tab:green', alpha=0.7, edgecolor='black')
                ax2.axvline(x=mean_val, color='tab:red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax2.set_title(f"Distribution ($\mu={mean_val:.1f}, \sigma={std_val:.1f}$)")
                ax2.set_xlabel("Reward Range")
                ax2.set_ylabel("Frequency")
                ax2.grid(True, axis='y', linestyle=':', alpha=0.5)
                ax2.legend()

                plt.tight_layout()
                
                # Sanitize filename
                safe_name = sim_name.replace("|", "_").replace("/", "_").replace(" ", "_")
                save_path = os.path.join(output_dir, f"{safe_name}_analysis.png")
                
                plt.savefig(save_path)
                plt.close(fig) # Prevent memory leaks
                print(f"Generated Analysis: {save_path}")

if __name__ == "__main__":
    # Replace with your actual sweep file name
    PATH_TO_FILE = "observation_and_windspeed_threshold_config_20250622_221018.h5"
    generate_reward_analysis(PATH_TO_FILE)