# # import pandas as pd

# # data = pd.read_pickle(r"continuous_vs_analytical_results.pkl")

# # print(data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the DataFrame from the pickle file.
file_path = "continuous_vs_analytical_results.pkl"
df = pd.read_pickle(file_path)

# Display the first few rows to understand the data structure.
print("DataFrame preview:")
print(df.head())

# -------------------------------
# Plot 1: Total Rewards per Episode
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df['Episode'], df['Analytical_TotalReward'], label="Analytical Total Reward", marker='o', linestyle='-')
plt.plot(df['Episode'], df['Continuous_TotalReward'], label="Continuous Total Reward", marker='x', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode: Analytical vs Continuous")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Figure 1: Histograms of Total Rewards
# -------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Histogram for Analytical Total Reward.
axs[0].hist(df['Analytical_TotalReward'], bins=20, edgecolor='black')
axs[0].set_title("Histogram: Analytical Total Reward")
axs[0].set_xlabel("Analytical Total Reward")
axs[0].set_ylabel("Frequency")
axs[0].grid(True)

# Histogram for Continuous Total Reward.
axs[1].hist(df['Continuous_TotalReward'], bins=20, edgecolor='black')
axs[1].set_title("Histogram: Continuous Total Reward")
axs[1].set_xlabel("Continuous Total Reward")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Figure 2: Histograms of the Last Step Number in Each Trajectory
# -------------------------------
# Compute the last step number in each trajectory using the length of the trajectory list/array.
analytical_last_steps = df['Analytical_Trajectory'].apply(len)
continuous_last_steps = df['Continuous_Trajectory'].apply(len)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Histogram for Analytical Trajectory last step numbers.
axs[0].hist(analytical_last_steps, bins=range(int(analytical_last_steps.min()), int(analytical_last_steps.max()) + 2), edgecolor='black')
axs[0].set_title("Analytical Trajectory Last Step Numbers")
axs[0].set_xlabel("Last Step Number")
axs[0].set_ylabel("Frequency")
axs[0].grid(True)

# Histogram for Continuous Trajectory last step numbers.
axs[1].hist(continuous_last_steps, bins=range(int(continuous_last_steps.min()), int(continuous_last_steps.max()) + 2), edgecolor='black')
axs[1].set_title("Continuous Trajectory Last Step Numbers")
axs[1].set_xlabel("Last Step Number")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Define a helper function to compute the failure step in a trajectory.
# The trajectory is assumed to be a sequence of states, where each state is
# represented as an array or list with the second element (index 1) indicating the mode.
# Mode 2 is failure. We count steps starting at 1.
def compute_failure_step(traj):
    for i, state in enumerate(traj):
        # Ensure the state is indexable and mode can be extracted.
        if state[1] == 2:
            return i + 1  # counting steps starting with 1
    return np.nan  # Return NaN if failure does not occur in the trajectory

# Compute failure steps for each episode.
df["Analytical_FailureStep"] = df["Analytical_Trajectory"].apply(compute_failure_step)
df["Continuous_FailureStep"] = df["Continuous_Trajectory"].apply(compute_failure_step)

# Calculate the average failure step across episodes for each method.
analytical_avg_failure_step = df["Analytical_FailureStep"].mean()
continuous_avg_failure_step = df["Continuous_FailureStep"].mean()

print("Average Analytical Failure Step:", analytical_avg_failure_step)
print("Average Continuous Failure Step:", continuous_avg_failure_step)

# Plot the average failure steps as a bar chart.
methods = ['Analytical', 'Continuous']
avg_failure_steps = [analytical_avg_failure_step, continuous_avg_failure_step]

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, avg_failure_steps, color=['blue', 'orange'], edgecolor='black')
plt.ylabel("Average Failure Step")
plt.title("Average Failure Step by Method")
# plt.grid(True, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x-coordinate: center of the bar
        height,                             # y-coordinate: at the top of the bar
        f'{height:.2f}',                    # label (formatted to 2 decimal places)
        ha='center', va='bottom'
    )
plt.tight_layout()
plt.show()

# Calculate the percentage of episodes where a failure occurred.
total_episodes = len(df)

analytical_failure_count = df["Analytical_FailureStep"].notna().sum()
continuous_failure_count = df["Continuous_FailureStep"].notna().sum()

analytical_failure_pct = (analytical_failure_count / total_episodes) * 100
continuous_failure_pct = (continuous_failure_count / total_episodes) * 100

print(f"Percentage of episodes with failure (Analytical): {analytical_failure_pct:.2f}%")
print(f"Percentage of episodes with failure (Continuous): {continuous_failure_pct:.2f}%")

# -------------------------------
# Plot 3: Sample Trajectories for a Selected Episode
# -------------------------------
# Select an episode to inspect (for example, the first episode).
episode_index = 0  # change this index to inspect another episode
analytical_traj = df.loc[episode_index, 'Analytical_Trajectory']
continuous_traj = df.loc[episode_index, 'Continuous_Trajectory']

# Ensure the trajectories are in array format
analytical_traj = np.array(analytical_traj)
continuous_traj = np.array(continuous_traj)

plt.figure(figsize=(10, 6))
plt.plot(analytical_traj, label="Analytical Trajectory", marker='o')
plt.plot(continuous_traj, label="Continuous Trajectory", marker='x')
plt.xlabel("Time step")
plt.ylabel("State Value")
plt.title(f"Trajectory Comparison for Episode {df.loc[episode_index, 'Episode']}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Select an episode to plot. (Change the index as needed.)
episode_index = 0  # For example, the first episode

# Extract data for the selected episode.
analytical_rewards = np.array(df.loc[episode_index, 'Analytical_Rewards'])
continuous_rewards = np.array(df.loc[episode_index, 'Continuous_Rewards'])

analytical_solar = np.array(df.loc[episode_index, 'Analytical_Solar'])
continuous_solar = np.array(df.loc[episode_index, 'Continuous_Solar'])

analytical_wind = np.array(df.loc[episode_index, 'Analytical_Wind'])
continuous_wind = np.array(df.loc[episode_index, 'Continuous_Wind'])

analytical_whale = np.array(df.loc[episode_index, 'Analytical_Whale'])
continuous_whale = np.array(df.loc[episode_index, 'Continuous_Whale'])

# Create a 2x2 subplot for rewards, solar, wind, and whale.
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot Rewards.
axs[0, 0].plot(analytical_rewards, label='Analytical', marker='o')
axs[0, 0].plot(continuous_rewards, label='Continuous', marker='x')
axs[0, 0].set_title('Rewards')
axs[0, 0].set_xlabel('Time Step')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot Solar.
axs[0, 1].plot(analytical_solar, label='Analytical', marker='o')
axs[0, 1].plot(continuous_solar, label='Continuous', marker='x')
axs[0, 1].set_title('Solar')
axs[0, 1].set_xlabel('Time Step')
axs[0, 1].set_ylabel('Solar Value')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot Wind.
axs[1, 0].plot(analytical_wind, label='Analytical', marker='o')
axs[1, 0].plot(continuous_wind, label='Continuous', marker='x')
axs[1, 0].set_title('Wind')
axs[1, 0].set_xlabel('Time Step')
axs[1, 0].set_ylabel('Wind Value')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot Whale.
axs[1, 1].plot(analytical_whale, label='Analytical', marker='o')
axs[1, 1].plot(continuous_whale, label='Continuous', marker='x')
axs[1, 1].set_title('Whale')
axs[1, 1].set_xlabel('Time Step')
axs[1, 1].set_ylabel('Whale Value')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()