import matplotlib.pyplot as plt

# Example data (labels & average rewards)
simulation_labels = [
    "Threshold (obs=0.0, wind=1.0)",
    "Threshold (obs=0.5, wind=1.0)",
    "Threshold (obs=1.0, wind=1.0)",
    "Optimal Policy"
]
average_rewards = [9.467, 10.426, 12.704, 19.393]

# Create the bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(simulation_labels, average_rewards, color=['gray', 'gray', 'gray', 'green'])

# Highlight the best performing policy (e.g., in green)
# If you already colored it in the 'color' list, you can skip this step.
best_index = average_rewards.index(max(average_rewards))
bars[best_index].set_color('green')

# Labeling and layout
plt.title("Comparison of Simulation Runs by Average Reward")
plt.xlabel("Simulation Configuration")
plt.ylabel("Average Reward")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
