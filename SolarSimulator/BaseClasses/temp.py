import matplotlib.pyplot as plt

# Data from the screenshot
simulation_names = [
    "Threshold (wind=10,obs=0.5)",
    "Threshold (wind=10,obs=0.9)",
    "Threshold (wind=10,obs=0.0)",
    "OptimalPolicy",
]
average_rewards = [
    17.5212079679542,
    19.192609076344,
    15.89439578484623,
    46.75878945514499,
]

# Create the figure and axis
plt.figure(figsize=(8, 5))

# Plot a bar chart
bars = plt.bar(simulation_names, average_rewards, color=["grey","grey","grey","#2ca02c"])

# Optionally, add the numerical values above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

# Labeling
plt.title("Comparison of Average Rewards")
plt.ylabel("Average Reward")
plt.ylim(0, max(average_rewards) + 5)  # A little extra space on top

# Show the plot
plt.tight_layout()
plt.show()
