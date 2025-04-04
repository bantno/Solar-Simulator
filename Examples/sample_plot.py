import matplotlib.pyplot as plt

# Data
sample_sizes = [100, 1000]
mcs_results = [2.039348, 27.583280]
ni_results  = [2.062604, 28.004335]

# Set up bar width and positions
bar_width = 0.35
x_positions = range(len(sample_sizes))

# Create the plot
plt.figure(figsize=(6, 4))

# Plot MCS and NI bars
bars_mcs = plt.bar(
    [x - bar_width/2 for x in x_positions], 
    mcs_results, 
    width=bar_width, 
    label='Monte Carlo Simulation', 
    color='skyblue'
)
bars_ni = plt.bar(
    [x + bar_width/2 for x in x_positions], 
    ni_results, 
    width=bar_width, 
    label='Numerical Integration', 
    color='salmon'
)

# Customize the x-axis labels and ticks
plt.xticks(x_positions, sample_sizes)
plt.xlabel('Time Horizon (Steps)')
plt.ylabel('Average Total Reward')
plt.title('Comparison of MCS vs. Numerical Integration (10000 Episodes)')

# Add data labels on top of each MCS bar
for bar in bars_mcs:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height, 
        f'{height:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9
    )

# Add data labels on top of each NI bar
for bar in bars_ni:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height, 
        f'{height:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=9
    )

# Show legend and display plot
plt.legend()
plt.tight_layout()
plt.show()
