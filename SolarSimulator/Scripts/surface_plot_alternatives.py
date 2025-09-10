import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import matplotlib

# --- Load data ---
path = r"configs\final_journal_configs\mission_start_date_sweep\value_function\mission_start_date_sweep_config_300.0Wh_8500h_15.0p.npy"
data = np.load(path)

# --- Extract dimensions ---
n_rows, T = data.shape
n_soc = (n_rows - 1) // 2
moored = data[:n_soc, :]
flying = data[n_soc:2*n_soc, :]
soc = np.linspace(0, 100, n_soc)
dsoc = soc[1] - soc[0]

# --- Time axis (15-min increments from start date) ---
start_date = datetime(2025, 7, 1, 0, 0)
times = np.array([start_date + timedelta(minutes=15*i) for i in range(T)])

# --- Option 1: Optimal value at each (SoC, time) ---
optimal_surface = np.maximum(moored, flying)

# --- Option 2: Max optimal value over SoC at each time ---
vmax_t = np.max(optimal_surface, axis=0)

# --- Option 4: Values for selected SoC slices over time ---
soc_slices = [100]
slice_indices = [np.argmin(np.abs(soc - val)) for val in soc_slices]
moored_slices = moored[slice_indices, :]
flying_slices = flying[slice_indices, :]
optimal_slices = optimal_surface[slice_indices, :]

# --- Option 5: Action preference map ---
preferred_action = np.where(flying > moored, 1, 0)  # 1 = flying, 0 = moored

time = np.arange(T)  # integer time for 3D plot
X, Y = np.meshgrid(time, soc)
Z = optimal_surface

# --- Option 6: 3D surface of optimal value ---
fig6 = plt.figure(figsize=(10, 6))
ax6 = fig6.add_subplot(111, projection='3d')
surf = ax6.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax6.set_title("3D Surface of Optimal Value V*(SoC, t)")
ax6.set_xlabel("Timestep")
ax6.set_ylabel("SoC (%)")
ax6.set_zlabel("Optimal Value")
fig6.colorbar(surf, ax=ax6, shrink=0.5, aspect=10)

plt.show()

# --- Create plots ---
figs = []

# # Option 1: Heatmap of optimal value
# fig1, ax1 = plt.subplots(figsize=(10, 4))
# c1 = ax1.pcolormesh(times, soc, optimal_surface, shading='auto', cmap='viridis')
# fig1.colorbar(c1, ax=ax1, label="Optimal Value")
# ax1.set_title("Option 1: Heatmap of Optimal Value V*(SoC, t)")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("SoC (%)")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
# ax1.set_xlim(times[0], times[-1])
# figs.append(("Optimal Value Heatmap", fig1))

# Option 2: Line plot of max value over time
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(times, vmax_t, label="Max V* over SoC", color='tab:green')
ax2.set_title("Option 2: Max Optimal Value Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
ax2.grid(True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
ax2.set_xlim(times[0], times[-1])
figs.append(("Max Value Over Time", fig2))

# Option 4: Line plots for fixed SoC levels
fig4, ax4 = plt.subplots(figsize=(10, 4))
for i, soc_val in enumerate(soc_slices):
    ax4.plot(times, moored_slices[i], '--', label=f'Moored {soc_val:.0f}%')
    ax4.plot(times, flying_slices[i], '-', label=f'Flying {soc_val:.0f}%')
ax4.set_title("V*(SoC, t) for Fixed SoC Levels")
ax4.set_xlabel("Time")
ax4.set_ylabel("Value")
ax4.grid(True)
ax4.legend(ncol=2)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
ax4.set_xlim(times[0], times[-1])
figs.append(("Fixed SoC Time Series", fig4))


plt.show()
