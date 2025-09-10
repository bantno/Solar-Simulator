import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# --- Config ---
path = r"configs\final_journal_configs\horizon_sweep\value_function\horizon_sweep_config_300.0Wh_3000h_20.0p.npy"
start_date = datetime(2025, 1, 1, 0, 0)  # ← REQUIRED: Set start datetime here
dt_minutes = 15  # time resolution

# --- Load and unpack ---
data = np.load(path)
n_rows, T = data.shape
n_soc = (n_rows - 1) // 2

moored = data[:n_soc, :]
flying = data[n_soc:2*n_soc, :]

soc = np.linspace(0, 100, n_soc)
dsoc = soc[1] - soc[0]

# --- Time axis ---
times = [start_date + timedelta(minutes=i * dt_minutes) for i in range(T)]

# --- Plot 1: Where flying > mooring ---
flying_better_threshold = np.full(T, np.nan)
for t in range(T):
    diff = flying[:, t] - moored[:, t]
    idx = np.where(diff > 0)[0]
    if idx.size > 0:
        flying_better_threshold[t] = soc[idx[0]]

# --- Plot 2: Failure cliff (jump in flying value under 20%) ---
cliff_threshold = np.full(T, np.nan)
jump_threshold = 1.0
soc_cutoff = 20.0

for t in range(T):
    values = flying[:, t]
    delta = np.diff(values)
    mask = soc[:-1] < soc_cutoff
    idx = np.where((delta > jump_threshold) & mask)[0]
    if idx.size > 0:
        cliff_threshold[t] = soc[idx[0] + 1]

# --- Plot both curves ---
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(times, flying_better_threshold, label="Takeoff Threshold", color='tab:blue')
ax.plot(times, cliff_threshold, label="Landing Threshold", color='tab:red')

# Highlight missing takeoff thresholds
no_takeoff = np.isnan(flying_better_threshold)
ax.scatter(np.array(times)[no_takeoff], [100] * np.sum(no_takeoff),
           color='gray', s=10, label="No Takeoff", zorder=3)

# Format x-axis as datetime with no padding
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
ax.set_xlim(times[0], times[-1])  # <--- no padding

# Labels and layout
ax.set_xlabel("Time")
ax.set_ylabel("SoC (%)")
ax.set_title("Flying vs. Mooring Thresholds")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

