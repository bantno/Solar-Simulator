import h5py
import numpy as np
import matplotlib.pyplot as plt

# ——— USER CONFIG ———
file_path       = r'simulation_results\sim_1000_eps_20250430_022508.h5'
sim_group_names = [
    'optimalcontinuousanalyticalpolicysimulation_c640',
    'observationthresholdcontinuoussimulation_c640_t0.0_w8.0',
    'observationthresholdcontinuoussimulation_c640_t0.25_w8.0',
]
episode_name    = 'episode 4'
dataset_names   = [
    'solar_series',
    'wind_series',
    'whale_series',
    'energy_series',
    'actions',
    'rewards',
]
x_label         = 'Decision Stage'
# —————————————————

# load all data
data = {sim: {} for sim in sim_group_names}
with h5py.File(file_path, 'r') as f:
    for sim in sim_group_names:
        grp = f[sim]['episodes'][episode_name]
        for ds in dataset_names:
            data[sim][ds] = grp[ds][:]

# make one big figure
fig, axes = plt.subplots(len(dataset_names), 1,
                         sharex=True,
                         figsize=(10, 2.5*len(dataset_names)),
                         constrained_layout=True)

for ax, ds in zip(axes, dataset_names):
    for sim in sim_group_names:
        y = data[sim][ds]
        x = np.arange(1, y.shape[0]+1)      # <-- per-sim stage axis

        # Only label once (in the first panel)
        label = sim if ds == dataset_names[0] else None

        # step for binary/integer, line otherwise
        if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0,1}):
            ax.step(x, y, where='mid', label=label)
        else:
            ax.plot(x, y, lw=1.5, label=label)

    ax.set_ylabel(ds, fontsize=10)
    if ds == dataset_names[0]:
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel(x_label, fontsize=12)
fig.suptitle(f"{episode_name} across simulations", fontsize=14)
plt.show()
