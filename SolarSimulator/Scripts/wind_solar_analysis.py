import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Specify the directory containing your .pkl files
data_folder = 'Data\EXPECTED_DATA'  # ← change to your folder path

# Specify the list of latitudes to include
lat_list = [20.0, 26.0, 30.0, 33.0]  # ← change to desired latitudes

# Time step in seconds (15 minutes)
dt_seconds = 15 * 60

# Automatically find all .pkl files in the folder
file_paths = [
    os.path.join(data_folder, f)
    for f in os.listdir(data_folder)
    if f.endswith('.pkl')
]

# Dictionaries to hold monthly stats keyed by latitude
wind_means = {}
solar_totals_mj = {}

for file in file_paths:
    # Extract latitude from filename
    m = re.search(r'lat([-0-9.]+)', file)
    lat = float(m.group(1)) if m else None
    if lat not in lat_list:
        continue  # skip files not in the specified latitudes
    df = pd.read_pickle(file)
    monthly = df.groupby('month')
    wind_means[lat] = monthly['expected_wind_speed'].mean()
    # Calculate total solar energy: sum of radiation * time step → Joules/m², then convert to MJ
    total_joules = monthly['expected_solar_rad'].sum() * dt_seconds
    solar_totals_mj[lat] = total_joules / 1e6

# Create subplots
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plot mean wind speed
for lat, wind in sorted(wind_means.items()):
    axes[0].plot(wind.index, wind.values, label=f'Lat {lat}°')
axes[0].set_title('Mean Wind Speed by Month')
axes[0].set_ylabel('Wind Speed (m/s)')
axes[0].set_xticks(range(1, 13))
axes[0].legend(title='Latitude')
axes[0].grid(True)

# Plot total solar energy in MJ
for lat, energy_mj in sorted(solar_totals_mj.items()):
    axes[1].plot(energy_mj.index, energy_mj.values, label=f'Lat {lat}°')
axes[1].set_title('Total Monthly Solar Energy')
axes[1].set_ylabel('Total Energy (MJ/m²)')
axes[1].set_xlabel('Month')
axes[1].set_xticks(range(1, 13))
axes[1].legend(title='Latitude')
axes[1].grid(True)

plt.tight_layout()
plt.show()
