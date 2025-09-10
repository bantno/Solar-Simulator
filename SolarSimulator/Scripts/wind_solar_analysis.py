import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Specify the directory containing your .pkl files
data_folder = r'Data\EXPECTED_DATA'  # ← change to your folder path

# Latitudes and longitudes to include
lat_list = [20,30,35,40,58]
lon_list = [-159,-75,14,138,-161]   # ← change to desired longitudes

# Time step in seconds (15 minutes)
dt_seconds = 15 * 60

# Find all .pkl files in the folder
file_paths = [
    os.path.join(data_folder, f)
    for f in os.listdir(data_folder)
    if f.endswith('.pkl')
]

# Dictionaries to hold monthly stats keyed by (lat, lon)
wind_means      = {}
solar_totals_mj = {}

for i in range(len(lat_list)):
    for file in file_paths:
        # Extract latitude and longitude from filename
        m_lat = re.search(r'lat([-0-9.]+)', file)
        m_lon = re.search(r'lon([-0-9.]+)', file)
        if not (m_lat and m_lon):
            continue
        lat = float(m_lat.group(1))
        lon = float(m_lon.group(1))

        # Skip if not in our lists
        if not lat == lat_list[i] or not lon == lon_list[i]:
            continue

        df = pd.read_pickle(file)
        monthly = df.groupby('month')

        # Store by (lat, lon) tuple
        wind_means     [(lat, lon)] = monthly['expected_wind_speed'].mean()
        total_joules               = monthly['expected_solar_rad'].sum() * dt_seconds
        solar_totals_mj[(lat, lon)] = total_joules / 1e6

# Now plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Mean wind speed
for (lat, lon), wind in sorted(wind_means.items()):
    axes[0].plot(wind.index, wind.values,
                 marker=".",
                 label=f'Lat {lat}°, Lon {lon}°')
axes[0].set_title('Mean Wind Speed by Month')
axes[0].set_ylabel('Wind Speed (m/s)')
axes[0].set_xticks(range(1, 13))
axes[0].legend(title='Location')
axes[0].grid(True)

# Total solar energy in MJ
for (lat, lon), energy_mj in sorted(solar_totals_mj.items()):
    axes[1].plot(energy_mj.index, energy_mj.values,
                 marker=".",
                 label=f'Lat {lat}°, Lon {lon}°')
axes[1].set_title('Total Monthly Solar Energy')
axes[1].set_ylabel('Total Energy (MJ/m²)')
axes[1].set_xlabel('Month')
axes[1].set_xticks(range(1, 13))
axes[1].legend(title='Location')
axes[1].grid(True)

plt.tight_layout()
plt.show()
