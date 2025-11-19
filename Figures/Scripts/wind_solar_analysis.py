import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Specify the directory containing your .pkl files
data_folder = r'Data\EXPECTED_DATA'  # ← change to your folder path

# Latitudes and longitudes to include
lat_list = [20, 30, 35, 40, 58]
lon_list = [-159, -75, 14, 138, -161]   # ← change to desired longitudes

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
        wind_means[(lat, lon)] = monthly['expected_wind_speed'].mean()
        total_joules = monthly['expected_solar_rad'].sum() * dt_seconds
        solar_totals_mj[(lat, lon)] = total_joules / 1e6

# Month labels (3-letter abbreviations)
month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

# ----------- Plot 1: Mean Wind Speed -----------
fig, ax = plt.subplots(figsize=(8, 4))
for (lat, lon), wind in sorted(wind_means.items()):
    ax.plot(
        wind.index,
        wind.values,
        marker=".",
        label=f'Lat {lat}°, Lon {lon}°'
    )
# ax.set_title('Mean Wind Speed by Month')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_xlabel('Month')
ax.legend(title='Location')
ax.grid(False)

plt.tight_layout()
plt.savefig("mean_wind_speed.png", dpi=300)
plt.close(fig)

# ----------- Plot 2: Total Solar Energy -----------
fig, ax = plt.subplots(figsize=(8, 4))
for (lat, lon), energy_mj in sorted(solar_totals_mj.items()):
    ax.plot(
        energy_mj.index,
        energy_mj.values,
        marker=".",
        label=f'Lat {lat}°, Lon {lon}°'
    )
# ax.set_title('Total Monthly Solar Energy')
ax.set_ylabel('Monthly Insolation (MJ/m²)')
ax.set_xlabel('Month')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.legend(title='Location')
ax.grid(False)

plt.tight_layout()
plt.savefig("total_monthly_solar_energy.png", dpi=500)
plt.close(fig)
