import pandas as pd

# Example DataFrame
data = {
    "timestamp": [
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 00:10:00+00:00",
        "2024-01-01 00:20:00+00:00",
        "2024-01-01 00:30:00+00:00",
    ],
    "wind_speed_10m": [29.598919, 29.535015, 29.471113, 29.407211],
    "wind_direction_10m": [265.81516, 267.10132, 268.3875, 269.6737],
    "shortwave_radiation": [0.0, 0.0, 0.0, 0.0],
}

df = pd.DataFrame(data)

# Convert timestamp to datetime with timezone awareness
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert to local timezone (e.g., US/Pacific)
df["timestamp_local"] = df["timestamp"].dt.tz_convert("US/Pacific")

print(df)


df = pd.read_pickle(
    r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\EXPECTED_DATA\data_expected_60min.pkl"
)
print(1)
