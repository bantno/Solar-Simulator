import numpy as np
from BaseClasses.weather_processor import WeatherDataProcessor


if __name__ == "__main__":
    lat_south = 29.0  # Southern latitude boundary
    lat_north = 45.0  # Northern latitude boundary
    lat_step = 1.0    # Step size for latitude
    for lat in np.arange(lat_south, lat_north, lat_step):
        # Adjust the range as needed for your specific use case
        print(f"Processing data for latitude: {lat}")
        processor = WeatherDataProcessor()
        lon = -75.
        timestep_min = 15
        processor.fetch_weather_data(
            lat,
            lon,
            "1950-01-01",
            "2022-12-31",
            ["wind_speed_10m", "wind_direction_10m", "shortwave_radiation"],
        )
        hourly_df = processor.process_hourly_data()
        hourly_df.to_pickle(rf"Data\HISTORICAL_DATA\data_{lat}_{lon}")
        resampled_df = processor.resample_data(timestep_min)

        expected_data_filename = rf"Data\EXPECTED_DATA\data_expected_lat{lat:.1f}_lon{lon:.1f}_{timestep_min}min.pkl"
        processor.fit_distributions(resampled_df, expected_data_filename)