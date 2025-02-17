import pandas as pd
import matplotlib.pyplot as plt


def plot_weather_data(file1, file2):
    """
    Plots weather data (wind speed, wind direction, and shortwave radiation) from two files.

    Parameters:
        file1 (str): Path to the first CSV or pickle file containing the weather data.
        file2 (str): Path to the second CSV or pickle file containing the weather data.
    """
    # Load the data from the files (assumes pickle format, change to read_csv() for CSV format)
    data1 = pd.read_pickle(file1)
    data2 = pd.read_pickle(file2)

    # Ensure the data is time-indexed
    if not isinstance(data1.index, pd.DatetimeIndex):
        data1.index = pd.to_datetime(data1.index)
    if not isinstance(data2.index, pd.DatetimeIndex):
        data2.index = pd.to_datetime(data2.index)

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    # Wind Speed Plot (both files)
    axes[0, 0].plot(data1.index, data1["wind_speed_10m"], label="File 1", color="b")
    axes[0, 0].set_title("Wind Speed (File 1)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Wind Speed (m/s)")
    axes[0, 0].legend()

    axes[0, 1].plot(data2.index, data2["wind_speed_10m"], label="File 2", color="r")
    axes[0, 1].set_title("Wind Speed (File 2)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Wind Speed (m/s)")
    axes[0, 1].legend()

    # Wind Direction Plot (both files)
    axes[1, 0].plot(data1.index, data1["wind_direction_10m"], label="File 1", color="b")
    axes[1, 0].set_title("Wind Direction (File 1)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Wind Direction (°)")
    axes[1, 0].legend()

    axes[1, 1].plot(data2.index, data2["wind_direction_10m"], label="File 2", color="r")
    axes[1, 1].set_title("Wind Direction (File 2)")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Wind Direction (°)")
    axes[1, 1].legend()

    # Shortwave Radiation Plot (both files)
    axes[2, 0].plot(data1.index, data1["shortwave_radiation"], label="File 1", color="b")
    axes[2, 0].set_title("Shortwave Radiation (File 1)")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("Shortwave Radiation (W/m²)")
    axes[2, 0].legend()

    axes[2, 1].plot(data2.index, data2["shortwave_radiation"], label="File 2", color="r")
    axes[2, 1].set_title("Shortwave Radiation (File 2)")
    axes[2, 1].set_xlabel("Time")
    axes[2, 1].set_ylabel("Shortwave Radiation (W/m²)")
    axes[2, 1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_first_year_weather_data(file1, file2):
    """
    Plots the first year of weather data (wind speed, wind direction, and shortwave radiation)
    from two files, considering the first year in either file.

    Parameters:
        file1 (str): Path to the first CSV or pickle file containing the weather data.
        file2 (str): Path to the second CSV or pickle file containing the weather data.
    """
    # Load the data from the files (assumes pickle format, change to read_csv() for CSV format)
    data1 = pd.read_pickle(file1)
    data2 = pd.read_pickle(file2)

    # Ensure the data is time-indexed
    if not isinstance(data1.index, pd.DatetimeIndex):
        data1.index = pd.to_datetime(data1.index)
    if not isinstance(data2.index, pd.DatetimeIndex):
        data2.index = pd.to_datetime(data2.index)

    # Get the first year from both datasets and select the earliest one
    first_year_data1 = data1.index.year.min()
    first_year_data2 = data2.index.year.min()

    # Filter the data to only include the first year from either dataset
    data1_first_year = data1[data1.index.year == first_year_data1]
    data2_first_year = data2[data2.index.year == first_year_data2]

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    # Wind Speed Plot (both files)
    axes[0, 0].plot(
        data1_first_year.index,
        data1_first_year["wind_speed_10m"],
        label=f"Synthetic Data ({first_year_data1})",
        color="b",
    )
    axes[0, 0].set_title("Wind Speed (Synthetic Data)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Wind Speed (m/s)")
    axes[0, 0].legend()

    axes[0, 1].plot(
        data2_first_year.index,
        data2_first_year["wind_speed_10m"],
        label=f"Recorded Data ({first_year_data2})",
        color="r",
    )
    axes[0, 1].set_title(f"Wind Speed ({first_year_data2} Data)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Wind Speed (m/s)")
    axes[0, 1].legend()

    # Wind Direction Plot (both files)
    axes[1, 0].plot(
        data1_first_year.index,
        data1_first_year["wind_direction_10m"],
        label=f"Synthetic Data ({first_year_data1})",
        color="b",
    )
    axes[1, 0].set_title("Wind Direction (Synthetic Data)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Wind Direction (°)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        data2_first_year.index,
        data2_first_year["wind_direction_10m"],
        label=f"Recorded Data ({first_year_data2})",
        color="r",
    )
    axes[1, 1].set_title(f"Wind Direction ({first_year_data2} Data)")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Wind Direction (°)")
    axes[1, 1].legend()

    # Shortwave Radiation Plot (both files)
    axes[2, 0].plot(
        data1_first_year.index,
        data1_first_year["shortwave_radiation"],
        label=f"Synthetic Data ({first_year_data1})",
        color="b",
    )
    axes[2, 0].set_title("Shortwave Radiation (Synthetic Data)")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("Shortwave Radiation (W/m²)")
    axes[2, 0].legend()

    axes[2, 1].plot(
        data2_first_year.index,
        data2_first_year["shortwave_radiation"],
        label=f"Recorded Data ({first_year_data2})",
        color="r",
    )
    axes[2, 1].set_title(f"Shortwave Radiation ({first_year_data2} Data)")
    axes[2, 1].set_xlabel("Time")
    axes[2, 1].set_ylabel("Shortwave Radiation (W/m²)")
    axes[2, 1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Call the function with paths to the two pickle files
    plot_first_year_weather_data(
        r"Data\SYNTHETIC_DATA\data_30min_0.pkl", r"Data\EXPECTED_DATA\data_expected_30min.pkl"
    )
