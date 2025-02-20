import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

class EnvironmentalDataProcessor:
    def __init__(self, data_path="Data/SYNTHETIC_DATA", output_path="Output_Heatmaps"):
        self.data_path = data_path  # Path to the data files
        self.output_path = output_path  # Directory to save the heatmaps
        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def process_data_for_latitude(self, lat):
        """Process data for a single latitude."""
        file_paths = glob.glob(f"{self.data_path}/lat{lat}/*.pkl")
        all_data = []

        # Read and process each file
        for file in tqdm(file_paths, f"Processing data for latitude {lat}: "):
            df = pd.read_pickle(file)

            # Extract hour and month
            df['hour'] = df.index.hour
            df['month'] = df.index.month
            df['minute'] = df.index.minute

            # Aggregate within this file and store results
            all_data.append(df.groupby(['month', 'hour', 'minute'])[['shortwave_radiation', 'wind_speed_10m']].mean())

        # Combine all data and compute average across all files
        final_df = pd.concat(all_data).groupby(['month', 'hour', 'minute']).mean()
        return final_df

    def plot_heatmaps(self, final_df, lat):
        """Generate and save heatmaps for solar radiation and wind speed."""
        # Pivot the data with hours on y-axis and months on x-axis
        solar_heatmap = final_df["shortwave_radiation"].unstack().T  # Transpose for correct orientation
        wind_heatmap = final_df["wind_speed_10m"].unstack().T  # Transpose for correct orientation

        # Save Solar Radiation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(solar_heatmap, cmap="coolwarm", annot=False)
        plt.title(f"Solar Radiation Heatmap (Month vs. Hour) for latitude {lat}")
        plt.xlabel("Month")
        plt.ylabel("Hour of the Day")
        # Reduce the number of x-axis and y-axis ticks
        plt.xticks(ticks=range(0, len(solar_heatmap.columns), 2), labels=solar_heatmap.columns[::2])  # Adjust step as needed
        plt.yticks(ticks=range(0, len(solar_heatmap.index), 2), labels=solar_heatmap.index[::2])  # Adjust step as needed
        plt.tight_layout()  # Adjust layout to prevent clipping
        solar_filename = os.path.join(self.output_path, f"solar_radiation_lat{lat}.png")
        plt.savefig(solar_filename)
        plt.close()  # Close the figure to avoid memory issues

        # Save Wind Speed Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(wind_heatmap, cmap="coolwarm", annot=False)
        plt.title(f"Wind Speed Heatmap (Month vs. Hour) for latitude {lat}")
        plt.xlabel("Month")
        plt.ylabel("Hour of the Day")
        plt.xticks(ticks=range(0, len(wind_heatmap.columns), 2), labels=wind_heatmap.columns[::2])  # Adjust step as needed
        plt.yticks(ticks=range(0, len(wind_heatmap.index), 2), labels=wind_heatmap.index[::2])  # Adjust step as needed
        plt.tight_layout()  # Adjust layout to prevent clipping
        wind_filename = os.path.join(self.output_path, f"wind_speed_lat{lat}.png")
        plt.savefig(wind_filename)
        plt.close()  # Close the figure to avoid memory issues

    def process_and_save_for_multiple_latitudes(self, latitudes):
        """Process data and save heatmaps for multiple latitudes."""
        for lat in latitudes:
            final_df = self.process_data_for_latitude(lat)
            self.plot_heatmaps(final_df, lat)

# Usage example:
latitudes = [30,0,-30]  # List of latitudes
processor = EnvironmentalDataProcessor()
processor.process_and_save_for_multiple_latitudes(latitudes)