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
        # Correctly pivot data: Months on x-axis, Hours on y-axis
        solar_heatmap = final_df["shortwave_radiation"].unstack(level=0)  # Month as columns
        wind_heatmap = final_df["wind_speed_10m"].unstack(level=0)

        # Save Solar Radiation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(solar_heatmap, cmap="coolwarm", annot=False)
        plt.title(f"Solar Radiation Heatmap (Month vs. Hour) for latitude {lat}")
        plt.xlabel("Month")
        plt.ylabel("Hour of the Day")
        plt.xticks(ticks=range(1, 13), labels=range(1, 13))  # Explicitly set months
        plt.yticks(ticks=range(0, 24, 2), labels=range(0, 24, 2))  # Adjust hour labels
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"solar_radiation_lat{lat}.png"))
        plt.close()

        # Save Wind Speed Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(wind_heatmap, cmap="coolwarm", annot=False)
        plt.title(f"Wind Speed Heatmap (Month vs. Hour) for latitude {lat}")
        plt.xlabel("Month")
        plt.ylabel("Hour of the Day")
        plt.xticks(ticks=range(1, 13), labels=range(1, 13))
        plt.yticks(ticks=range(0, 24, 2), labels=range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f"wind_speed_lat{lat}.png"))
        plt.close()


    def process_and_save_for_multiple_latitudes(self, latitudes):
        """Process data and save heatmaps for multiple latitudes."""
        for lat in latitudes:
            final_df = self.process_data_for_latitude(lat)
            self.plot_heatmaps(final_df, lat)

# Usage example:
latitudes = [30,0,-30]  # List of latitudes
processor = EnvironmentalDataProcessor()
processor.process_and_save_for_multiple_latitudes(latitudes)