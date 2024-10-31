import os
import xarray as xr
import pandas as pd

class GribDataProcessor:
    def __init__(self, target_directory):
        """
        Initialize the processor with the target directory containing GRIB files.
        """
        self.target_directory = target_directory

    def load_ssrd_data(self, grib_file_path):
        """
        Load SSRD (surface solar radiation downwards) data from a GRIB file.
        """
        try:
            ds = xr.open_dataset(
                grib_file_path,
                engine='cfgrib',
                filter_by_keys={'shortName': ['ssrd']},
                decode_times=False
            )
            df = ds.to_dataframe().reset_index()
            df['valid_time'] = pd.to_datetime(df['valid_time'], unit='s')
            df.set_index(['latitude', 'longitude', 'valid_time'], inplace=True)
            df.sort_index(inplace=True)
            print(f"Number of NaN values in SSRD: {df['ssrd'].isna().sum()}")
            return df
        except Exception as e:
            print(f"Error loading SSRD data from {grib_file_path}: {e}")
            return None

    def load_wind_data(self, grib_file_path):
        """
        Load wind data (u10 and v10) from a GRIB file.
        """
        try:
            ds = xr.open_dataset(
                grib_file_path,
                engine='cfgrib',
                decode_times=True
            )
            df = ds.to_dataframe().reset_index()
            df.set_index(['latitude', 'longitude', 'time'], inplace=True)
            df.sort_index(inplace=True)
            print(f"Wind data from {grib_file_path} loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading wind data from {grib_file_path}: {e}")
            return None

    def load_and_merge_ssrd_wind(self, grib_file_path):
        """
        Load SSRD and wind data separately and merge them on latitude, longitude, and time.
        """
        ssrd_df = self.load_ssrd_data(grib_file_path)
        wind_df = self.load_wind_data(grib_file_path)

        if ssrd_df is not None and wind_df is not None:
            
            print(f"Merged SSRD and wind data from {grib_file_path}.")
            pass
            # return merged_df
        else:
            print(f"Skipping merge for {grib_file_path} due to missing data.")
            return None

    def combine_grib_files(self):
        """
        Combine data from all GRIB files in the target directory into a single DataFrame.
        """
        all_dataframes = []

        # Iterate over all .grib files in the target directory
        for file_name in os.listdir(self.target_directory):
            if file_name.endswith(".grib"):
                file_path = os.path.join(self.target_directory, file_name)
                print(f"Processing {file_name}...")

                df = self.load_and_merge_ssrd_wind(file_path)
                if df is not None:
                    all_dataframes.append(df)

        # Combine all DataFrames if any data was loaded
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print("All GRIB files combined successfully.")
            return combined_df
        else:
            print("No valid data loaded from GRIB files.")
            return pd.DataFrame()

    def save_combined_data(self, output_file):
        """
        Save the combined GRIB data to a CSV file.
        """
        combined_df = self.combine_grib_files()

        if not combined_df.empty:
            combined_df.to_csv(output_file, index=False)
            print(f"Combined data saved to '{output_file}'.")
        else:
            print("No data to save.")


# Example usage
if __name__ == "__main__":
    # Specify the directory where the GRIB files are located
    target_directory = "."

    # Initialize the processor and save combined data to a CSV
    processor = GribDataProcessor(target_directory)
    processor.save_combined_data("combined_grib_data.csv")
