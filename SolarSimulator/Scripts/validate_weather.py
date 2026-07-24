import os
import pandas as pd

import os
import pandas as pd


def validate_dataframes_in_directory(directory):
    """Reads all pickle files in a directory and ensures the DataFrames they store are all the same size.

    Args:
        directory (str): Path to the directory containing the pickle files.

    Returns:
        dict: A dictionary with filenames as keys and DataFrames as values,
              or None if there is a size mismatch.
    """
    dataframes = {}
    common_shape = None

    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_pickle(file_path)
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"File {file} does not contain a DataFrame.")

                if common_shape is None:
                    common_shape = df.shape  # Initialize with the first DataFrame's shape
                elif df.shape != common_shape:
                    raise ValueError(
                        f"DataFrame in file {file} has a different size: {df.shape}. Expected: {common_shape}."
                    )

                dataframes[file] = df
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                return None  # Return None if there's an error

    print(
        f"All DataFrames in directory '{directory}' are validated to have the same size: {common_shape}."
    )
    return dataframes


validate_dataframes_in_directory(r"Data\SYNTHETIC_DATA\lat30")
