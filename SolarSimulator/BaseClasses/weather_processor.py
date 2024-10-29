import xarray as xr
import pandas as pd

def grib_to_dataframe_cfgrib(grib_file_path):
    """
    Fast conversion of GRIB to DataFrame using cfgrib + xarray.
    """
    # Open the GRIB file with xarray and cfgrib engine
    ds = xr.open_dataset(grib_file_path, engine='cfgrib')

    # Convert to a DataFrame, including lat, lon, and time coordinates
    df = ds.to_dataframe().reset_index()

    # Optional: Set multi-index for faster queries (if needed)
    df.set_index(['latitude', 'longitude', 'time'], inplace=True)
    df.sort_index(inplace=True)

    return df


def load_ssrd_data(grib_file_path):
    try:
        # Filter to only include specific variables if provided
        filter_keys = {}
        
        filter_keys['shortName'] = ['ssrd']

        # Open the dataset with filtering
        ds = xr.open_dataset(
            grib_file_path,
            engine='cfgrib',
            filter_by_keys=filter_keys,
            decode_times=False  # Set to False if time decoding is causing issues
        )
        
        # Convert to DataFrame
        df = ds.to_dataframe().reset_index()
        df['valid_time'] =  pd.to_datetime(df['valid_time'], unit='s')
        df.set_index(['latitude', 'longitude', 'valid_time'], inplace=True)
        df.sort_index(inplace=True)
        
        # Check for NaN values in SSRD
        nan_ssrd = df['ssrd'].isna().sum()
        print(f"Number of NaN values in SSRD: {nan_ssrd}")

        return df

    except Exception as e:
        print(f"Error loading SSRD data: {e}")

    except Exception as e:
        print(f"Error loading SSRD data: {e}")


def load_ssrd_and_wind(grib_file):
    try:
        # Load the dataset without filtering to check available variables
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        
        # Print the available variables and dimensions for diagnosis
        print("Available variables in the dataset:", ds.data_vars)
        
        # Load the SSRD variable
        if 'ssrd' in ds.data_vars:
            ssrd_data = ds['ssrd'].squeeze()
        else:
            print("SSRD variable not found in the dataset.")
            return None

        # Load wind data if available
        wind_data = {}
        for var in ['u10', 'v10']:
            if var in ds.data_vars:
                wind_data[var] = ds[var].squeeze()
            else:
                print(f"{var} variable not found in the dataset.")

        # If wind data is found, merge it with SSRD data
        if wind_data:
            combined_data = xr.merge([ssrd_data] + list(wind_data.values()))
            print("Data loaded and consolidated successfully.")
            return combined_data
        else:
            print("No wind data available to merge with SSRD.")
            return None
            
    except Exception as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    grib_file_path = r'janfeb.grib'

    # If you need to convert to DataFrame
    
    # df_wind = grib_to_dataframe_cfgrib(grib_file_path)
    # df_ssrd = load_ssrd_data(grib_file_path)
    # print("GRIB data successfully processed!")

    # Save for later use
    # df.to_pickle("grib_data_ssrd.pkl")

    df = pd.read_pickle("grib_data_wind.pkl")
    df.to_csv("test.csv")
    print(df)
    df = pd.read_pickle("grib_data_ssrd.pkl")
    print(df)
    # print(df)
