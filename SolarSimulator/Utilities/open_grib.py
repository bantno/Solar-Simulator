import pygrib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class GribOpener:
    def __init__(self, grib_file, write=False):
        """
        Initializes the class with the GRIB file and reads the data into a DataFrame.
        
        Parameters:
        grib_file (str): Path to the GRIB file containing the desired data.
        """
        self.grib_file = grib_file
        self.csv_write = write
        self.df = None

    def split_grib_by_variable(self):
        """
        Reads the GRIB file and splits it into three separate GRIB files based on variables:
        - wind_data.grib: Contains 10 metre U and V wind components.
        - solar_radiation.grib: Contains Surface short-wave (solar) radiation downwards.
        - tcc.grib: Contains Total cloud cover.
        """
        grbs = pygrib.open(self.grib_file)

        # Create lists to hold selected GRIB messages
        wind_gribs = []
        solar_radiation_gribs = []
        tcc_gribs = []

        # Define variable names for splitting
        wind_variables = ['10 metre U wind component', '10 metre V wind component']
        solar_radiation_variable = 'Surface short-wave (solar) radiation downwards'
        tcc_variable = 'Total cloud cover'

        # Iterate through GRIB messages
        for grb in tqdm(grbs.select()):
            if grb.name in wind_variables:
                wind_gribs.append(grb)
            elif grb.name == solar_radiation_variable:
                solar_radiation_gribs.append(grb)
            elif grb.name == tcc_variable:
                tcc_gribs.append(grb)

        # Close original GRIB file
        grbs.close()

        # Save wind-related variables
        self.write_grib_file(wind_gribs, 'wind_data.grib')

        # Save solar radiation variable
        self.write_grib_file(solar_radiation_gribs, 'solar_radiation.grib')

        # Save total cloud cover variable
        self.write_grib_file(tcc_gribs, 'tcc.grib')

    def write_grib_file(self, grib_list, output_file):
        """
        Writes a list of GRIB messages to a new GRIB file.

        Parameters:
        grib_list (list): List of GRIB messages to write.
        output_file (str): Name of the output GRIB file.
        """
        if len(grib_list) == 0:
            print(f"No data found for {output_file}")
            return

        with open(output_file, 'wb') as out_file:
            for grb in grib_list:
                out_file.write(grb.tostring())
        
        print(f"GRIB data written to {output_file}")

    def list_grib_variables(self,grib_file):
        """
        Lists all unique variables in the provided GRIB file.

        Parameters:
        grib_file (str): Path to the GRIB file.
        """
        grbs = pygrib.open(grib_file)
        
        # Create a set to hold the unique variable names
        variables = set()
        
        for grb in tqdm(grbs.select()):
            variables.add(grb.name)  # Add the variable name to the set

        # Close the GRIB file
        grbs.close()
        
        # Display the unique variables
        print("Unique Variables in the GRIB file:")
        for var in sorted(variables):
            print(var)

    def read_grib_to_df(self, variable_name):
        """
        Reads the GRIB file and extracts values with latitude, longitude, and time.

        Parameters:
        variable_name (str): The name of the variable to extract.
        csv_write (bool): If True, writes the resulting DataFrame to a CSV file.

        Returns:
        pd.DataFrame: DataFrame containing the extracted data.
        """
        # Open the GRIB file
        grbs = pygrib.open(self.grib_file)
        
        # Initialize list to store data
        data = []
        
        # Iterate through the GRIB messages for the specified variable
        for grb in tqdm(grbs.select(name=variable_name)):
            # Extract the necessary details
            date = grb.validityDate
            hour = grb.validityTime
            valid_time = pd.to_datetime(f'{date} {hour:04}', format='%Y%m%d %H%M')
            latitudes, longitudes = grb.latlons()
            values = grb.values
            
            # Flatten lat/lon and values for ease of storage in DataFrame
            for i in range(len(latitudes)):
                for j in range(len(latitudes[i])):
                    data.append([valid_time, latitudes[i][j], longitudes[i][j], values[i][j]])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['time', 'latitude', 'longitude', 'value'])
        
        # Set time as index
        df.set_index('time', inplace=True)

        if self.csv_write:
            # Write to CSV
            filename = r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\extracted_data.csv"
            print(f"Saved to CSV at {filename}")
            df.to_csv(filename)
        
        self.df = df
        grbs.close()
        return df
    
    def assign_values_to_15_min_intervals(self):
        """
        Assigns values to 15-minute intervals using linear interpolation.
        
        Returns:
        pd.DataFrame: DataFrame with interpolated 15-minute data.
        """
        if self.df is None:
            print("No data available to interpolate. Please run read_grib_to_df first.")
            return None
        
        # Create a new DataFrame to hold the interpolated data
        interpolated_data = []

        # Group by latitude and longitude to handle each pair separately
        grouped = self.df.groupby(['latitude', 'longitude'])

        for (lat, lon), group in tqdm(grouped):
            # Resample to 15-minute intervals using linear interpolation
            group_15_min = group.resample('15min').interpolate('linear')
            
            # Retain latitude and longitude for all 15-minute intervals
            group_15_min['latitude'] = lat
            group_15_min['longitude'] = lon
            
            # Add the interpolated data to the list
            interpolated_data.append(group_15_min)

        # Concatenate all the interpolated data into a single DataFrame
        df_interpolated = pd.concat(interpolated_data)

        if self.csv_write:
            # Write to CSV
            filename = r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\interpolated_data.csv"
            print("Interpolation completed. Data is now at 15-minute intervals.")
            print(f"Saved to CSV at {filename}")
            df_interpolated.to_csv(filename)
        
        return df_interpolated

    def plot_data(self, latitude, longitude, num_days=1):
        """
        Plots both the original and interpolated data for a given latitude and longitude.
        
        Parameters:
        latitude (float): Latitude of the desired location.
        longitude (float): Longitude of the desired location.
        num_days (int): Number of days from the start of the data to plot.
        """
        # Filter the DataFrame for the specified latitude and longitude
        original_data = self.df[(self.df['latitude'] == latitude) & (self.df['longitude'] == longitude)]
        
        if original_data.empty:
            print("No data available for the specified latitude and longitude.")
            return

        # Keep only the specified number of days of data
        start_time = original_data.index.min()
        end_time = start_time + pd.Timedelta(days=num_days)
        original_data = original_data[(original_data.index >= start_time) & (original_data.index < end_time)]
        
        # Interpolate to 15-minute intervals
        interpolated_data = self.assign_values_to_15_min_intervals()
        interpolated_data = interpolated_data[(interpolated_data['latitude'] == latitude) & 
                                              (interpolated_data['longitude'] == longitude)]
        interpolated_data = interpolated_data[(interpolated_data.index >= start_time) & 
                                              (interpolated_data.index < end_time)]
        
        # Plot the original and interpolated data
        plt.figure(figsize=(12, 6))
        plt.plot(original_data.index, original_data['value'], label='Original Data', marker='o', linestyle='-', color='b')
        plt.plot(interpolated_data.index, interpolated_data['value'], label='Interpolated Data', marker='x', linestyle='--', color='r')
        plt.title(f'Data for Latitude: {latitude}, Longitude: {longitude} (First {num_days} Days)')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Specify the path to your GRIB file
    grib_file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\JanuaryData.grib'
    
    # Create an instance of the GribOpener
    opener = GribOpener(grib_file_path, write=True)
       # Split the GRIB file into separate files based on variables
    opener.split_grib_by_variable()
    # opener.list_grib_variables(grib_file_path)
    
    # # Read the GRIB data for a specific variable
    # # variable_name = "Surface short-wave (solar) radiation downwards"
    # variable_name = "10m u-component of wind"
    # opener.read_grib_to_df(variable_name)
    
    # # Example: Plot original and interpolated data for a specific latitude, longitude, and number of days
    # example_latitude = 27.75  # Replace with your desired latitude
    # example_longitude = -85.25  # Replace with your desired longitude
    # opener.plot_data(example_latitude, example_longitude, num_days=1)
