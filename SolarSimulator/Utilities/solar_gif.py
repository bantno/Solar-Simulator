import pygrib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

class SolarRadiationPlotter:
    def __init__(self, grib_file):
        """
        Initializes the class with the GRIB file and reads the SSRD data into a DataFrame.
        
        Parameters:
        grib_file (str): Path to the GRIB file containing SSRD data.
        """
        self.grib_file = grib_file
        self.df = self.read_ssrd_grib_to_df()

    def read_ssrd_grib_to_df(self):
        """
        Reads the GRIB file and extracts SSRD values with latitude, longitude, and time.

        Returns:
        pd.DataFrame: DataFrame containing SSRD data with time as the index.
        """
        # Open the GRIB file
        grbs = pygrib.open(self.grib_file)
        
        # Initialize list to store data
        data = []
        
        # Iterate through the GRIB messages for SSRD
        for grb in tqdm(grbs):
            if grb.name == 'Surface short-wave (solar) radiation downwards' and grb.typeOfLevel == 'surface':
                # Extract the necessary details
                time = grb.validDate
                latitudes, longitudes = grb.latlons()
                ssrd_values = grb.values
                
                # Flatten lat/lon and SSRD for ease of storage in DataFrame
                for i in range(len(latitudes)):
                    for j in range(len(latitudes[i])):
                        data.append([time, latitudes[i][j], longitudes[i][j], ssrd_values[i][j]])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['time', 'latitude', 'longitude', 'ssrd'])
        
        # Set time as index
        df.set_index('time', inplace=True)

        df.to_csv(r"C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\srad")
        
        return df

    def plot_ssrd_contour_with_slider(self):
        """
        Creates a contour plot using Plotly with latitude and longitude as axes and SSRD as the value.
        A slider is provided to animate over time with a consistent color scale across time steps.
        """
        # Get unique times
        times = self.df.index.unique()

        # Determine global min and max for SSRD values to ensure consistent colors
        zmin = self.df['ssrd'].min()
        zmax = self.df['ssrd'].max()
        
        # Initialize the figure
        fig = go.Figure()

        # Create a contour plot for each time step
        for time in times:
            df_time = self.df.loc[time]
            fig.add_trace(go.Contour(
                z=df_time.pivot_table(index='latitude', columns='longitude', values='ssrd').values,
                x=df_time['longitude'].unique(),
                y=df_time['latitude'].unique(),
                zmin=zmin,  # Set fixed min for color scale
                zmax=zmax,  # Set fixed max for color scale
                contours_coloring='heatmap',
                colorbar_title="SSRD",
                showscale=True,
                visible=False  # Initially hide all traces
            ))

        # Make the first contour visible by default
        fig.data[0].visible = True

        # Create slider steps
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],  # Hide all traces
                label=str(times[i])  # Set label to time
            )
            step["args"][0]["visible"][i] = True  # Show only the current time trace
            steps.append(step)

        # Define the slider layout
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]

        # Update figure layout
        fig.update_layout(
            title="SSRD Contour Plot over Time",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            sliders=sliders
        )

        # Display the plot
        fig.show()

    def create_gif_from_ssrd(self, output_gif_path='ssrd_animation.gif', font_size=20, duration=2, show_time=True):
        """
        Create a GIF from SSRD data and optionally add time at the bottom of each frame.

        Parameters:
        - output_gif_path: str, path where the GIF will be saved.
        - font_size: int, size of the font for the time label (default is 20).
        - duration: int, duration between frames in the GIF (default is 200ms).
        - show_time: bool, whether to include the time text in the GIF frames (default is True).
        """
        # Initialize a list to store frames
        frames = []

        # Get unique times
        times = self.df.index.unique()

        # Determine global min and max for SSRD values to ensure consistent color normalization
        ssrd_min = self.df['ssrd'].min()
        ssrd_max = self.df['ssrd'].max()

        # Create colormap (using matplotlib)
        colormap = plt.get_cmap('viridis')

        # Try to load a default font for time text; fall back if not available
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Iterate through each time step and generate the corresponding frame
        for time in tqdm(times):
            # Get data for the specific time
            df_time = self.df.loc[time]

            # Pivot the data to create a 2D array of SSRD values
            ssrd_values = df_time.pivot_table(index='latitude', columns='longitude', values='ssrd').values

            # Normalize SSRD data to the range [0, 255] for image creation
            normalized_data = ((ssrd_values - ssrd_min) / (ssrd_max - ssrd_min) * 255).astype(np.uint8)

            # Apply the colormap to create an RGB image
            colored_image = colormap(normalized_data / 255.0)  # Normalize to [0, 1] for colormap
            colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB format

            # Convert the RGB array into an image
            img = Image.fromarray(colored_image)

            # Draw the time on the image if show_time is True
            if show_time:
                draw = ImageDraw.Draw(img)
                text = time.strftime("%Y-%m-%d %H:%M:%S")  # Convert time to string format
                
                # Use textbbox to calculate the size of the text
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Position the text at the bottom center of the image
                img_width, img_height = img.size
                text_x = (img_width - text_width) // 2
                text_y = img_height - text_height - 10  # Slight padding from bottom
                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))  # White text color

            # Append the image with optional time annotation to the frames
            frames.append(img)

        # Save the frames as a GIF
        if frames:
            frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)
            print(f"GIF saved as '{output_gif_path}'.")
        else:
            print("No frames found for the SSRD data.")

if __name__ == "__main__":
    # Usage Example
    plotter = SolarRadiationPlotter(r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\solar_data.grib')
    # plotter.plot_ssrd_contour_with_slider()
    plotter.create_gif_from_ssrd('ssrd_animation.gif',show_time=False)
