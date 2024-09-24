import pygrib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import BytesIO
from tqdm import tqdm

def create_gif_with_coastline_and_time(file_path, output_gif_path):
    """
    Create a GIF from GRIB data for total wind speed, overlayed with coastline and time.
    
    Parameters:
    - file_path: str, path to the GRIB file.
    - output_gif_path: str, path where the GIF will be saved.
    """
    # Open the GRIB file
    grbs = pygrib.open(file_path)
    
    # Initialize a list to store frames
    frames = []

    # Initialize lists to store u and v components and corresponding times
    u_components = []
    v_components = []
    times = []

    # Extract messages for u and v components and times
    for g in grbs:
        if g.name == '10 metre U wind component':
            u_data, lats, lons = g.data()
            u_components.append(u_data)
            times.append(g.validDate)  # Store the time for each frame
        elif g.name == '10 metre V wind component':
            v_data, _, _ = g.data()
            v_components.append(v_data)

    # Close the GRIB file
    grbs.close()

    # Create a matplotlib figure once
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')

    # Plot coastline first, so it appears on all frames
    ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=ccrs.PlateCarree())

    # Calculate total wind speed and create frames with a progress bar
    for u_data, v_data, time in tqdm(zip(u_components, v_components, times), total=len(u_components), desc="Creating GIF frames"):
        # Calculate wind speed
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        
        # Normalize data to the range [0, 1] for image creation
        normalized_data = (wind_speed - np.min(wind_speed)) / (np.max(wind_speed) - np.min(wind_speed))
        
        # Create a color image using a colormap
        colormap = plt.get_cmap('viridis')
        colored_image = colormap(normalized_data)  # This gives RGBA
        
        # Convert the RGBA array to RGB (ignoring alpha channel)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

        # Plot the wind speed data
        ax.imshow(colored_image, extent=[np.min(lons), np.max(lons), np.min(lats), np.max(lats)],
                  origin='upper', transform=ccrs.PlateCarree(), alpha=0.5)

        # Add timestamp text
        ax.text(0.5, -0.1, time.strftime('%Y-%m-%d %H:%M:%S'), ha='center', va='top', fontsize=12,
                transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.7))

        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.axis('off')  # Turn off the axis
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)  # Move to the beginning of the BytesIO buffer
        frames.append(Image.open(buf))
        ax.clear()  # Clear the axis for the next frame
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')  # Re-add coastline for each frame
        ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=ccrs.PlateCarree())

    # Save the frames as a GIF
    if frames:
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=200, loop=0)
        print(f"GIF saved as '{output_gif_path}'.")
    else:
        print("No frames found for the specified variables.")

# Example usage
create_gif_with_coastline_and_time(
    file_path=r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\MeteorDataTest\test.grib',
    output_gif_path='output_with_coastline_and_time.gif'
)