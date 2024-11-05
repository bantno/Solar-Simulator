import pygrib
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_gif_from_grib(file_path, output_gif_path):
    """
    Create a GIF from GRIB data for total wind speed.
    
    Parameters:
    - file_path: str, path to the GRIB file.
    - output_gif_path: str, path where the GIF will be saved.
    """
    # Open the GRIB file
    grbs = pygrib.open(file_path)
    
    # Initialize a list to store frames
    frames = []

    # Initialize lists to store u and v components
    u_components = []
    v_components = []

    # Extract messages for u and v components
    for g in grbs:
        if g.name == '10 metre U wind component':
            u_data, lats, lons = g.data()
            u_components.append(u_data)
        elif g.name == '10 metre V wind component':
            v_data, _, _ = g.data()
            v_components.append(v_data)

    # Close the GRIB file
    grbs.close()

    # Calculate total wind speed and create frames
    for u_data, v_data in tqdm(zip(u_components, v_components)):
        # Calculate wind speed
        wind_speed = np.sqrt(u_data**2 + v_data**2)
        
        # Normalize data to the range [0, 255] for image creation
        normalized_data = ((wind_speed - np.min(wind_speed)) / (np.max(wind_speed) - np.min(wind_speed)) * 255).astype(np.uint8)
        
        # Create a color image using a colormap
        colormap = plt.get_cmap('viridis')
        colored_image = colormap(normalized_data / 255.0)  # Normalize to [0, 1] for colormap
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB format
        
        # Convert the RGB array into an image
        img = Image.fromarray(colored_image)
        frames.append(img)

    # Save the frames as a GIF
    if frames:
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=200, loop=0)
        print(f"GIF saved as '{output_gif_path}'.")
    else:
        print("No frames found for the specified variables.")

# Example usage
create_gif_from_grib(
    file_path=r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\Data\MeteorDataTest\test.grib',
    output_gif_path='output.gif'
)
