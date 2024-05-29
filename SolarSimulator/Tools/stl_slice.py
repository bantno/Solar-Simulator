import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Load the STL file
file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SampleData\STL\test.stl'  # Adjust the path if necessary
mesh = trimesh.load(file_path)

# Define the plane for the cross-section
plane_origin = [0, 0, 0]  # Origin of the plane
plane_normal = [0, 1, 0]  # Normal to the plane (XY plane)

# Get the cross-section
cross_section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

def get_enclosed_area(cross_section):
    if cross_section is None:
        print("No cross section found at the given plane.")
        return 0

    slice_2D, to_3D = cross_section.to_planar()

    polygons = []
    for path in slice_2D.entities:
        if hasattr(path, 'discrete'):
            vertices = slice_2D.vertices
            polygon = Polygon(path.discrete(vertices))
            if polygon.is_valid:
                polygons.append(polygon)

    # Use unary_union to merge overlapping polygons and avoid double counting
    merged_polygons = unary_union(polygons)

    # Calculate the total enclosed area
    total_area = merged_polygons.area
    return total_area

total_enclosed_area = get_enclosed_area(cross_section)
print(f"Total enclosed area: {total_enclosed_area}")

# Plot the cross-section for visualization
def plot_cross_section(cross_section):
    if cross_section is None:
        print("No cross section found at the given plane.")
        return

    slice_2D, to_3D = cross_section.to_planar()

    fig, ax = plt.subplots()
    for path in slice_2D.entities:
        if hasattr(path, 'discrete'):
            vertices = slice_2D.vertices
            ax.plot(*path.discrete(vertices).T)

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Cross Section of STL File')
    plt.grid(True)
    plt.show()

plot_cross_section(cross_section)
