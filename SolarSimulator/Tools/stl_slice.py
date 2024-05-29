from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Load the STL file
file_path = r'SampleData\STL\WhalePlane.stl'  # Adjust the path if necessary
mesh = trimesh.load(file_path)

# Define the plane for the cross-section
plane_origin = [0.4, 0.0, 0.1]  # Origin of the plane
plane_normal = [0.0, 0.0, 1.0]  # Normal to the plane (XY plane)

# Get the cross-section
cross_section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

def merge_polygons(cross_section):
    if cross_section is None:
        print("No cross section found at the given plane.")
        return 0, None

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

    return merged_polygons, polygons

# Plot the merged geometry
def plot_merged_geometry(merged_polygon):
    fig, ax = plt.subplots()

    if isinstance(merged_polygon,Polygon):
        x, y = merged_polygon.exterior.xy
        ax.plot(y, x)
    elif isinstance(merged_polygon,MultiPolygon):
        for i, polygon in enumerate(merged_polygon.geoms):  # Use .geoms to iterate over MultiPolygon
            x, y = polygon.exterior.xy  # Extract the exterior coordinates
            ax.plot(y, x, label=f'Polygon {i + 1}')

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Merged Cross Section Geometry')
    plt.grid(True)
    plt.show()

def plot_geometry(polygons):
    fig, ax = plt.subplots()
    i = 0
    for polygon in polygons:
        x, y = polygon.exterior.xy
        # x = [xi * -1 for xi in x]
        ax.plot(y, x, label = i)
        i+=1

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Cross Section Geometry')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

# Example Usage
merged_polygon, polygons = merge_polygons(cross_section)
print(f"Total enclosed area: {merged_polygon.area}")
plot_geometry(polygons)
plot_merged_geometry(merged_polygon)