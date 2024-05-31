from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import split
from shapely.ops import unary_union
import numpy as np
import trimesh
import matplotlib.pyplot as plt

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
        ax.plot(x, y)
    elif isinstance(merged_polygon,MultiPolygon):
        for i, polygon in enumerate(merged_polygon.geoms):  # Use .geoms to iterate over MultiPolygon
            x, y = polygon.exterior.xy  # Extract the exterior coordinates
            ax.plot(x, y, label=f'Polygon {i + 1}')

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
        ax.plot(x, y, label = i)
        i+=1

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Cross Section Geometry')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def second_moment_of_area(polygon):
    if isinstance(polygon, Polygon):
        polygons = [polygon]
    elif isinstance(polygon, MultiPolygon):
        polygons = polygon.geoms
    else:
        raise TypeError("Input must be a Polygon or MultiPolygon")

    ix_total = 0
    iy_total = 0
    ixy_total = 0

    for poly in polygons:
        x, y = poly.exterior.xy
        x = np.array(x)
        y = np.array(y)
        
        # Calculate the second moments of area using Green's theorem
        a = 0
        ix = 0
        iy = 0
        ixy = 0
        
        for i in range(len(x) - 1):
            xi = x[i]
            yi = y[i]
            xi1 = x[i + 1]
            yi1 = y[i + 1]
            
            ai = xi * yi1 - xi1 * yi
            a += ai
            ix += (yi**2 + yi * yi1 + yi1**2) * ai
            iy += (xi**2 + xi * xi1 + xi1**2) * ai
            ixy += (xi * yi1 + 2 * xi * yi + 2 * xi1 * yi1 + xi1 * yi) * ai
        
        a /= 2
        ix = abs(ix) / 12
        iy = abs(iy) / 12
        ixy = abs(ixy) / 24

        ix_total += ix
        iy_total += iy
        ixy_total += ixy
    
    return ix_total, iy_total, ixy_total

def cut_polygon_at_x(polygon, x_cutoff, cut_direction):
    # Create a vertical line at the x_cutoff
    min_x, min_y, max_x, max_y = polygon.bounds
    cutting_line = LineString([(x_cutoff, min_y), (x_cutoff, max_y)])
    
    # Split the polygon with the cutting line
    split_polygons = split(polygon, cutting_line)
    
    # Filter the parts based on the cut direction
    if cut_direction == "left":
        remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[0] < x_cutoff]
    elif cut_direction == "right":
        remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[2] > x_cutoff]
    else:
        raise ValueError("cut_direction must be 'left' or 'right'")
    
    if not remaining_polygons:
        return None  # Return None if nothing remains
    
    # Combine remaining parts into a single polygon or multipolygon
    if len(remaining_polygons) == 1:
        return remaining_polygons[0]
    else:
        return MultiPolygon(remaining_polygons)

def cut_at_plane(geometry: Polygon | MultiPolygon, x_cutoff, cut_direction):
    if isinstance(geometry, Polygon):
        return cut_polygon_at_x(geometry, x_cutoff, cut_direction)
    elif isinstance(geometry, MultiPolygon):
        remaining_parts = [cut_polygon_at_x(poly, x_cutoff, cut_direction) for poly in geometry.geoms]
        remaining_parts = [part for part in remaining_parts if part is not None]
        if not remaining_parts:
            return None  # Return None if nothing remains
        if len(remaining_parts) == 1:
            return remaining_parts[0]
        else:
            return remaining_parts
    else:
        raise TypeError("Input must be a Polygon or MultiPolygon")
    
def plot_polygon(polygon: Polygon | MultiPolygon, ax, **kwargs):
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        ax.plot(x, y, **kwargs)
        ax.set_aspect('equal', adjustable='box')

    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, **kwargs)
        ax.set_aspect('equal', adjustable='box')

def plot_xsec(polygon: list | Polygon | MultiPolygon, ax, **kwargs):
    if isinstance(polygon, list):
        for poly in polygon:
            plot_polygon(poly, ax, **kwargs)
    if polygon.is_empty:
        return
    if isinstance(polygon, Polygon) or isinstance(polygon,MultiPolygon):
        plot_polygon(polygon, ax,**kwargs)

def calculate_centroid(polgon: Polygon | MultiPolygon):
    if isinstance(polgon,polgon):
        polgon.c

# Load the STL file
file_path = r'SampleData\STL\WhalePlane.stl'  # Adjust the path if necessary
mesh = trimesh.load(file_path)

# Define the plane for the cross-section
plane_origin = [0.4, 0.0, 0.0]  # Origin of the plane
plane_normal = [1.0, 0.0, 0.0]  # Normal to the plane (XY plane)

# Get the cross-section
cross_section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

# Example Usage
merged_polygon, polygons = merge_polygons(cross_section)
# print(second_moment_of_area(merged_polygon))
# print(f"Total enclosed area: {merged_polygon.area}")
plot_geometry(polygons)
plot_merged_geometry(merged_polygon)

x_cutoff = 0.0
cut_direction = "left"
result = cut_at_plane(merged_polygon, x_cutoff, cut_direction)

fig, ax = plt.subplots()
plot_polygon(merged_polygon, ax, label='Original', color='blue')
if result:
    plot_polygon(result, ax, label='Cut', color='red')
ax.axvline(x=x_cutoff, color='gray', linestyle='--', label='Cutting line')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Polygon Cut Example')
ax.legend(loc='upper right')
plt.show()

plot_merged_geometry(result)