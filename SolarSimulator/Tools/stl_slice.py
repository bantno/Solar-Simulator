from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.affinity import translate
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
        y, x = merged_polygon.exterior.xy
        ax.plot(x, y)
    elif isinstance(merged_polygon,MultiPolygon):
        for i, polygon in enumerate(merged_polygon.geoms):  # Use .geoms to iterate over MultiPolygon
            y, x = polygon.exterior.xy  # Extract the exterior coordinates
            ax.plot(x, y, label=f'Polygon {i + 1}')

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('Y-axis')
    plt.ylabel('X-axis')
    plt.title('Merged Cross Section Geometry')
    plt.grid(True)
    plt.show()

def plot_geometry(polygons):
    fig, ax = plt.subplots()
    i = 0
    for polygon in polygons:
        y, x = polygon.exterior.xy
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
    """Determine second moment of area of a polygon
    
    Parameters:
    polygon (Polygon): The original Shapely polygon.

    Returns:
    tuple: The second moments of area (Ixx, Iyy, Ixy).
    
    """

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

def cut_polygon(polygon, cutoff, plane, cut_direction):
    """
    Cuts a polygon along a specified plane (x or y) at a given cutoff coordinate.

    Parameters:
    polygon (Polygon): The polygon to be cut.
    cutoff (float): The coordinate along the plane at which to cut.
    plane (str): The plane along which to cut ('x' or 'y').
    cut_direction (str): The direction to keep ('left' or 'right' for plane 'x', 'below' or 'above' for plane 'y').

    Returns:
    Polygon or MultiPolygon: The remaining part of the polygon after the cut.
    None: If no part remains after the cut.
    """
    
    # Create a cutting line based on the specified plane and cutoff
    min_x, min_y, max_x, max_y = polygon.bounds
    if plane == 'x':
        cutting_line = LineString([(cutoff, min_y), (cutoff, max_y)])
    elif plane == 'y':
        cutting_line = LineString([(min_x, cutoff), (max_x, cutoff)])
    else:
        raise ValueError("plane must be 'x' or 'y'")
    
    # Split the polygon with the cutting line
    split_polygons = split(polygon, cutting_line)
    
    # Filter the parts based on the cut direction and plane
    if plane == 'x':
        if cut_direction == "left":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[0] < cutoff]
        elif cut_direction == "right":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[2] > cutoff]
        else:
            raise ValueError("cut_direction must be 'left' or 'right'")
    elif plane == 'y':
        if cut_direction == "below":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[1] < cutoff]
        elif cut_direction == "above":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[3] > cutoff]
        else:
            raise ValueError("cut_direction must be 'below' or 'above'")
    
    if not remaining_polygons:
        return None  # Return None if nothing remains
    
    # Combine remaining parts into a single polygon or multipolygon
    if len(remaining_polygons) == 1:
        return remaining_polygons[0]
    else:
        return MultiPolygon(remaining_polygons)

def cut_at_plane(geometry, cutoff, plane, cut_direction):
    if isinstance(geometry, Polygon):
        return cut_polygon(geometry, cutoff, plane, cut_direction)
    elif isinstance(geometry, MultiPolygon):
        remaining_parts = [cut_polygon(poly, cutoff, plane, cut_direction) for poly in geometry.geoms]
        remaining_parts = [part for part in remaining_parts if part is not None]
        if not remaining_parts:
            return None  # Return None if nothing remains
        if len(remaining_parts) == 1:
            return remaining_parts[0]
        else:
            return MultiPolygon(remaining_parts)
    else:
        raise TypeError("Input must be a Polygon or MultiPolygon")
    
def plot_polygon(polygon: Polygon | MultiPolygon, ax, **kwargs):
    if isinstance(polygon, Polygon):
        y, x = polygon.exterior.xy
        ax.plot(x, y, **kwargs)
        ax.set_aspect('equal', adjustable='box')

    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            y, x = poly.exterior.xy
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

def set_new_origin(polygon, new_origin_x, new_origin_y):
    """
    Shift the origin of a Shapely polygon to a new origin (new_origin_x, new_origin_y).

    Parameters:
    polygon (Polygon): The original Shapely polygon.
    new_origin_x (float): The x-coordinate of the new origin.
    new_origin_y (float): The y-coordinate of the new origin.

    Returns:
    Polygon: A new polygon with the origin shifted to (0, 0).
    """
    # Calculate the offsets to shift the new origin to (0, 0)
    x_shift = -new_origin_x
    y_shift = -new_origin_y

    # Translate the polygon
    shifted_polygon = translate(polygon, xoff=x_shift, yoff=y_shift)

    return shifted_polygon

def calculate_mc(Izz: float, W: float, h_cb: float, rho_w: float):
    return rho_w*(Izz/W)-h_cb

def calculate_draft(weight: float):
    pass

def calculate_hstab(file_path, plane_origin, plane_normal,cutoff,plane,cut_direction):
    # Get the cross-section
    cross_section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    merged_polygon, polygons = merge_polygons(cross_section)
    # plot_geometry(polygons)
    # plot_merged_geometry(merged_polygon)
    result = cut_at_plane(merged_polygon, cutoff, plane, cut_direction)

    fig, ax = plt.subplots()
    plot_polygon(merged_polygon, ax, label='Original', color='blue')
    if result:
        plot_polygon(result, ax, label='Cut', color='red')
    # ax.axvline(x=cutoff, color='gray', linestyle='--', label='Cutting line')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_title('Polygon Cut Example')
    # ax.legend(loc='upper right')


    # plot_merged_geometry(result)
    submerged = set_new_origin(result,result.centroid.x,result.centroid.y)

    # print("Second Moment of Area: \n{0}\n{1}".format({second_moment_of_area(result)},{second_moment_of_area(submerged)}))
    Izz = np.max(second_moment_of_area(submerged))
    print("Izz: {0}".format(Izz))


    # weight = 1334.47/9.81 # for metric this is mass in kg
    weight = 8
    rho_w = 1001.15 # density of water [kg/m^3]
    # h_cb = 0.2286-0.13289/2 #h_cg-0.066 # needs to be the vertical distance between the CG and the CB
    # h_cg = (0.3048)/4 # height of center of gravity
    cg = 0.0
    cb = result.centroid.x
    h_cb = cg - cb
    h_mc = calculate_mc(Izz,weight,h_cb,rho_w)
    ax.plot(0,cg,color='red',marker = 'o')
    ax.plot(0,cb,color='green',marker='o')
    fig.tight_layout()
    plt.show()


    print("Center of Buoyancy [m]: {0}".format(h_cb))
    print("Height of Metacenter [ft]: {0}".format(h_mc*3.281))



# Example

# Load the STL file
file_path = r'SampleData\STL\WhalePlane.stl'  # Adjust the path if necessary
mesh = trimesh.load(file_path)

# Define the plane for the cross-section
plane_origin = [0.4, 0.0, 0.0]  # Origin of the plane
plane_normal = [1.0, 0.0, 0.0]  # Normal to the plane (XY plane)

# TODO: write function to determine waterline (cuttoff value)
# cutoff = -0.3048/2+0.1328928 
cutoff = -0.07
plane = "x"  # or "y"
cut_direction = "left" # or "right" for plane "x", "below" or "above" for plane "y"

calculate_hstab(file_path,plane_origin,plane_normal,cutoff,plane,cut_direction)


# Define the plane for the cross-section
plane_normal = [0.0, 1.0, 0.0]  # Normal to the plane (XY plane)

calculate_hstab(file_path,plane_origin,plane_normal,cutoff,plane,cut_direction)
