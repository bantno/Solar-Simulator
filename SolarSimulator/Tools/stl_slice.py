from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.affinity import translate
from shapely.ops import split, transform, unary_union

import numpy as np

import trimesh

import matplotlib.pyplot as plt

def merge_polygons(cross_section):
    """
    Merge overlapping polygons from a given cross section.

    This function takes a cross section, converts it to a 2D planar representation, 
    and then extracts and merges polygons that do not overlap. It uses the unary_union 
    operation from the Shapely library to merge any overlapping polygons to avoid 
    double counting.

    Parameters:
    cross_section (CrossSection): An object representing the cross section of a 
                                   3D object. This object must have a `to_planar` 
                                   method that converts the cross section to a 2D 
                                   planar representation.

    Returns:
    tuple: A tuple containing:
        - merged_polygons (Polygon or MultiPolygon): A Shapely polygon or 
          multipolygon resulting from merging the individual polygons in the 
          cross section.
        - polygons (list of Polygon): A list of individual Shapely polygons 
          before merging.

    Notes:
    - If the cross_section is None, the function will print a message and return 
      (0, None).
    """

    if cross_section is None:
        print("No cross section found at the given plane.")
        return 0, None
    to_2D = trimesh.geometry.align_vectors(plane_normal, [0,0,-1])
    slice_2D, to_3D = cross_section.to_planar(to_2D=to_2D)
    
    polygons = []
    for poly in slice_2D.polygons_closed:
        polygons.append(transform(lambda x, y: (y, x), poly))

    # # Use unary_union to merge overlapping polygons and avoid double counting
    merged_polygons = unary_union(polygons)

    return merged_polygons, polygons

# Plot the merged geometry
def plot_merged_geometry(merged_polygon):
    """
    Plot a merged polygon or multipolygon using Matplotlib.

    This function takes a Shapely polygon or multipolygon resulting from a 
    cross-sectional merge operation and plots it using Matplotlib. It 
    distinguishes between single polygons and multipolygons and plots them 
    accordingly.

    Parameters:
    merged_polygon (Polygon or MultiPolygon): A Shapely polygon or multipolygon 
                                              representing the merged geometry 
                                              of a cross section.
    """
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
    """
    Plot a Shapely polygon using Matplotlub

    Parameters:
    polygons (polygon or multipolygon): A Shapely polygon representing a cross section
    """
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
    """Determine second moment of area of a polygon.
    
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
    cut_direction (str): The direction to keep ('left' or 'right' for plane 'x',
                         'below' or 'above' for plane 'y').

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
    """
    Cuts a geometry (polygon or multipolygon) along a specified plane (x or y) at a given cutoff coordinate.

    Parameters:
    geometry (Polygon or MultiPolygon): The geometry to be cut.
    cutoff (float): The coordinate along the plane at which to cut.
    plane (str): The plane along which to cut ('x' or 'y').
    cut_direction (str): The direction to keep ('left' or 'right' for plane 'x', 'below' or 'above' for plane 'y').

    Returns:
    Polygon or MultiPolygon: The remaining part of the geometry after the cut.
    None: If no part remains after the cut.
    """
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
    """
    Plots a polygon or multipolygon on the given axes.

    Parameters:
    polygon (Polygon or MultiPolygon): The polygon to plot.
    ax (matplotlib.axes.Axes): The axes on which to plot the polygon.
    **kwargs: Additional keyword arguments to pass to the plot function.
    """
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
    """
    Plots a cross section on the given axes.

    Parameters:
    polygon (list, Polygon or MultiPolygon): The polygon(s) to plot.
    ax (matplotlib.axes.Axes): The axes on which to plot the polygon.
    **kwargs: Additional keyword arguments to pass to the plot function.
    """
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
    """
    Calculate metacenter as per Equation XX in Gudmundsson

    Parameters:
    Izz (float): Second moment of area.
    W (float): Mass or weight of object. [kg or lbf]
    h_cb (float): Distance between center of buoyancy and center of gravity.
    rho_w (float): Density of fluid. [kg/m^3 or lbf/ft^3]
    
    Returns:
    mc (float): Metacenter height. [m or ft]

    """
    mc = rho_w*(Izz/W)-h_cb
    return mc

def calculate_draft(weight: float):
    """
    Calculate the draft of the object

    Parameters:
    weight (float): Weight of the object.
    """
    pass

def plot_mesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    plt.show()
    
    return ax

def calculate_hstab(file_path, plane_origin, plane_normal,cutoff,plane,cut_direction,weight,cg,rho_w = 1001.15):
    # Get the cross-section
    mesh = trimesh.load(file_path)
    # Find the vertex with the smallest x value
    min_x_vertex = mesh.vertices[mesh.vertices[:, 0].argmin()]

    # Calculate the translation vector needed to move this vertex to the origin
    translation_vector = -min_x_vertex

    # Apply the translation to the mesh
    mesh.apply_translation(translation_vector)
    # plot_mesh(mesh)



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
    ax.set_xlabel('X-Axis [m]')
    ax.set_ylabel('Y-Axis [m]')
    ax.set_title('Submerged Plane')
    # ax.legend(loc='upper right')
    submerged = set_new_origin(result,result.centroid.x,result.centroid.y)

    # print("Second Moment of Area: \n{0}\n{1}".format({second_moment_of_area(result)},{second_moment_of_area(submerged)}))
    Izz = np.max(second_moment_of_area(submerged))
    print("Izz: {0}".format(Izz))
    cb = (result.centroid.x,result.centroid.y)
    if plane_normal[0] == 1.0:
        h_cb = cg[2] - cb[1]
        ax.plot(cg[1],cg[2],color='red',marker = 'o')
    elif plane_normal[1] == 1.0:
        h_cb = cg[2] - cb[1]
        ax.plot(-cg[0],cg[2],color='red',marker = 'o')
    
    h_mc = calculate_mc(Izz,weight,h_cb,rho_w)
    
    ax.plot(cb[0],cb[1],color='green',marker='o')
    plt.tight_layout()
    plt.show()


    print("Center of Buoyancy [m]: {0}".format(h_cb))
    print("Height of Metacenter [ft]: {0}".format(h_mc*3.281))



# Example

# Load the STL file
file_path = r'SampleData\STL\WhalePlane2.stl'  # Adjust the path if necessary



# Transverse Stability
# Define the plane 
# for the cross-section
plane_origin = [0.45, 0.0, 0.0]  # Origin of the plane
plane_normal = [1.0, 0.0, 0.0]  # Normal to the plane (XY plane)

# TODO: write function to determine waterline (cuttoff value)
# cutoff = -0.3048/2+0.1328928 
cutoff = 0.00
plane = "y"  # or "y"
cut_direction = "below" # or "right" for plane "x", "below" or "above" for plane "y"

# weight = 1334.47/9.81 # for metric this is mass in kg
weight = 8 # [kg]
rho_w = 1001.15 # density of water [kg/m^3]
# h_cb = 0.2286-0.13289/2 #h_cg-0.066 # needs to be the vertical distance between the CG and the CB
# h_cg = (0.3048)/4 # height of center of gravity
cg = (0.337,0.000,0.053)


calculate_hstab(file_path,plane_origin,plane_normal,cutoff,plane,cut_direction,weight,cg)


# Longitudinal Stability
# Define the plane for the cross-section
plane_origin = [0.0, 0.0, 0.0]  # Origin of the plane
plane_normal = [0.0, 1.0, 0.0]  # Normal to the plane (XY plane)

calculate_hstab(file_path,plane_origin,plane_normal,cutoff,plane,cut_direction,weight,cg)
