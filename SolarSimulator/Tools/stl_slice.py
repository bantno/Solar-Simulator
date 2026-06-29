import os

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import split, transform, unary_union

import numpy as np

import trimesh

import matplotlib.pyplot as plt


def merge_polygons(cross_section, normal):
    """Merge overlapping polygons from a given cross section.

    This function takes a cross section, converts it to a 2D planar representation,
    and then extracts and merges polygons that do not overlap. It uses the unary_union
    operation from the Shapely library to merge any overlapping polygons to avoid
    double counting.

    Args:
        cross_section (CrossSection): An object representing the cross section of a
            3D object. This object must have a `to_planar` method that converts the
            cross section to a 2D planar representation.

    Returns:
        tuple: A tuple containing:
            - merged_polygons (Polygon or MultiPolygon): A Shapely polygon or
              multipolygon resulting from merging the individual polygons in the
              cross section.
            - polygons (list of Polygon): A list of individual Shapely polygons
              before merging.

    Notes:
        If the cross_section is None, the function will print a message and return
        (0, None).
    """

    if cross_section is None:
        print("No cross section found at the given plane.")
        return 0, None
    planar = trimesh.geometry.align_vectors(normal, [0, 0, -1])
    slice_, _ = cross_section.to_planar(to_2D=planar)
    polygons = []
    for poly in slice_.polygons_closed:
        polygons.append(transform(lambda x, y: (y, x), poly))

    # # Use unary_union to merge overlapping polygons and avoid double counting
    merged_polygons = unary_union(polygons)

    return merged_polygons, polygons


# Plot the merged geometry
def plot_merged_geometry(merged_polygon):
    """Plot a merged polygon or multipolygon using Matplotlib.

    This function takes a Shapely polygon or multipolygon resulting from a
    cross-sectional merge operation and plots it using Matplotlib. It
    distinguishes between single polygons and multipolygons and plots them
    accordingly.

    Args:
        merged_polygon (Polygon or MultiPolygon): A Shapely polygon or multipolygon
            representing the merged geometry of a cross section.
    """
    _, ax = plt.subplots()

    if isinstance(merged_polygon, Polygon):
        x, y = merged_polygon.exterior.xy
        ax.plot(x, y)
    elif isinstance(merged_polygon, MultiPolygon):
        for i, polygon in enumerate(merged_polygon.geoms):
            x, y = polygon.exterior.xy  # Extract the exterior coordinates
            ax.plot(x, y, label=f"Polygon {i + 1}")

    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.title('Merged Cross Section Geometry')
    plt.grid(True)
    plt.show()


def plot_geometry(polygons):
    """Plot a Shapely polygon using Matplotlub

    Args:
        polygons (polygon or multipolygon): A Shapely polygon representing a cross section
    """
    _, ax = plt.subplots()
    i = 0
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, label=i)
        i += 1

    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.title('Cross Section Geometry')
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def second_moment_of_area(polygon):
    """Determine second moment of area of a polygon.

    Args:
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


def cut_polygon(polygon, cutoff, plane, direction):
    """Cuts a polygon along a specified plane (x or y) at a given cutoff coordinate.

    Args:
        polygon (Polygon): The polygon to be cut.
        cutoff (float): The coordinate along the plane at which to cut.
        plane (str): The plane along which to cut ('x' or 'y').
        direction (str): The direction to keep ('left' or 'right' for plane 'x',
            'below' or 'above' for plane 'y').

    Returns:
        Polygon or MultiPolygon: The remaining part of the polygon after the cut.
        None: If no part remains after the cut.
    """

    # Create a cutting line based on the specified plane and cutoff
    min_x, min_y, max_x, max_y = polygon.bounds
    if plane == "x":
        cutting_line = LineString([(cutoff, min_y), (cutoff, max_y)])
    elif plane == "y":
        cutting_line = LineString([(min_x, cutoff), (max_x, cutoff)])
    else:
        raise ValueError("plane must be 'x' or 'y'")

    # Split the polygon with the cutting line
    split_polygons = split(polygon, cutting_line)

    # Filter the parts based on the cut direction and plane
    if plane == "x":
        if direction == "left":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[0] < cutoff]
        elif direction == "right":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[2] > cutoff]
        else:
            raise ValueError("cut_direction must be 'left' or 'right'")
    elif plane == "y":
        if direction == "below":
            remaining_polygons = [poly for poly in split_polygons.geoms if poly.bounds[1] < cutoff]
        elif direction == "above":
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


def cut_at_plane(geometry, cutoff, plane, direction):
    """Cuts a geometry (polygon or multipolygon) along a specified plane (x or y)
    at a given cutoff coordinate.

    Args:
        geometry (Polygon or MultiPolygon): The geometry to be cut.
        cutoff (float): The coordinate along the plane at which to cut.
        plane (str): The plane along which to cut ('x' or 'y').
        direction (str): The direction to keep ('left' or 'right' for plane 'x',
            'below' or 'above' for plane 'y').

    Returns:
        Polygon or MultiPolygon: The remaining part of the geometry after the cut.
        None: If no part remains after the cut.
    """
    if isinstance(geometry, Polygon):
        return cut_polygon(geometry, cutoff, plane, direction)
    elif isinstance(geometry, MultiPolygon):
        remaining_parts = [cut_polygon(poly, cutoff, plane, direction) for poly in geometry.geoms]
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
    """Plots a polygon or multipolygon on the given axes.

    Args:
        polygon (Polygon or MultiPolygon): The polygon to plot.
        ax (matplotlib.axes.Axes): The axes on which to plot the polygon.
        **kwargs: Additional keyword arguments to pass to the plot function.
    """
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        ax.plot(x, y, **kwargs)
        ax.set_aspect("equal", adjustable="box")

    elif isinstance(polygon, MultiPolygon):
        for i, poly in enumerate(polygon.geoms):
            x, y = poly.exterior.xy
            if i == 0:
                ax.plot(x, y, **kwargs)
                kwargs.pop("label", None)
            ax.plot(x, y, **kwargs)
        ax.set_aspect("equal", adjustable="box")


def plot_xsec(polygon: list | Polygon | MultiPolygon, ax, **kwargs):
    """Plots a cross section on the given axes.

    Args:
        polygon (list, Polygon or MultiPolygon): The polygon(s) to plot.
        ax (matplotlib.axes.Axes): The axes on which to plot the polygon.
        **kwargs: Additional keyword arguments to pass to the plot function.
    """
    if isinstance(polygon, list):
        for poly in polygon:
            plot_polygon(poly, ax, **kwargs)
    if polygon.is_empty:
        return
    if isinstance(polygon, Polygon) or isinstance(polygon, MultiPolygon):
        plot_polygon(polygon, ax, **kwargs)


def calculate_mc(inertia: float, w: float, h_cb: float, rho: float):
    """Calculate metacenter as per Equation XX in Gudmundsson

    Args:
        Izz (float): Second moment of area.
        W (float): Mass or weight of object. [kg or lbf]
        h_cb (float): Distance between center of buoyancy and center of gravity.
        rho_w (float): Density of fluid. [kg/m^3 or lbf/ft^3]

    Returns:
        mc (float): Metacenter height. [m or ft]
    """
    mc = rho * (inertia / w) - h_cb
    return mc


def calculate_draft(m_kg: float, file_path):
    """Calculate the draft of the object

    Args:
        m_kg (float): mass of the object.
    """
    done = False
    mesh = trimesh.load(file_path)

    plane_origin = [0.0, 0.0, 0.0]
    plane_normal = trimesh.unitize([0.0, 0.0, -1.0])
    m_tol = 0.1

    while not done:
        submerged = mesh.slice_plane(plane_origin, plane_normal)
        displaced_mass = submerged.volume * 1000
        if displaced_mass > m_kg - m_tol and displaced_mass < m_kg + m_tol:
            done = True
        elif displaced_mass > m_kg:
            plane_origin[2] -= 0.001
        elif displaced_mass < m_kg:
            plane_origin[2] += 0.001

    waterline = plane_origin[2]
    min_ind = mesh.vertices[:, 2].argmin()
    draft = waterline - mesh.vertices[min_ind, 2]
    return draft, waterline


def calculate_hstab(
    file_path, filename, origin, normal, cutoff, plane, cut_direction, weight, cg, rho_w=1001.15
):
    """Calculates the height of the hydrostatic stabilizer for a given 3D model.

    Args:
        filename (str): Path to the 3D model file.
        origin (array_like): Origin point of the plane section.
        normal (array_like): Normal vector of the plane section.
        cutoff (float): Cutoff value for the plane section.
        plane (str): Plane type ('xz', 'xy', or 'yz').
        cut_direction (str): Direction of the cut ('positive' or 'negative').
        weight (float): Weight of the object.
        cg (array_like): Center of gravity coordinates.
        rho_w (float, optional): Density of water. Default is 1001.15 kg/m^3.

    Returns:
        None: Prints the height of the center of buoyancy and the height of the metacenter.

    Raises:
        NotImplementedError: If the plane type is not supported.
    """

    # Get the cross-section
    mesh = trimesh.load(file_path)
    # Find the vertex with the smallest x value
    min_x_vertex = mesh.vertices[mesh.vertices[:, 0].argmin()]

    # Calculate the translation vector needed to move this vertex to the origin
    translation_vector = -min_x_vertex

    # Apply the translation to the mesh
    mesh.apply_translation(translation_vector)
    # plot_mesh(mesh)

    cross_section = mesh.section(plane_origin=origin, plane_normal=normal)
    merged_polygon, polygons = merge_polygons(cross_section, normal)

    result = cut_at_plane(merged_polygon, cutoff, plane, cut_direction)

    fig, ax = plt.subplots()
    plot_polygon(merged_polygon, ax, color="blue")
    if result:
        plot_polygon(result, ax, color="red", label="Submerged")

    i_zz = np.max(second_moment_of_area(result))
    print(f"Izz: {i_zz}")
    cb = (result.centroid.x, result.centroid.y)
    if normal[0] == 1.0:
        h_cb = cg[2] - cb[1]
        ax.plot(cg[1], cg[2], color="red", marker="o", label="Center of Gravity")
        # ax.set_title('Lateral Hydrostatic Stability')
        ax.set_xlabel("Y-Axis [m]")
        ax.set_ylabel("Z-Axis [m]")
    elif normal[1] == 1.0:
        h_cb = cg[2] - cb[1]
        ax.plot(-cg[0], cg[2], color="red", marker="o", label="Center of Gravity")
        # ax.set_title('Longitudinal Hydrostatic Stability')
        ax.set_xlabel("X-Axis [m]")
        ax.set_ylabel("Z-Axis [m]")

    h_mc = calculate_mc(i_zz, weight, h_cb, rho_w)

    ax.plot(cb[0], cb[1], color="green", marker="o", label="Center of Buoyancy")
    # ax.plot(cb[0],cb[1]+h_mc,color='purple',marker='o',label='Metacenter')
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plot_path = os.path.join("Figures", f"{filename}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Center of Buoyancy [m]: {h_cb}")
    print(f"Height of Metacenter [ft]: {h_mc*3.281}")
