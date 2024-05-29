import numpy as np
import trimesh
import matplotlib.pyplot as plt

def load_stl(file_path):
    mesh = trimesh.load(file_path)
    return mesh

def get_cross_section(mesh, plane_origin, plane_normal):
    # Find the intersection of the mesh with the plane
    cross_sections = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    return cross_sections

# Adjust the plotting function to handle the missing argument error
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
            discrete_path = path.discrete(vertices)
            # Remove intersecting regions by checking the bounds
            if np.all(discrete_path[:, 0] >= np.min(vertices[:, 0])) and np.all(discrete_path[:, 0] <= np.max(vertices[:, 0])):
                ax.plot(*discrete_path.T)

    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Cross Section of STL File')
    plt.grid(True)
    plt.show()

# Example usage
file_path = r'C:\Users\brian\OneDrive\Documents\Georgia Tech\Research\Whale Plane\SolarSim\SampleData\STL\WhalePlane.stl'  # Replace with your STL file path
plane_origin = [0, 0, 0]  # Define the origin of the plane
plane_normal = [0, 1, 0]  # Define the normal to the plane

mesh = load_stl(file_path)
cross_section = get_cross_section(mesh, plane_origin, plane_normal)
plot_cross_section(cross_section)
