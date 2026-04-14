import os
import getpass
import morphic
import pyvista as pv
import numpy as np
import trimesh
import itertools

# Set a consistent path for matplotlib used by pyvista.
os.environ['MPLCONFIGDIR'] = "/tmp/" + getpass.getuser()


def generate_xi_grid_fem(num_points=[4, 4, 4], dim=3):
    """
    Generate a grid of points within each element, either in 2D or 3D space.

    Parameters:
    -----------
    num_points : list of int, optional (default=[4, 4, 4])
        A list of numbers specifying how many points to generate along each axis. For example,
        [4, 4, 4] will generate a 4x4x4 grid of points in 3D space.

    dim : int, optional (default=3)
        The number of dimensions for the grid. 2 for 2D grid and 3 for 3D grid.

    Returns:
    --------
    XiNd : numpy.ndarray
        A 2D array where each row contains the coordinates of a point in the grid.
    """
    xi1 = np.linspace(0., 1., num_points[0])
    xi2 = np.linspace(0., 1., num_points[1])

    if dim == 2:
        X, Y = np.meshgrid(xi1, xi2)
        XiNd = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    else:
        xi3 = np.linspace(0., 1., num_points[2])
        X, Y, Z = np.meshgrid(xi1, xi2, xi3)
        XiNd = np.array([
            Z.reshape((Z.size)),
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T

    return XiNd


def generate_points_morphic_elements(mesh, xi, element_ids=[], dim=3):
    """
    Generate a grid of points within selected morphic mesh elements.

    Parameters:
    -----------
    mesh : morphic.Mesh
        The mesh to evaluate points in.

    xi : numpy.ndarray
        The local coordinates (xi) for which to generate the points.

    element_ids : list of int, optional (default=[])
        List of element IDs for which the points should be generated. If empty, points are generated
        for all elements.

    dim : int, optional (default=3)
        The number of dimensions in the xi grid.

    Returns:
    --------
    points : numpy.ndarray
        A 2D array of points with shape (total_points, dim), where each row represents a point's coordinates.

    all_xi : numpy.ndarray
        A 2D array of the xi coordinates for each point.

    all_ne : numpy.ndarray
        A 1D array of element IDs for each generated point.
    """
    if not element_ids:
        # If no element IDs are provided, evaluate points for all elements.
        element_ids = mesh.elements.ids

    num_ne = len(element_ids)
    ne_num_points = len(xi)
    total_num_points = num_ne * ne_num_points

    points = np.zeros((num_ne, ne_num_points, dim))
    all_xi = np.zeros((num_ne, ne_num_points, dim))
    all_ne = np.zeros((num_ne, ne_num_points))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(xi)
        all_xi[idx, :, :] = xi
        all_ne[idx, :] = element_id

    points = np.reshape(points, (total_num_points, dim))
    all_xi = np.reshape(all_xi, (total_num_points, dim))
    all_ne = np.reshape(all_ne, (total_num_points))

    return points, all_xi, all_ne


def draw_elem_lines(mesh, res=2):
    """
    Draw the element lines from a mesh, where each element is represented by a series of connected lines.

    Parameters:
    -----------
    mesh : morphic.Mesh
        A morphic Mesh object that contains the geometric data of the model.

    res : int, optional (default=2)
        The resolution for creating the lines. A higher value will result in more points per line.

    Returns:
    --------
    pyvista.PolyData
        A PyVista PolyData object containing the points and lines representing the mesh elements.
    """
    # Get the lines from the mesh, with resolution applied.
    mesh_lines = np.array(mesh.get_lines(res))

    # Flatten points into a single array (each point once).
    n_lines, n_points_per_line, _ = mesh_lines.shape
    points = mesh_lines.reshape(-1, 3)

    # Create connectivity for each line (using n_points_per_line).
    lines = []
    for i in range(n_lines):
        start_idx = i * n_points_per_line
        point_indices = list(range(start_idx, start_idx + n_points_per_line))
        lines.append([n_points_per_line] + point_indices)

    # Flatten connectivity array (for PyVista).
    lines = np.array(lines, dtype=np.int64).flatten()

    # Create PolyData object for PyVista.
    poly_data = pv.PolyData()
    poly_data.points = points
    poly_data.lines = lines

    return poly_data


if __name__ == "__main__":
    """
    Main function to generate and visualize a 3D mesh, including nodes, faces, and element lines.

    This script:
    - Creates a mesh with nodes and elements.
    - Generates the mesh structure.
    - Visualizes the nodes, faces, and element lines using PyVista.
    """
    # Initialize a Morphic mesh object and configure it.
    mesh = morphic.Mesh()
    mesh.auto_add_faces = True

    # Define the node coordinates for the mesh.
    Xn = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]])

    # Add nodes to the mesh.
    for node_idx, node_coordinates in enumerate(Xn):
        mesh.add_stdnode(node_idx + 1, node_coordinates, group='_default')

    # Add elements to the mesh.
    mesh.add_element(1, ['L1', 'L1', 'L1'], [1, 2, 3, 4, 5, 6, 7, 8])

    # Generate the mesh and its lines and faces.
    mesh.generate()

    # Interpolate mesh geometry at a given local element coordinate.
    element_number = 1
    interpolated_point = mesh.elements[element_number].evaluate([0.5, 0.5, 0.5])

    # Update node coordinates.
    for node_idx in range(len(Xn)):
        mesh.nodes[node_idx + 1].values += np.random.rand(3) * 0.1

    # Generate grid of local element points.
    xi = generate_xi_grid_fem(num_points=[4, 4, 4], dim=3)

    interpolated_points, _, _ = generate_points_morphic_elements(mesh, xi)

    # Get node coordinates.
    node_points = mesh.get_nodes(group='_default')

    # Visualization setup using PyVista.
    plotter = pv.Plotter()

    # Visualize the nodes.
    plotter.add_points(
        node_points, style='points', color='orange',
        point_size=12, label='nodes', render_points_as_spheres=True)

    # Visualize the interpolated points.
    plotter.add_points(
        interpolated_points, style='points', color='green',
        point_size=10, label='nodes', render_points_as_spheres=True)

    # Visualize the mesh faces.
    mesh_faces = pv.wrap(trimesh.Trimesh(*mesh.get_faces()))
    plotter.add_mesh(mesh_faces, color="lightblue", opacity=0.25, show_edges=False)

    # Visualize the element lines.
    mesh_lines = draw_elem_lines(mesh, res=2)
    plotter.add_mesh(mesh_lines, color="grey", show_edges=False)

    _ = plotter.add_axes(line_width=5, labels_off=True)

    # Display the plot.
    plotter.show()
