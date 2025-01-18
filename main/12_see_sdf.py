import numpy as np
import polyscope as ps
import pyvista as pv
import matplotlib.pyplot as plt
import h5py
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
import argparse
import os
import sys
import igl
from dolfinx import mesh
from typing import Tuple

os.chdir(sys.path[0])

# Hardcoded paths
DISPLACEMENT_DIRECTORY = "./deformed_bunny_files_tunned"
BUNNY_FILE = "bunny.xdmf"
SDF_DIRECTORY = "./calculated_sdf"
POINTS_TO_TAKE_SDF_FILE = "points_to_take_sdf.npy"


def load_file(filename: str) -> mesh.Mesh:
    """
    Load the mesh from an XDMF file.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        mesh.Mesh: The loaded mesh object.
    """
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain: mesh.Mesh = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain


def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
        conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).

    Returns:
        np.ndarray: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [connectivity_array[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    return np.array(connectivity_2d, dtype=object)


def get_mesh(filename: str) -> Tuple[mesh.Mesh, np.ndarray, np.ndarray]:
    """
    Extract points and connectivity from the mesh.

    Parameters:
        filename (str): Path to the XDMF file.

    Returns:
        Tuple[mesh.Mesh, np.ndarray, np.ndarray]: The mesh object, points, and connectivity array.
    """
    domain = load_file(filename)
    points = domain.geometry.x  # Array of vertex coordinates
    conn = domain.topology.connectivity(3, 0)
    connectivity = get_array_from_conn(conn).astype(np.int64)  # Convert to 2D numpy array

    return domain, points, connectivity


def load_deformations(h5_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load deformation data from an HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file containing deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: An array of time steps and a 3D tensor of displacements [time_index][point_index][x, y, z].
    """
    with h5py.File(h5_file, "r") as f:
        # Access the 'Function' -> 'f' group
        function_group = f["Function"]
        f_group = function_group["f"]

        # Extract time steps and displacements
        time_steps = np.array(sorted(f_group.keys(), key=lambda x: float(x)), dtype=float)
        displacements = np.array([f_group[time_step][...] for time_step in f_group.keys()])
        print(f"Loaded {len(time_steps)} time steps, Displacement tensor shape: {displacements.shape}")

    return time_steps, displacements


def load_mesh_and_deformations(xdmf_file: str, h5_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh points and deformation data.

    Parameters:
        xdmf_file (str): Path to the XDMF file for the mesh.
        h5_file (str): Path to the HDF5 file for deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The mesh points, connectivity, time steps, and deformation tensor.
        return points, connectivity, time_steps, deformations
    """
    # Load the mesh
    _, points, connectivity = get_mesh(xdmf_file)

    # Load the deformations
    time_steps, deformations = load_deformations(h5_file)

    return points, connectivity, time_steps, deformations


def compute_small_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the smallest bounding box for the vertices."""
    b_min = np.min(mesh_points, axis=0)
    b_max = np.max(mesh_points, axis=0)
    return b_min, b_max


def compute_bounding_box(mesh_points: np.ndarray, box_ratio: float = 1.5) -> (np.ndarray, np.ndarray):
    """Compute an expanded bounding box for the vertices."""
    b_min, b_max = compute_small_bounding_box(mesh_points)

    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2

    # Expand the bounding box by the given ratio
    b_min = center - half_lengths * box_ratio
    b_max = center + half_lengths * box_ratio

    return b_min, b_max


def draw_bounding_box(
    b_min: np.ndarray,
    b_max: np.ndarray,
    name: str,
    color: tuple = (0.0, 1.0, 0.0),
    radius: float = 0.002,
):
    """Draw a bounding box in Polyscope given min and max points."""
    # Create corners of the bounding box
    box_corners = np.array(
        [
            [b_min[0], b_min[1], b_min[2]],
            [b_max[0], b_min[1], b_min[2]],
            [b_max[0], b_max[1], b_min[2]],
            [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]],
            [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_max[2]],
            [b_min[0], b_max[1], b_max[2]],
        ]
    )

    # Define edges for the bounding box
    box_edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )  # Vertical edges

    # Register the bounding box as a curve network
    ps_bounding_box = ps.register_curve_network(name, box_corners, box_edges)
    ps_bounding_box.set_radius(radius)  # Adjust bounding box line thickness
    ps_bounding_box.set_color(color)  # Set the color for the bounding box


def get_surface_mesh(points: np.ndarray, connectivity: np.ndarray):
    """Extract the surface mesh from the tetrahedral mesh."""
    cells = np.hstack([np.full((connectivity.shape[0], 1), 4), connectivity]).flatten()
    cell_types = np.full(connectivity.shape[0], 10, dtype=np.uint8)  # Tetrahedron type
    tetra_mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # Extract surface mesh
    surface_mesh = tetra_mesh.extract_surface()

    # Get vertices and faces
    vertices = surface_mesh.points
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix

    return vertices, faces


def show_result_in_polyscope(
    mesh_points,
    mesh_connectivity,
    filtered_points,
    filtered_signed_distances,
    finger_position,
    R,
):

    # ----------------------------------- Start and add the Bunny MESH
    ps.init()
    ps_mesh = ps.register_surface_mesh("Bunny", mesh_points, mesh_connectivity)

    # ------------------------------------ Add the point cloud where sdf are calculated
    NUMBER_OF_POINTS = min(20_000, len(filtered_points))
    ps_cloud = ps.register_point_cloud("Filtered Points", filtered_points[:NUMBER_OF_POINTS], radius=0.0025)
    ps_cloud.add_scalar_quantity("Signed Distances", filtered_signed_distances[:NUMBER_OF_POINTS])

    # ------------------------------------- Add the lines connecting the point cloud to the surface
    NUMBER_OF_LINES = min(100, len(filtered_points))

    # Calculate the nearest points only for the points used in the lines
    line_filtered_points = filtered_points[:NUMBER_OF_LINES]
    vertices, faces = get_surface_mesh(mesh_points, mesh_connectivity)
    _, _, filtered_nearest = igl.signed_distance(line_filtered_points, vertices, faces)

    # Combine filtered points and their nearest points into a single array
    all_points = np.vstack((line_filtered_points, filtered_nearest))
    print(f"\n\n\nall_points = \n{all_points}\nShape={np.shape(all_points)}")

    # Create edges that connect filtered_points to filtered_nearest
    edges = np.column_stack(
        (np.arange(NUMBER_OF_LINES), np.arange(NUMBER_OF_LINES) + NUMBER_OF_LINES)
    )  # Adjust edges for the combined array
    print(f"\n\n\nedges = \n{edges}\nShape={np.shape(edges)}")

    # Register the curve network to show lines from filtered_points to filtered_nearest
    ps_lines = ps.register_curve_network("Lines to Nearest Points", all_points, edges)

    # Optional: Customize appearance of the lines
    ps_lines.set_radius(0.001)  # Adjust line thickness
    ps_lines.set_color((0.0, 1.0, 1.0))  # Cyan color for the lines

    # -------------------------------------------- Add a single point to represent the sphere's center
    ps_finger = ps.register_point_cloud("Finger Position", np.array([finger_position]), radius=3 * R)
    ps_finger.set_color((1.0, 0.0, 0.0))  # Red color for the point

    # Compute and draw the larger bounding box
    b_min, b_max = compute_bounding_box(mesh_points)
    draw_bounding_box(b_min, b_max, "Large Bounding Box", color=(0.0, 1.0, 0.0), radius=0.002)

    # Compute and draw the smaller bounding box
    small_b_min, small_b_max = compute_small_bounding_box(mesh_points)
    draw_bounding_box(
        small_b_min,
        small_b_max,
        "Small Bounding Box",
        color=(0.0, 0.0, 1.0),
        radius=0.001,
    )

    ps.show()


def validate_indices(index, finger_index):
    """Validate if both index and finger_index are provided and consistent."""
    if index is not None and finger_index is not None and index != finger_index:
        raise ValueError(
            f"Inconsistent indices: --index={index} and --finger_index={finger_index}. "
            f"Please provide matching values or only one of these arguments."
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize mesh displacement using precomputed SDF.")
    parser.add_argument(
        "positional_index",
        type=int,
        nargs="?",
        help="Positional index of the displacement file.",
    )
    parser.add_argument(
        "positional_time_index",
        type=int,
        nargs="?",
        help="Positional index of the time step.",
    )
    parser.add_argument("--index", type=int, help="Index of the displacement file to use.")
    parser.add_argument(
        "--finger_index",
        type=int,
        help="Alternative index of the displacement file to use.",
    )
    parser.add_argument("--time_index", type=int, help="Index of the time step to visualize.")
    parser.add_argument("--sdf_only", action="store_true", help="Use precomputed SDF-only data.")

    args = parser.parse_args()

    # Resolve indices
    finger_index = args.index or args.finger_index or args.positional_index
    time_index = args.time_index or args.positional_time_index

    # Ensure defaults if not provided
    finger_index = finger_index if finger_index is not None else 0
    time_index = time_index if time_index is not None else 90

    # Validate consistency
    validate_indices(args.index, args.finger_index)

    sdf_only = args.sdf_only

    print(f"Using displacement index: {finger_index}")
    print(f"Using time index: {time_index}")
    print(f"SDF-only mode: {sdf_only}")

    # File containing finger_positions (after filtering) [Active Filter: Z>Z_min | no finger of grounded foot]
    # Wanted filter: Z > Z_min and Z < Z_max | points are on the head or near root of ears
    FINGER_POSITIONS_FILES = "filtered_points_of_force_on_boundary.txt"
    finger_positions = np.loadtxt(FINGER_POSITIONS_FILES, skiprows=1)
    # Swap Y and Z because poylscope use weird data
    finger_positions[:, [1, 2]] = finger_positions[:, [2, 1]]
    finger_position = finger_positions[finger_index]
    R = 0.003  # Radius of the FINGER

    # Construct file paths
    displacement_file = f"{DISPLACEMENT_DIRECTORY}/displacement_{finger_index}.h5"
    mesh_points, mesh_connectivity, time_steps, deformations = load_mesh_and_deformations(xdmf_file=BUNNY_FILE, h5_file=DISPLACEMENT_FILE)

    print(f"np.shape(mesh_points) = {np.shape(mesh_points)}")
    print(f"mesh_points = \n{mesh_points}\n")

    print(f"np.shape(mesh_connectivity) = {np.shape(mesh_connectivity)}")
    print(f"mesh_connectivity = \n{mesh_connectivity}\n")

    # Load precomputed SDF data
    points, sdf = load_sdf_data(finger_index, time_index, sdf_only)
    print(f"np.shape(points) = {np.shape(points)}")
    print(f"points = \n{points}\n")

    print(f"np.shape(sdf) = {np.shape(sdf)}")
    print(f"sdf = \n{sdf}\n")

    # Swap Y and Z because poylscope use weird data
    mesh_points[:, [1, 2]] = mesh_points[:, [2, 1]]
    points[:, [1, 2]] = points[:, [2, 1]]

    print(f"Loaded points shape: {points.shape}")
    print(f"Loaded SDF shape: {sdf.shape}")

    show_result_in_polyscope(mesh_points, mesh_connectivity, points, sdf, finger_position, R)


if __name__ == "__main__":
    main()
