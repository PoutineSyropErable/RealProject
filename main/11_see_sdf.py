import numpy as np
import igl
import polyscope as ps
import matplotlib.pyplot as plt
import h5py
import pyvista as pv
from dolfinx.io import XDMFFile
from dolfinx import mesh
from mpi4py import MPI
import os
import sys
import argparse

# Hardcoded paths
DISPLACEMENT_DIRECTORY = "./deformed_bunny_files"
BUNNY_FILE = "bunny.xdmf"

def load_mesh_domain(xdmf_file):
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain

def load_displacement_data(h5_file):
    with h5py.File(h5_file, "r") as f:
        displacements = f["displacements"][:]
        print(f"Loaded displacement data with shape: {displacements.shape}")
    return displacements

def load_mesh_t(xdmf_file, h5_file, time_index):
    domain = load_mesh_domain(xdmf_file)
    points = domain.geometry.x.astype(np.float32)
    conn = domain.topology.connectivity(3, 0)
    connectivity = [
        conn.array[start:end] for start, end in zip(conn.offsets[:-1], conn.offsets[1:])
    ]
    connectivity = np.array(connectivity)
    displacements_all_times = load_displacement_data(h5_file)
    points_moved = points + displacements_all_times[time_index]
    return points_moved, connectivity


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


def compute_bounding_box(mesh_points: np.ndarray):
    """Compute the bounding box for the vertices."""
    b_min = mesh_points.min(axis=0)
    b_max = mesh_points.max(axis=0)

    BOX_RATIO = 1.5
    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2 * BOX_RATIO
    b_min = center - half_lengths
    b_max = center + half_lengths

    return b_min, b_max

def generate_random_points(b_min: np.ndarray, b_max: np.ndarray, num_points: int):
    """Generate random points within the bounding box."""
    return np.random.uniform(b_min, b_max, size=(num_points, 3))

def compute_signed_distances(point_list: np.ndarray, mesh_points: np.ndarray, mesh_faces: np.ndarray):
    """Compute signed distances from points to the triangle mesh."""
    signed_distances, nearest_face, nearest_points = igl.signed_distance(point_list, mesh_points, mesh_faces)
    return signed_distances, nearest_face, nearest_points





def weight_function(signed_distance: float, weight_exponent: float = 10) -> float:
    """Takes a signed_distances and return a probability of taking said points"""
    return (1 + abs(signed_distance)) ** (-weight_exponent)

def filter_function(signed_distance: float, weight_exponent: float) -> bool:
    "Returns a bool or not, deciding weither or not to take the point"

    random_number = np.random.rand()
    return random_number < weight_function(signed_distance, weight_exponent)


def filter_points(signed_distances: np.ndarray, weight_exponent: float) -> np.ndarray:
    """Filter points based on their signed distances."""

    filtered_index = np.array(
        [
            i
            for i in range(len(signed_distances))
            if filter_function(signed_distances[i], weight_exponent)
        ]
    )
    return filtered_index

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



def show_result_in_polyscope(mesh_points, mesh_connectivity, filtered_points, filtered_signed_distances, filtered_nearest):
    ps.init()
    ps_mesh = ps.register_surface_mesh("Bunny", mesh_points, mesh_connectivity)

    NUMBER_OF_POINTS = 20_000
    ps_cloud = ps.register_point_cloud(
        "Flitered Points", filtered_points[0:NUMBER_OF_POINTS], radius=0.0025
    )
    ps_cloud.add_scalar_quantity(
        "Signed Distances", filtered_signed_distances[0:NUMBER_OF_POINTS]
    )

    NUMBER_OF_LINES = 100
    # Create edges for the curve network
    edges = np.column_stack(
        (np.arange(NUMBER_OF_LINES), np.arange(NUMBER_OF_LINES))
    )  # Connecting filtered points to their nearest points

    # Combine filtered points and filtered nearest points into a single array
    all_points = np.vstack(
        (filtered_points[0:NUMBER_OF_LINES], filtered_nearest[0:NUMBER_OF_LINES])
    )
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
    ps_lines.set_color((1.0, 0.0, 0.0))  # Red color for the lines

    # Compute and draw the larger bounding box
    b_min, b_max = compute_bounding_box(mesh_points)
    draw_bounding_box(
        b_min, b_max, "Large Bounding Box", color=(0.0, 1.0, 0.0), radius=0.002
    )

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


# Define other helper functions as in your original script...

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh displacement using polyscope.")
    parser.add_argument(
        "index",
        type=int,
        nargs="?",
        help="Index of the displacement file to use.",
    )
    parser.add_argument(
        "time_index",
        type=int,
        nargs="?",
        help="Index of the time step to visualize.",
    )
    parser.add_argument(
        "--index", "--displacement_index",
        type=int,
        help="Index of the displacement file to use.",
    )
    parser.add_argument(
        "--time_index",
        type=int,
        help="Index of the time step to visualize.",
    )
    
    args = parser.parse_args()

    # Determine displacement and time indices
    displacement_index = args.index or args.displacement_index
    time_index = args.time_index

    if displacement_index is None and len(sys.argv) > 1:
        displacement_index = int(sys.argv[1])
    if time_index is None and len(sys.argv) > 2:
        time_index = int(sys.argv[2])

    # Default values if not provided
    displacement_index = displacement_index if displacement_index is not None else 0
    time_index = time_index if time_index is not None else 10

    # Construct file paths
    displacement_file = f"{DISPLACEMENT_DIRECTORY}/displacement_{displacement_index}.h5"
    
    print(f"Using displacement file: {displacement_file}")
    print(f"Using bunny file: {BUNNY_FILE}")
    print(f"Time index: {time_index}")

    # Load and process the mesh
    mesh_points, mesh_connectivity = load_mesh_t(BUNNY_FILE, displacement_file, time_index)

    vertices, faces = get_surface_mesh(mesh_points, mesh_connectivity)

    b_min, b_max = compute_bounding_box(vertices)
    point_list = generate_random_points(b_min, b_max, 1_000_000)
    signed_distances, nearest_face, nearest_points = compute_signed_distances(
        point_list, vertices, faces
    )

    filtered_index = filter_points(signed_distances, weight_exponent=20)
    filtered_signed_distances = signed_distances[filtered_index]
    filtered_points = point_list[filtered_index]
    filtered_nearest = nearest_points[filtered_index]

    show_result_in_polyscope(
        mesh_points, mesh_connectivity, filtered_points, filtered_signed_distances, filtered_nearest
    )

if __name__ == "__main__":
    main()

