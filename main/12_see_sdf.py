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

os.chdir(sys.path[0])

# Hardcoded paths
DISPLACEMENT_DIRECTORY = "./deformed_bunny_files"
BUNNY_FILE = "bunny.xdmf"
SDF_DIRECTORY = "./calculated_sdf"
POINTS_TO_TAKE_SDF_FILE = "points_to_take_sdf.npy"




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



def load_sdf_data(displacement_index, time_index, sdf_only):
    """
    Load SDF data for a specific displacement index and time index.

    Args:
        displacement_index (int): Index of the displacement file.
        time_index (int): Time index for SDF data.
        sdf_only (bool): Whether to load only the SDF values.

    Returns:
        tuple: (points, sdf) if not sdf_only, otherwise just sdf.
    """
    file_suffix = "_sdf_only" if sdf_only else ""
    input_file = f"./calculated_sdf/sdf_points_{displacement_index}{file_suffix}.h5"

    print(f"Loading SDF data from: {input_file}")
    
    with h5py.File(input_file, "r") as f:
        dataset_name = f"time_{time_index}"
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in file {input_file}.")
        
        data = f[dataset_name][:]
        
        if sdf_only:
            # Return only the SDF values
            sdf = data
            points = np.load(POINTS_TO_TAKE_SDF_FILE)
        else:
            # Return points and SDF values
            points = data[:, :3]
            sdf = data[:, 3]

        return points, sdf



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


def draw_bounding_box(b_min: np.ndarray, b_max: np.ndarray,name: str, color: tuple = (0.0, 1.0, 0.0), radius: float = 0.002):
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





def show_result_in_polyscope(mesh_points, mesh_connectivity, filtered_points, filtered_signed_distances, finger_position, R):

    #----------------------------------- Start and add the Bunny MESH
    ps.init()
    ps_mesh = ps.register_surface_mesh("Bunny", mesh_points, mesh_connectivity)

    #------------------------------------ Add the point cloud where sdf are calculated
    NUMBER_OF_POINTS = min(20_000, len(filtered_points))
    ps_cloud = ps.register_point_cloud(
        "Filtered Points", filtered_points[:NUMBER_OF_POINTS], radius=0.0025
    )
    ps_cloud.add_scalar_quantity(
        "Signed Distances", filtered_signed_distances[:NUMBER_OF_POINTS]
    )

    #------------------------------------- Add the lines connecting the point cloud to the surface
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


    #-------------------------------------------- Add a single point to represent the sphere's center
    ps_finger = ps.register_point_cloud("Finger Position", np.array([finger_position]), radius=3*R)
    ps_finger.set_color((1.0, 0.0, 0.0))  # Red color for the point

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

def validate_indices(index, displacement_index):
    """Validate if both index and displacement_index are provided and consistent."""
    if index is not None and displacement_index is not None and index != displacement_index:
        raise ValueError(
            f"Inconsistent indices: --index={index} and --displacement_index={displacement_index}. "
            f"Please provide matching values or only one of these arguments."
        )

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh displacement using precomputed SDF.")
    parser.add_argument("positional_index", type=int, nargs="?", help="Positional index of the displacement file.")
    parser.add_argument("positional_time_index", type=int, nargs="?", help="Positional index of the time step.")
    parser.add_argument("--index", type=int, help="Index of the displacement file to use.")
    parser.add_argument("--displacement_index", type=int, help="Alternative index of the displacement file to use.")
    parser.add_argument("--time_index", type=int, help="Index of the time step to visualize.")
    parser.add_argument("--sdf_only", action="store_true", help="Use precomputed SDF-only data.")

    args = parser.parse_args()

    # Resolve indices
    displacement_index = args.index or args.displacement_index or args.positional_index
    time_index = args.time_index or args.positional_time_index

    # Ensure defaults if not provided
    displacement_index = displacement_index if displacement_index is not None else 0
    time_index = time_index if time_index is not None else 10

    # Validate consistency
    validate_indices(args.index, args.displacement_index)

    sdf_only = args.sdf_only

    print(f"Using displacement index: {displacement_index}")
    print(f"Using time index: {time_index}")
    print(f"SDF-only mode: {sdf_only}")

    # File containing finger_positions (after filtering) [Active Filter: Z>Z_min | no finger of grounded foot]
    # Wanted filter: Z > Z_min and Z < Z_max | points are on the head or near root of ears
    FINGER_POSITIONS_FILES = "filtered_points_of_force_on_boundary.txt"
    finger_positions = np.loadtxt(FINGER_POSITIONS_FILES , skiprows=1)
    # Swap Y and Z because poylscope use weird data
    finger_positions[:, [1, 2]] = finger_positions[:, [2, 1]]
    finger_position = finger_positions[displacement_index]
    R = 0.003  # Radius of the FINGER

    # Construct file paths
    displacement_file = f"{DISPLACEMENT_DIRECTORY}/displacement_{displacement_index}.h5"
    mesh_points, mesh_connectivity = load_mesh_t(BUNNY_FILE, displacement_file, time_index)

    print(f"np.shape(mesh_points) = {np.shape(mesh_points)}")
    print(f"mesh_points = \n{mesh_points}\n")

    print(f"np.shape(mesh_connectivity) = {np.shape(mesh_connectivity)}")
    print(f"mesh_connectivity = \n{mesh_connectivity}\n")





    # Load precomputed SDF data
    points, sdf = load_sdf_data(displacement_index, time_index, sdf_only)
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

