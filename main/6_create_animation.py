import h5py
import numpy as np
import pyvista as pv
from dolfinx import mesh
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
from typing import Tuple
import argparse
import os, sys

# Ensure the working directory is correct
os.chdir(sys.path[0])


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


def get_finger_position(index: int) -> np.ndarray:
    """
    Retrieve the finger position based on the given index.

    Parameters:
        index (int): The index for the deformation scenario.

    Returns:
        np.ndarray: The finger position as a NumPy array.
    """
    FILTERED_POINTS_FILE = "./filtered_points_of_force_on_boundary.txt"
    filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)
    finger_position = filtered_points[index]
    return finger_position


def load_mesh_and_deformations(xdmf_file: str, h5_file: str):
    """
    Load mesh points and deformation data.

    Parameters:
        xdmf_file (str): Path to the XDMF file for the mesh.
        h5_file (str): Path to the HDF5 file for deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The mesh points, connectivity, time steps, and deformation tensor.
    """
    # Load the mesh
    _, points, connectivity = get_mesh(xdmf_file)

    # Load the deformations
    time_steps, deformations = load_deformations(h5_file)

    return points, connectivity, time_steps, deformations


def animate_deformation(
    points: np.ndarray, connectivity: np.ndarray, deformations: np.ndarray, finger_position: np.ndarray, output_file: str, offscreen: bool
):
    """
    Animate the deformation of the mesh using PyVista.

    Parameters:
        points (np.ndarray): The initial points of the mesh.
        connectivity (np.ndarray): The connectivity of the mesh.
        deformations (np.ndarray): The deformation tensor [time_index][point_index][x, y, z].
        finger_position (np.ndarray): The position of the finger marker.
        output_file (str): Path to save the animation.
        offscreen (bool): Whether to enable offscreen rendering.
    """
    R = 0.003  # Radius of the sphere

    # Create a PyVista UnstructuredGrid
    num_cells = connectivity.shape[0]
    cells = np.hstack([np.full((num_cells, 1), connectivity.shape[1]), connectivity]).flatten()
    cell_types = np.full(num_cells, 10, dtype=np.uint8)  # 10 corresponds to tetrahedrons in PyVista
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Setup PyVista plotter
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.add_axes()
    plotter.add_text("Deformation Animation", font_size=12)

    # Add the mesh
    actor = plotter.add_mesh(grid, show_edges=True, colormap="coolwarm")

    # Add the finger marker as a sphere
    finger_marker = pv.Sphere(radius=R, center=finger_position)
    plotter.add_mesh(finger_marker, color="green", label="Finger Position", opacity=1.0)

    # Open movie file for writing
    plotter.open_movie(output_file, framerate=20)

    # Animate through all time steps
    for t_index in range(deformations.shape[0]):
        displacement = deformations[t_index]
        grid.points = points + displacement

        # Write frame
        plotter.write_frame()

        # (Optional) Print progress
        print(f"Rendered time step {t_index + 1}/{deformations.shape[0]}")

    # Close the plotter
    plotter.close()


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create animation from deformation data.")
    parser.add_argument("--index", type=int, help="Index of the deformation scenario.")
    parser.add_argument("index_pos", type=int, nargs="?", help="Index of the deformation scenario (positional).")
    parser.add_argument("--offscreen", action="store_true", help="Enable offscreen rendering for the animation.")
    args = parser.parse_args()

    # Determine the index from either --index or positional argument
    if args.index is not None:
        INDEX = args.index
    elif args.index_pos is not None:
        INDEX = args.index_pos
    else:
        parser.error("Index must be provided either as '--index' or as a positional argument.")

    # Fixed XDMF file
    XDMF_FILE = "bunny.xdmf"

    # Construct HDF5 file path based on index
    H5_FILE = f"./deformed_bunny_files_tunned/displacement_{INDEX}.h5"
    OUTPUT_DIR = f"./Animations_tunned"
    OUTPUT_FILE = f"{OUTPUT_DIR}/deformation_{INDEX}.mp4"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load mesh and deformations
    points, connectivity, time_steps, deformations = load_mesh_and_deformations(XDMF_FILE, H5_FILE)

    # Get finger position
    finger_position = get_finger_position(INDEX)

    print(f"Loaded mesh points: {points.shape}")
    print(f"Loaded connectivity: {connectivity.shape}")
    print(f"Loaded {len(time_steps)} deformation time steps.")

    # Animate deformation
    animate_deformation(points, connectivity, deformations, finger_position, OUTPUT_FILE, args.offscreen)
