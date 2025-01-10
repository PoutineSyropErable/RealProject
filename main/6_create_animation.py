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


def load_deformations(h5_file: str) -> dict:
    """
    Load deformation data from an HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file containing deformation data.

    Returns:
        dict: A dictionary where keys are time steps and values are displacement arrays.
    """
    deformations = {}
    with h5py.File(h5_file, "r") as f:
        # Access the 'Function' -> 'f' group
        function_group = f["Function"]
        f_group = function_group["f"]

        # Extract datasets for each time step
        for time_step in f_group.keys():
            dataset = f_group[time_step]
            deformations[time_step] = dataset[...]
            print(f"Loaded deformation for time step {time_step}, Shape: {dataset.shape}")

    return deformations


def load_mesh_and_deformations(xdmf_file: str, h5_file: str):
    """
    Load mesh points and deformation data.

    Parameters:
        xdmf_file (str): Path to the XDMF file for the mesh.
        h5_file (str): Path to the HDF5 file for deformation data.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: The mesh points, connectivity, and deformation dictionary.
    """
    # Load the mesh
    _, points, connectivity = get_mesh(xdmf_file)

    # Load the deformations
    deformations = load_deformations(h5_file)

    return points, connectivity, deformations


def animate_deformation(
    points: np.ndarray, connectivity: np.ndarray, deformations: dict, finger_position: np.ndarray, output_file: str, offscreen: bool
):
    """
    Animate the deformation of the mesh using PyVista.

    Parameters:
        points (np.ndarray): The initial points of the mesh.
        connectivity (np.ndarray): The connectivity of the mesh.
        deformations (dict): The deformation data for each time step.
        finger_position (np.ndarray): The position of the finger marker.
        output_file (str): Path to save the animation.
        offscreen (bool): Whether to enable offscreen rendering.
    """
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
    finger_marker = pv.Sphere(radius=0.003, center=finger_position)
    plotter.add_mesh(finger_marker, color="green", label="Finger Position", opacity=1.0)

    # Open movie file for writing
    plotter.open_movie(output_file, framerate=20)

    deformation_keys = sorted(deformations.keys(), key=lambda x: float(x))

    # Animate through all time steps
    for time_step in deformation_keys:
        displacement = deformations[time_step]
        grid.points = points + displacement

        # Write frame
        plotter.write_frame()

        # (Optional) Print progress
        print(f"Rendered time step {time_step}")

    # Close the plotter
    plotter.close()


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Animate mesh deformation.")
    parser.add_argument("--xdmf_file", required=True, help="Path to the XDMF file.")
    parser.add_argument("--h5_file", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--output", required=True, help="Path to save the animation.")
    parser.add_argument("--finger_position", type=float, nargs=3, required=True, help="Finger position as x, y, z.")
    parser.add_argument("--offscreen", action="store_true", help="Enable offscreen rendering.")
    args = parser.parse_args()

    # Load mesh and deformations
    points, connectivity, deformations = load_mesh_and_deformations(args.xdmf_file, args.h5_file)

    print(f"Loaded mesh points: {points.shape}")
    print(f"Loaded connectivity: {connectivity.shape}")
    print(f"Loaded {len(deformations)} deformation time steps.")

    # Animate deformation
    animate_deformation(points, connectivity, deformations, np.array(args.finger_position), args.output, args.offscreen)
