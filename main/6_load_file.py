import h5py
import numpy as np
import pyvista as pv
from dolfinx.io import XDMFFile
from dolfinx import mesh
from mpi4py import MPI



INDEX = 0
DIRECTORY = "./deformed_bunny_files"
ANIMATION_DIRECTORY = "./Animations/"
ANIMATION_FILE = f"{ANIMATION_DIRECTORY}/bunny_deformation_animation_{INDEX}.mp4"
mesh_file = "bunny.xdmf"
displacement_file = f"{DIRECTORY}/displacement_{INDEX}.h5"
FILTERED_POINTS_FILE = "./filtered_points_of_force_on_boundary.txt"


def load_mesh(xdmf_file):
    """
    Load the mesh from an XDMF file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain

def load_displacement_data(h5_file):
    """
    Load displacement data from an HDF5 file.
    """
    with h5py.File(h5_file, "r") as f:
        displacements = f["displacements"][:]
        print(f"Loaded displacement data with shape: {displacements.shape}")
    return displacements


def get_finger_position(index):    
    filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)
    finger_position = filtered_points[index]
    return finger_position

def animate_displacement(mesh, displacements):
    """
    Animate the displacement data on the mesh using PyVista.
    """
    points = mesh.geometry.x.copy()  # Original mesh points
    num_steps = displacements.shape[0]

    # Create PyVista grid
    connectivity = mesh.topology.connectivity(3, 0).array.reshape((-1, 4))  # Assuming tetrahedral mesh
    cell_types = np.full(connectivity.shape[0], 10, dtype=np.uint8)  # PyVista tetrahedron cell type is 10
    cells = np.hstack([np.full((connectivity.shape[0], 1), 4), connectivity]).flatten()  # Add node count per cell
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Add initial displacement norms as scalar data
    displacement_norms = np.linalg.norm(displacements[0], axis=1)
    grid.point_data["Displacement"] = displacement_norms

    # Setup PyVista plotter
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_text("Displacement Animation", font_size=12)

    # Add mesh to the plotter
    actor = plotter.add_mesh(grid, show_edges=True, scalars="Displacement", colormap="coolwarm")
    plotter.show_grid()

    # Open movie file for writing
    plotter.open_movie(ANIMATION_FILE, framerate=10)

    for step in range(num_steps):
        # Update points with displacement
        grid.points = points + displacements[step]

        # Update scalar values (displacement norms) for coloring
        displacement_norms = np.linalg.norm(displacements[step], axis=1)
        grid.point_data["Displacement"] = displacement_norms

        # Update the scene and write frame
        plotter.write_frame()

        # (Optional) Print progress
        if step % 10 == 0:
            print(f"Rendered time step {step}/{num_steps}")

    # Close movie and show
    plotter.close()


# Load mesh and displacement data
domain = load_mesh(mesh_file)
displacement_data = load_displacement_data(displacement_file)

# Animate displacement
animate_displacement(domain, displacement_data)

