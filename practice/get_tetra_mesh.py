from dolfinx.fem import coordinate_element
from dolfinx import plot
import meshio
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import ufl
import basix.ufl
import pyvista as pv

# Function to extract mesh points and connectivity
def get_tetra_mesh_data(file_path):
    mesh = meshio.read(file_path)
    points = mesh.points
    connectivity = mesh.cells_dict.get("tetra", None)
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")
    return points, connectivity


# Load mesh
mesh_file = "../Bunny/bunny.mesh"
points, connectivity = get_tetra_mesh_data(mesh_file)
points[:, [1, 2]] = points[:, [2, 1]] # Switch y and z
points[:, 0] = -points[:, 0] # So the bunny face forward

# Get the min and max for each column
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()


# Print the results
print("____________before_______________")
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")

# Calculate required shifts
x_shift = -x_min
z_shift = -z_min
y_shift = -(y_min + y_max) / 2  # Center y around 0

# Apply transformations
points[:, 0] += x_shift  # Shift x so x_min = 0
points[:, 1] += y_shift  # Center y
points[:, 2] += z_shift  # Shift z so z_min = 0

# Print the results
print("____________after____________")
# Get the min and max for each column
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")

print("\n\n--------START OF PROGRAM---------\n\n")

print("Mesh name = Bunny")
print("Mesh path = {mesh_file}\n")

print("Mesh = (points, connectivity)\n")

print(f"type(points) = {type(points)}")
print(f"type(connectivity) = {type(connectivity)}\n")

print(f"shape(points) = {np.shape(points)}")
print(f"shape(connectivity) = {np.shape(connectivity)}\n\n")

print(f"points = \n{points}\n")
print(f"connectivity = \n{connectivity}\n")



#---------------------------------
print("\n------Doing dolfinx things-----\n")

points = points.astype(np.float64)
connectivity = connectivity.astype(np.int64)

# Define the coordinate element
coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(points.shape[1],)))


# Create the DOLFINx mesh
tetra_mesh = mesh.create_mesh(MPI.COMM_WORLD, connectivity, points, coordinate_element)


# Output mesh information
print(f"Mesh geometry:\n{tetra_mesh.geometry.x}")
print(f"Mesh topology:\n{tetra_mesh.topology.cell_type}\n")


# Extract mesh data from DOLFINx for Pyvista
topology, cell_types, geometry = plot.vtk_mesh(tetra_mesh)

# Create a Pyvista UnstructuredGrid
grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# Plot the mesh
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightblue", label="Tetrahedral Mesh")
plotter.add_axes()
plotter.show()


# Save the mesh to XDMF
meshio.write_points_cells( "bunny.xdmf",    points, [("tetra", connectivity)] )
print("Mesh saved to bunny.xdmf")
