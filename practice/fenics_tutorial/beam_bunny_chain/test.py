import numpy as np
from dolfinx.io import XDMFFile
from dolfinx import fem
from scipy.spatial import Delaunay
from mpi4py import MPI

def load_file(filename):
    # Load the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain


def get_array_from_conn(conn) -> np.ndarray:
    """
    Convert mesh topology connectivity to a 2D numpy array.

    Parameters:
    - conn: The mesh topology connectivity (dolfinx mesh.topology.connectivity).

    Returns:
    - connectivity_2d: A 2D numpy array where each row contains the vertex indices for a cell.
    """
    connectivity_array = conn.array
    offsets = conn.offsets

    # Convert the flat connectivity array into a list of arrays
    connectivity_2d = [
        connectivity_array[start:end]
        for start, end in zip(offsets[:-1], offsets[1:])
    ]

    # Convert to numpy array with dtype=object to handle variable-length rows
    return np.array(connectivity_2d, dtype=object)

# Load the mesh
domain = load_file("bunny.xdmf")

# Extract points and cells from the mesh
points = domain.geometry.x  # Array of vertex coordinates
topology = domain.topology.connectivity(3, 0)
cells = get_array_from_conn(topology)  # 2D numpy array of cell connectivity

print(f"\nnp.shape(cells) = {np.shape(cells)}")
print(f"cells = \n{cells}\n")

# Create a Delaunay triangulation
delaunay = Delaunay(points)

# Define a query point
query_points = np.array([[0.1, 0, 0], [0.11,0,0]])  # Example point
query_point = np.array([0.1, 0, 0])  # Example point
print(f"query_point = {query_point}")
print(f"query_points = \n{query_points}\n")

# Find the simplex (cell) containing the point
simplex = delaunay.find_simplex(query_point)

print(f"\nsimplex = {simplex}\n")

if np.any(simplex == -1):
    print("a Point is outside the mesh.")
else:
    print(f"Point is inside cell index: {simplex}")

    # Define a function space and function over the domain
    # V = fem.functionspace(domain, ("Lagrange", 1, (1,) ))
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(V)
    
    # Correctly interpolate a function
    def interpolation_function(x):
        # Return an array of shape (dim, num_points) to match vector-valued function space
        return x[0]**2 + x[1] - x[2]

    # Apply the interpolation
    u.interpolate(interpolation_function)

    # Identify the cell containing the query point
    point = np.atleast_2d(query_point)
    containing_cell = np.array([simplex], dtype=np.int32)
    
    print(f"point = {point}")

    # Compute the function value at the query point
    function_value = u.eval(point, containing_cell)
    print(f"Value of the function at {query_point}: {function_value}")
    print(f"exact Value of the function at {query_point}: {interpolation_function(point[0])}")


