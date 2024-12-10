import meshio
import numpy as np
from scipy.spatial import KDTree
from dolfin import *


# Function to extract mesh points and connectivity from file
def get_tetra_mesh_data(file_path):
    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading mesh file: {e}")

    points = mesh.points
    connectivity = mesh.cells_dict.get("tetra", None)

    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")
    return points, connectivity


# Normalize points
def normalize_points(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    normalized_points = (points - min_coords) / (max_coords - min_coords)
    return normalized_points


# Load and normalize bunny mesh
mesh_file = "Bunny/bunny.mesh"
points, connectivity = get_tetra_mesh_data(mesh_file)
points = normalize_points(points)
print(f"Mesh loaded: {len(points)} points, {len(connectivity)} tetrahedra.")

# Convert mesh to FEniCS format
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(points))
editor.init_cells(len(connectivity))

for i, point in enumerate(points):
    editor.add_vertex(i, point)

for i, cell in enumerate(connectivity):
    editor.add_cell(i, cell)

editor.close()
print("Mesh successfully converted to FEniCS format.")

# Function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Boundary condition (fixed displacement at boundaries)
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary)

# Define force location (nearest vertex to the given deformation point)
force_point = np.array([0.01413998, 0.5773221, 0.7305683])  # Given deformation point
force_vector = [937770.0, 288890.0, -193000.0]  # Increased force

kdtree = KDTree(points)
force_vertex_id = kdtree.query(force_point)[1]
force_vertex_coords = points[force_vertex_id]
print(f"Applying force at vertex {force_vertex_id}, location: {force_vertex_coords}")

# Material properties (lowered Young's modulus for more deformation)
E = 1000       # Reduced Young's modulus for a more deformable material
nu = 0.3       # Poisson's ratio
rho = 1000     # Density of the material

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Define stress and strain
def epsilon(u):
    return sym(grad(u))


def sigma(u):
    return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)


# Time-stepping parameters
dt = 0.01       # Time step size
T = 1.0         # Total simulation time
time = 0.0      # Start time

# Define trial, test, and previous solution
u = TrialFunction(V)
v = TestFunction(V)
u_n = Function(V)  # Displacement at previous time step
u_new = Function(V)  # Displacement at current time step

# External point force (applied at the vertex)
subspaces = [V.sub(i) for i in range(3)]
sources = [
    PointSource(subspaces[i], Point(*force_vertex_coords), force_vector[i])
    for i in range(3)
]

# Variational form for dynamic elasticity (Newmark-beta method)
a = rho / dt**2 * inner(u, v) * dx + inner(sigma(u), epsilon(v)) * dx
L = rho / dt**2 * inner(u_n, v) * dx

# Output file
file = XDMFFile("bunny_dynamic_deformation.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True

# Time-stepping loop
while time < T:
    time += dt
    print(f"Time step: {time:.2f}")

    # Solve system
    solve(a == L, u_new, bc)

    # Apply the point force
    for source in sources:
        source.apply(u_new.vector())

    # Save results
    file.write(u_new, time)

    # Update for next time step
    u_n.assign(u_new)

print("Simulation complete. Results saved to bunny_dynamic_deformation.xdmf.")

