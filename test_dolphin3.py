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


# Define boundary condition (fixed displacement at boundaries)
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary)

# Define force location (nearest vertex to the given deformation point)
force_point = np.array([0.01413998, 0.5773221, 0.7305683])  # Given deformation point
force_vector = [93.77047826000594, 28.889294419396272, -19.30041644211834]

kdtree = KDTree(points)
force_vertex_id = kdtree.query(force_point)[1]
force_vertex_coords = points[force_vertex_id]
print(f"Applying force at vertex {force_vertex_id}, location: {force_vertex_coords}")

# Variational problem: Linear elasticity
E = 2100  # Young's modulus
nu = 0.3  # Poisson's ratio
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))


def epsilon(u):
    return sym(grad(u))


def sigma(u):
    return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)


# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define bilinear and linear forms
a = inner(sigma(u), epsilon(v)) * dx  # Bilinear form (stiffness matrix)
L = dot(Constant((0.0, 0.0, 0.0)), v) * dx  # No distributed body forces

# Solve the system
u_sol = Function(V)
solve(a == L, u_sol, bc)

# Apply point force component-by-component
subspaces = [V.sub(i) for i in range(3)]  # Access the scalar subspaces of V
sources = [PointSource(subspaces[i], Point(*force_vertex_coords), force_vector[i]) for i in range(3)]

# Apply the point sources
for source in sources:
    source.apply(u_sol.vector())

# Save results
file = XDMFFile("bunny_deformation.xdmf")
file.write(u_sol)
print("Results saved to bunny_deformation.xdmf.")

# Output displacement at force point
displacement = u_sol(force_vertex_coords)
print(f"Displacement at force point {force_vertex_coords}: {displacement}")
