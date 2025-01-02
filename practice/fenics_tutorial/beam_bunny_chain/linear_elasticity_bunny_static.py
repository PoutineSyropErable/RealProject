# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.io.utils import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

# Material properties
E = 10e6  # Young's modulus for rubber in Pascals (Pa)
nu = 0.45  # Poisson's ratio for rubber
rho = 0.5
g = 0.0000001

# Lamé parameters
mu = E / (2 * (1 + nu))
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))


# Load the mesh from the XDMF file
with XDMFFile(MPI.COMM_WORLD, "bunny.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    print("Mesh loaded successfully!")


V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))



def grounded_bunny(x):
    return np.isclose(x[2],0) # no movement at z = 0


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, grounded_bunny)

u_D = np.array([0, 0, 0], dtype=default_scalar_type) # no displacement on boundary
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V) # create a condition where u = 0 on z = 0




def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)



f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g))) # every point is submit to -rho*g force
T = fem.Constant(domain, default_scalar_type((0, 0, 0))) # No external force
ds = ufl.Measure("ds", domain=domain)
# dx = ufl.dx
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds



problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()



# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)

# Add title
p.add_text("Wireframe: Original | Colored Solid: Deformed", 
           position="upper_edge", 
           font_size=12, 
           color="black", 
           shadow=True)

p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")



print("\n\n-----End of Program-----\n\n")
