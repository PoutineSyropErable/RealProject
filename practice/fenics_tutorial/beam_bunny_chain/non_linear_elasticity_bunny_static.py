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



USE_FINGER_PRESSURE = False # My H(x,u) condition
if not USE_FINGER_PRESSURE:
    f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g))) # every point is submit to -rho*g force
    T = fem.Constant(domain, default_scalar_type((0, 0, 0))) # No external force
    ds = ufl.Measure("ds", domain=domain)
    # dx = ufl.dx
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds


if USE_FINGER_PRESSURE: # Not linear due to boundary condition and H(x,u), since finger moves with boundary
    # Deformed position x + u
    x = ufl.SpatialCoordinate(domain)  # Reference position (Without deformation)
    x_deformed = x + u  # Deformed position

    # Finger position (center of the sphere) and radius
    finger_position = (0.5, 0.5, 0.5)  # Example vec3 for the finger position
    R = 0.02  # Radius of the sphere

    # Distance from the sphere center (finger position) in the deformed configuration
    distance = ufl.sqrt(
        (x_deformed[0] - finger_position[0])**2 +
        (x_deformed[1] - finger_position[1])**2 +
        (x_deformed[2] - finger_position[2])**2
    )

    # Indicator function H(x, u)
    H = ufl.conditional(ufl.lt(distance, R), 1.0, 0.0)
    # H(x,u) is sadly what causes a to not be Bilinear, it is not linear in u since a(u,v) = bilinear(u,v) - H(u)*v*p*(...) and H(u) is non linear
    # Hence, I'll need a time dependant, linear elastic, but non linear boundary conditions pde.

    # Pressure
    pressure = 1.0
    p = fem.Constant(domain, default_scalar_type(-pressure))  # Negative for compression

    # Traction term with H(x, u)
    traction_term = H * p * ufl.dot(v, ufl.FacetNormal(domain)) * ufl.ds

    # Bilinear form
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx - traction_term

    # Linear form
    L = ufl.dot(f, v) * ufl.dx

    # No external force applied at every point
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))

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
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")



print("\n\n-----End of Program-----\n\n")
