from dolfinx.io.utils import XDMFFile
from mpi4py import MPI
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl


#Material property
E = 5000  # Young's modulus (Pa)
nu = 0.35  # Poisson's ratio
rho = 1000  # Density (kg/m^3)

mu = E / (2 * (1 + nu))
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))



# Load the mesh from the XDMF file
with XDMFFile(MPI.COMM_WORLD, "bunny.xdmf", "r") as xdmf:
    tetra_mesh = xdmf.read_mesh(name="Grid")
    print("Mesh loaded successfully!")


domain = tetra_mesh

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

z_max = 0.15416191518306732


def near_floor_boundary(point):
    return np.isclose(point[2],0)

fdim = tetra_mesh.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, near_floor_boundary)


def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


