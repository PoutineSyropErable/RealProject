# Scaled variable
import pyvista as pv
from dolfinx import mesh, fem, plot, io, default_scalar_type, nls, log
from dolfinx.io.utils import XDMFFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
DEBUG_ = True

# Material properties
E = 10e4  # Young's modulus for rubber in Pascals (Pa)
nu = 0.40  # Poisson's ratio for rubber
rho = 1 # Some rubber that weighs as much as metal

# Finger position (center of the sphere) and radius
finger_position = np.array([0.098629, 0.004755, 0.118844])
R = 0.003  # Radius of the sphere
# Pressure
pressure = 1


# Time parameters
t = 0.0
T = 0.001  # Total simulation time
dt = 5e-8
num_steps = int(T/dt)
print(f"dt={dt}")

# Initialize arrays to store time and maximum displacement
t_array = np.linspace(0, T, num_steps)
max_displacement_array = np.zeros(num_steps)

# Lamé parameters
mu = E / (2 * (1 + nu))
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))



def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)



def load_file(filename):
    # Load the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain

domain = load_file("bunny.xdmf")



Z_GROUND = 0
def grounded_bunny(x):
    return x[2] <= Z_GROUND


V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

tdim = domain.topology.dim # 3D
fdim = tdim - 1 # 2D
boundary_facets = mesh.locate_entities_boundary(domain, fdim , grounded_bunny)
u_D = np.array([0, 0, 0], dtype=default_scalar_type) # no displacement on boundary
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V) # create a condition where u = 0 on z = 0


v = ufl.TestFunction(V)
u_t = ufl.TrialFunction(V)

u_n = fem.Function(V) # u_n
u_n1 = fem.Function(V) # u_n+1

u_tn = fem.Function(V) # du/dt (n)
u_tn1 = fem.Function(V) #du/dt (n+1)

# set u_n, a_tn to 0
# Initialize displacement and velocity to zero
u_n.interpolate(lambda x: np.zeros_like(x))
u_tn.interpolate(lambda x: np.zeros_like(x))
u_n1.interpolate(lambda x: np.zeros_like(x))
u_tn1.interpolate(lambda x: np.zeros_like(x))


# Body force
f = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))

p = fem.Constant(domain, PETSc.ScalarType(-pressure))  # Negative for compression

# Create the solver
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Create the matrix and vector
a = (rho) * ufl.inner(u_t, v) * ufl.dx
bilinear_form = fem.form(a)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
solver.setOperators(A)


PYVISTA_SHOW = False
if PYVISTA_SHOW:
    #------------------ pyvista initialisation

    # Create the PyVista plotter
    plotter = pv.Plotter()

    #------------------------------------------------------ FINGER PRESSURE POINT
    # Define sphere marker for the finger position
    finger_marker = pv.Sphere(radius=R, center=finger_position)
    plotter.add_mesh(finger_marker, color="red", label="Finger Position", opacity=1.0)

    #------------------------------------------------------ BUNNY MESH 
    # Extract mesh data
    connectivity, cell_types, points = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(connectivity, cell_types, points)

    # Add the initial mesh
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    plotter.show_axes()

    # Open a movie file for the animation
    plotter.open_movie("bunny_deformation_animation.mp4", framerate=10)

# Open an XDMF file to save u(t)
xdmf_file = io.XDMFFile(domain.comm, "bunny_displacement.xdmf", "w")
xdmf_file.write_mesh(domain)  # Write the mesh structure once

print("")
for step in range(num_steps):
    t += dt
    print("-----------------------------------------------")
    print(f"Time step {step+1}/{num_steps}, t = {t:.6f}")

    # Update H(x + u_n)
    x = ufl.SpatialCoordinate(domain)
    x_deformed = x + u_n
    distance = ufl.sqrt(
        (x_deformed[0] - finger_position[0])**2 +
        (x_deformed[1] - finger_position[1])**2 +
        (x_deformed[2] - finger_position[2])**2
    )
    H = ufl.conditional(ufl.lt(distance, R), 1.0, 0.0) # If touching finger

    # Traction term
    traction_term = H * p * ufl.dot(v, ufl.FacetNormal(domain)) * ufl.ds

    # Linear form
    L = (
        rho * ufl.inner(u_tn, v) * ufl.dx
        + dt * ufl.inner(f, v) * ufl.dx
        - dt * ufl.inner(sigma(u_n), epsilon(v)) * ufl.dx
        + dt * traction_term
    )
    linear_form = fem.form(L)
    b = create_vector(linear_form) # L(v)

    # Assemble RHS vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)


    # Solve for velocity update (u_tn1)
    solver.solve(b, u_tn1.x.petsc_vec)
    u_tn1.x.scatter_forward()

    # Update displacement
    u_n.x.array[:] += dt * u_tn1.x.array
    # Reshape the displacement array to match the (num_points, 3) structure

    # Update velocity for the next time step
    u_tn.x.array[:] = u_tn1.x.array
    # Save displacement field to XDMF
    xdmf_file.write_function(u_n, t)

    displacements = u_n.x.array.reshape(domain.geometry.x.shape)
    if abs(t - 0.02) < 0.01:
        print(f"type(displacements) = {type(displacements)}\n")
        print(f"np.shape(displacements) = {np.shape(displacements)}\n")
    
    # Print displacement information for debugging
    displacement_norms = np.linalg.norm(displacements, axis=1)
    max_norm = np.max(displacement_norms)
    max_index = np.argmax(displacement_norms)
    point_with_max_displacement = domain.geometry.x[max_index]
    max_displacement_array[step] = max_norm

    print(f"displacements = \n{displacements}\n")
    print(f"Maximum displacement norm: {max_norm}")
    print(f"Point with maximum displacement: {point_with_max_displacement}")
    print(f"Point of finger pressure:        {finger_position}")
    
    print(f"type(finger_position) = {type(finger_position)}\n")
    print(f"np.shape(finger_position) = {np.shape(finger_position)}\n")
    print(f"finger_position = \n{finger_position}\n")

    # Ensure finger_position is reshaped to (num_points, 3)
    evaluation_point = finger_position.reshape((1, 3))  # Shape (1, 3)
    # Compute the bounding box tree for the mesh
    tree = domain.geometry.bounding_box_tree()

    # Compute collisions
    collisions = compute_collisions(tree, evaluation_point)
    cells = select_colliding_cells(collisions, evaluation_point)

    # Check if the point is inside the mesh
    if len(cells) > 0:
        # Evaluate the function at the given point
        displacement = u_n.eval(evaluation_point.T, cells)
        print(f"Displacement at finger position {finger_position}: {displacement}")
    else:
        print(f"Point {finger_position} is outside the mesh!")




    if PYVISTA_SHOW:
        deformed_points = points + displacements
        grid.points = deformed_points  # Update PyVista grid points

        # Add the deformed mesh to the plotter
        plotter.add_mesh(grid, show_edges=True, color="lightblue")
        plotter.write_frame()  # Write the frame to the movie file

    print("\n-----------------------------------------------")



print("\n\n-----End of Simulation-----\n\n")# Close the movie file


if PYVISTA_SHOW:
    # Close the movie file
    plotter.close()

# Close the XDMF file
xdmf_file.close()
