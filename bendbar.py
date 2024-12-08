import polyfempy as pf

# Initialize solver
solver = pf.Solver()

# Load bar mesh
mesh_path = "bar.mesh"  # Replace with actual mesh file path
solver.load_mesh_from_path(mesh_path, normalize_mesh=True)

# Set solver settings
settings = pf.Settings()
settings.set_pde(pf.PDEs.LinearElasticity)  # Linear Elasticity problem
settings.set_material_params("E", 200.0)  # Young's modulus
settings.set_material_params("nu", 0.3)  # Poisson's ratio

# Define boundary conditions
problem = pf.Problem()

# Fix one end (Dirichlet BC) - Assume sideset ID 1
problem.add_dirichlet_value(id=1, value=[0.0, 0.0, 0.0])

# Apply force on another end (Neumann BC) - Assume sideset ID 2
force_vector = [0.0, -10.0, 0.0]  # Downward force in the Y-direction
problem.add_neumann_value(id=2, value=force_vector)

# Assign problem and settings
settings.set_problem(problem)
solver.set_settings(settings)

# Solve the problem
solver.solve()

# Export results to visualize (VTU file)
solver.export_vtu("bent_bar.vtu")
print("Solution exported to 'bent_bar.vtu'")
