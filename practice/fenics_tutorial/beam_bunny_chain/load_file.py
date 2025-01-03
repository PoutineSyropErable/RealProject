import numpy as np
from dolfinx import fem, io
from mpi4py import MPI

def load_mesh(mesh_file):
    """
    Load the mesh from an XDMF file.

    Parameters:
    - mesh_file: Name of the XDMF file containing the mesh.

    Returns:
    - Mesh object.
    """
    with io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf_file:
        domain = xdmf_file.read_mesh(name="Grid")
        print("Mesh loaded successfully.")
    return domain


def load_displacement(displacement_file, domain):
    """
    Load the displacement function u(t) from an XDMF file using the given mesh.

    Parameters:
    - displacement_file: Name of the XDMF file containing the displacement data.
    - domain: Mesh object.

    Returns:
    - List of time steps and their corresponding displacement functions.
    """
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

    with io.XDMFFile(MPI.COMM_WORLD, displacement_file, "r") as xdmf_file:
        u = fem.Function(V)  # Create a function to store the displacement
        times = []  # List to store time steps
        displacements = []  # List to store displacement functions

        # Manually iterate through stored functions
        while True:
            try:
                time = xdmf_file.read_function(u, -1)  # Read the next function with its time step
                times.append(time)
                displacements.append(u.copy(deepcopy=True))  # Store a deep copy of the function
            except RuntimeError:
                # End of file reached
                break

    return times, displacements


# File paths
mesh_file = "bunny.xdmf"
displacement_file = "bunny_displacement.xdmf"

# Load mesh
domain = load_mesh(mesh_file)

# Load displacement data
times, displacements = load_displacement(displacement_file, domain)

# Output results
print(f"Number of time steps: {len(times)}")
for i, time in enumerate(times):
    print(f"Time step {i + 1}: t = {time}")
    print(f"Displacement norm: {np.linalg.norm(displacements[i].x.array)}")

# Example: Save the displacement norm over time
displacement_norms = [np.linalg.norm(u.x.array) for u in displacements]
np.savetxt("displacement_norms.csv", np.column_stack((times, displacement_norms)), delimiter=",", header="Time,Displacement_Norm")

print("Displacement norms saved to displacement_norms.csv.")

