import polyfempy as pf
import meshio
import numpy as np


def get_tetra_mesh_data(file_path):
    """
    Extracts points and tetrahedral connectivity from a mesh file.

    Args:
        file_path (str): Path to the input mesh file.

    Returns:
        tuple: A tuple (points, connectivity), where:
            - points: numpy.ndarray of shape (N, 3), the mesh vertex coordinates.
            - connectivity: numpy.ndarray of shape (M, 4), the tetrahedral cell indices.

    Raises:
        ValueError: If no tetrahedral cells are found in the mesh.
        FileNotFoundError: If the file does not exist.
    """
    # Load the mesh file
    try:
        mesh = meshio.read(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading mesh file: {e}")

    # Extract points
    points = mesh.points

    # Find tetrahedral cells
    connectivity = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            connectivity = cell_block.data
            break

    # Raise error if no tets are found
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")

    return points, connectivity


def main():
    # Load mesh using meshio
    mesh_path = "Bunny/bunny.mesh"  # Replace with actual input mesh path
    points, connectivity = get_tetra_mesh_data(mesh_path)

    print(f"\nPoints.shape() = {np.shape(points)}")
    print(f"Connectivity.shape() = {np.shape(connectivity)}\n")

    print(f"\nPoints = \n{points}\n")
    print(f"\nConnectivity = \n{connectivity}\n")


    # Initialize the PolyFEM solver
    solver = pf.Solver()

    # Set the mesh using vertices and faces
    solver.set_mesh(vertices=points, connectivity=connectivity, normalize_mesh=True)
    print("Mesh successfully loaded into PolyFEM.")

    # Set solver settings
    settings = pf.Settings()
    settings.set_pde(pf.PDEs.LinearElasticity)  # Linear Elasticity problem
    settings.set_material_params("E", 200.0)  # Young's modulus
    settings.set_material_params("nu", 0.3)  # Poisson's ratio

    # Define boundary conditions
    problem = pf.Problem()
    problem.add_dirichlet_value(id=1, value=[0.0, 0.0, 0.0])  # Fix one end
    problem.add_neumann_value(id=2, value=[0.0, -1.0, 0.0])   # Apply downward force

    # Assign problem and settings
    settings.set_problem(problem)
    solver.set_settings(settings)

    # Solve the problem
    print("Solving the problem...")
    solver.solve()
    print("Problem solved.")

    # Export results to visualize (VTU file)
    solver.export_vtu("modified_bunny.vtu")
    print("Solution exported to 'bent_bar.vtu'.")


if __name__ == "__main__":
    main()
