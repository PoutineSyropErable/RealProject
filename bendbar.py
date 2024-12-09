import polyfempy as pf
import meshio
import numpy as np


def load_mesh_data(mesh_path):
    """
    Load a mesh file and return vertices and faces/cells.

    Args:
        mesh_path (str): Path to the input mesh file.

    Returns:
        tuple: (vertices, faces) as numpy arrays.
               vertices - Nx3 array of vertex coordinates.
               faces    - MxK array of cell indices (K=3 for triangles, K=4 for tets).

    Raises:
        ValueError: If no supported cells (triangle/tetra) are found.
    """
    # Load the mesh using meshio
    mesh = meshio.read(mesh_path)

    # Extract vertices
    vertices = mesh.points

    # Extract faces/cells
    faces = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":  # For 2D meshes
            faces = cell_block.data
            print("Loaded triangular mesh.")
            break
        elif cell_block.type == "tetra":  # For 3D meshes
            faces = cell_block.data
            print("Loaded tetrahedral mesh.")
            break

    # If no supported cells are found, raise an error
    if faces is None:
        raise ValueError("No supported cells (triangle or tetrahedral) found in the mesh file.")

    return vertices, faces


def main():
    # Load mesh using meshio
    mesh_path = "bunny.mesh"  # Replace with actual input mesh path
    vertices, faces = load_mesh_data(mesh_path)

    print(np.shape(vertices))
    print(np.shape(faces))

    # Initialize the PolyFEM solver
    solver = pf.Solver()

    # Set the mesh using vertices and faces
    solver.set_mesh(vertices=vertices, connectivity=faces, normalize_mesh=True)
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
    solver.export_vtu("bent_bar.vtu")
    print("Solution exported to 'bent_bar.vtu'.")

    # Get solution (displacements)
    solution_vertices = solver.get_sampled_solution()[0]
    print("Number of points in Solution:", len(solution_vertices))

    # Export the resulting mesh using meshio
    output_mesh = "exported_bent_bar.mesh"
    meshio.write_points_cells(
        output_mesh,
        points=solution_vertices,
        cells=[("tetra", faces)]
    )
    print(f"Exported the resulting mesh to: {output_mesh}")


if __name__ == "__main__":
    main()
