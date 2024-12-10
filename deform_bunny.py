import polyfempy as pf
import meshio
import igl
from collections import defaultdict
import numpy as np
import os, sys

# master_venv on windows
# conda_venv on linux


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
    connectivity = mesh.cells_dict.get("tetra", None)

    # Raise error if no tets are found
    if connectivity is None:
        raise ValueError("No tetrahedral cells found in the mesh file.")

    return points, connectivity


def compute_cell_center(points, connectivity, cell_id):
    """
    Computes the center of a tetrahedral cell.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        cell_id (int): Index of the tetrahedral cell in the connectivity.

    Returns:
        numpy.ndarray: Coordinates of the center of the specified cell.
    """
    # Extract the vertex indices for the given cell
    vertex_indices = connectivity[cell_id]

    # Retrieve the positions of the vertices
    cell_vertices = points[vertex_indices]

    # Compute the center as the average of the vertices
    center = np.mean(cell_vertices, axis=0)
    return center

def compute_location(points, connectivity, cell_id, barycentric_coords):
    """
    Computes the Cartesian coordinates of a point given its barycentric coordinates 
    in a specified tetrahedral cell.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        cell_id (int): Index of the tetrahedral cell in the connectivity.
        barycentric_coords (list or numpy.ndarray): Barycentric coordinates [w0, w1, w2, w3].

    Returns:
        numpy.ndarray: Cartesian coordinates of the point.
    """
    # Extract vertex indices for the cell
    vertex_indices = connectivity[cell_id]
    
    # Retrieve the positions of the vertices
    cell_vertices = points[vertex_indices]  # Shape: (4, 3)
    
    # Check if barycentric coordinates sum to approximately 1
    if not np.isclose(np.sum(barycentric_coords), 1.0):
        raise ValueError("Barycentric coordinates must sum to 1.")
    
    # Compute the Cartesian coordinates using the barycentric formula. T for 
    location = np.dot(barycentric_coords, cell_vertices)
    
    return location

def compute_sdf_and_normals(points, connectivity, query_points):
    """
    Computes the signed distance field (SDF) and surface normals (from gradient of SDF)
    for a set of query points relative to a tetrahedral mesh.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        query_points (numpy.ndarray): Qx3 array of points where SDF and normals are computed.

    Returns:
        tuple: A tuple (sdf, normals) where:
            - sdf: Qx1 numpy.ndarray of signed distances.
            - normals: Qx3 numpy.ndarray of normalized surface normals.
    """
    # Convert tetrahedral mesh to triangular surface mesh
    V = points  # Vertices of the mesh
    T = connectivity  # Tetrahedral cell connectivity

    # Extract surface triangular mesh
    F = igl.boundary_facets(T)

    # Compute the SDF and gradients
    sdf, normals, _ = igl.signed_distance(query_points, V, F, return_normals=True)

    # Normalize the normals (to ensure unit length)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return sdf, normals

def compute_sdf_and_normal_single_point(points, connectivity, query_point):
    """
    Computes the signed distance field (SDF) and surface normal (gradient of SDF)
    for a single query point relative to a tetrahedral mesh.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        query_point (numpy.ndarray): 1x3 array representing the point to query.

    Returns:
        tuple: A tuple (sdf, normal) where:
            - sdf: Signed distance as a float.
            - normal: 1x3 numpy.ndarray of the normalized surface normal.
    """
    # Convert tetrahedral mesh to triangular surface mesh
    V = points  # Vertices of the mesh
    T = connectivity  # Tetrahedral cell connectivity

    # Extract surface triangular mesh
    F = igl.boundary_facets(T)

    # Ensure query_point is a 2D array of shape (1, 3)
    query_point = np.array(query_point).reshape(1, 3)

    # Compute the SDF and normal for the single point
    sdf, normals, _ = igl.signed_distance(query_point, V, F, return_normals=True)

    # Normalize the normal (ensure it's a unit vector)
    normal = normals[0] / np.linalg.norm(normals[0])

    return sdf[0], normal

def find_surface_cells(points, connectivity):
    """
    Finds the surface triangular faces and the bordering tetrahedral cells in a tetrahedral mesh.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.

    Returns:
        tuple:
            - surface_faces: List of unique surface triangular faces.
            - surface_cells: Set of indices of tetrahedral cells that share surface faces.
    """
    face_count = defaultdict(int)  # Track the number of times each face appears
    face_to_cell = defaultdict(list)  # Map faces to the tetrahedral cells they belong to

    # Function to sort a face (ensures consistent representation)
    def sorted_face(v1, v2, v3):
        return tuple(sorted([v1, v2, v3]))

    # Iterate over all tetrahedral cells
    for cell_idx, cell in enumerate(connectivity):
        # Extract the 4 faces of the tetrahedron
        faces = [
            sorted_face(cell[0], cell[1], cell[2]),
            sorted_face(cell[0], cell[1], cell[3]),
            sorted_face(cell[0], cell[2], cell[3]),
            sorted_face(cell[1], cell[2], cell[3]),
        ]

        # Update the face count and map faces to cells
        for face in faces:
            face_count[face] += 1
            face_to_cell[face].append(cell_idx)

    # Identify faces that appear only once (surface faces)
    surface_faces = [face for face, count in face_count.items() if count == 1]

    # Identify cells that share these surface faces
    surface_cells = set()
    for face in surface_faces:
        surface_cells.update(face_to_cell[face])

    return surface_faces, surface_cells


def main():

    os.chdir(sys.path[0])

    # Load mesh using meshio
    mesh_path = "Bunny/bunny.mesh"  # Replace with actual input mesh path
    points, connectivity = get_tetra_mesh_data(mesh_path)

    surface_face, surface_cells = find_surface_cells(points, connectivity)

    print(f"The surface cells are: \n{surface_cells}\n")

    print(f"\nPoints.shape() = {np.shape(points)}")
    print(f"Connectivity.shape() = {np.shape(connectivity)}\n")

    print(f"\nPoints = \n{points}\n")
    print(f"\nConnectivity = \n{connectivity}\n")

    center = compute_cell_center(points, connectivity, 100)
    print(f"Center = {center}\n")


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
    problem.add_neumann_value(id=100, value=[0.0, -1.0, 0.0])   # Apply downward force

    # Assign problem and settings
    settings.set_problem(problem)
    solver.set_settings(settings)

    # Solve the problem
    print("Solving the problem...")
    solver.solve()
    print("Problem solved.")

    # Export results to visualize (VTU file)
    solver.export_vtu("./Bunny/modified_bunny.vtu")
    print("Solution exported to 'Bunny/modified_bunny.vtu'.")


if __name__ == "__main__":
    main()
