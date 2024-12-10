import polyfempy as pf
import meshio
import igl
from collections import defaultdict
import numpy as np
import os, sys
import pyvista as pv

np.set_printoptions(suppress=True, precision=8)

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

def extract_surface_mesh(points, connectivity):
    """
    Extracts the surface triangular mesh from a tetrahedral mesh using PyVista.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.

    Returns:
        tuple: (surface_points, surface_faces)
            - surface_points: Nx3 array of vertex positions.
            - surface_faces: Kx3 array of triangular face indices.
    """
    # Add cell sizes as a prefix to the connectivity
    cells = np.hstack([[4, *tet] for tet in connectivity])  # Prefix 4 indicates tetrahedron
    cell_types = [pv.CellType.TETRA] * len(connectivity)

    # Create an unstructured grid using PyVista
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Extract the surface mesh
    surface_mesh = grid.extract_surface()

    # Surface points and faces
    surface_points = surface_mesh.points
    surface_faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Drop the face size prefix

    return surface_points, surface_faces

def compute_sdf_and_normals(points, faces, query_points):
    """
    Computes the signed distance field (SDF) and surface normals at query points.
    Args:
        points (numpy.ndarray): Nx3 array of vertex positions (float).
        faces (numpy.ndarray): Mx3 array of surface triangle indices (int).
        query_points (numpy.ndarray): Qx3 array of points where SDF and normals are computed.
    Returns:
        tuple: SDF values and normals at query points.
    """
    # Ensure data types are correct
    points = points.astype(np.float64)       # Vertex positions
    faces = faces.astype(np.int32)           # Triangle indices
    query_points = query_points.astype(np.float64)  # Query points

    # Print the input arrays for debugging
    print("\n--- Input Data for Signed Distance Calculation ---\n")
    print(f"Query Points Shape: {query_points.shape}")
    print(f"Query Points:\n{query_points}")

    print(f"\nSurface Vertices Shape: {points.shape}")
    print(f"Surface Vertices:\n{points}")

    print(f"\nSurface Faces Shape: {faces.shape}")
    print(f"Surface Faces:\n{faces}")

    # Compute SDF and normals using libigl
    output = igl.signed_distance(query_points, points, faces, return_normals=True)
    # Extract the results
    sdf = output[0]  # Signed distances
    indices = output[1]  # Indices of closest faces
    closest_points = output[2]  # Closest points on the mesh
    normals = output[3]  # 

    # Print the results in a formatted way
    print("\n--- Signed Distance Output ---\n")
    print(f"Signed Distances (SDF):\n{np.array2string(sdf, precision=4, separator=', ', threshold=10)}\n")
    print(f"Closest Face Indices:\n{np.array2string(indices, separator=', ', threshold=10)}\n")
    print(f"Closest Points on Mesh:\n{np.array2string(closest_points, precision=4, separator=', ', threshold=10)}\n")
    print(f"Normals :\n{np.array2string(normals, precision=4, separator=', ', threshold=10)}\n")


    # Normalize normals to ensure unit vectors
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return sdf, normals


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



def compute_cells_center(points, connectivity, cell_ids):
    """
    Computes the centers of multiple tetrahedral cells.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        cell_ids (list or numpy.ndarray): List of indices of tetrahedral cells.

    Returns:
        numpy.ndarray: Qx3 array of cell centers, where Q is the number of cells.
    """
    # Extract vertex indices for all specified cells
    vertex_indices = connectivity[cell_ids]

    # Retrieve positions of vertices for all cells
    cell_vertices = points[vertex_indices]  # Shape: (Q, 4, 3)

    # Compute the centers as the mean of the vertices along axis 1
    centers = np.mean(cell_vertices, axis=1)  # Shape: (Q, 3)

    return centers

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

    surface_cells = np.array(list(surface_cells)) 
    surface_faces = np.array(surface_faces) 

    return surface_faces, surface_cells


def compute_mesh_center(points, connectivity):
    """
    Computes the geometric center of a tetrahedral mesh.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.

    Returns:
        numpy.ndarray: 1x3 array representing the coordinates of the mesh center.
    """
    # Retrieve all vertices for all tetrahedral cells
    all_cell_vertices = points[connectivity]  # Shape: (M, 4, 3)

    # Compute the center of each tetrahedral cell
    cell_centers = np.mean(all_cell_vertices, axis=1)  # Shape: (M, 3)

    # Compute the overall mesh center as the mean of all cell centers
    mesh_center = np.mean(cell_centers, axis=0)  # Shape: (3,)

    return mesh_center

def find_closest_cell(points, connectivity, target_point):
    """
    Finds the tetrahedral cell whose center is closest to a given point.

    Args:
        points (numpy.ndarray): Nx3 array of vertex positions.
        connectivity (numpy.ndarray): Mx4 array of tetrahedral cell indices.
        target_point (numpy.ndarray): 1x3 array representing the target point.

    Returns:
        int: Index of the closest tetrahedral cell in the connectivity array.
    """
    # Retrieve all vertices for all tetrahedral cells
    all_cell_vertices = points[connectivity]  # Shape: (M, 4, 3)

    # Compute the center of each tetrahedral cell
    cell_centers = np.mean(all_cell_vertices, axis=1)  # Shape: (M, 3)

    # Compute the Euclidean distance between each cell center and the target point
    distances = np.linalg.norm(cell_centers - target_point, axis=1)

    # Find the index of the cell with the minimum distance
    closest_cell_id = np.argmin(distances)

    return closest_cell_id


def main():
    print("\n\n---------------------Start of Program-----------------\n")

    os.chdir(sys.path[0])

    # Load mesh using meshio
    mesh_path = "Bunny/bunny.mesh"  # Replace with actual input mesh path
    points, connectivity = get_tetra_mesh_data(mesh_path)

    print(f"\nPoints.shape() = {np.shape(points)}")
    print(f"Connectivity.shape() = {np.shape(connectivity)}\n")

    print(f"\nPoints = \n{points}\n")
    print(f"\nConnectivity = \n{connectivity}\n")


    # This is to calcualate sdf and sdf gradient. 
    # Sfgt gradient = normal 
    obj_surface_points, obj_surface_faces = extract_surface_mesh(points, connectivity)
    print(f"\nobj_surface_faces = \n{obj_surface_faces}")
    print(f"\nobj_surface_points = \n{obj_surface_points}\n")

    # Here we won't really use surface_face. Surface cells is just obtained so we can apply force on them.
    surface_face, surface_cells = find_surface_cells(points, connectivity) 



    
    surface_cells_center = compute_cells_center(points, connectivity, surface_cells)
    sdfs_surface, normals_surface = compute_sdf_and_normals(obj_surface_points, obj_surface_faces, surface_cells_center)
    print("\n\n\n_______COMPUTED SDF_______________\n")


    
    # See 12 lines above : find_surface_cells
    print(f"The number of surface cells is: \n{len(surface_cells)}\n")
    print(f"surface_cells = \n{surface_cells}\n")
    print("\n_______COMPUTED SURFACE CELLS_______________\n")



    SOME_BORDER_CELL_INDEX_IN_ARRAY = 100
    some_border_cell_id = int(surface_cells[SOME_BORDER_CELL_INDEX_IN_ARRAY])
    some_border_cell_position = surface_cells_center[SOME_BORDER_CELL_INDEX_IN_ARRAY]
    some_border_cell_sdf = sdfs_surface[SOME_BORDER_CELL_INDEX_IN_ARRAY]
    some_border_cell_normal = normals_surface[SOME_BORDER_CELL_INDEX_IN_ARRAY]

    INWARD_FORCE_MULTIPLIER = -1
    FORCE_NORM = 0.5
    force_on_some_border_cell = list(INWARD_FORCE_MULTIPLIER * FORCE_NORM * some_border_cell_normal)
    print(f"\nTaking the {SOME_BORDER_CELL_INDEX_IN_ARRAY}th Border Cell")
    print(f"some_border_cell_id = {some_border_cell_id}")
    print(f"some_border_cell_position = {some_border_cell_position}")
    print(f"some_border_cell_sdf = {some_border_cell_sdf:.4f}")
    print(f"some_border_cell_normal = {some_border_cell_normal}")
    print(f"force_on_some_border_cell = {force_on_some_border_cell}")

    mesh_center = compute_mesh_center(points, connectivity)
    mesh_center_cell_id = find_closest_cell(points, connectivity, mesh_center)
    

    print("\n\n\n_______COMPUTED FORCES_______________\n\n")

    



    center = compute_cell_center(points, connectivity, 100)
    print("__ The center won't move __")
    print(f"mesh_center_cell_id = {mesh_center_cell_id}")
    print(f"Center = {center}")

    print("\n\n\n_______START OF SOLVER_______________\n\n")
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

    print("\n\n")

    

    problem.add_dirichlet_value(id=int(mesh_center_cell_id), value=[0.0, 0.0, 0.0])  # Fix the center
    problem.add_neumann_value(id=some_border_cell_id, value=force_on_some_border_cell)  # Apply inward force on some border cell

    # Assign problem and settings
    settings.set_problem(problem)
    solver.set_settings(settings)

    # Solve the problem
    print("\n\n\nSolving the problem...")
    solver.solve()
    print("\n\nProblem solved.")

    # Export results to visualize (VTU file)
    # solver.export_vtu("./Bunny/modified_bunny.vtu")
    # print("Solution exported to 'Bunny/modified_bunny.vtu'.")


if __name__ == "__main__":
    main()
