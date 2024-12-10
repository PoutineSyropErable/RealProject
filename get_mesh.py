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


# Example Usage
if __name__ == "__main__":
    import sys

    # Check command-line argument
    if len(sys.argv) != 2:
        print("Usage: python get_tetra_mesh_data.py <input_mesh_file>")
        sys.exit(1)

    # File path
    mesh_file = sys.argv[1]

    # Process the mesh file
    try:
        points, connectivity = get_tetra_mesh_data(mesh_file)
        print("Mesh Data Extracted Successfully!")
        print(f"Number of Points: {points.shape[0]}")
        print(f"Number of Tetrahedra: {connectivity.shape[0]}")
    except Exception as e:
        print(f"Error: {e}")
