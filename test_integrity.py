import meshio

def check_mesh_integrity(mesh_path):
    """
    Checks the mesh integrity using meshio.
    """
    try:
        mesh = meshio.read(mesh_path)
        print(f"Mesh loaded successfully: {mesh_path}")
        print(f"Number of points: {mesh.points.shape[0]}")
        print(f"Number of cells: {len(mesh.cells_dict)}")
    except Exception as e:
        print(f"Error reading mesh: {e}")

# Check bunny.stl
check_mesh_integrity("bunny.stl")
