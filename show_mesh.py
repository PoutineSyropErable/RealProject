import polyscope as ps
import meshio
import os, sys
import numpy as np

def show_mesh_with_meshio(file_path: str):
    # Start Polyscope
    ps.init()

    # Load mesh using meshio
    mesh = meshio.read(file_path)

    # Extract vertices and faces
    vertices = mesh.points
    faces = mesh.cells_dict.get("triangle", None)  # Extract triangle faces

    if faces is None:
        print("No triangle faces found in the mesh. Polyscope requires triangular meshes.")
        return

    # Register the mesh with Polyscope
    ps.register_surface_mesh(
        "Mesh Viewer",
        vertices,
        faces
    )

    # Show the mesh in the viewer
    ps.show()

# Example usage
def main():
    # Change directory to script location
    os.chdir(sys.path[0])

    # Path to the mesh file
    file_path = "bunny.obj"  # Replace with your mesh file
    show_mesh_with_meshio(file_path)

if __name__ == "__main__":
    main()
