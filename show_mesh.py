import polyscope as ps
import meshio
import os, sys
import numpy as np

# Function to check if an array is empty
def is_empty(array):
    return array.size == 0

def show_mesh_with_meshio(file_path: str):

    # Load mesh using meshio
    mesh = meshio.read(file_path)

    # Extract vertices and faces
    points = mesh.points
    dict_result = mesh.cells_dict
    faces = mesh.cells_dict.get("triangle", None)  # Extract triangle faces
    connectivity = mesh.cells_dict.get("tetra", None)  # Extract triangle faces

    print(f"Types of faces and conectivity: {type(faces)}, {type(connectivity)}\n")
    print(f"Faces = \n{faces}\n")
    print(f"Connectivity = \n{connectivity}\n")

    if not is_empty(connectivity) and is_empty(faces):
        edges = connectivity
        print("It's a tetra shape")
    elif is_empty(connectivity) and not is_empty(faces):
        edges = faces
        print("It's a triangle shape\n")
    else:
        AssertionError("No face or edges")

    
    return points, edges


def show_mesh(points, edges):
    # Start Polyscope
    ps.init()


    # Register the mesh with Polyscope
    ps.register_surface_mesh(
        "Mesh Viewer",
        points,
        edges
    )

    # Show the mesh in the viewer
    ps.show()

# Example usage
def main():

    print("\n\n\n-----------Start of Program---------------------\n")
    # Change directory to script location
    os.chdir(sys.path[0])

    # Default path to the mesh file
    file_path = "Bunny/bunny.mesh" 

    # Use command-line argument if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]


    points, edges = show_mesh_with_meshio(file_path)
    show_mesh(points,edges)

if __name__ == "__main__":
    main()
