import numpy as np
import igl
import polyscope as ps

import matplotlib.pyplot as plt


import h5py
import pyvista as pv
from dolfinx.io.utils import XDMFFile
from dolfinx import mesh
from mpi4py import MPI

import os, sys
import argparse



def load_mesh_domain(xdmf_file):
    """
    Load the mesh from an XDMF file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain

def load_displacement_data(h5_file):
    """
    Load displacement data from an HDF5 file.
    """
    with h5py.File(h5_file, "r") as f:
        displacements = f["displacements"][:]
        print(f"Loaded displacement data with shape: {displacements.shape}")
    return displacements

def load_mesh_t(xdmf_file, h5_file, time_index):
    domain = load_mesh_domain(xdmf_file)
    
    points = domain.geometry.x
    points = points.astype(np.float32)

    conn = domain.topology.connectivity(3,0)
    connectivity_array = conn.array
    offsets = conn.offsets
    # Convert the flat connectivity array into a list of arrays
    connectivity = [
        connectivity_array[start:end]
        for start, end in zip(offsets[:-1], offsets[1:])
    ]

    connectivity = np.array(connectivity)

    displacements_all_times = load_displacement_data(h5_file)
    points_moved = points + displacements_all_times[time_index]

    return points_moved, connectivity

def get_surface_mesh(points: np.ndarray, connectivity: np.ndarray) -> (np.ndarray, np.ndarray):

    # Convert to PyVista UnstructuredGrid
    cells = np.hstack([np.full((connectivity.shape[0], 1), 4), connectivity]).flatten()
    cell_types = np.full(connectivity.shape[0], 10, dtype=np.uint8)  # Tetrahedron type
    tetra_mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # Extract surface mesh
    surface_mesh = tetra_mesh.extract_surface()

    # Get vertices and faces
    vertices = surface_mesh.points
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix

    return vertices, faces
    

def compute_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the bounding box for the vertices."""
    b_min = np.zeros(3)
    b_max = np.zeros(3)

    for i in range(3):
        min_val, max_val = mesh_points[:, i].min(), mesh_points[:, i].max()
        print("min, max=", min_val, max_val)

        center = (min_val + max_val) / 2
        half_length = max_val - center

        BOX_RATIO = 1.5

        print("c,h=", center, half_length)

        b_min[i] = center - half_length * BOX_RATIO
        b_max[i] = center + half_length * BOX_RATIO

    return b_min, b_max


def compute_small_bounding_box(mesh_points: np.ndarray) -> (np.ndarray, np.ndarray):
    b_min = np.zeros(3)
    b_max = np.zeros(3)

    for i in range(3):
        min_val, max_val = mesh_points[:, i].min(), mesh_points[:, i].max()
        b_min[i] = min_val
        b_max[i] = max_val

    return b_min, b_max


def generate_random_points(
    b_min: np.ndarray, b_max: np.ndarray, num_points: int
) -> np.ndarray:
    """Generate random points within the bounding box."""
    point_list = np.zeros(shape=(num_points, 3))

    # Working with one column at a time. IE: All X -> All Y -> All Z
    for i in range(3):
        random_points_i = np.random.uniform(b_min[i], b_max[i], num_points)
        print(f"rand_{i}=", random_points_i)
        point_list[:, i] = random_points_i
        print("---")

    return point_list


def compute_signed_distances(
    point_list: np.ndarray, mesh_points: np.ndarray, mesh_connectivity: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute signed distances from points to the triangle mesh."""

    signed_distances, nearest_face, nearest_points = igl.signed_distance(point_list, mesh_points, mesh_connectivity)
    return signed_distances, nearest_face, nearest_points


def draw_bounding_box(
    b_min: np.ndarray,
    b_max: np.ndarray,
    name: str,
    color: tuple = (0.0, 1.0, 0.0),
    radius: float = 0.002,
):
    """Draw a bounding box in Polyscope given min and max points."""
    # Create corners of the bounding box
    box_corners = np.array(
        [
            [b_min[0], b_min[1], b_min[2]],
            [b_max[0], b_min[1], b_min[2]],
            [b_max[0], b_max[1], b_min[2]],
            [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]],
            [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_max[2]],
            [b_min[0], b_max[1], b_max[2]],
        ]
    )

    # Define edges for the bounding box
    box_edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )  # Vertical edges

    # Register the bounding box as a curve network
    ps_bounding_box = ps.register_curve_network(name, box_corners, box_edges)
    ps_bounding_box.set_radius(radius)  # Adjust bounding box line thickness
    ps_bounding_box.set_color(color)  # Set the color for the bounding box




def weight_function(signed_distance: float, weight_exponent: float = 8) -> float:
    """Takes a signed_distances and return a probability of taking said points"""
    return (1 + abs(signed_distance)) ** (-weight_exponent)


def filter_function(signed_distance: float, weight_exponent: float) -> bool:
    "Returns a bool or not, deciding weither or not to take the point"

    random_number = np.random.rand()
    return random_number < weight_function(signed_distance, weight_exponent)


def filter_points(signed_distances: np.ndarray, weight_exponent: float) -> np.ndarray:
    """Filter points based on their signed distances."""

    filtered_index = np.array(
        [
            i
            for i in range(len(signed_distances))
            if filter_function(signed_distances[i], weight_exponent)
        ]
    )
    return filtered_index


def show_histogram(filtered_signed_distances):
    # Show Histogram of signed_distances
    plt.figure()
    plt.title("Histogram of Signed Distances")
    plt.grid()
    plt.xlabel("Signed Distances")
    plt.ylabel("Counts")
    plt.hist(filtered_signed_distances)
    plt.show()


# ---------------------- Start of ML ----------------------


def main():

    print(f"\n\n{'-'*10} Start of Program{'-'*10}\n\n")
    os.chdir(sys.path[0])


    # Load the mesh
    
    BUNNY_FILE = "bunny.xdmf" 
    TIME_INDEX = 10

    DISPLACEMENT_INDEX = 0
    DISPLACEMENT_DIRECTORY = "./deformed_bunny_files"
    DISPLACEMENT_FILE = f"{DISPLACEMENT_DIRECTORY}/displacement_{DISPLACEMENT_INDEX}.h5"




    
    mesh_points, mesh_connectivity = load_mesh_t(BUNNY_FILE, DISPLACEMENT_FILE, TIME_INDEX)
    print("\n\n------------------Start of Program----------------------------\n\n")
    print(f"\ntype(mesh_points) = {type(mesh_points)}")
    print(f"np.shape(mesh_points) = {np.shape(mesh_points)}")
    print(f"mesh_points = \n{mesh_points}\n")

    print(f"\ntype(mesh_connectivity) = {type(mesh_connectivity)}")
    print(f"np.shape(mesh_connectivity) = {np.shape(mesh_connectivity)}")
    print(f"mesh_connectivity = \n{mesh_connectivity}\n")


    vertices, faces = get_surface_mesh(mesh_points, mesh_connectivity)



    # Compute bounding box
    b_min, b_max = compute_bounding_box(vertices)
    print("------\n\n\n")
    print(f"b_min = {b_min},  b_max ={b_max}\n\n\n")

    # Generate random points
    NUMBER_POINT = 1_000_000
    point_list = generate_random_points(b_min, b_max, NUMBER_POINT)

    print("\n\n")
    print(f"point_list=\n{point_list}")

    # Compute signed distances
    signed_distances, nearest_face, nearest_points = compute_signed_distances(point_list, vertices, faces)

    diff = nearest_points - point_list
    l2_norms = np.linalg.norm(diff, axis=1)

    print(f"\n\n  signed distance   = {signed_distances}")
    print(f"Norm(nearest-point) = {l2_norms}\n")
    # print(f"nearest_face = {nearest_face}\n")
    # print(f"nearest_points = {nearest_points}\n")
    print(f"3 nearest_shape = {np.shape(nearest_points)}")

    # Print min/max signed distances
    min_dn, max_dn = np.min(signed_distances[signed_distances < 0]), np.max(
        signed_distances[signed_distances < 0]
    )
    min_dp, max_dp = np.min(signed_distances[signed_distances > 0]), np.max(
        signed_distances[signed_distances > 0]
    )
    print(f"min_dn={min_dn}, max_dn={max_dn}")
    print(f"min_dp={min_dp}, max_dp={max_dp}")

    # Filter points
    filtered_index = filter_points(signed_distances, weight_exponent=20)
    filtered_signed_distances = signed_distances[filtered_index]
    filtered_points = point_list[filtered_index]
    filtered_nearest = nearest_points[filtered_index]

    print(f"\n\n\nfiltered_index= {filtered_index}, {len(filtered_index)} \n")
    print(f"filtered_points = \n{filtered_points} \n")
    print(f"filtered_signed_distances = \n{filtered_signed_distances} \n")



if __name__ == "__main__":
    main()

