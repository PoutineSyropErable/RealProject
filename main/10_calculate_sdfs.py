import numpy as np
import igl
import h5py
import pyvista as pv
from dolfinx.io.utils import XDMFFile
from dolfinx import mesh
from mpi4py import MPI

import argparse
import os, sys

def load_mesh_domain(xdmf_file):
    """Load the mesh from an XDMF file."""
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        print("Mesh loaded successfully!")
    return domain

def load_displacement_data(h5_file):
    """Load displacement data from an HDF5 file."""
    with h5py.File(h5_file, "r") as f:
        displacements = f["displacements"][:]
        print(f"Loaded displacement data with shape: {displacements.shape}")
    return displacements

def get_surface_mesh(points: np.ndarray, connectivity: np.ndarray):
    """Extract the surface mesh from the tetrahedral mesh."""
    cells = np.hstack([np.full((connectivity.shape[0], 1), 4), connectivity]).flatten()
    cell_types = np.full(connectivity.shape[0], 10, dtype=np.uint8)  # Tetrahedron type
    tetra_mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # Extract surface mesh
    surface_mesh = tetra_mesh.extract_surface()

    # Get vertices and faces
    vertices = surface_mesh.points
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove face size prefix

    return vertices, faces

def compute_bounding_box(mesh_points: np.ndarray):
    """Compute the bounding box for the vertices."""
    b_min = mesh_points.min(axis=0)
    b_max = mesh_points.max(axis=0)

    BOX_RATIO = 1.5
    center = (b_min + b_max) / 2
    half_lengths = (b_max - b_min) / 2 * BOX_RATIO
    b_min = center - half_lengths
    b_max = center + half_lengths

    return b_min, b_max

def generate_random_points(b_min: np.ndarray, b_max: np.ndarray, num_points: int):
    """Generate random points within the bounding box."""
    return np.random.uniform(b_min, b_max, size=(num_points, 3))

def compute_signed_distances(point_list: np.ndarray, mesh_points: np.ndarray, mesh_faces: np.ndarray):
    """Compute signed distances from points to the triangle mesh."""
    signed_distances, nearest_face, nearest_points = igl.signed_distance(point_list, mesh_points, mesh_faces)
    return signed_distances, nearest_face, nearest_points





def weight_function(signed_distance: float, weight_exponent: float = 10) -> float:
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

def main():
    print(f"\n\n{'-'*10} Start of Program{'-'*10}\n\n")

    parser = argparse.ArgumentParser(description="Calculate SDF for deformed meshes.")
    parser.add_argument("--index", type=int, required=True, help="Index of the deformation file.")
    parser.add_argument("--sdf_only", action="store_true", help="Only save the SDF values.")
    args = parser.parse_args()

    INDEX = args.index
    SDF_ONLY = args.sdf_only
    print(f"\n\n{'-' * 10} Processing INDEX: {INDEX} {'-' * 10}\n\n")
    print(f"SDF only mode: {'Enabled' if SDF_ONLY else 'Disabled'}")

    os.chdir(sys.path[0])

    # Input files
    BUNNY_FILE = "bunny.xdmf"
    DISPLACEMENT_FILE = f"./deformed_bunny_files/displacement_{INDEX}.h5"
    OUTPUT_FILE = f"./calculated_sdf/sdf_points_{INDEX}{'_sdf_only' if SDF_ONLY else ''}.h5"
    os.makedirs("./calculated_sdf", exist_ok=True)

    # Load mesh and displacements
    domain = load_mesh_domain(BUNNY_FILE)
    displacements_all_times = load_displacement_data(DISPLACEMENT_FILE)

    # Get initial points and connectivity
    points = domain.geometry.x
    connectivity = np.array([domain.topology.connectivity(3, 0).array[i:i+4] for i in range(0, len(domain.topology.connectivity(3, 0).array), 4)])

    # Extract bounding box from the undeformed mesh
    b_min, b_max = compute_bounding_box(points)
    print(f"Bounding box:\nMin: {b_min}\nMax: {b_max}\n")

    # File to store filtered points
    SDF_POINTS_FILE = "points_to_take_sdf.npy"
    if os.path.exists(SDF_POINTS_FILE):
        # Load precomputed points if the file exists
        filtered_points = np.load(SDF_POINTS_FILE)
        print(f"Loaded {len(filtered_points)} filtered points from {SDF_POINTS_FILE}.")
    else:
        # Generate one set of random points for SDF evaluation
        NUM_POINTS = 1_000_000
        point_list = generate_random_points(b_min, b_max, NUM_POINTS)
        print(f"Generated {NUM_POINTS} random points for SDF evaluation.")
        
        # Extract surface mesh from the original (undeformed) mesh
        vertices, faces = get_surface_mesh(points, connectivity)
        
        # Compute signed distances
        signed_distances, _, _ = compute_signed_distances(point_list, vertices, faces)
        
        # Filter points based on signed distances
        filtered_index = filter_points(signed_distances, weight_exponent=20)
        filtered_points = point_list[filtered_index]
        
        # Save filtered points for future use
        np.save(SDF_POINTS_FILE, filtered_points)
        print(f"Saved {len(filtered_points)} filtered points to {SDF_POINTS_FILE}.")


    # Prepare to store SDF results
    sdf_results = []

    for t_index in range(displacements_all_times.shape[0]):
        # Deform points using displacement
        deformed_points = points + displacements_all_times[t_index]

        # Extract surface mesh from the deformed mesh
        vertices, faces = get_surface_mesh(deformed_points, connectivity)

        # Compute signed distances for the pre-generated points
        signed_distances, _, _ = compute_signed_distances(filtered_points, vertices, faces)

        if SDF_ONLY:
            # Save only signed distances
            sdf_results.append(signed_distances)
        else:
            # Combine points and signed distances
            sdf_with_points = np.hstack((filtered_points, signed_distances[:, None]))
            sdf_results.append(sdf_with_points)

        print(f"Processed time step {t_index}/{displacements_all_times.shape[0]}.")

    with h5py.File(OUTPUT_FILE, "w") as f:
        for t_index, sdf_data in enumerate(sdf_results):
            f.create_dataset(f"time_{t_index}", data=sdf_data)
        print(f"Saved SDF results to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()

