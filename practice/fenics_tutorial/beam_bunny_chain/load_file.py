import h5py
import numpy as np

def load_displacement_file(h5_filename):
    """
    Load displacement data from the HDF5 file.

    Parameters:
    - h5_filename: The name of the HDF5 file.

    Returns:
    - displacements: A numpy array representing the displacements.
    """
    try:
        # Open the HDF5 file in read mode
        with h5py.File(h5_filename, "r") as h5_file:
            # Ensure the dataset exists
            if "displacements" not in h5_file:
                raise ValueError("The dataset 'displacements' does not exist in the file.")

            # Load the displacements dataset
            displacements = h5_file["displacements"][:]
            print(f"Loaded displacement data with shape: {displacements.shape}")
            return displacements

    except Exception as e:
        print(f"Error loading displacement file: {e}")
        return None


def main():
    # File name
    h5_filename = "displacements.h5"

    # Load displacement data
    displacements = load_displacement_file(h5_filename)
    if displacements is None:
        print("Failed to load displacement data.")
        return

    # Display metadata and example data
    print("\nDataset dimensions: ", displacements.shape)  # (num_steps, num_points, 3)

    # Show the first 5 time steps and the first 5 points
    num_t, num_points, _ = displacements.shape
    print("\nFirst 5 time steps and first 5 points:\n")
    for t in range(min(5, num_t)):
        print(f"Time step {t}:")
        print(displacements[t, :5, :])  # First 5 points for this time step
        print()

    # Example: Access displacement for a specific time and point
    t_index = 0  # Example time step
    point_index = 0  # Example point
    print(f"Displacement at time step {t_index}, point {point_index}:")
    print(displacements[t_index, point_index, :])  # x, y, z displacement


if __name__ == "__main__":
    main()

