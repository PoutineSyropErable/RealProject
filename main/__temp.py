import h5py
import numpy as np

# Path to HDF5 file
H5_FILE = "./deformed_bunny_files_tunned/displacement_1.h5"

# Open the HDF5 file
with h5py.File(H5_FILE, "r") as f:
    # Access the 'Function' -> 'f' group
    function_group = f["Function"]
    f_group = function_group["f"]

    # Extract all datasets (time steps) and their displacement data
    displacements = {}
    for time_step in f_group.keys():
        dataset = f_group[time_step]
        displacements[time_step] = dataset[...]
        print(f"Time step: {time_step}, Data shape: {dataset.shape}")

# Example: Access the displacement data for the first time step
first_time_step = sorted(displacements.keys())[0]  # Get the earliest time step
first_displacement = displacements[first_time_step]
print(f"\nDisplacement at first time step ({first_time_step}):")
print(first_displacement)
