import subprocess
import argparse
import time
import os
import h5py
import numpy as np

# File containing filtered points
FILTERED_POINTS_FILE = "filtered_points_of_force_on_boundary.txt"

def load_displacement_data(h5_file):
    """Load displacement data from an HDF5 file with a lock check."""
    while True:
        try:
            with h5py.File(h5_file, "r") as f:
                displacements = f["displacements"][:]
                print(f"Loaded displacement data with shape: {displacements.shape}")
            return displacements
        except OSError as e:
            print(f"File {h5_file} is locked. Retrying in 10 seconds...")
            time.sleep(10)

def validate_indices(starting_index, stopping_index, filtered_points):
    """Ensure starting and stopping indices are valid."""
    num_points = len(filtered_points)
    if starting_index < 0 or starting_index >= num_points:
        raise ValueError(f"Starting index {starting_index} is out of range (0, {num_points - 1}).")
    if stopping_index != -1 and (stopping_index < 0 or stopping_index >= num_points):
        raise ValueError(f"Stopping index {stopping_index} is out of range (0, {num_points - 1}).")
    if stopping_index != -1 and stopping_index < starting_index:
        raise ValueError(f"Stopping index {stopping_index} cannot be less than starting index {starting_index}.")

def main():
    parser = argparse.ArgumentParser(description="Iterate SDF calculation over a range of indices.")
    parser.add_argument("--starting_index", type=int, required=True, help="Starting index (inclusive).")
    parser.add_argument("--stopping_index", type=int, required=True, help="Stopping index (inclusive). Use -1 to process till the end.")
    parser.add_argument("--sdf_only", action="store_true", help="Only save SDF values.")
    args = parser.parse_args()

    starting_index = args.starting_index
    stopping_index = args.stopping_index
    sdf_only = args.sdf_only

    # Check if the filtered points file exists
    if not os.path.exists(FILTERED_POINTS_FILE):
        raise FileNotFoundError(f"Filtered points file not found: {FILTERED_POINTS_FILE}")

    # Load the filtered points
    filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)
    print(f"Loaded {len(filtered_points)} filtered points.")

    # Validate indices
    validate_indices(starting_index, stopping_index, filtered_points)

    if stopping_index == -1:
        stopping_index = len(filtered_points) - 1

    print(f"Processing indices from {starting_index} to {stopping_index} (inclusive).")
    print(f"SDF only mode: {'Enabled' if sdf_only else 'Disabled'}")

    # Iterate through the specified index range
    current_index = starting_index
    while current_index <= stopping_index:
        print(f"Processing index: {current_index}")

        # Build the command for running `10_calculate_sdfs.py`
        command = [
            "python", "10_calculate_sdfs.py",
            "--index", str(current_index)
        ]
        if sdf_only:
            command.append("--sdf_only")

        try:
            # Run the command
            subprocess.run(command, check=True)
            print(f"Index {current_index} processed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error processing index {current_index}: {e}\n")

        # Increment index
        current_index += 1

    print("SDF calculation iteration completed.")

if __name__ == "__main__":
    main()

