import subprocess
import numpy as np
import os

STARTING_INDEX = 15

# File containing filtered points
FILTERED_POINTS_FILE = "filtered_points_of_force_on_boundary.txt"

# Path to the simulation script
SIMULATION_SCRIPT = "./4_linear_elasticity_finger_pressure_bunny_dynamic.py"

# Output directory for simulation results
OUTPUT_DIR = "./deformed_bunny_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if the filtered points file exists
if not os.path.exists(FILTERED_POINTS_FILE):
    raise FileNotFoundError(f"Filtered points file not found: {FILTERED_POINTS_FILE}")

# Load the filtered points
filtered_points = np.loadtxt(FILTERED_POINTS_FILE, skiprows=1)


# Iterate over each row and run the simulation
for index, finger_position in enumerate(filtered_points[STARTING_INDEX:], start=STARTING_INDEX):
    # Convert finger position to string for passing as arguments
    finger_position_str = ' '.join(map(str, finger_position))
    
    print(f"Running simulation for index: {index}, finger_position: {finger_position_str}")
    
    # Run the simulation script with subprocess
    command = [
        "python", SIMULATION_SCRIPT,
        "--index", str(index),
        "--finger_position", *map(str, finger_position)
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Simulation {index} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Simulation {index} failed with error: {e}\n")
        continue  # Skip to the next simulation

print("All simulations completed.")

