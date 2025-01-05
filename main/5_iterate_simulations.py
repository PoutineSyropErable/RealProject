import subprocess
import numpy as np
import os

STARTING_INDEX = 270
STOPPING_INDEX = -1  # Included; set to -1 to process until the end.

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

# Adjust stopping index if -1
if STOPPING_INDEX == -1:
    STOPPING_INDEX = len(filtered_points) - 1

# Validate indices
if STARTING_INDEX < 0 or STARTING_INDEX >= len(filtered_points):
    raise ValueError(f"STARTING_INDEX ({STARTING_INDEX}) is out of range.")
if STOPPING_INDEX < 0 or STOPPING_INDEX >= len(filtered_points):
    raise ValueError(f"STOPPING_INDEX ({STOPPING_INDEX}) is out of range.")
if STOPPING_INDEX < STARTING_INDEX:
    raise ValueError(f"STOPPING_INDEX ({STOPPING_INDEX}) cannot be less than STARTING_INDEX ({STARTING_INDEX}).")

print(f"Processing range: STARTING_INDEX={STARTING_INDEX}, STOPPING_INDEX={STOPPING_INDEX}")

# Iterate over each row and run the simulation
for index, finger_position in enumerate(filtered_points[STARTING_INDEX:STOPPING_INDEX + 1], start=STARTING_INDEX):
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
        result = subprocess.run(command, check=True, text=False, capture_output=False)
        print(f"Simulation {index} completed successfully.\nOutput:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Simulation {index} failed with error: {e}\nStderr:\n{e.stderr}")
        continue  # Skip to the next simulation

print("All simulations completed.")

