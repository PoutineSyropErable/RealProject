import os
import signal
import time
import copy
import sys

# Change to the script's directory
os.chdir(sys.path[0])

# Constants
RESULT_DIR = "./result_saved"
FILENAME = "model"

# Global variables to track the current state
epoch = 0
weights = 0
weights_previous = 0
stop_requested = False  # Tracks if a custom signal is received

# Ensure the results directory exists
os.makedirs(RESULT_DIR, exist_ok=True)


def save_state(filename, epoch, weights):
    """Save the current state to a file."""
    print(f"\nSaving state to {filename}...")
    if epoch == 0:
        print("No previous state to save.")
        return
    with open(filename, "w") as file:
        file.write(f"Epoch: {epoch}\nWeights: {weights}\n")
    print(f"State saved successfully to {filename}!")


def handle_kill_signal(signum, frame):
    """Handle the kill signal (SIGUSR1)."""
    print(f"\nKill signal ({signum}) received. Terminating immediately without saving.")
    os._exit(3)  # Immediately terminate the program without saving state


def handle_term_signal(signum, frame):
    """Handle the termination signal (SIGTERM) or keyboard interrupt."""
    global epoch, weights_previous
    print(f"\nTermination signal ({signum}) received. Saving previous state and terminating.")
    if epoch > 0:
        filename = f"{RESULT_DIR}/{FILENAME}_{epoch}_terminate"
        save_state(filename, epoch, weights_previous)
    else:
        print("No previous state to save.")
    os._exit(2)  # Immediately terminate the program


def handle_custom_signal(signum, frame):
    """Handle the custom signal (SIGUSR2)."""
    global stop_requested
    print(f"\nCustom signal ({signum}) received. Stopping after current iteration.")
    stop_requested = True


def simulate_execution(epoch):
    """Simulate some execution logic."""
    global weights
    print(f"Simulating {epoch}th epoch of machine learning.")
    total = 0
    for i in range(10):
        time.sleep(0.5)  # Simulate work
        weights += 0.01  # Update weights incrementally
        result = epoch * i + weights / 100  # Dependence on epoch and updated weights
        print(f"Simulating backpropagation of the {i}th neural network with weight {result}")
        total += result
    return total


def main():
    """Main program logic."""
    global epoch, weights, weights_previous, stop_requested

    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_kill_signal)  # Kill signal
    signal.signal(signal.SIGTERM, handle_term_signal)  # Termination signal
    signal.signal(signal.SIGUSR2, handle_custom_signal)  # Custom stop signal
    signal.signal(signal.SIGINT, handle_term_signal)  # Keyboard interrupt treated as terminate

    print("Starting main loop...")

    while epoch < 100:
        # Simulate code execution
        print(f"Simulating epoch {epoch}...")
        weights_previous = copy.deepcopy(weights)
        weights = simulate_execution(epoch)
        epoch += 1

        # Handle custom signal to stop gracefully
        if stop_requested:
            filename = f"{RESULT_DIR}/{FILENAME}_{epoch}_stop"
            save_state(filename, epoch, weights)
            return 1

    print("Loop finished.")
    if not stop_requested:
        print("Execution completed without interruption.")
        filename = f"{RESULT_DIR}/{FILENAME}_{epoch}_complete"
        save_state(filename, epoch, weights)
        return 0


if __name__ == "__main__":
    ret = main()
    exit(ret)
