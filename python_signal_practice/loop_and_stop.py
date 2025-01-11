import os
import signal
import time
import copy
import sys
import argparse
import pickle

# Change to the script's directory
os.chdir(sys.path[0])

# Constants
RESULT_DIR = "./result_saved"
FILENAME = "model"

# Global variables to track the current state
epoch = 0
weights: float = 0.0
weights_previous: float = 0.0
stop_requested = False  # Tracks if a custom signal is received

# Ensure the results directory exists
os.makedirs(RESULT_DIR, exist_ok=True)


def save_state(filename, epoch, weights):
    """Save the current state to both .txt and .pickle files."""
    print(f"\nSaving state to {filename}...")
    if epoch == 0:
        print("No previous state to save.")
        return

    # Save as text file
    text_file = f"{filename}.txt"
    with open(text_file, "w") as tf:
        tf.write(f"Epoch: {epoch}\nWeights: {weights}\n")

    # Save as pickle file
    pickle_file = f"{filename}.pickle"
    with open(pickle_file, "wb") as pf:
        pickle.dump((epoch, weights), pf)

    print(f"State saved successfully to {text_file} and {pickle_file}!")


def load_state(filename):
    """Load the state from a pickle file."""
    pickle_file = f"{filename}.pickle"
    try:
        with open(pickle_file, "rb") as pf:
            epoch, weights = pickle.load(pf)
            print(f"Loaded state from {pickle_file}: Epoch: {epoch}, Weights: {weights}")
            return epoch, weights
    except FileNotFoundError:
        print(f"Error: {pickle_file} not found.")
        return None, None


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


def get_max_epoch(state_type=None):
    """Get the maximum epoch from the saved files."""
    try:
        files = os.listdir(RESULT_DIR)
        if state_type:
            files = [f for f in files if f.endswith(f"_{state_type}.pickle")]
        if not files:
            return None
        epochs = [int(f.split("_")[1]) for f in files if f.startswith(FILENAME)]
        return max(epochs)
    except Exception as e:
        print(f"Error while retrieving epochs: {e}")
        return None


def main(continue_run=False, state_type=None, epoch_override=None):
    """Main program logic."""
    global epoch, weights, weights_previous, stop_requested

    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_kill_signal)  # Kill signal
    signal.signal(signal.SIGTERM, handle_term_signal)  # Termination signal
    signal.signal(signal.SIGUSR2, handle_custom_signal)  # Custom stop signal
    signal.signal(signal.SIGINT, handle_term_signal)  # Keyboard interrupt treated as terminate

    if continue_run:
        print("Continuing from a saved state...")

        if epoch_override is not None:
            epoch = epoch_override
        else:
            epoch = get_max_epoch(state_type)

        if epoch is None:
            print(f"No saved state found to continue from.")
            return 1

        # Load weights from the saved state
        filename = f"{RESULT_DIR}/{FILENAME}_{epoch}_{state_type or 'stop'}"
        epoch, weights = load_state(filename)
        if epoch is None:
            print(f"Failed to load state from {filename}.")
            return 1

        print(f"Resuming from epoch {epoch} ({'all states' if state_type is None else state_type}).")
    else:
        print("Starting fresh training run...")

    while epoch < 100:
        print(f"Simulating epoch {epoch}...")
        weights_previous = copy.deepcopy(weights)
        weights = simulate_execution(epoch)
        epoch += 1

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
    parser = argparse.ArgumentParser(description="Machine Learning Simulation Script")
    parser.add_argument(
        "--continue_run",
        action="store_true",
        help="Continue training from a saved state.",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Resume training from a terminate state.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Resume training from a stop state.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specify the epoch to start from if continuing.",
    )
    args = parser.parse_args()

    # Validation
    if not args.continue_run and (args.stop or args.terminate or args.epoch is not None):
        print("Error: --continue_run is needed for --stop, --terminate and --epoch")
        sys.exit(1)
    if args.terminate and args.stop:
        print("Error: Cannot specify both --terminate and --stop.")
        sys.exit(1)

    state_type = "terminate" if args.terminate else "stop" if args.stop else None
    ret = main(continue_run=args.continue_run, state_type=state_type, epoch_override=args.epoch)
    sys.exit(ret)
