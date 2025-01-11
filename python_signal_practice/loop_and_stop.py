import os
import signal
import time

# Global variables to track the current state
current_index = None
weights_previous = None
weights_i = None
stop_requested = False  # Tracks if a custom signal is received


def save_state(filename, i, weights):
    """Save the current state to a file."""
    if i is None or weights is None:
        print(f"\nNo state to save. Skipping write to {filename}.")
        return
    print(f"\nSaving state to {filename}...")
    with open(f"./result_saved/{filename}", "w") as file:
        file.write(f"Epoch: {i}\nWeights: {weights}\n")
    print(f"State saved successfully to {filename}!")


def handle_term_signal(signum, frame):
    """Handle the termination signal (SIGTERM)."""
    global current_index, weights_previous
    print(f"\nTermination signal ({signum}) received.")
    if current_index is not None and weights_previous is not None:
        save_state("filename_terminated", current_index - 1, weights_previous)
    else:
        print("No previous state to save. Exiting without saving.")
    os._exit(1)  # Terminate the program


def handle_custom_signal(signum, frame):
    """Handle the custom signal (SIGUSR2)."""
    global stop_requested
    print(f"\nCustom signal ({signum}) received. Stopping gracefully after the current epoch.")
    stop_requested = True


def simulate_execution(epoch):
    """Simulate some execution logic."""
    print(f"Simulating epoch {epoch} of machine learning.")
    total_weights = 0
    for i in range(10):
        time.sleep(0.3)  # Simulate work
        weight = 2 * i
        print(f"Simulating backpropagation of step {i}: Weight update = {weight}")
        total_weights += weight
    return total_weights


def main():
    """Main program logic."""
    global current_index, weights_previous, weights_i, stop_requested

    # Register signal handlers
    # signal.signal(signal.SIGUSR1, handle_kill_signal)  # Kill signal
    signal.signal(signal.SIGTERM, handle_term_signal)  # Termination signal
    signal.signal(signal.SIGUSR2, handle_custom_signal)  # Custom stop signal

    print("Starting main loop...")

    for current_index in range(1, 101):
        # Save the previous weights and compute the new weights
        weights_previous = weights_i
        weights_i = simulate_execution(current_index)

        # Handle custom signal to stop gracefully
        if stop_requested:
            save_state("filename_custom", current_index, weights_i)
            break

    if not stop_requested:
        print("\nExecution completed without interruption.")
    else:
        print("\nStopped gracefully by custom signal.")


if __name__ == "__main__":
    main()
