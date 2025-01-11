import argparse
import os
import signal
import subprocess

# Constant: The filename of the process to control
FILENAME = "loop_and_stop.py"


def get_pids_from_filename(filename):
    """Retrieve the PIDs of processes running the specified file."""
    try:
        result = subprocess.run(["pgrep", "-f", filename], check=True, text=True, capture_output=True)
        pids = list(map(int, result.stdout.strip().split()))
        if not pids:
            print(f"No processes found running {filename}.")
        return pids
    except subprocess.CalledProcessError:
        print(f"No processes found running {filename}.")
        return []


def send_signal(pid, signal_type):
    """Send the specified signal to the process with the given PID."""
    try:
        os.kill(pid, signal_type)
        print(f"Signal {signal_type} sent to process {pid}.")
    except ProcessLookupError:
        print(f"No process found with PID {pid}.")
    except PermissionError:
        print(f"Permission denied to send signal {signal_type} to process {pid}.")
    except Exception as e:
        print(f"An error occurred while sending the signal to PID {pid}: {e}")


def main(stop, terminate, kill, all):
    # Get the PIDs of the target process
    pids = get_pids_from_filename(FILENAME)
    if not pids:
        return 1

    # Handle multiple PIDs
    if len(pids) > 1:
        if all:
            print(f"Multiple PIDs found: {pids}. Sending signal to all.")
        else:
            print(f"Multiple PIDs found: {pids}. Use --all to send the signal to all.")
            return 2

    # Determine which signal to send
    signal_type = None
    if stop:
        signal_type = signal.SIGUSR2  # Custom stop signal
    elif terminate:
        signal_type = signal.SIGTERM  # Termination signal
    elif kill:
        signal_type = signal.SIGUSR1  # Kill signal

    # Send signal to the PIDs
    for pid in pids:
        send_signal(pid, signal_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the loop_and_stop.py process.")
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Send the custom stop signal (SIGUSR2) to gracefully stop the process.",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Send the termination signal (SIGTERM) to save state and terminate the process.",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Send the kill signal (SIGUSR1) to immediately terminate the process without saving.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If multiple PIDs are found, send the signal to all of them. If not provided, exit on multiple PIDs.",
    )
    args = parser.parse_args()

    # Ensure only one signal option is provided
    if sum([args.stop, args.terminate, args.kill]) != 1:
        print("Error: You must specify exactly one of --stop, --terminate, or --kill.")
        exit(1)

    ret = main(args.stop, args.terminate, args.kill, args.all)
    exit(ret)
