import argparse
import os
import signal
import subprocess

# Constant: The filename of the process to control
FILENAME = "loop_and_stop.py"


def get_process_details(filename):
    """
    Retrieve details of processes running the specified file.

    Args:
        filename (str): The filename to search for.

    Returns:
        list: A list of tuples containing PID and process details.
    """
    try:
        # Use ps to get details of processes matching the filename
        result = subprocess.run(
            ["ps", "axo", "pid,cmd", "--no-headers"],
            check=True,
            text=True,
            capture_output=True,
        )
        lines = result.stdout.strip().split("\n")
        processes = [(int(line.split(maxsplit=1)[0]), line) for line in lines if filename in line]
        return processes
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


def select_process_with_fzf(processes):
    """
    Use fzf to allow the user to select a process from multiple options.

    Args:
        processes (list): A list of tuples containing PID and process details.

    Returns:
        int: The selected PID, or None if no selection was made.
    """
    try:
        process_list = "\n".join([details for _, details in processes])
        result = subprocess.run(["fzf"], input=process_list, text=True, capture_output=True, check=True)
        selected_line = result.stdout.strip()
        selected_pid = int(selected_line.split(maxsplit=1)[0])
        return selected_pid
    except subprocess.CalledProcessError:
        print("No process selected or fzf was canceled.")
        return None


def main(stop, terminate, kill, all):
    # Get the processes matching the filename
    processes = get_process_details(FILENAME)
    if not processes:
        print(f"No processes found running {FILENAME}.")
        return 1

    # Extract PIDs from the process details
    pids = [pid for pid, _ in processes]

    # Handle multiple processes
    if len(processes) > 1:
        if all:
            print(f"Multiple processes found:\n" + "\n".join(details for _, details in processes))
            print("Sending signal to all.")
        else:
            print(f"Multiple processes found:\n" + "\n".join(details for _, details in processes))
            print("Use --all to send the signal to all, or select one using fzf.")
            selected_pid = select_process_with_fzf(processes)
            if not selected_pid:
                return 2  # Exit if no PID is selected
            pids = [selected_pid]  # Only send the signal to the selected PID

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
        help="If multiple processes are found, send the signal to all of them. If not provided, fzf will be used to select one.",
    )
    args = parser.parse_args()

    # Ensure only one signal option is provided
    if sum([args.stop, args.terminate, args.kill]) != 1:
        print("Error: You must specify exactly one of --stop, --terminate, or --kill.")
        exit(1)

    ret = main(args.stop, args.terminate, args.kill, args.all)
    exit(ret)
