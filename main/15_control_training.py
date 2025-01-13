import argparse
import os
import signal
import psutil

# Constants
SCRIPT_NAME = "14_train_nn_family.py"

# Map argument flags to their respective signals
SIGNAL_MAP = {
    "terminate": signal.SIGTERM,
    "stop_next_epoch": signal.SIGUSR2,
    "stop_next_time": signal.SIGUSR1,
    "save_last_epoch": signal.SIGRTMIN,
    "save_last_time": signal.SIGRTMIN + 1,
}


def send_signal(pid, sig):
    """
    Send a signal to the process with the given PID.

    Args:
        pid (int): Process ID of the target process.
        sig (signal.Signals): Signal to send.
    """
    try:
        os.kill(pid, sig)
        print(f"Signal {sig} sent to process {pid}.")
    except ProcessLookupError:
        print(f"No process with PID {pid} found.")
    except PermissionError:
        print(f"Permission denied to send signal {sig} to process {pid}.")


def find_processes_by_script(script_name):
    """
    Find all processes running a specific Python script.

    Args:
        script_name (str): Name of the target Python script.
    Returns:
        list: List of PIDs of processes running the script.
    """
    pids = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline:  # Ensure cmdline is not None or empty
                # Match script name against the last part of the command line (file name)
                if os.path.basename(script_name) in [os.path.basename(arg) for arg in cmdline]:
                    pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def main(signal_name):
    """
    Main function to find processes by script and send the specified signal.

    Args:
        signal_name (str): Name of the signal to send.
    """
    sig = SIGNAL_MAP[signal_name]
    pids = find_processes_by_script(SCRIPT_NAME)

    if not pids:
        print(f"No processes found for script '{SCRIPT_NAME}'.")
        return

    for pid in pids:
        send_signal(pid, sig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Send signals to {SCRIPT_NAME}.")

    # Add mutually exclusive group for signals
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--terminate", action="store_true", help="Send SIGTERM to terminate the process.")
    group.add_argument("--stop_next_epoch", action="store_true", help="Send SIGUSR2 to stop after the next epoch.")
    group.add_argument("--stop_next_time", action="store_true", help="Send SIGUSR1 to stop after the next time iteration.")
    group.add_argument("--save_last_epoch", action="store_true", help="Send SIGRTMIN to save weights for the last epoch.")
    group.add_argument("--save_last_time", action="store_true", help="Send SIGRTMIN+1 to save weights for the last time iteration.")

    args = parser.parse_args()

    # Determine which signal to send
    if args.terminate:
        signal_name = "terminate"
    elif args.stop_next_epoch:
        signal_name = "stop_next_epoch"
    elif args.stop_next_time:
        signal_name = "stop_next_time"
    elif args.save_last_epoch:
        signal_name = "save_last_epoch"
    elif args.save_last_time:
        signal_name = "save_last_time"
    else:
        parser.error("No valid signal option selected.")

    # Call main with the resolved signal name
    main(signal_name)
