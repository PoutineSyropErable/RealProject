import subprocess


def get_matching_pids(pattern):
    """
    Grep processes based on a given pattern and extract PIDs.

    Args:
        pattern (str): The pattern to search for in the process list.

    Returns:
        list: A list of strings representing PIDs and their details.
    """
    try:
        # Get the list of processes matching the pattern
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            text=True,
            capture_output=True,
            check=True,
        )
        # Split lines and return the matching processes
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        # If no processes match, return an empty list
        return []


def choose_pid_with_fzf(processes):
    """
    Use fzf to allow the user to select a process.

    Args:
        processes (list): A list of process strings to present in fzf.

    Returns:
        str: The selected process string, or None if no selection was made.
    """
    try:
        # Send the list of processes to fzf for selection
        result = subprocess.run(
            ["fzf"],
            input="\n".join(processes),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # If fzf is canceled or no selection is made, return None
        return None


def main():
    # Define the pattern to search for
    pattern = "loop_and_stop.py"

    # Get the list of matching processes
    processes = get_matching_pids(pattern)

    if not processes:
        print(f"No processes found matching pattern: {pattern}")
        return

    print("Select a process using fzf:")
    selected_process = choose_pid_with_fzf(processes)

    if selected_process:
        # Extract the PID from the selected process
        pid = selected_process.split()[0]
        print(f"Selected PID: {pid}")
    else:
        print("No process selected.")


if __name__ == "__main__":
    main()
