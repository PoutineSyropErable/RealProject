import pickle
import argparse
import os
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy

# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"

# Global variables to store weights and indices
previous_encoder_weights_epoch = None
previous_calculator_weights_epoch = None
previous_encoder_weights_time = None
previous_calculator_weights_time = None
previous_time_index = None
previous_epoch_index = None


# Global values for signals
stop_time_signal = False
stop_epoch_signal = False


# Signal handlers
def handle_stop_time_signal(signum, frame):
    global stop_time_signal
    stop_time_signal = True
    print("Received stop time signal. Will stop after the current time iteration.")


def handle_stop_epoch_signal(signum, frame):
    global stop_epoch_signal
    stop_epoch_signal = True
    print("Received stop epoch signal. Will stop after the current epoch iteration.")


def handle_save_epoch_signal(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """

    print("Received termination signal. Saving weights and continuing...")
    save_model_weights(previous_encoder_weights_epoch, previous_calculator_weights_epoch, previous_epoch_index, 0)


def handle_save_time_signal(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """

    print("Received termination signal. Saving weights and continuing...")
    save_model_weights(previous_encoder_weights_time, previous_calculator_weights_time, previous_epoch_index, previous_time_index)


def handle_termination(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    global previous_epoch_index
    if previous_time_index == None:
        print("Nothing to save, no acceptable state met yet")
        exit(-1)

    if previous_epoch_index == None:
        previous_epoch_index = 0
        save_model_weights(previous_encoder_weights_time, previous_calculator_weights_time, previous_epoch_index, previous_time_index)
    print("Received termination signal. Saving weights before exiting...")
    save_model_weights(previous_encoder_weights_time, previous_calculator_weights_time, previous_epoch_index + 1, previous_time_index)
    exit(1)


def get_latest_saved_indices():
    """
    Fetch the latest saved epoch and time index from the saved weights directory.
    Returns:
        (int, int): Latest epoch index and time index.
    """
    weights_files = [f for f in os.listdir(NEURAL_WEIGHTS_DIR) if f.endswith(".pth")]
    if not weights_files:
        return 0, 0  # No weights saved yet
    epochs_times = [tuple(map(int, f.split("_")[2:5:2])) for f in weights_files]  # Extract epoch and time indices
    return max(epochs_times)  # Return the latest epoch and time index


def get_latest_saved_time_for_epoch(epoch):
    """
    Fetch the latest saved time index for a specific epoch.
    Args:
        epoch (int): The epoch for which to find the latest time index.
    Returns:
        int: Latest time index for the given epoch.
    """
    weights_files = [f for f in os.listdir(NEURAL_WEIGHTS_DIR) if f.endswith(".pth") and f"epoch_{epoch}_" in f]
    if not weights_files:
        raise ValueError(f"No saved weights found for epoch {epoch}.")
    time_indices = [int(f.split("_")[4]) for f in weights_files]  # Extract time index
    return max(time_indices)


def load_preprocessed_data():
    # File paths
    files = {
        "sdf_points": os.path.join(LOAD_DIR, "sdf_points.pkl"),
        "sdf_values": os.path.join(LOAD_DIR, "sdf_values.pkl"),
        "vertices_tensor": os.path.join(LOAD_DIR, "vertices_tensor.pkl"),
        "faces": os.path.join(LOAD_DIR, "faces.pkl"),
    }

    # Load each object
    data = {}
    for name, path in files.items():
        with open(path, "rb") as file:
            data[name] = pickle.load(file)
            print(f"Loaded {name} from {path}")

    # Access loaded data
    sdf_points = data["sdf_points"]
    sdf_values = data["sdf_values"]
    vertices_tensor = data["vertices_tensor"]
    faces = data["faces"]

    print(f"\n\n")
    # Verify the loaded shapes
    print(f"sdf_points.shape: {sdf_points.shape}")
    print(f"sdf_values.shape: {sdf_values.shape}")
    print(f"vertices_tensor.shape: {vertices_tensor.shape}")
    print(f"faces.shape: {faces.shape}")

    return faces, vertices_tensor, sdf_points, sdf_values


def read_pickle(directory, filename):
    long_file_name = f"{directory}/{filename}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def reload_data_placeholder():
    """
    Placeholder function to get the index when --continue is used, but no index is provided.
    """
    return 5, 3  # Example return values: index = 5, some additional data


class MeshEncoder(nn.Module):
    """
    Neural network that encodes a 3D mesh into a latent vector.
    Args:
        input_dim (int): Dimensionality of the input vertices.
        latent_dim (int): Dimensionality of the latent vector.
    """

    def __init__(self, input_dim: int = 9001, latent_dim: int = 256):
        super(MeshEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, latent_dim)

    def forward(self, vertices):
        """
        Forward pass for encoding vertices into a latent vector.
        Args:
            vertices (torch.Tensor): Input tensor of shape (batch_size, num_vertices, input_dim).
        Returns:
            torch.Tensor: Latent vector of shape (batch_size, latent_dim).
        """
        x = F.relu(self.fc1(vertices))
        x = F.relu(self.fc2(x))
        latent_vector = self.fc3(x)  # Aggregate over vertices
        return latent_vector


class SDFCalculator(nn.Module):
    """
    Neural network that calculates SDF values from a latent vector and 3D coordinates.
    Args:
        latent_dim (int): Dimensionality of the latent vector.
        input_dim (int): Dimensionality of the 3D coordinates (default 3 for x, y, z).
    """

    def __init__(self, latent_dim: int = 256, input_dim: int = 3):
        super(SDFCalculator, self).__init__()
        self.fc1 = nn.Linear(latent_dim + input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, latent_vector, coordinates):
        """
        Forward pass to calculate SDF values.
        Args:
            latent_vector (torch.Tensor): Latent vector of shape (batch_size, latent_dim).
            coordinates (torch.Tensor): Input tensor of shape (batch_size, num_points, input_dim).
        Returns:
            torch.Tensor: SDF values of shape (batch_size, num_points, 1).
        """
        batch_size, num_points, _ = coordinates.size()
        latent_repeated = latent_vector.unsqueeze(1).repeat(1, num_points, 1)  # Repeat latent vector for each point
        inputs = torch.cat([latent_repeated, coordinates], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        sdf_values = self.fc4(x)
        return sdf_values


def save_model_weights(encoder_weights, calculator_weights, epoch_index, time_index):
    """
    Save the weights of the encoder and calculator models.

    Args:
        encoder_weights (dict): State dictionary of the encoder model.
        calculator_weights (dict): State dictionary of the calculator model.
        epoch_index (int): Current epoch index.
        time_index (int): Current time index.
    """
    encoder_weights_path = os.path.join(NEURAL_WEIGHTS_DIR, f"encoder_epoch_{epoch_index}_time_{time_index}.pth")
    calculator_weights_path = os.path.join(NEURAL_WEIGHTS_DIR, f"calculator_epoch_{epoch_index}_time_{time_index}.pth")

    os.makedirs(NEURAL_WEIGHTS_DIR, exist_ok=True)  # Ensure the directory exists

    torch.save(encoder_weights, encoder_weights_path)
    torch.save(calculator_weights, calculator_weights_path)

    print(f"Saved encoder weights to {encoder_weights_path}.")
    print(f"Saved calculator weights to {calculator_weights_path}.")


def load_model_weights(encoder, calculator, epoch_index, time_index):
    """
    Placeholder function to load the weights for the encoder and calculator models.
    """
    # Replace with the actual mechanism to load weights, e.g., from files or a database.
    encoder_weights_path = os.path.join(NEURAL_WEIGHTS_DIR, f"encoder_epoch_{epoch_index}_time_{time_index}.pth")
    calculator_weights_path = os.path.join(NEURAL_WEIGHTS_DIR, f"calculator_epoch_{epoch_index}_time_{time_index}.pth")

    if os.path.exists(encoder_weights_path):
        encoder.load_state_dict(torch.load(encoder_weights_path))
        print(f"Loaded encoder weights from {encoder_weights_path}.")
    else:
        raise FileNotFoundError(f"Encoder weights not found at {encoder_weights_path}.")

    if os.path.exists(calculator_weights_path):
        calculator.load_state_dict(torch.load(calculator_weights_path))
        print(f"Loaded calculator weights from {calculator_weights_path}.")
    else:
        raise FileNotFoundError(f"Calculator weights not found at {calculator_weights_path}.")


def train_model(
    vertices_tensor,
    sdf_points,
    sdf_values,
    latent_dim=256,
    epochs=100,
    learning_rate=1e-3,
    mesh_encoder=None,
    sdf_calculator=None,
    start_epoch=0,
    start_time=0,
):
    """
    Train the mesh encoder and SDF calculator sequentially over time steps.

    Args:
        vertices_tensor (torch.Tensor): Vertices of the shapes (num_time_steps, num_vertices, vertex_dim).
        sdf_points (torch.Tensor): Points for SDF computation (num_time_steps, num_points, 3).
        sdf_values (torch.Tensor): Ground truth SDF values (num_time_steps, num_points).
        latent_dim (int): Dimensionality of the latent vector.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        mesh_encoder (nn.Module): Preinitialized mesh encoder.
        sdf_calculator (nn.Module): Preinitialized SDF calculator.
        start_epoch (int): Epoch to start training from.
        start_time (int): Time index to start training from.
    """
    global stop_time_signal, stop_epoch_signal
    global previous_encoder_weights_epoch, previous_calculator_weights_epoch
    global previous_encoder_weights_time, previous_calculator_weights_time
    global previous_time_index, previous_epoch_index

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor, dtype=torch.float32)  # (time_steps, num_vertices, 3)
    sdf_points = torch.tensor(sdf_points, dtype=torch.float32)  # (time_steps, num_points, 3)
    sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)  # (time_steps, num_points, 1)

    # Initialize models if not provided
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]  # num_vertices * 3
    if mesh_encoder is None:
        mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=latent_dim)
    if sdf_calculator is None:
        sdf_calculator = SDFCalculator(latent_dim=latent_dim)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        list(mesh_encoder.parameters()) + list(sdf_calculator.parameters()),
        lr=learning_rate,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=4, verbose=True)
    criterion = nn.MSELoss()
    # Initialize loss tracker
    loss_tracker = []

    print("\n-------Start of Training----------\n")
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"start of epoch {epoch}")
        total_loss = 0
        epoch_losses = []  # Store losses for this epoch
        for t_index in range(start_time if epoch == start_epoch else 0, vertices_tensor.shape[0]):
            optimizer.zero_grad()

            # Get data for the current time step
            vertices = vertices_tensor[t_index].view(1, -1)  # Flatten vertices (1, num_vertices * 3)
            points = sdf_points[t_index].unsqueeze(0)  # Add batch dimension (1, num_points, 3)
            ground_truth_sdf = sdf_values[t_index].unsqueeze(0)  # (1, num_points, 1)

            # Encode vertices to latent vector
            latent_vector = mesh_encoder(vertices)  # (1, latent_dim)

            # Predict SDF values
            predicted_sdf = sdf_calculator(latent_vector, points)  # (1, num_points, 1)

            # Compute loss
            loss = criterion(predicted_sdf, ground_truth_sdf)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Custom logging for the learning rate
            current_lr = scheduler._last_lr if hasattr(scheduler, "_last_lr") else scheduler.get_last_lr()
            print(f"\t\tTime Iteration {t_index}, Loss: {loss.item()}, Learning Rate: {current_lr}")

            epoch_losses.append(loss.item())  # Track loss for this time step

            # Store weights for the previous time step
            previous_encoder_weights_time = copy.deepcopy(mesh_encoder.state_dict())
            previous_calculator_weights_time = copy.deepcopy(sdf_calculator.state_dict())
            previous_time_index = t_index

            # Handle stop time signal
            if stop_time_signal:
                print(f"Stopping after time iteration {t_index + 1}/{vertices_tensor.shape[0]}.")
                save_model_weights(previous_encoder_weights_time, previous_calculator_weights_time, epoch, t_index + 1)
                exit(3)  # Exit with code 3

        # Store weights for the previous epoch
        previous_encoder_weights_epoch = copy.deepcopy(mesh_encoder.state_dict())
        previous_calculator_weights_epoch = copy.deepcopy(sdf_calculator.state_dict())
        previous_epoch_index = epoch

        loss_tracker.append(epoch_losses)  # Add losses for this epoch
        # Step the scheduler
        scheduler.step(total_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        # Handle stop epoch signal
        if stop_epoch_signal:
            print(f"Stopping after epoch {epoch + 1}.")
            save_model_weights(previous_encoder_weights_epoch, previous_calculator_weights_epoch, epoch + 1, 0)
            exit(4)  # Exit with code 4

    print("Training complete.")
    return mesh_encoder, sdf_calculator


def main(start_from_zero=True, continue_training=False, epoch_index=None, time_index=None):
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_termination)  # Kill (no -9)
    signal.signal(signal.SIGINT, handle_termination)  # KeyboardInterrupt
    signal.signal(signal.SIGTSTP, handle_termination)  # Ctrl+Z
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)  # Example: SIGUSR1 for stopping time iteration
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)  # Example: SIGUSR2 for stopping epoch iteration
    signal.signal(signal.SIGRTMIN, handle_save_epoch_signal)
    signal.signal(signal.SIGRTMIN + 1, handle_save_time_signal)

    # Ensure the weights directory exists
    os.makedirs(NEURAL_WEIGHTS_DIR, exist_ok=True)

    vertices_tensor = read_pickle(LOAD_DIR, "vertices_tensor")
    sdf_points = read_pickle(LOAD_DIR, "sdf_points")
    sdf_values = read_pickle(LOAD_DIR, "sdf_values")

    print(f"sdf_points.shape: {sdf_points.shape}")
    print(f"sdf_values.shape: {sdf_values.shape}")
    print(f"vertices_tensor.shape: {vertices_tensor.shape}")

    # Initialize models
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]
    print(f"mesh encoder input_dim = {input_dim}")
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=256)
    sdf_calculator = SDFCalculator(latent_dim=256)

    # Load weights if continuing training
    if continue_training:
        load_model_weights(mesh_encoder, sdf_calculator, epoch_index, time_index)

    # Train model
    train_model(
        vertices_tensor,
        sdf_points,
        sdf_values,
        latent_dim=256,
        epochs=100,
        learning_rate=1e-3,
        mesh_encoder=mesh_encoder,
        sdf_calculator=sdf_calculator,
        start_epoch=epoch_index or 0,
        start_time=time_index or 0,
    )

    return 0


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process preprocessed data with options to start or continue.")

    # Mutually exclusive group for start or continue training
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start_from_zero", action="store_true", help="Start processing from the beginning.")
    group.add_argument("--continue_training", action="store_true", help="Continue processing from the last state.")

    # Arguments for epoch and time indices
    parser.add_argument("--epoch_index", type=int, help="Specify the epoch index to continue processing from.")
    parser.add_argument("--time_index", type=int, help="Specify the time index to continue processing from.")

    args = parser.parse_args()

    # Validation for argument combinations
    if args.start_from_zero and (args.epoch_index is not None or args.time_index is not None):
        parser.error("--start_from_zero cannot be used with --epoch_index or --time_index.")

    if args.time_index is not None and args.epoch_index is None:
        parser.error("--time_index can only be used if --epoch_index is specified.")

    epoch_index, time_index = None, None

    if args.continue_training:
        if args.epoch_index is not None:
            epoch_index = args.epoch_index
            time_index = args.time_index or get_latest_saved_time_for_epoch(epoch_index)
        else:
            epoch_index, time_index = get_latest_saved_indices()
    elif args.start_from_zero:
        epoch_index, time_index = 0, 0

    # Call main and exit with the returned code
    ret = main(args.start_from_zero, args.continue_training, epoch_index, time_index)
    exit(ret)
