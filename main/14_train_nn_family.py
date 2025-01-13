import pickle
import argparse
import os
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def handle_termination(signum, frame):
    """
    Handle termination signals (e.g., SIGTERM, SIGINT).
    """
    print(f"Termination signal received: {signum}. Cleaning up before exit.")
    # Add your termination handling logic here
    save_state("terminate")
    exit(0)


def save_state(reason):
    """
    Simulate saving the state.
    """
    print(f"Saving state due to {reason}.")
    # Add actual saving logic here
    time.sleep(1)  # Simulate save time
    print("State saved successfully!")


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
    return 5, None  # Example return values: index = 5, some additional data


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
        latent_vector = torch.mean(self.fc3(x), dim=1)  # Aggregate over vertices
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


def train_model(vertices_tensor, sdf_points, sdf_values, latent_dim=256, epochs=100, learning_rate=1e-3):
    """
    Train the mesh encoder and SDF calculator sequentially over time steps.

    Args:
        vertices_tensor (np.ndarray): Vertices of the shapes (num_time_steps, num_vertices, vertex_dim).
        sdf_points (np.ndarray): Points for SDF computation (num_time_steps, num_points, 3).
        sdf_values (np.ndarray): Ground truth SDF values (num_time_steps, num_points).
        latent_dim (int): Dimensionality of the latent vector.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    global stop_time_signal, stop_epoch_signal

    global previous_encoder_weights_epoch, previous_calculator_weights_epoch
    global previous_encoder_weights_time, previous_calculator_weights_time
    global previous_time_index, previous_epoch_index

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor, dtype=torch.float32)  # (time_steps, num_vertices, 3)
    sdf_points = torch.tensor(sdf_points, dtype=torch.float32)  # (time_steps, num_points, 3)
    sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)  # (time_steps, num_points, 1)

    # Initialize models
    input_dim = vertices_tensor.shape[1] * vertices_tensor.shape[2]  # num_vertices * 3
    mesh_encoder = MeshEncoder(input_dim=input_dim, latent_dim=latent_dim)
    sdf_calculator = SDFCalculator(latent_dim=latent_dim)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        list(mesh_encoder.parameters()) + list(sdf_calculator.parameters()),
        lr=learning_rate,
    )
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for t_index in range(vertices_tensor.shape[0]):  # Iterate over time steps
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
            print(f"\t\tTime Itteration loss: {loss.item()}")

            # Store weights for the previous time step
            previous_encoder_weights_time = copy.deepcopy(mesh_encoder.state_dict())
            previous_calculator_weights_time = copy.deepcopy(sdf_calculator.state_dict())
            previous_time_index = t_index

            # Handle stop time signal
            if stop_time_signal:
                print(f"Stopping after time iteration {t_index + 1}/{vertices_tensor.shape[0]}.")
                exit(3)  # Exit with code 3

        # Store weights for the previous epoch
        previous_encoder_weights_epoch = copy.deepcopy(mesh_encoder.state_dict())
        previous_calculator_weights_epoch = copy.deepcopy(sdf_calculator.state_dict())
        previous_epoch_index = epoch

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        # Handle stop epoch signal
        if stop_epoch_signal:
            print(f"Stopping after epoch {epoch + 1}.")
            exit(4)  # Exit with code 4

    print("Training complete.")
    return mesh_encoder, sdf_calculator


def main(start_from_zero=True, continue_training=False, index=None):
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_termination)  # Terminate
    signal.signal(signal.SIGINT, handle_termination)  # KeyboardInterrupt
    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)  # Example: SIGUSR1 for stopping time iteration
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)  # Example: SIGUSR2 for stopping epoch iteration

    vertices_tensor = read_pickle(LOAD_DIR, "vertices_tensor")
    sdf_points = read_pickle(LOAD_DIR, "sdf_points")
    sdf_values = read_pickle(LOAD_DIR, "sdf_values")

    print(f"sdf_points.shape: {sdf_points.shape}")
    print(f"sdf_values.shape: {sdf_values.shape}")
    print(f"vertices_tensor.shape: {vertices_tensor.shape}")

    train_model(vertices_tensor, sdf_points, sdf_values, latent_dim=256, epochs=100, learning_rate=1e-3)

    return 0


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process preprocessed data with options to start or continue.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start_from_zero", action="store_true", help="Start processing from the beginning.")
    group.add_argument("--continue_training", action="store_true", help="Continue processing from the last state.")

    parser.add_argument("--index", type=int, help="Specify the index to continue processing from.")

    args = parser.parse_args()

    # Ensure mutual exclusivity between --start_from_zero and --continue
    if args.start_from_zero and args.continue_training:
        parser.error("--start_from_zero and --continue_training cannot be used together.(--continue_training XOR --start_from_zero)")

    if not args.start_from_zero and not args.continue_training:
        parser.error("You must continue_training or start training from zero (--continue_training XOR --start_from_zero)")

    # Handle index determination for --continue-training
    if args.continue_training:
        if args.index is None:
            print("Restarting from the latest available training data...")
            args.index, _ = reload_data_placeholder()
    elif args.index is not None:
        parser.error("The --index argument can only be used with --continue-training.")

    ret = main(args.start_from_zero, args.continue_training, args.index)
    exit(ret)
