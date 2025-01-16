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
from typing import Optional

from enum import Enum

# Directory containing the pickle files
LOAD_DIR = "./training_data"

# Directory where we save and load the neural weights
NEURAL_WEIGHTS_DIR = "./neural_weights"
LATENT_DIM = 64


# Global values for signals
stop_time_signal = False
stop_epoch_signal = False


class SaveMode(Enum):
    NowTime = 1
    NowEpoch = 2
    NextTimeItteration = 3
    NextEpochItteration = 4
    End = 5


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
    global training_context
    training_context.save_model_weights(SaveMode.NowEpoch)


def handle_save_time_signal(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    global training_context
    training_context.save_model_weights(SaveMode.NowTime)


def handle_termination_time(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    global training_context
    training_context.save_model_weights(SaveMode.NowTime)
    exit(1)


def handle_termination_epoch(signum, frame):
    """
    Handle termination signals (SIGTERM, SIGINT).
    Save weights at the current epoch and time index before exiting.
    """
    global training_context
    training_context.save_model_weights(SaveMode.NowEpoch)
    exit(2)


def get_latest_saved_indices():
    """
    Fetch the latest saved epoch and time index from the saved weights directory.
    Returns:
        (int, int): Latest epoch index and time index.
    """
    weights_files = [f for f in os.listdir(NEURAL_WEIGHTS_DIR) if f.endswith(".pth")]
    if not weights_files:
        return 0, 0  # No weights saved yet
    epochs_times = [
        tuple(map(int, f.split("_")[2:5:2])) for f in weights_files
    ]  # Extract epoch and time indices
    return max(epochs_times)  # Return the latest epoch and time index


def get_latest_saved_time_for_epoch(epoch):
    """
    Fetch the latest saved time index for a specific epoch.
    Args:
        epoch (int): The epoch for which to find the latest time index.
    Returns:
        int: Latest time index for the given epoch.
    """
    weights_files = [
        f
        for f in os.listdir(NEURAL_WEIGHTS_DIR)
        if f.endswith(".pth") and f"epoch_{epoch}_" in f
    ]
    if not weights_files:
        raise ValueError(f"No saved weights found for epoch {epoch}.")
    time_indices = [int(f.split("_")[4]) for f in weights_files]  # Extract time index
    return max(time_indices)


def read_pickle(directory, filename):
    long_file_name = f"{directory}/{filename}.pkl"

    with open(long_file_name, "rb") as file:
        output = pickle.load(file)
        print(f"Loaded {type(output)} from {long_file_name}")

    return output


def save_pickle(path: str, object1):
    with open(path, "wb") as f:
        pickle.dump(object1, f)


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
        latent_repeated = latent_vector.unsqueeze(1).repeat(
            1, num_points, 1
        )  # Repeat latent vector for each point
        inputs = torch.cat([latent_repeated, coordinates], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        sdf_values = self.fc4(x)
        return sdf_values


def get_path(name: str, epoch_index: int, time_index: int, extension="pth"):
    return os.path.join(
        NEURAL_WEIGHTS_DIR, f"{name}_epoch_{epoch_index}_time_{time_index}.{extension}"
    )


def load_dict_from_path(object1, path):
    if os.path.exists(path):
        object1.load_state_dict(torch.load(path))
        print(f"Loaded encoder weights from {path}.")
    else:
        raise FileNotFoundError(f"Weights/State file not found: {path} Doesn't exist")


class TrainingContext:
    def __init__(self, encoder: MeshEncoder, sdf_calculator: SDFCalculator):
        self.previous_time_index: Optional[int] = None
        self.previous_epoch_index: Optional[int] = None

        self.previous_encoder_weights_epoch = None
        self.previous_calculator_weights_epoch = None
        self.previous_encoder_weights_time = None
        self.previous_calculator_weights_time = None
        self.previous_time_index = None
        self.previous_epoch_index = None

        self.previous_optimizer_state_epoch = None
        self.previous_scheduler_state_epoch = None
        self.previous_optimizer_state_time = None
        self.previous_scheduler_state_time = None

        self.mesh_encoder = encoder
        self.sdf_calculator = sdf_calculator

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None

        self.loss_trackem: list[list[float]] = []

    def load_model_weights(self, epoch_index, time_index):
        encoder_weights_path = get_path("encoder", epoch_index, time_index)
        calculator_weights_path = get_path("sdf_calculator", epoch_index, time_index)
        optimizer_state_path = get_path("optimizer", epoch_index, time_index)
        scheduler_state_path = get_path("scheduler", epoch_index, time_index)
        loss_tracker_path = get_path("loss_tracker", epoch_index, time_index, ".pkl")

        load_dict_from_path(self.mesh_encoder, encoder_weights_path)
        load_dict_from_path(self.sdf_calculator, calculator_weights_path)
        load_dict_from_path(self.optimizer, optimizer_state_path)
        load_dict_from_path(self.scheduler, scheduler_state_path)
        with open(loss_tracker_path, "rb") as file:
            self.loss_tracker = pickle.load(file)

    def save_model_weights(self, mode: SaveMode):

        if self.previous_time_index == None or self.previous_epoch_index == None:
            print("Nothing to save, nothing was done yet")
            exit(3)

        if mode == SaveMode.NowTime:
            epoch_index = self.previous_epoch_index
            time_index = self.previous_time_index
            encoder_weights = self.previous_encoder_weights_time
            sdf_calculator_weights = self.previous_calculator_weights_time
            optimizer_state = self.previous_optimizer_state_time
            scheduler_state = self.previous_scheduler_state_time
            loss_tracker = self.loss_tracker

        if mode == SaveMode.NowEpoch:
            epoch_index = self.previous_epoch_index
            time_index = 0
            encoder_weights = self.previous_encoder_weights_epoch
            sdf_calculator_weights = self.previous_calculator_weights_epoch
            optimizer_state = self.previous_optimizer_state_epoch
            scheduler_state = self.previous_scheduler_state_epoch
            loss_tracker = self.loss_tracker[: epoch_index + 1]

        elif mode == SaveMode.NextTimeItteration:
            epoch_index = self.previous_epoch_index
            time_index = self.previous_time_index + 1
            encoder_weights = self.previous_encoder_weights_time
            sdf_calculator_weights = self.previous_calculator_weights_time
            optimizer_state = self.previous_optimizer_state_time
            scheduler_state = self.previous_scheduler_state_time
            loss_tracker = self.loss_tracker

        elif mode == SaveMode.NextEpochItteration or SaveMode.End:
            epoch_index = self.previous_epoch_index + 1
            time_index = 0
            encoder_weights = self.previous_encoder_weights_epoch
            sdf_calculator_weights = self.previous_calculator_weights_epoch
            optimizer_state = self.previous_optimizer_state_epoch
            scheduler_state = self.previous_scheduler_state_epoch
            loss_tracker = self.loss_tracker

        encoder_weights_path = get_path("encoder", epoch_index, time_index)
        calculator_weights_path = get_path("sdf_calculator", epoch_index, time_index)
        optimizer_state_path = get_path("optimizer", epoch_index, time_index)
        scheduler_state_path = get_path("scheduler", epoch_index, time_index)
        loss_tracker_path = get_path("loss_tracker", epoch_index, time_index, ".pkl")

        torch.save(encoder_weights, encoder_weights_path)
        torch.save(sdf_calculator_weights, calculator_weights_path)
        torch.save(optimizer_state, optimizer_state_path)
        torch.save(scheduler_state, scheduler_state_path)
        save_pickle(loss_tracker_path, loss_tracker)

    def time_update(self, time_index):
        self.previous_encoder_weights_time = copy.deepcopy(
            self.mesh_encoder.state_dict()
        )

        self.previous_calculator_weights_time = copy.deepcopy(
            self.sdf_calculator.state_dict()
        )
        self.previous_optimizer_state_time = copy.deepcopy(self.optimizer.state_dict())
        self.previous_scheduler_state_time = copy.deepcopy(self.scheduler.state_dict())
        self.previous_time_index = time_index

    def epoch_update(self, epoch_index):
        self.previous_encoder_weights_epoch = copy.deepcopy(
            self.mesh_encoder.state_dict()
        )
        self.previous_calculator_weights_epoch = copy.deepcopy(
            self.sdf_calculator.state_dict()
        )
        self.previous_optimizer_state_epoch = copy.deepcopy(self.optimizer.state_dict())
        self.previous_scheduler_state_epoch = copy.deepcopy(self.scheduler.state_dict())
        self.previous_epoch_index = epoch_index


def train_model(
    training_context: TrainingContext,
    vertices_tensor,
    sdf_points,
    sdf_values,
    latent_dim=64,
    epochs=1000,
    learning_rate=5e-4,
    start_epoch=0,
    start_time=0,
):
    """
    Train the mesh encoder and SDF calculator sequentially over time steps.

    Args:
        training_conext (TrainingContext): The context of the training. It has the neural networks, optimizer and scheduler and previous data
        vertices_tensor (torch.Tensor): Vertices of the shapes (num_time_steps, num_vertices, vertex_dim).
        sdf_points (torch.Tensor): Points for SDF computation (num_time_steps, num_points, 3).
        sdf_values (torch.Tensor): Ground truth SDF values (num_time_steps, num_points).
        latent_dim (int): Dimensionality of the latent vector.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        start_epoch (int): Epoch to start training from.
        start_time (int): Time index to start training from.
    """
    global stop_time_signal, stop_epoch_signal

    # Convert inputs to PyTorch tensors
    vertices_tensor = torch.tensor(vertices_tensor, dtype=torch.float32)
    # (time_steps, num_vertices, 3)
    sdf_points = torch.tensor(sdf_points, dtype=torch.float32)
    # (time_steps, num_points, 3)
    sdf_values = torch.tensor(sdf_values, dtype=torch.float32).unsqueeze(-1)
    # (time_steps, num_points, 1)

    # Optimizer and loss function
    if training_context.optimizer == None or training_context.scheduler == None:
        training_context.optimizer = torch.optim.Adam(
            list(training_context.mesh_encoder.parameters())
            + list(training_context.sdf_calculator.parameters()),
            lr=learning_rate,
        )
        training_context.scheduler = ReduceLROnPlateau(
            training_context.optimizer, mode="min", factor=0.7, patience=4, verbose=True
        )

    criterion = nn.MSELoss()

    print("\n-------Start of Training----------\n")
    # Training loop
    for epoch in range(start_epoch, epochs):

        if len(training_context.loss_tracker) < epoch + 1:
            training_context.loss_tracker.append([])
        print(f"start of epoch {epoch}")
        total_loss = 0
        for t_index in range(
            start_time if epoch == start_epoch else 0, vertices_tensor.shape[0]
        ):
            training_context.optimizer.zero_grad()

            # ------------ Get data for the current time step
            # Flatten vertices (1, num_vertices * 3)
            vertices = vertices_tensor[t_index].view(1, -1)
            # Add batch dimension (1, num_points, 3)
            points = sdf_points[t_index].unsqueeze(0)
            ground_truth_sdf = sdf_values[t_index].unsqueeze(0)  # (1, num_points, 1)

            # Encode vertices to latent vector
            latent_vector = training_context.mesh_encoder(vertices)  # (1, latent_dim)

            # Predict SDF values
            predicted_sdf = training_context.sdf_calculator(latent_vector, points)
            # (1, num_points, 1)

            # Compute loss
            loss = criterion(predicted_sdf, ground_truth_sdf)
            loss.backward()
            training_context.optimizer.step()

            total_loss += loss.item()
            # Custom logging for the learning rate
            current_lr = training_context.scheduler.get_last_lr()
            print(
                f"\t\tTime Iteration {t_index}, Loss: {loss.item()}, Learning Rate: {current_lr}"
            )

            try:
                training_context.loss_tracker[epoch].append(loss.item())
            except:
                pass

            # Store weights for the previous time step
            training_context.time_update(t_index)

            # Handle stop time signal
            if stop_time_signal:
                print(
                    f"Stopping after time iteration {t_index + 1}/{vertices_tensor.shape[0]}."
                )
                training_context.save_model_weights(SaveMode.NextTimeItteration)
                return 4  # return with code 4

        training_context.epoch_update(epoch)

        training_context.loss_tracker.append([])  # Add losses for this epoch
        # Step the scheduler
        training_context.scheduler.step(total_loss)
        print(f" End of Epoch {epoch}/{epochs -1}, Loss: {total_loss}")
        # Handle stop epoch signal
        if stop_epoch_signal:
            print(f"Stopping after epoch {epoch + 1}.")
            training_context.save_model_weights(SaveMode.NextEpochItteration)
            return 5  # Exit with code 5

    print("Training complete.")

    training_context.save_model_weights(SaveMode.End)

    return 0


def main(
    start_from_zero=True, continue_training=False, epoch_index=None, time_index=None
):
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_termination_epoch)  # Kill (no -9)
    signal.signal(signal.SIGINT, handle_termination_time)  # KeyboardInterrupt
    signal.signal(signal.SIGTSTP, handle_termination_time)  # Ctrl+Z
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)
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

    global training_context
    training_context = TrainingContext(mesh_encoder, sdf_calculator)

    # Load weights if continuing training
    if continue_training:
        training_context.load_model_weights(epoch_index, time_index)

    # Train model
    ret = train_model(
        training_context,
        vertices_tensor,
        sdf_points,
        sdf_values,
        latent_dim=LATENT_DIM,
        epochs=1000,
        learning_rate=1e-3,
        start_epoch=epoch_index or 0,
        start_time=time_index or 0,
    )

    return ret


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process preprocessed data with options to start or continue."
    )

    # Mutually exclusive group for start or continue training
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--start_from_zero",
        action="store_true",
        help="Start processing from the beginning.",
    )
    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue processing from the last state.",
    )

    # Arguments for epoch and time indices
    parser.add_argument(
        "--epoch_index",
        type=int,
        help="Specify the epoch index to continue processing from.",
    )
    parser.add_argument(
        "--time_index",
        type=int,
        help="Specify the time index to continue processing from.",
    )

    args = parser.parse_args()

    # Validation for argument combinations
    if args.start_from_zero and (
        args.epoch_index is not None or args.time_index is not None
    ):
        parser.error(
            "--start_from_zero cannot be used with --epoch_index or --time_index."
        )

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
