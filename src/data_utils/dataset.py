"""PyTorch Dataset for thermal simulation data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.physics.materials import create_property_field

logger = logging.getLogger(__name__)


class ThermalDataset(Dataset):
    """PyTorch Dataset for loading thermal simulation trajectories.

    Each sample is a pair of consecutive temperature fields (input, target)
    along with material masks, SDFs, and physical parameters.

    Args:
        data_path: Path to HDF5 dataset file.
        trajectory_indices: List of trajectory indices to include.
            If None, use all trajectories.
        transform: Optional transform to apply to samples.
    """

    def __init__(
        self,
        data_path: Path,
        trajectory_indices: Optional[np.ndarray] = None,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform

        # Load metadata and geometry (keep in memory)
        with h5py.File(self.data_path, "r") as f:
            self.mask = f["mask"][:]
            self.sdf_cell = f["sdf_cell"][:]
            self.sdf_coolant = f["sdf_coolant"][:]

            self.dx = f.attrs["dx"]
            self.dy = f.attrs["dy"]
            self.dt = f.attrs["dt"]
            self.T_amb = f.attrs["T_amb"]

            # Temperature data shape: (n_traj, n_steps, H, W)
            temp_shape = f["temperature"].shape
            self.n_trajectories = temp_shape[0]
            self.n_steps_per_traj = temp_shape[1]
            self.grid_size = temp_shape[2]

        # Determine which trajectories to use
        if trajectory_indices is None:
            self.trajectory_indices = np.arange(self.n_trajectories)
        else:
            self.trajectory_indices = np.asarray(trajectory_indices)

        # Each trajectory contributes (n_steps - 1) training pairs
        self.samples_per_traj = self.n_steps_per_traj - 1
        self.total_samples = len(self.trajectory_indices) * self.samples_per_traj

        logger.info(
            f"ThermalDataset initialized: {len(self.trajectory_indices)} trajectories, "
            f"{self.total_samples} samples"
        )

    def __len__(self) -> int:
        """Return total number of training samples."""
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
              - "input": Input tensor of shape ``(9, H, W)``
              - "target": Target temperature of shape ``(1, H, W)``
              - "physics": Physics parameters of shape ``(3,)`` [k_mean, q_mean, h_mean]
        """
        # Map flat index to (trajectory_idx, time_step)
        traj_local_idx = idx // self.samples_per_traj
        time_step = idx % self.samples_per_traj

        traj_global_idx = self.trajectory_indices[traj_local_idx]

        # Load data from HDF5
        with h5py.File(self.data_path, "r") as f:
            # Temperature fields at t and t+1
            T_t = f["temperature"][traj_global_idx, time_step, :, :]       # (H, W)
            T_next = f["temperature"][traj_global_idx, time_step + 1, :, :] # (H, W)

            # Physical parameters for this trajectory
            params = f["parameters"][traj_global_idx, :]  # [k_cell, q0, h_conv, freq]

        # Unpack parameters
        k_cell, q0, h_conv, freq = params

        # Build material property fields
        k_values = np.array([k_cell, 0.6, 0.04])  # cell, coolant, insulation
        k_field = create_property_field(self.mask, k_values)

        # Heat generation field (only in battery cells)
        q_field = q0 * (self.mask == 0).astype(np.float32)

        # Convection coefficient (uniform)
        h_field = np.full_like(k_field, h_conv)

        # Material masks (one-hot encoding)
        mask_cell = (self.mask == 0).astype(np.float32)
        mask_coolant = (self.mask == 1).astype(np.float32)
        mask_insulation = (self.mask == 2).astype(np.float32)

        # Assemble 9-channel input
        input_channels = np.stack([
            T_t,                # Channel 0: Temperature
            mask_cell,          # Channel 1: Cell mask
            mask_coolant,       # Channel 2: Coolant mask
            mask_insulation,    # Channel 3: Insulation mask
            k_field,            # Channel 4: Conductivity
            q_field,            # Channel 5: Heat generation
            h_field,            # Channel 6: Convection coefficient
            self.sdf_cell,      # Channel 7: SDF to cell
            self.sdf_coolant,   # Channel 8: SDF to coolant
        ], axis=0)  # (9, H, W)

        target = T_next[np.newaxis, :, :]  # (1, H, W)

        # Physics parameter vector (for physics conditioning)
        # Use spatial averages over the relevant regions
        k_mean = k_field[self.mask == 0].mean() if (self.mask == 0).sum() > 0 else k_cell
        q_mean = q_field[self.mask == 0].mean() if (self.mask == 0).sum() > 0 else q0
        h_mean = h_conv

        physics_vector = np.array([k_mean, q_mean, h_mean], dtype=np.float32)

        # Convert to torch tensors
        sample = {
            "input": torch.from_numpy(input_channels).float(),
            "target": torch.from_numpy(target).float(),
            "physics": torch.from_numpy(physics_vector).float(),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_data_splits(
    data_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split trajectories into train/val/test sets.

    Args:
        data_path: Path to HDF5 dataset.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    with h5py.File(data_path, "r") as f:
        n_trajectories = f["temperature"].shape[0]

    # Shuffle trajectory indices
    rng = np.random.RandomState(seed)
    indices = np.arange(n_trajectories)
    rng.shuffle(indices)

    # Split
    n_train = int(n_trajectories * train_ratio)
    n_val = int(n_trajectories * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    logger.info(
        f"Data splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    return train_indices, val_indices, test_indices
