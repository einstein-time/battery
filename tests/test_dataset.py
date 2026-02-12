"""Tests for dataset utilities."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data_utils.dataset import ThermalDataset, create_data_splits
from src.physics.materials import create_material_mask, compute_signed_distance


@pytest.fixture
def dummy_dataset_path():
    """Create a dummy HDF5 dataset for testing."""
    grid_size = 32
    n_traj = 10
    n_steps = 20

    mask = create_material_mask(grid_size=grid_size)
    sdf_cell = compute_signed_distance(mask, material_id=0)
    sdf_coolant = compute_signed_distance(mask, material_id=1)

    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = Path(f.name)

    with h5py.File(temp_path, "w") as f:
        # Create dummy data
        temperature = np.random.randn(n_traj, n_steps, grid_size, grid_size).astype(np.float32) + 300.0
        parameters = np.random.rand(n_traj, 4).astype(np.float32)

        f.create_dataset("temperature", data=temperature)
        f.create_dataset("parameters", data=parameters)
        f.create_dataset("mask", data=mask)
        f.create_dataset("sdf_cell", data=sdf_cell)
        f.create_dataset("sdf_coolant", data=sdf_coolant)

        f.attrs["grid_size"] = grid_size
        f.attrs["dx"] = 0.001
        f.attrs["dy"] = 0.001
        f.attrs["dt"] = 0.001
        f.attrs["T_amb"] = 300.0

    yield temp_path

    # Cleanup
    temp_path.unlink()


def test_dataset_loading(dummy_dataset_path):
    """Test basic dataset loading."""
    dataset = ThermalDataset(dummy_dataset_path)

    assert len(dataset) > 0
    assert dataset.grid_size == 32


def test_dataset_getitem(dummy_dataset_path):
    """Test getting individual samples."""
    dataset = ThermalDataset(dummy_dataset_path)

    sample = dataset[0]

    assert "input" in sample
    assert "target" in sample
    assert "physics" in sample

    assert sample["input"].shape == (9, 32, 32)
    assert sample["target"].shape == (1, 32, 32)
    assert sample["physics"].shape == (3,)


def test_dataset_splits(dummy_dataset_path):
    """Test train/val/test splitting."""
    train_idx, val_idx, test_idx = create_data_splits(
        dummy_dataset_path,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    # Check that splits are non-overlapping
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # Check that we used all trajectories
    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == 10  # From fixture


def test_dataset_with_indices(dummy_dataset_path):
    """Test dataset with specific trajectory indices."""
    indices = np.array([0, 1, 2])
    dataset = ThermalDataset(dummy_dataset_path, trajectory_indices=indices)

    # Each trajectory has 19 samples (n_steps - 1)
    assert len(dataset) == 3 * 19


def test_dataset_channels(dummy_dataset_path):
    """Test that input channels are correctly constructed."""
    dataset = ThermalDataset(dummy_dataset_path)

    sample = dataset[0]
    input_tensor = sample["input"]

    # Check channel order
    # Channel 0: Temperature
    assert input_tensor[0].mean() > 200  # Reasonable temperature range

    # Channels 1-3: Material masks (binary)
    for i in [1, 2, 3]:
        assert input_tensor[i].min() >= 0
        assert input_tensor[i].max() <= 1

    # Channel 4: Conductivity (positive)
    assert (input_tensor[4] >= 0).all()

    # Channel 5: Heat generation (non-negative)
    assert (input_tensor[5] >= 0).all()
