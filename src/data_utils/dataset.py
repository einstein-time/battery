"""PyTorch Dataset and DataLoader utilities for thermal simulation data.

Provides lazy-loading access to 2D battery thermal simulation trajectories stored
in HDF5 format.  Each sample consists of a 9-channel spatial input tensor
(current temperature, material masks, conductivity/heat-source/convection fields,
and signed-distance fields) and a single-channel target tensor representing the
temperature change over one time step (delta_T).

Typical usage::

    from src.data_utils.dataset import ThermalDataset, create_data_loaders

    dataset = ThermalDataset("data/raw/thermal_dataset.h5", split="train",
                             indices=list(range(1400)))
    sample = dataset[0]  # dict with 'input', 'target', 'params', ...

    loaders = create_data_loaders("data/raw/thermal_dataset.h5", config, device)
    for batch in loaders["train"]:
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ThermalDataset(Dataset):
    """PyTorch Dataset for 2D battery thermal simulation trajectories in HDF5.

    Each sample maps a ``(trajectory, timestep)`` pair to a dict containing a
    9-channel input tensor and a 1-channel delta-T target tensor.  Data is
    lazily read from HDF5 so the full dataset does not need to fit in memory.

    HDF5 layout expected::

        thermal_dataset.h5
        +-- temperature   (n_trajectories, n_steps, nx, ny)  float32
        +-- parameters    (n_trajectories, 4)                float32
        +-- masks         (nx, ny)                           int8
        +-- sdf_cell      (nx, ny)                           float32
        +-- sdf_coolant   (nx, ny)                           float32
        +-- attrs: {grid_size, dx, dy, dt, n_steps, T_amb, rho, cp}

    Input channels (C = 9)::

        0  T_t               -- current temperature field [K]
        1  mask_cell         -- binary mask for battery cell regions
        2  mask_coolant      -- binary mask for coolant channel regions
        3  mask_insulation   -- binary mask for insulation regions
        4  k_field           -- spatially varying thermal conductivity [W/(m*K)]
        5  q_field           -- volumetric heat generation rate [W/m^3]
        6  h_field           -- convective heat transfer coefficient [W/(m^2*K)]
        7  sdf_cell          -- signed distance field to cell boundaries
        8  sdf_coolant       -- signed distance field to coolant boundaries

    Target (C = 1)::

        T_{t+1} - T_t  (temperature change over one time step)

    Attributes:
        h5_path: Resolved path to the HDF5 file.
        split: Name of the data split (``"train"``, ``"val"``, or ``"test"``).
        trajectory_indices: Trajectory indices that belong to this split.
        n_steps_per_traj: Number of *usable* time-step transitions per
            trajectory (``n_steps - 1``).
        total_samples: Total number of ``(trajectory, timestep)`` pairs.
    """

    # Material IDs matching src.physics.materials
    _MATERIAL_BATTERY: int = 0
    _MATERIAL_COOLANT: int = 1
    _MATERIAL_INSULATION: int = 2

    def __init__(
        self,
        h5_path: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialise the dataset from an HDF5 file.

        Args:
            h5_path: Path to the HDF5 dataset produced by
                :class:`~src.data_utils.generator.DataGenerator`.
            split: Human-readable split name used for logging.  Has no effect
                on data selection -- use *indices* for that.
            transform: Optional callable applied to the sample dict after
                construction.  Receives and must return a
                ``Dict[str, torch.Tensor]``.
            indices: Trajectory indices that belong to this split.  If
                ``None``, all trajectories in the file are used.

        Raises:
            FileNotFoundError: If *h5_path* does not exist.
            KeyError: If the HDF5 file is missing a required dataset.
        """
        self.h5_path = Path(h5_path).resolve()
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        self.split = split
        self.transform = transform

        # Open file once to read metadata and cache static fields.
        # We do NOT keep the file handle open -- it will be reopened
        # lazily in __getitem__ so that the dataset is picklable for
        # multi-worker DataLoader.
        with h5py.File(str(self.h5_path), "r") as h5f:
            self._read_metadata(h5f, indices)

    def _read_metadata(self, h5f: h5py.File, indices: Optional[Sequence[int]]) -> None:
        """Read and cache metadata and static fields from the open HDF5 file.

        Args:
            h5f: Open HDF5 file handle.
            indices: Trajectory indices for this split.
        """
        # Validate required datasets
        required_datasets = ["temperature", "parameters", "masks", "sdf_cell", "sdf_coolant"]
        for name in required_datasets:
            if name not in h5f:
                raise KeyError(
                    f"Required dataset '{name}' not found in {self.h5_path}. "
                    f"Available keys: {list(h5f.keys())}"
                )

        # Read shapes
        temp_dset = h5f["temperature"]
        n_trajectories: int = temp_dset.shape[0]
        n_steps: int = temp_dset.shape[1]
        self._nx: int = temp_dset.shape[2]
        self._ny: int = temp_dset.shape[3]

        # Trajectory indices for this split
        if indices is not None:
            self.trajectory_indices: np.ndarray = np.asarray(indices, dtype=np.int64)
        else:
            self.trajectory_indices = np.arange(n_trajectories, dtype=np.int64)

        # Number of usable transitions per trajectory
        self.n_steps_per_traj: int = n_steps - 1
        self.total_samples: int = len(self.trajectory_indices) * self.n_steps_per_traj

        # Cache static spatial fields (small: single 2-D arrays)
        mask_raw = h5f["masks"][:]  # (nx, ny) int8
        self._mask_cell = torch.from_numpy((mask_raw == self._MATERIAL_BATTERY).astype(np.float32))
        self._mask_coolant = torch.from_numpy((mask_raw == self._MATERIAL_COOLANT).astype(np.float32))
        self._mask_insulation = torch.from_numpy((mask_raw == self._MATERIAL_INSULATION).astype(np.float32))
        self._sdf_cell = torch.from_numpy(h5f["sdf_cell"][:].astype(np.float32))
        self._sdf_coolant = torch.from_numpy(h5f["sdf_coolant"][:].astype(np.float32))

        # Cache parameters (small: n_traj x 4)
        self._parameters = h5f["parameters"][:]  # (n_traj, 4) float32

        # Read HDF5 attributes for physics constants
        self._T_amb: float = float(h5f.attrs.get("T_amb", 298.15))

        logger.info(
            "ThermalDataset [%s]: %d trajectories, %d steps/traj, "
            "%d total samples, grid (%d, %d)",
            self.split,
            len(self.trajectory_indices),
            self.n_steps_per_traj,
            self.total_samples,
            self._nx,
            self._ny,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of (trajectory, timestep) sample pairs.

        Returns:
            Integer count of available samples in this split.
        """
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve one sample by linear index.

        The linear index is mapped to a ``(trajectory_idx, time_step)`` pair.
        Temperature data is read lazily from HDF5; static fields (masks, SDFs)
        are served from the in-memory cache.

        Args:
            idx: Integer index in ``[0, len(self))``.

        Returns:
            Dictionary with the following keys:

            * ``'input'`` -- ``(9, H, W)`` float32 tensor.
            * ``'target'`` -- ``(1, H, W)`` float32 tensor (delta T).
            * ``'k_field'`` -- ``(1, H, W)`` float32 conductivity field.
            * ``'q_field'`` -- ``(1, H, W)`` float32 heat source field.
            * ``'h_field'`` -- ``(1, H, W)`` float32 convection field.
            * ``'params'`` -- ``(4,)`` float32 tensor of physical parameters.

        Raises:
            IndexError: If *idx* is out of range.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.total_samples}"
            )

        traj_local, time_step = self._linear_to_traj_step(idx)
        traj_global = int(self.trajectory_indices[traj_local])

        # Read temperature lazily -- open and close per access so the
        # dataset stays picklable for multi-worker DataLoaders.
        with h5py.File(str(self.h5_path), "r") as h5f:
            temp_dset = h5f["temperature"]
            T_t = torch.from_numpy(
                temp_dset[traj_global, time_step].astype(np.float32)
            )  # (H, W)
            T_next = torch.from_numpy(
                temp_dset[traj_global, time_step + 1].astype(np.float32)
            )  # (H, W)

        # Physical parameters for this trajectory: [k_cell, q0, h_conv, freq]
        params = self._parameters[traj_global]  # (4,)
        k_cell, q0, h_conv, _freq = params

        # Build spatially-varying physical fields
        k_field = self._build_conductivity_field(k_cell)  # (H, W)
        q_field = self._build_heat_source_field(q0)  # (H, W)
        h_field = self._build_convection_field(h_conv)  # (H, W)

        # Assemble 9-channel input: [T_t, masks(3), k, q, h, sdf(2)]
        input_tensor = torch.stack(
            [
                T_t,                    # 0: current temperature
                self._mask_cell,        # 1: battery cell mask
                self._mask_coolant,     # 2: coolant mask
                self._mask_insulation,  # 3: insulation mask
                k_field,                # 4: conductivity field
                q_field,                # 5: heat source field
                h_field,                # 6: convection field
                self._sdf_cell,         # 7: SDF to cell
                self._sdf_coolant,      # 8: SDF to coolant
            ],
            dim=0,
        )  # (9, H, W)

        # Target: delta T
        delta_T = (T_next - T_t).unsqueeze(0)  # (1, H, W)

        sample: Dict[str, torch.Tensor] = {
            "input": input_tensor,
            "target": delta_T,
            "k_field": k_field.unsqueeze(0),
            "q_field": q_field.unsqueeze(0),
            "h_field": h_field.unsqueeze(0),
            "params": torch.from_numpy(params.astype(np.float32)),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _linear_to_traj_step(self, idx: int) -> Tuple[int, int]:
        """Convert a flat sample index to a (trajectory_local, time_step) pair.

        Args:
            idx: Linear index in ``[0, total_samples)``.

        Returns:
            Tuple of ``(trajectory_local_index, time_step)`` where
            ``trajectory_local_index`` indexes into
            :attr:`trajectory_indices`.
        """
        traj_local = idx // self.n_steps_per_traj
        time_step = idx % self.n_steps_per_traj
        return traj_local, time_step

    def _build_conductivity_field(self, k_cell: float) -> torch.Tensor:
        """Build a spatially-varying thermal conductivity field.

        Battery cells receive *k_cell*; coolant and insulation use fixed
        reference values.

        Args:
            k_cell: Thermal conductivity for battery cell material [W/(m*K)].

        Returns:
            Float32 tensor of shape ``(H, W)``.
        """
        # Reference conductivities (consistent with src.physics.materials)
        k_coolant: float = 0.6   # water at ~25 degC
        k_insulation: float = 0.04

        k_field = (
            self._mask_cell * k_cell
            + self._mask_coolant * k_coolant
            + self._mask_insulation * k_insulation
        )
        return k_field

    def _build_heat_source_field(self, q0: float) -> torch.Tensor:
        """Build the volumetric heat generation field.

        Only battery cells generate heat.

        Args:
            q0: Volumetric heat generation rate [W/m^3].

        Returns:
            Float32 tensor of shape ``(H, W)``.
        """
        return self._mask_cell * q0

    def _build_convection_field(self, h_conv: float) -> torch.Tensor:
        """Build the convective heat transfer coefficient field.

        Convection is active only in coolant regions.

        Args:
            h_conv: Convective heat transfer coefficient [W/(m^2*K)].

        Returns:
            Float32 tensor of shape ``(H, W)``.
        """
        return self._mask_coolant * h_conv


# ======================================================================
# DataLoader factory
# ======================================================================

def create_data_loaders(
    h5_path: Union[str, Path],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, DataLoader]:
    """Create train / validation / test :class:`~torch.utils.data.DataLoader` objects.

    Trajectory-level splitting is performed so that no trajectory appears in
    more than one split.  Split indices are generated deterministically from
    the seed in *config*.

    Args:
        h5_path: Path to the HDF5 dataset file.
        config: Project configuration dictionary.  Expected keys:

            * ``config["data"]["train_ratio"]`` -- float, default 0.7
            * ``config["data"]["val_ratio"]`` -- float, default 0.15
            * ``config["data"]["test_ratio"]`` -- float, default 0.15
            * ``config["training"]["batch_size"]`` -- int, default 16
            * ``config["training"]["seed"]`` -- int, default 42
            * ``config["data"].get("num_workers", 0)`` -- int

        device: Target compute device (used only for logging; tensors are
            moved to device in the training loop, not here).

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        :class:`~torch.utils.data.DataLoader` instances.

    Raises:
        FileNotFoundError: If *h5_path* does not exist.
    """
    from src.data_utils.preprocess import create_splits

    h5_path = Path(h5_path).resolve()

    # Read configuration values with sensible defaults
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    train_ratio: float = data_cfg.get("train_ratio", 0.7)
    val_ratio: float = data_cfg.get("val_ratio", 0.15)
    test_ratio: float = data_cfg.get("test_ratio", 0.15)
    batch_size: int = training_cfg.get("batch_size", 16)
    seed: int = training_cfg.get("seed", 42)
    num_workers: int = data_cfg.get("num_workers", 0)

    # Determine the number of trajectories from the file
    with h5py.File(str(h5_path), "r") as f:
        n_trajectories: int = f["temperature"].shape[0]

    # Create trajectory-level splits
    splits = create_splits(
        n_samples=n_trajectories,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    logger.info(
        "Data splits: train=%d, val=%d, test=%d trajectories (device=%s)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        device,
    )

    # Build datasets
    datasets: Dict[str, ThermalDataset] = {}
    for split_name in ("train", "val", "test"):
        datasets[split_name] = ThermalDataset(
            h5_path=h5_path,
            split=split_name,
            indices=splits[split_name],
        )

    # Build data loaders
    # Pin memory only when targeting a CUDA device
    pin_memory = device.type == "cuda" and num_workers > 0

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
    }

    return loaders
