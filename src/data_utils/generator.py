"""Parallel data generation for 2D battery thermal simulation trajectories.

Generates randomised thermal simulation trajectories by sampling physical
parameters from specified ranges, running a finite-difference heat-equation
solver, and writing the results to a compressed HDF5 file.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
from tqdm import tqdm

from src.physics.materials import create_material_mask, compute_signed_distance
from src.physics.solver import HeatSolver2D

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Container for a single simulation trajectory and its metadata.

    Attributes:
        temperature: Temperature field history of shape ``(n_steps, ny, nx)``.
        parameters: Physical parameters ``[k_cell, q0, h_conv, freq]``.
        seed: Random seed used for reproducibility.
    """
    temperature: np.ndarray  # (n_steps, nx, nx)
    parameters: np.ndarray   # (4,) float32
    seed: int


class DataGenerator:
    """Generate synthetic thermal simulation datasets.

    Args:
        config: Configuration dictionary with keys 'data' and 'physics'.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_cfg = config["data"]
        self.physics_cfg = config["physics"]

        # Grid parameters
        self.grid_size = self.data_cfg["grid_size"]
        self.dx = self.physics_cfg["dx"]
        self.dy = self.physics_cfg["dy"]
        self.dt_target = self.data_cfg["dt"]
        self.total_time = self.data_cfg["total_time"]

        # Material properties
        self.rho_cell = self.physics_cfg["rho"]
        self.cp_cell = self.physics_cfg["cp"]
        self.T_amb = self.physics_cfg["T_amb"]

        # Parameter ranges
        self.k_range = tuple(self.physics_cfg["k_range"])
        self.q_range = tuple(self.physics_cfg["q_range"])
        self.h_range = tuple(self.physics_cfg["h_range"])
        self.freq_range = tuple(self.physics_cfg.get("freq_range", [0.0, 0.0]))

        # Create shared material mask
        self.mask = create_material_mask(grid_size=self.grid_size, layout="grid")
        self.source_mask = (self.mask == 0).astype(np.float64)  # Heat only in cells

        # Compute signed distance fields
        self.sdf_cell = compute_signed_distance(self.mask, material_id=0)
        self.sdf_coolant = compute_signed_distance(self.mask, material_id=1)

        logger.info(f"DataGenerator initialized (grid_size={self.grid_size}, dt={self.dt_target})")

    def generate_single_trajectory(self, seed: int) -> TrajectoryResult:
        """Generate one simulation trajectory with random parameters.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            TrajectoryResult containing temperature history and parameters.
        """
        rng = np.random.RandomState(seed)

        # Sample physical parameters
        k_cell = rng.uniform(*self.k_range)
        q0 = rng.uniform(*self.q_range)
        h_conv = rng.uniform(*self.h_range)
        freq = rng.uniform(*self.freq_range)

        params = np.array([k_cell, q0, h_conv, freq], dtype=np.float32)

        # Material property arrays (indexed by material id)
        # 0=battery cell, 1=coolant (water), 2=insulation
        k_values = np.array([k_cell, 0.6, 0.04])
        rho_values = np.array([self.rho_cell, 998.0, 30.0])
        cp_values = np.array([self.cp_cell, 4182.0, 1400.0])

        # Create solver
        solver = HeatSolver2D.from_material_fields(
            mask=self.mask,
            k_values=k_values,
            rho_values=rho_values,
            cp_values=cp_values,
            dx=self.dx,
            dy=self.dy,
            dt=0.0,  # Will be set below
            T_amb=self.T_amb,
            h_conv=h_conv,
        )

        # Use stable time step (50% of CFL limit for safety)
        solver.dt = min(solver.max_stable_dt * 0.5, self.dt_target)

        # Initial condition: uniform ambient temperature
        T0 = np.full((self.grid_size, self.grid_size), self.T_amb, dtype=np.float64)

        # Determine number of solver steps needed
        n_solver_steps = int(self.total_time / solver.dt)
        n_output_steps = self.data_cfg["n_steps"]
        save_every = max(1, n_solver_steps // n_output_steps)

        # Run simulation
        trajectory = solver.solve(
            T0=T0,
            n_steps=n_solver_steps,
            q0=q0,
            source_mask=self.source_mask,
            freq=freq,
            save_every=save_every,
        )

        return TrajectoryResult(
            temperature=trajectory.astype(np.float32),
            parameters=params,
            seed=seed,
        )

    def generate_dataset(
        self,
        n_trajectories: Optional[int] = None,
        output_path: Optional[Path] = None,
        n_workers: int = 1,
    ) -> None:
        """Generate full dataset and save to HDF5 file.

        Args:
            n_trajectories: Number of trajectories to generate (default from config).
            output_path: Output HDF5 file path (default from config).
            n_workers: Number of parallel workers (default 1 for compatibility).
        """
        if n_trajectories is None:
            n_trajectories = self.data_cfg["n_trajectories"]

        if output_path is None:
            output_path = Path(self.data_cfg["output_path"])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {n_trajectories} trajectories...")
        start_time = time.time()

        # Generate trajectories sequentially (simpler for cross-platform)
        all_trajectories = []
        all_parameters = []

        for traj_idx in tqdm(range(n_trajectories), desc="Generating trajectories"):
            result = self.generate_single_trajectory(seed=traj_idx + 42)
            all_trajectories.append(result.temperature)
            all_parameters.append(result.parameters)

        # Stack into arrays
        temperature_data = np.stack(all_trajectories, axis=0)  # (n_traj, n_steps, H, W)
        parameter_data = np.stack(all_parameters, axis=0)      # (n_traj, 4)

        elapsed = time.time() - start_time
        logger.info(f"Generated {n_trajectories} trajectories in {elapsed:.1f}s")

        # Save to HDF5
        logger.info(f"Saving dataset to {output_path}")
        with h5py.File(output_path, "w") as f:
            # Main datasets
            f.create_dataset(
                "temperature",
                data=temperature_data,
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset("parameters", data=parameter_data)

            # Geometry and metadata
            f.create_dataset("mask", data=self.mask.astype(np.int8))
            f.create_dataset("sdf_cell", data=self.sdf_cell)
            f.create_dataset("sdf_coolant", data=self.sdf_coolant)

            # Attributes
            f.attrs["grid_size"] = self.grid_size
            f.attrs["dx"] = self.dx
            f.attrs["dy"] = self.dy
            f.attrs["dt"] = self.dt_target
            f.attrs["n_steps"] = temperature_data.shape[1]
            f.attrs["T_amb"] = self.T_amb
            f.attrs["rho"] = self.rho_cell
            f.attrs["cp"] = self.cp_cell

        file_size_mb = output_path.stat().st_size / 1e6
        logger.info(f"Dataset saved: {file_size_mb:.1f} MB")
