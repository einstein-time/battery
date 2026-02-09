"""Parallel data generation for 2D battery thermal simulation trajectories.

Generates randomised thermal simulation trajectories by sampling physical
parameters from specified ranges, running a finite-difference heat-equation
solver, and writing the results to a compressed HDF5 file suitable for
consumption by :class:`~src.data_utils.dataset.ThermalDataset`.

The output HDF5 structure is::

    thermal_dataset.h5
    +-- temperature   (n_traj, n_steps, nx, ny)  float32, gzip-4
    +-- parameters    (n_traj, 4)                float32  [k_cell, q0, h_conv, freq]
    +-- masks         (nx, ny)                   int8
    +-- sdf_cell      (nx, ny)                   float32
    +-- sdf_coolant   (nx, ny)                   float32
    +-- attrs: {grid_size, dx, dy, dt, n_steps, T_amb, rho, cp}

Typical usage::

    from src.data_utils.generator import DataGenerator

    config = {
        "data": {"grid_size": 128, "dt": 0.001, "total_time": 1.0,
                 "n_trajectories": 2000},
        "physics": {"dx": 0.001, "dy": 0.001, "rho": 2500.0, "cp": 700.0,
                    "k_range": [0.5, 5.0], "q_range": [1e5, 5e6],
                    "h_range": [10.0, 500.0], "T_amb": 298.15},
    }
    gen = DataGenerator(config)
    gen.generate_dataset(n_trajectories=2000, output_path="data/raw/thermal_dataset.h5")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from src.physics.materials import (
    MaterialProperties,
    compute_signed_distance,
    create_material_mask,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Internal data container
# ======================================================================

@dataclass
class TrajectoryResult:
    """Container for a single simulation trajectory and its metadata.

    Attributes:
        temperature: Temperature field history of shape ``(n_steps, nx, ny)``.
        parameters: Physical parameters ``[k_cell, q0, h_conv, freq]``.
        seed: Random seed used for reproducibility.
    """

    temperature: np.ndarray  # (n_steps, nx, ny)
    parameters: np.ndarray   # (4,) float32
    seed: int


# ======================================================================
# Stand-alone trajectory worker (must be top-level for pickling)
# ======================================================================

def _generate_single_trajectory_worker(args: Tuple[Dict[str, Any], int]) -> TrajectoryResult:
    """Process-safe wrapper around trajectory generation.

    This function lives at module level so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    Args:
        args: Tuple of ``(config_dict, seed)``.

    Returns:
        A :class:`TrajectoryResult` containing the temperature history and
        sampled physical parameters.
    """
    config, seed = args
    gen = DataGenerator(config)
    return gen.generate_single_trajectory(seed)


# ======================================================================
# DataGenerator
# ======================================================================

class DataGenerator:
    """Generates 2D battery thermal simulation datasets.

    The generator samples physical parameters from uniform distributions,
    constructs the material geometry, runs the finite-difference heat solver
    for a prescribed number of time steps, and stores the results in HDF5.

    Args:
        config: Project configuration dictionary.  Expected nested keys:

            * ``config["data"]["grid_size"]`` -- int (default 128)
            * ``config["data"]["dt"]`` -- float, time step [s]
            * ``config["data"]["total_time"]`` -- float [s]
            * ``config["physics"]["dx"]`` -- float, grid spacing x [m]
            * ``config["physics"]["dy"]`` -- float, grid spacing y [m]
            * ``config["physics"]["rho"]`` -- float, density [kg/m^3]
            * ``config["physics"]["cp"]`` -- float, heat capacity [J/(kg*K)]
            * ``config["physics"]["k_range"]`` -- [min, max] conductivity
            * ``config["physics"]["q_range"]`` -- [min, max] heat generation
            * ``config["physics"]["h_range"]`` -- [min, max] convection coeff
            * ``config["physics"]["T_amb"]`` -- float, ambient temperature [K]

    Attributes:
        grid_size: Spatial resolution (cells per side).
        dt: Simulation time step [s].
        n_steps: Total number of time steps per trajectory.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        rho: Reference density [kg/m^3].
        cp: Reference heat capacity [J/(kg*K)].
        T_amb: Ambient temperature [K].
    """

    # Parameter sampling bounds (overridden from config)
    _k_range: Tuple[float, float]
    _q_range: Tuple[float, float]
    _h_range: Tuple[float, float]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the data generator from a configuration dictionary.

        Args:
            config: Nested configuration dict (see class docstring).
        """
        data_cfg: Dict[str, Any] = config.get("data", {})
        physics_cfg: Dict[str, Any] = config.get("physics", {})

        # Grid and time parameters
        self.grid_size: int = int(data_cfg.get("grid_size", 128))
        self.dt: float = float(data_cfg.get("dt", 0.001))
        total_time: float = float(data_cfg.get("total_time", 1.0))
        self.n_steps: int = int(round(total_time / self.dt))

        # Spatial discretisation
        self.dx: float = float(physics_cfg.get("dx", 0.001))
        self.dy: float = float(physics_cfg.get("dy", 0.001))

        # Material properties
        self.rho: float = float(physics_cfg.get("rho", 2500.0))
        self.cp: float = float(physics_cfg.get("cp", 700.0))
        self.T_amb: float = float(physics_cfg.get("T_amb", 298.15))

        # Parameter sampling ranges
        k_range = physics_cfg.get("k_range", [0.5, 5.0])
        q_range = physics_cfg.get("q_range", [1e5, 5e6])
        h_range = physics_cfg.get("h_range", [10.0, 500.0])
        self._k_range = (float(k_range[0]), float(k_range[1]))
        self._q_range = (float(q_range[0]), float(q_range[1]))
        self._h_range = (float(h_range[0]), float(h_range[1]))

        # Store full config for worker pickling
        self._config = config

        logger.info(
            "DataGenerator: grid=%d, dt=%.4f, n_steps=%d, "
            "k_range=%s, q_range=%s, h_range=%s",
            self.grid_size, self.dt, self.n_steps,
            self._k_range, self._q_range, self._h_range,
        )

    # ------------------------------------------------------------------
    # Single trajectory generation
    # ------------------------------------------------------------------

    def generate_single_trajectory(self, seed: int) -> TrajectoryResult:
        """Generate one thermal simulation trajectory.

        Samples random physical parameters from the configured uniform
        distributions, constructs the material field, runs the heat-equation
        solver, and returns the full temperature history.

        Args:
            seed: Random seed for reproducibility of parameter sampling.

        Returns:
            A :class:`TrajectoryResult` with temperature history of shape
            ``(n_steps, nx, ny)`` and the sampled parameters.
        """
        rng = np.random.RandomState(seed)

        # Sample physical parameters from uniform distributions
        k_cell: float = float(rng.uniform(*self._k_range))
        q0: float = float(rng.uniform(*self._q_range))
        h_conv: float = float(rng.uniform(*self._h_range))
        # Frequency of heat-source pulsation (0 = steady)
        freq: float = float(rng.uniform(0.0, 10.0))

        params = np.array([k_cell, q0, h_conv, freq], dtype=np.float32)

        # Create material mask and properties
        mask = create_material_mask(self.grid_size)
        mat_props = MaterialProperties(coolant="water", battery_k=np.clip(k_cell, 1.0, 5.0))

        # Build conductivity field
        k_arr = mat_props.conductivity_array()
        k_field = k_arr[mask].astype(np.float64)  # (nx, ny)

        # Build density and heat-capacity fields
        rho_arr = mat_props.density_array()
        cp_arr = mat_props.heat_capacity_array()
        rho_field = rho_arr[mask].astype(np.float64)
        cp_field = cp_arr[mask].astype(np.float64)

        # Binary masks
        mask_cell = (mask == 0)
        mask_coolant = (mask == 1)

        # Initialise temperature at ambient
        T = np.full((self.grid_size, self.grid_size), self.T_amb, dtype=np.float64)

        # Pre-compute finite-difference coefficients
        alpha_field = k_field / (rho_field * cp_field)  # thermal diffusivity

        # Storage for trajectory
        trajectory = np.empty(
            (self.n_steps, self.grid_size, self.grid_size), dtype=np.float32
        )
        trajectory[0] = T.astype(np.float32)

        # Time-stepping with explicit Euler finite-difference scheme
        for step in range(1, self.n_steps):
            t_phys = step * self.dt

            # Heat source with optional pulsation
            q_current = q0 * (1.0 + 0.5 * np.sin(2.0 * np.pi * freq * t_phys))

            # Laplacian via central differences (interior points)
            T_padded = np.pad(T, 1, mode="edge")
            laplacian = (
                (T_padded[2:, 1:-1] - 2.0 * T + T_padded[:-2, 1:-1]) / (self.dy ** 2)
                + (T_padded[1:-1, 2:] - 2.0 * T + T_padded[1:-1, :-2]) / (self.dx ** 2)
            )

            # Source term: only in battery cells
            source = np.zeros_like(T)
            source[mask_cell] = q_current / (rho_field[mask_cell] * cp_field[mask_cell])

            # Convection cooling: model as volumetric sink in coolant channels
            # q_conv ~ h * (T - T_amb) / characteristic_length
            # Using dx as the characteristic length for simplicity
            conv_sink = np.zeros_like(T)
            conv_sink[mask_coolant] = (
                h_conv * (T[mask_coolant] - self.T_amb)
                / (rho_field[mask_coolant] * cp_field[mask_coolant] * self.dx)
            )

            # Update: dT/dt = alpha * laplacian + source - conv_sink
            dTdt = alpha_field * laplacian + source - conv_sink
            T = T + self.dt * dTdt

            # Clamp for numerical stability
            T = np.clip(T, self.T_amb - 100.0, self.T_amb + 500.0)

            trajectory[step] = T.astype(np.float32)

        return TrajectoryResult(temperature=trajectory, parameters=params, seed=seed)

    # ------------------------------------------------------------------
    # Full dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_trajectories: int,
        output_path: Union[str, Path],
        n_workers: int = 4,
    ) -> Path:
        """Generate a complete dataset and write it to HDF5.

        Trajectories can optionally be generated in parallel using
        :class:`~concurrent.futures.ProcessPoolExecutor`.  Progress is
        displayed with a :mod:`tqdm` progress bar.

        Args:
            n_trajectories: Number of trajectories to generate.
            output_path: Destination HDF5 file path.
            n_workers: Number of parallel worker processes.  Set to 0 or 1
                for sequential execution (safest).

        Returns:
            Resolved :class:`~pathlib.Path` to the written HDF5 file.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating %d trajectories (%d steps each) -> %s",
            n_trajectories, self.n_steps, output_path,
        )
        t_start = time.time()

        # Collect trajectory results
        results: List[TrajectoryResult] = []

        if n_workers <= 1:
            # Sequential generation
            for i in tqdm(range(n_trajectories), desc="Generating trajectories"):
                result = self.generate_single_trajectory(seed=i)
                results.append(result)
        else:
            # Parallel generation with ProcessPoolExecutor
            args_list = [(self._config, seed) for seed in range(n_trajectories)]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_generate_single_trajectory_worker, args): idx
                    for idx, args in enumerate(args_list)
                }
                # Use a list pre-sized for ordered insertion
                results = [None] * n_trajectories  # type: ignore[list-item]
                with tqdm(total=n_trajectories, desc="Generating trajectories") as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                        pbar.update(1)

        elapsed = time.time() - t_start
        logger.info("Generation complete in %.1f s (%.2f s/traj)", elapsed, elapsed / n_trajectories)

        # Build material mask and SDFs (shared across all trajectories)
        mask = create_material_mask(self.grid_size)
        sdf_cell = compute_signed_distance(mask, material_id=0)
        sdf_coolant = compute_signed_distance(mask, material_id=1)

        # Write HDF5
        self._write_hdf5(results, mask, sdf_cell, sdf_coolant, output_path)

        return output_path

    def generate_quick_test(self, output_path: Union[str, Path]) -> Path:
        """Generate a small test dataset for pipeline validation.

        Creates 10 trajectories with 50 time steps each -- enough to verify
        that the full data pipeline (generation -> dataset -> training loop)
        works end-to-end without waiting for a large generation run.

        Args:
            output_path: Destination HDF5 file path.

        Returns:
            Resolved :class:`~pathlib.Path` to the written HDF5 file.
        """
        # Override n_steps temporarily for a short run
        original_n_steps = self.n_steps
        self.n_steps = 50

        n_test_trajectories: int = 10

        logger.info(
            "Generating quick-test dataset: %d trajectories, %d steps -> %s",
            n_test_trajectories, self.n_steps, output_path,
        )

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results: List[TrajectoryResult] = []
        for i in tqdm(range(n_test_trajectories), desc="Quick test generation"):
            result = self.generate_single_trajectory(seed=i + 10000)
            results.append(result)

        mask = create_material_mask(self.grid_size)
        sdf_cell = compute_signed_distance(mask, material_id=0)
        sdf_coolant = compute_signed_distance(mask, material_id=1)

        self._write_hdf5(results, mask, sdf_cell, sdf_coolant, output_path)

        # Restore original n_steps
        self.n_steps = original_n_steps

        return output_path

    # ------------------------------------------------------------------
    # HDF5 writer
    # ------------------------------------------------------------------

    def _write_hdf5(
        self,
        results: List[TrajectoryResult],
        mask: np.ndarray,
        sdf_cell: np.ndarray,
        sdf_coolant: np.ndarray,
        output_path: Path,
    ) -> None:
        """Write generated trajectories to a compressed HDF5 file.

        Args:
            results: List of :class:`TrajectoryResult` objects.
            mask: Material mask of shape ``(nx, ny)``.
            sdf_cell: SDF to battery cells of shape ``(nx, ny)``.
            sdf_coolant: SDF to coolant channels of shape ``(nx, ny)``.
            output_path: Destination file path.
        """
        n_traj = len(results)
        n_steps = results[0].temperature.shape[0]

        logger.info(
            "Writing HDF5: %d trajectories, %d steps, grid %d x %d -> %s",
            n_traj, n_steps, self.grid_size, self.grid_size, output_path,
        )

        with h5py.File(str(output_path), "w") as f:
            # Temperature trajectories with gzip compression
            temp_dset = f.create_dataset(
                "temperature",
                shape=(n_traj, n_steps, self.grid_size, self.grid_size),
                dtype=np.float32,
                chunks=(1, min(n_steps, 50), self.grid_size, self.grid_size),
                compression="gzip",
                compression_opts=4,
            )
            for i, res in enumerate(results):
                temp_dset[i] = res.temperature

            # Physical parameters
            params_array = np.stack([res.parameters for res in results], axis=0)
            f.create_dataset(
                "parameters",
                data=params_array.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )

            # Material mask (shared across trajectories)
            f.create_dataset("masks", data=mask.astype(np.int8))

            # Signed distance fields
            f.create_dataset("sdf_cell", data=sdf_cell.astype(np.float32))
            f.create_dataset("sdf_coolant", data=sdf_coolant.astype(np.float32))

            # Metadata attributes
            f.attrs["grid_size"] = self.grid_size
            f.attrs["dx"] = self.dx
            f.attrs["dy"] = self.dy
            f.attrs["dt"] = self.dt
            f.attrs["n_steps"] = n_steps
            f.attrs["T_amb"] = self.T_amb
            f.attrs["rho"] = self.rho
            f.attrs["cp"] = self.cp
            f.attrs["n_trajectories"] = n_traj
            f.attrs["k_range"] = list(self._k_range)
            f.attrs["q_range"] = list(self._q_range)
            f.attrs["h_range"] = list(self._h_range)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("HDF5 written: %.1f MB", file_size_mb)
