"""2-D finite-difference heat equation solver with explicit Euler time integration.

Solves the energy equation on a uniform Cartesian grid:

.. math::

    \\rho\\,c_p\\,\\frac{\\partial T}{\\partial t}
    = \\nabla \\cdot (k\\,\\nabla T) + q(x,y,t)

using second-order central differences in space and forward Euler in time.
Robin (convective) boundary conditions are applied on all four outer surfaces.

The solver supports **spatially varying** thermal properties (k, rho, cp) via
material masks, making it suitable for multi-material battery-pack simulations.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class HeatSolver2D:
    """Explicit finite-difference solver for the 2-D heat equation.

    The solver operates on a regular grid of shape ``(ny, nx)`` with uniform
    spacing ``(dx, dy)``.  Thermal properties may be given as scalars
    (homogeneous medium) or as 2-D arrays of shape ``(ny, nx)`` for
    multi-material problems.

    Args:
        nx: Number of grid points in the *x*-direction.
        ny: Number of grid points in the *y*-direction.
        dx: Grid spacing in *x* [m].
        dy: Grid spacing in *y* [m].
        dt: Time step [s].
        k: Thermal conductivity [W/(m*K)].  Scalar or array ``(ny, nx)``.
        rho: Density [kg/m^3].  Scalar or array ``(ny, nx)``.
        cp: Specific heat capacity [J/(kg*K)].  Scalar or array ``(ny, nx)``.
        T_amb: Ambient temperature [K] used for convective BCs.
        h_conv: Convective heat transfer coefficient [W/(m^2*K)] on outer
            boundaries.

    Raises:
        ValueError: If array-valued properties do not match ``(ny, nx)``.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dt: float,
        k: Union[float, np.ndarray],
        rho: Union[float, np.ndarray],
        cp: Union[float, np.ndarray],
        T_amb: float = 300.0,
        h_conv: float = 10.0,
    ) -> None:
        self.nx: int = nx
        self.ny: int = ny
        self.dx: float = float(dx)
        self.dy: float = float(dy)
        self.dt: float = float(dt)
        self.T_amb: float = float(T_amb)
        self.h_conv: float = float(h_conv)

        # Broadcast scalar properties to full arrays for uniform code paths.
        self.k: np.ndarray = self._to_field(k, "k")
        self.rho: np.ndarray = self._to_field(rho, "rho")
        self.cp: np.ndarray = self._to_field(cp, "cp")

    def _to_field(self, value: Union[float, np.ndarray], name: str) -> np.ndarray:
        """Convert scalar or array to field of shape (ny, nx)."""
        if isinstance(value, (int, float)):
            return np.full((self.ny, self.nx), float(value), dtype=np.float64)
        else:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (self.ny, self.nx):
                raise ValueError(
                    f"{name} array has shape {arr.shape}, expected ({self.ny}, {self.nx})"
                )
            return arr

    @classmethod
    def from_material_fields(
        cls,
        mask: np.ndarray,
        k_values: np.ndarray,
        rho_values: np.ndarray,
        cp_values: np.ndarray,
        dx: float,
        dy: float,
        dt: float,
        T_amb: float = 300.0,
        h_conv: float = 10.0,
    ) -> HeatSolver2D:
        """Construct solver from material mask and per-material property arrays.

        Args:
            mask: Material ID mask of shape ``(ny, nx)`` with integer values.
            k_values: Thermal conductivity for each material ID, shape ``(n_materials,)``.
            rho_values: Density for each material ID, shape ``(n_materials,)``.
            cp_values: Specific heat for each material ID, shape ``(n_materials,)``.
            dx: Grid spacing in x [m].
            dy: Grid spacing in y [m].
            dt: Time step [s].
            T_amb: Ambient temperature [K].
            h_conv: Convective heat transfer coefficient [W/(m^2*K)].

        Returns:
            Initialized HeatSolver2D instance.
        """
        ny, nx = mask.shape
        k_field = k_values[mask]
        rho_field = rho_values[mask]
        cp_field = cp_values[mask]

        return cls(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dt=dt,
            k=k_field,
            rho=rho_field,
            cp=cp_field,
            T_amb=T_amb,
            h_conv=h_conv,
        )

    @property
    def max_stable_dt(self) -> float:
        """Maximum stable time step for explicit Euler scheme (CFL condition).

        Returns:
            Maximum stable dt [s] for the current grid and material properties.
        """
        # Stability criterion: dt <= (rho * cp) / (2 * k * (1/dx^2 + 1/dy^2))
        alpha = self.k / (self.rho * self.cp)  # Thermal diffusivity
        max_alpha = alpha.max()
        dt_stable = 1.0 / (2.0 * max_alpha * (1.0 / self.dx**2 + 1.0 / self.dy**2))
        return dt_stable

    def check_stability(self) -> bool:
        """Check if current time step satisfies CFL stability condition.

        Returns:
            True if stable, False otherwise.

        Warns:
            If the time step exceeds the stability limit.
        """
        dt_max = self.max_stable_dt
        is_stable = self.dt <= dt_max
        if not is_stable:
            logger.warning(
                f"Time step dt={self.dt:.2e} exceeds stability limit {dt_max:.2e}. "
                f"Solver may be unstable. Consider reducing dt."
            )
        return is_stable

    def step(
        self,
        T: np.ndarray,
        q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance temperature field by one time step using explicit Euler.

        Args:
            T: Current temperature field of shape ``(ny, nx)`` [K].
            q: Volumetric heat generation rate ``(ny, nx)`` [W/m^3].
                If None, assumes q=0 everywhere.

        Returns:
            Updated temperature field of shape ``(ny, nx)`` [K].
        """
        if q is None:
            q = np.zeros_like(T)

        T_new = T.copy()

        # Compute Laplacian using 5-point stencil (interior points)
        laplacian = np.zeros_like(T)

        # Interior points
        laplacian[1:-1, 1:-1] = (
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / self.dx**2
            + (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / self.dy**2
        )

        # Apply Robin BC (convection) on boundaries
        # Bottom edge (y=0)
        laplacian[0, :] = (T[1, :] - T[0, :]) / self.dy**2 - (
            self.h_conv / (self.k[0, :] * self.dy)
        ) * (T[0, :] - self.T_amb)

        # Top edge (y=ny-1)
        laplacian[-1, :] = (T[-2, :] - T[-1, :]) / self.dy**2 - (
            self.h_conv / (self.k[-1, :] * self.dy)
        ) * (T[-1, :] - self.T_amb)

        # Left edge (x=0)
        laplacian[:, 0] = (T[:, 1] - T[:, 0]) / self.dx**2 - (
            self.h_conv / (self.k[:, 0] * self.dx)
        ) * (T[:, 0] - self.T_amb)

        # Right edge (x=nx-1)
        laplacian[:, -1] = (T[:, -2] - T[:, -1]) / self.dx**2 - (
            self.h_conv / (self.k[:, -1] * self.dx)
        ) * (T[:, -1] - self.T_amb)

        # Time update: dT/dt = (k / (rho * cp)) * Laplacian(T) + q / (rho * cp)
        alpha = self.k / (self.rho * self.cp)
        T_new = T + self.dt * (alpha * laplacian + q / (self.rho * self.cp))

        return T_new

    def solve(
        self,
        T0: np.ndarray,
        n_steps: int,
        q0: float = 0.0,
        source_mask: Optional[np.ndarray] = None,
        freq: float = 0.0,
        save_every: int = 1,
    ) -> np.ndarray:
        """Run time integration for multiple steps and return trajectory.

        Args:
            T0: Initial temperature field of shape ``(ny, nx)`` [K].
            n_steps: Number of time steps to simulate.
            q0: Baseline volumetric heat generation rate [W/m^3].
            source_mask: Binary mask ``(ny, nx)`` indicating where heat is generated.
                If None, heat is generated uniformly everywhere.
            freq: Oscillation frequency [Hz] for time-varying heat source.
                If 0, heat source is constant.
            save_every: Save temperature field every N steps (default 1 = save all).

        Returns:
            Temperature trajectory of shape ``(n_saved_steps, ny, nx)``.
        """
        if source_mask is None:
            source_mask = np.ones_like(T0)

        n_saved = (n_steps + save_every - 1) // save_every
        trajectory = np.zeros((n_saved, self.ny, self.nx), dtype=np.float32)

        T = T0.copy()
        save_idx = 0

        for step in range(n_steps):
            # Time-varying heat source
            if freq > 0:
                t = step * self.dt
                q_amplitude = q0 * (1.0 + 0.5 * np.sin(2 * np.pi * freq * t))
            else:
                q_amplitude = q0

            q = q_amplitude * source_mask

            # Advance one step
            T = self.step(T, q)

            # Save snapshot
            if step % save_every == 0:
                trajectory[save_idx] = T
                save_idx += 1

        logger.info(
            f"Completed {n_steps} time steps (dt={self.dt:.2e} s, "
            f"total_time={n_steps * self.dt:.3f} s). Saved {save_idx} snapshots."
        )

        return trajectory[:save_idx]
