"""2-D finite-difference heat equation solver with explicit Euler time integration.

Solves the energy equation on a uniform Cartesian grid:

.. math::

    \\rho\\,c_p\\,\\frac{\\partial T}{\\partial t}
    = \\nabla \\cdot (k\\,\\nabla T) + q(x,y,t)

using second-order central differences in space and forward Euler in time.
Robin (convective) boundary conditions are applied on all four outer surfaces.

The solver supports **spatially varying** thermal properties (k, rho, cp) via
material masks, making it suitable for multi-material battery-pack simulations.

Typical usage::

    from src.physics.solver import HeatSolver2D
    import numpy as np

    solver = HeatSolver2D(nx=128, ny=128, dx=1e-3, dy=1e-3,
                          dt=1e-3, k=3.0, rho=2500, cp=700,
                          T_amb=300.0, h_conv=10.0)
    solver.check_stability()
    T0 = np.full((128, 128), 300.0)
    source_mask = np.ones_like(T0)
    trajectory = solver.solve(T0, n_steps=1000, q0=1e4, source_mask=source_mask)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------ init
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

    # ------------------------------------------------------------ helpers
    def _to_field(self, value: Union[float, np.ndarray], name: str) -> np.ndarray:
        """Convert a scalar or array to a full ``(ny, nx)`` float64 field.

        Args:
            value: Scalar or ``ndarray`` of shape ``(ny, nx)``.
            name: Parameter name for error messages.

        Returns:
            Float64 array of shape ``(ny, nx)``.

        Raises:
            ValueError: If an array is supplied with the wrong shape.
        """
        if np.isscalar(value):
            return np.full((self.ny, self.nx), float(value), dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.ny, self.nx):
            raise ValueError(
                f"{name} array has shape {arr.shape}, expected ({self.ny}, {self.nx})"
            )
        return arr

    # --------------------------------------------------------- Laplacian
    def compute_laplacian(self, T: np.ndarray) -> np.ndarray:
        """Compute the discrete Laplacian of a temperature field.

        Uses second-order central finite differences with the *variable
        coefficient* form:

        .. math::

            \\nabla \\cdot (k\\,\\nabla T) \\approx
            \\frac{\\partial}{\\partial x}\\!\\left(k\\frac{\\partial T}{\\partial x}\\right)
            + \\frac{\\partial}{\\partial y}\\!\\left(k\\frac{\\partial T}{\\partial y}\\right)

        At interior points the x-component is approximated as::

            d/dx(k dT/dx) ≈ [ k_{i,j+1/2}*(T_{i,j+1} - T_{i,j})
                              - k_{i,j-1/2}*(T_{i,j} - T_{i,j-1}) ] / dx^2

        where ``k_{i,j+1/2} = 0.5*(k_{i,j} + k_{i,j+1})``.

        Boundary treatment: the outermost ring of cells is **not** updated by
        the Laplacian (handled separately by :meth:`apply_boundary_conditions`).
        Zero-gradient (Neumann) padding is used so that the stencil is valid
        everywhere.

        Args:
            T: Temperature field of shape ``(ny, nx)`` [K].

        Returns:
            Array of shape ``(ny, nx)`` with div(k grad T) [W/m^3].
        """
        ny, nx = self.ny, self.nx
        lap = np.zeros_like(T)

        # Pad T with zero-gradient (replicate) for stencil near boundaries.
        Tp = np.pad(T, 1, mode="edge")
        kp = np.pad(self.k, 1, mode="edge")

        # Half-point conductivities (harmonic mean for robustness).
        # Using arithmetic mean here for simplicity + stability.
        k_ip = 0.5 * (kp[1:-1, 1:-1] + kp[2:, 1:-1])    # k_{i+1/2, j}
        k_im = 0.5 * (kp[1:-1, 1:-1] + kp[:-2, 1:-1])    # k_{i-1/2, j}
        k_jp = 0.5 * (kp[1:-1, 1:-1] + kp[1:-1, 2:])     # k_{i, j+1/2}
        k_jm = 0.5 * (kp[1:-1, 1:-1] + kp[1:-1, :-2])    # k_{i, j-1/2}

        # Central slice of padded T is Tp[1:-1, 1:-1] == T.
        T_c = Tp[1:-1, 1:-1]
        T_ip = Tp[2:, 1:-1]
        T_im = Tp[:-2, 1:-1]
        T_jp = Tp[1:-1, 2:]
        T_jm = Tp[1:-1, :-2]

        lap_y = (k_ip * (T_ip - T_c) - k_im * (T_c - T_im)) / (self.dy ** 2)
        lap_x = (k_jp * (T_jp - T_c) - k_jm * (T_c - T_jm)) / (self.dx ** 2)

        lap = lap_x + lap_y
        return lap

    # ------------------------------------------------ boundary conditions
    def apply_boundary_conditions(
        self,
        T: np.ndarray,
        h: Optional[float] = None,
        T_amb: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Robin (convective) boundary conditions on all four outer surfaces.

        The discrete convective BC for a boundary cell is:

        .. math::

            T_{\\text{boundary}} = \\frac{k\\,T_{\\text{interior}} + h\\,\\Delta x\\,T_{\\text{amb}}}
            {k + h\\,\\Delta x}

        This is a first-order-accurate discretisation of
        ``-k dT/dn = h*(T - T_amb)`` at the boundary.

        Args:
            T: Temperature field of shape ``(ny, nx)``.  **Modified in-place.**
            h: Override convective HTC [W/(m^2*K)].  Defaults to
                :attr:`h_conv`.
            T_amb: Override ambient temperature [K].  Defaults to
                :attr:`T_amb`.

        Returns:
            The same array *T* (modified in-place) for convenience.
        """
        if h is None:
            h = self.h_conv
        if T_amb is None:
            T_amb = self.T_amb

        k = self.k

        # Bottom boundary (y = 0, row index 0).
        _apply_robin_edge(T, k, h, T_amb, self.dy, axis=0, side="low")
        # Top boundary (y = L, last row).
        _apply_robin_edge(T, k, h, T_amb, self.dy, axis=0, side="high")
        # Left boundary (x = 0, column index 0).
        _apply_robin_edge(T, k, h, T_amb, self.dx, axis=1, side="low")
        # Right boundary (x = L, last column).
        _apply_robin_edge(T, k, h, T_amb, self.dx, axis=1, side="high")

        return T

    # -------------------------------------------------------- heat source
    @staticmethod
    def heat_source(t: float, q0: float, freq: float = 1.0) -> float:
        """Compute the time-dependent volumetric heat source.

        .. math::

            q(t) = q_0\\,(1 + 0.3\\,\\sin(2\\pi f t))

        Args:
            t: Current simulation time [s].
            q0: Baseline volumetric heat generation rate [W/m^3].
            freq: Oscillation frequency [Hz].  Defaults to 1.0.

        Returns:
            Instantaneous heat generation rate [W/m^3].
        """
        return q0 * (1.0 + 0.3 * math.sin(2.0 * math.pi * freq * t))

    # --------------------------------------------------- single time step
    def step(
        self,
        T: np.ndarray,
        t: float,
        q0: float,
        source_mask: np.ndarray,
    ) -> np.ndarray:
        """Advance the temperature field by one explicit Euler time step.

        Implements:

        .. math::

            T^{n+1} = T^n + \\frac{\\Delta t}{\\rho\\,c_p}
            \\bigl[\\nabla\\cdot(k\\,\\nabla T^n) + q(t)\\,M(x,y)\\bigr]

        where *M(x,y)* is the ``source_mask``.

        Args:
            T: Current temperature field ``(ny, nx)`` [K].
            t: Current simulation time [s].
            q0: Base heat source magnitude [W/m^3].
            source_mask: Array ``(ny, nx)`` in [0, 1] indicating where heat
                is generated (e.g. battery cells only).

        Returns:
            Updated temperature field ``(ny, nx)`` for the next time level.
            A **new** array is returned; the input *T* is not modified.
        """
        lap = self.compute_laplacian(T)
        q = self.heat_source(t, q0) * source_mask

        rho_cp = self.rho * self.cp
        dTdt = (lap + q) / rho_cp

        T_new = T + self.dt * dTdt

        # Apply boundary conditions on the new field.
        self.apply_boundary_conditions(T_new)

        return T_new

    # ------------------------------------------------- multi-step solve
    def solve(
        self,
        T0: np.ndarray,
        n_steps: int,
        q0: float,
        source_mask: np.ndarray,
        save_every: int = 1,
    ) -> np.ndarray:
        """Run the solver for *n_steps* time steps and return the trajectory.

        Args:
            T0: Initial temperature field ``(ny, nx)`` [K].
            n_steps: Number of time steps to take.
            q0: Base heat source [W/m^3].
            source_mask: Heat-source spatial distribution ``(ny, nx)``.
            save_every: Store a snapshot every *save_every* steps.  Defaults
                to 1 (store every step).  The initial state is always stored.

        Returns:
            Temperature trajectory array of shape
            ``(n_saved + 1, ny, nx)`` where ``n_saved = n_steps // save_every``.
            Index 0 is the initial condition.

        Raises:
            ValueError: If *T0* or *source_mask* shapes do not match the grid.
        """
        if T0.shape != (self.ny, self.nx):
            raise ValueError(
                f"T0 shape {T0.shape} does not match grid ({self.ny}, {self.nx})"
            )
        if source_mask.shape != (self.ny, self.nx):
            raise ValueError(
                f"source_mask shape {source_mask.shape} does not match grid "
                f"({self.ny}, {self.nx})"
            )

        n_saved = n_steps // save_every
        trajectory = np.zeros((n_saved + 1, self.ny, self.nx), dtype=np.float64)
        trajectory[0] = T0.copy()

        T = T0.copy()
        t = 0.0
        save_idx = 1

        for step_i in range(1, n_steps + 1):
            T = self.step(T, t, q0, source_mask)
            t += self.dt

            if step_i % save_every == 0:
                trajectory[save_idx] = T
                save_idx += 1

        logger.info(
            "Solve complete: %d steps, final T range [%.2f, %.2f] K",
            n_steps,
            T.min(),
            T.max(),
        )
        return trajectory

    # ------------------------------------------------ stability check
    def check_stability(self) -> bool:
        """Verify the CFL-like stability condition for the explicit scheme.

        For a 2-D explicit heat equation the stability requirement is:

        .. math::

            \\Delta t \\le \\frac{\\Delta x^2\\,\\rho\\,c_p}{4\\,k_{\\max}}

        (assuming dx == dy; otherwise use the more restrictive direction).

        Returns:
            ``True`` if the current ``dt`` satisfies the stability limit.

        Raises:
            RuntimeWarning: Logged (not raised) if the condition is violated.
                Callers should inspect the return value.
        """
        k_max = float(self.k.max())
        rho_cp_min = float((self.rho * self.cp).min())

        # Use the smaller of dx, dy for the most restrictive bound.
        ds_min = min(self.dx, self.dy)
        dt_max = ds_min ** 2 * rho_cp_min / (4.0 * k_max)

        stable = self.dt <= dt_max
        if not stable:
            logger.warning(
                "Stability condition VIOLATED: dt=%.4e > dt_max=%.4e. "
                "Reduce dt or increase grid spacing.",
                self.dt,
                dt_max,
            )
        else:
            logger.info(
                "Stability OK: dt=%.4e <= dt_max=%.4e (margin %.1f%%).",
                self.dt,
                dt_max,
                100.0 * (dt_max - self.dt) / dt_max,
            )
        return stable

    # --------------------------------------------- convenience factories
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
    ) -> "HeatSolver2D":
        """Create a solver from a material mask and per-material property arrays.

        This is the preferred constructor for multi-material problems.  The
        property arrays are indexed by material id (matching the mask values).

        Args:
            mask: Integer array ``(ny, nx)`` of material ids.
            k_values: 1-D conductivity array indexed by material id.
            rho_values: 1-D density array indexed by material id.
            cp_values: 1-D heat-capacity array indexed by material id.
            dx: Grid spacing in *x* [m].
            dy: Grid spacing in *y* [m].
            dt: Time step [s].
            T_amb: Ambient temperature [K].
            h_conv: Convective HTC [W/(m^2*K)].

        Returns:
            Configured :class:`HeatSolver2D` instance.
        """
        ny, nx = mask.shape
        k_field = k_values[mask].astype(np.float64)
        rho_field = rho_values[mask].astype(np.float64)
        cp_field = cp_values[mask].astype(np.float64)

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

    # ------------------------------------------------------ repr / info
    def __repr__(self) -> str:
        return (
            f"HeatSolver2D(nx={self.nx}, ny={self.ny}, dx={self.dx:.2e}, "
            f"dy={self.dy:.2e}, dt={self.dt:.2e}, "
            f"k_range=[{self.k.min():.3g}, {self.k.max():.3g}], "
            f"T_amb={self.T_amb:.1f}, h_conv={self.h_conv:.1f})"
        )

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Return ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def domain_size(self) -> Tuple[float, float]:
        """Return physical domain size ``(Ly, Lx)`` in metres."""
        return (self.ny * self.dy, self.nx * self.dx)

    @property
    def max_stable_dt(self) -> float:
        """Return the maximum stable time step for the current grid and properties."""
        k_max = float(self.k.max())
        rho_cp_min = float((self.rho * self.cp).min())
        ds_min = min(self.dx, self.dy)
        return ds_min ** 2 * rho_cp_min / (4.0 * k_max)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _apply_robin_edge(
    T: np.ndarray,
    k: np.ndarray,
    h: float,
    T_amb: float,
    dn: float,
    axis: int,
    side: str,
) -> None:
    """Apply a Robin BC on one edge of the domain (in-place).

    The discrete form is:

        T_bnd = (k_bnd * T_int + h * dn * T_amb) / (k_bnd + h * dn)

    where *T_int* is the first interior cell adjacent to the boundary.

    Args:
        T: Full temperature array ``(ny, nx)``.
        k: Conductivity array ``(ny, nx)``.
        h: Convective HTC.
        T_amb: Ambient temperature.
        dn: Grid spacing normal to the boundary.
        axis: 0 for y-boundaries (rows), 1 for x-boundaries (columns).
        side: ``"low"`` for the index-0 boundary, ``"high"`` for the last.
    """
    if axis == 0:
        if side == "low":
            k_bnd = k[0, :]
            T_int = T[1, :]
            T[0, :] = (k_bnd * T_int + h * dn * T_amb) / (k_bnd + h * dn)
        else:
            k_bnd = k[-1, :]
            T_int = T[-2, :]
            T[-1, :] = (k_bnd * T_int + h * dn * T_amb) / (k_bnd + h * dn)
    else:
        if side == "low":
            k_bnd = k[:, 0]
            T_int = T[:, 1]
            T[:, 0] = (k_bnd * T_int + h * dn * T_amb) / (k_bnd + h * dn)
        else:
            k_bnd = k[:, -1]
            T_int = T[:, -2]
            T[:, -1] = (k_bnd * T_int + h * dn * T_amb) / (k_bnd + h * dn)
