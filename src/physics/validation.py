"""Method of Manufactured Solutions (MMS) validation for the heat equation solver.

This module provides tools to verify spatial and temporal convergence of
:class:`~src.physics.solver.HeatSolver2D`, check energy conservation, and
compare numerical results against analytical steady-state solutions.

The primary entry point is :func:`manufactured_solution_test`, which runs the
solver at multiple resolutions and verifies second-order spatial convergence.

Typical usage::

    from src.physics.solver import HeatSolver2D
    from src.physics.validation import manufactured_solution_test

    results = manufactured_solution_test()
    for gs, err, rate in zip(results["grid_sizes"], results["L2"], results["rates"]):
        print(f"N={gs:4d}  L2={err:.4e}  rate={rate:.2f}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.physics.solver import HeatSolver2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------

def analytical_steady_state(
    x: np.ndarray,
    y: np.ndarray,
    k: float,
    q: float,
) -> np.ndarray:
    """Analytical steady-state temperature for uniform heat generation in a square.

    Considers a unit-square domain ``[0, 1] x [0, 1]`` with constant
    conductivity *k*, uniform volumetric heat source *q*, and homogeneous
    Dirichlet BCs (T = 0 on all boundaries).

    The exact solution is obtained via a double Fourier sine series:

    .. math::

        T(x,y) = \\sum_{m\\ \\text{odd}} \\sum_{n\\ \\text{odd}}
        \\frac{16\\,q}{\\pi^4\\,k\\,m\\,n\\,(m^2+n^2)}
        \\sin(m\\pi x)\\sin(n\\pi y)

    We retain enough terms for the series to converge to machine precision
    for moderate *q/k* values.

    Args:
        x: 2-D array of x-coordinates in [0, 1].
        y: 2-D array of y-coordinates in [0, 1] (same shape as *x*).
        k: Thermal conductivity [W/(m*K)].
        q: Volumetric heat source [W/m^3].

    Returns:
        Temperature field of the same shape as *x*.
    """
    T = np.zeros_like(x, dtype=np.float64)
    n_terms = 30  # plenty for convergence

    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            coeff = 16.0 * q / (
                math.pi ** 4 * k * m * n * (m ** 2 + n ** 2)
            )
            T += coeff * np.sin(m * math.pi * x) * np.sin(n * math.pi * y)

    return T


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------

def compute_error_norms(
    numerical: np.ndarray,
    analytical: np.ndarray,
) -> Dict[str, float]:
    """Compute L1, L2, and L-infinity error norms between two fields.

    All norms are *grid-averaged* (divided by the number of grid points)
    so that they are resolution-independent.

    Args:
        numerical: Computed temperature field.
        analytical: Reference (exact) temperature field of the same shape.

    Returns:
        Dictionary with keys ``"L1"``, ``"L2"``, ``"Linf"``.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    if numerical.shape != analytical.shape:
        raise ValueError(
            f"Shape mismatch: numerical {numerical.shape} vs "
            f"analytical {analytical.shape}"
        )

    err = np.abs(numerical - analytical)
    n = float(err.size)

    return {
        "L1": float(np.sum(err) / n),
        "L2": float(np.sqrt(np.sum(err ** 2) / n)),
        "Linf": float(np.max(err)),
    }


# ---------------------------------------------------------------------------
# Convergence rate
# ---------------------------------------------------------------------------

def convergence_rate(
    errors: Sequence[float],
    grid_sizes: Sequence[int],
) -> List[float]:
    """Compute observed convergence rates from a sequence of errors and grid sizes.

    Between consecutive refinement levels the rate is:

    .. math::

        p = \\frac{\\log(e_{i} / e_{i+1})}{\\log(N_{i+1} / N_{i})}

    A second-order method should yield *p* close to 2.

    Args:
        errors: Error norms at each grid size (coarse to fine).
        grid_sizes: Corresponding grid sizes.

    Returns:
        List of convergence rates of length ``len(errors) - 1``.
        The first entry is ``float('nan')`` for convenience (no rate at
        the coarsest level).

    Raises:
        ValueError: If lengths differ or fewer than 2 entries.
    """
    if len(errors) != len(grid_sizes):
        raise ValueError("errors and grid_sizes must have the same length")
    if len(errors) < 2:
        raise ValueError("Need at least 2 data points to compute a rate")

    rates: List[float] = [float("nan")]
    for i in range(1, len(errors)):
        if errors[i] == 0.0 or errors[i - 1] == 0.0:
            rates.append(float("nan"))
            continue
        log_err = math.log(errors[i - 1] / errors[i])
        log_h = math.log(grid_sizes[i] / grid_sizes[i - 1])
        rates.append(log_err / log_h if log_h != 0.0 else float("nan"))

    return rates


# ---------------------------------------------------------------------------
# Energy conservation check
# ---------------------------------------------------------------------------

def check_energy_conservation(
    solver: HeatSolver2D,
    T_trajectory: np.ndarray,
    q_source: np.ndarray,
    q0: float = 0.0,
    freq: float = 1.0,
    rtol: float = 0.1,
) -> Dict[str, Union[float, bool]]:
    """Verify global energy balance over a simulation trajectory.

    For the 2-D heat equation with heat source *q* the energy balance is:

    .. math::

        \\int \\rho\\,c_p\\,\\frac{\\partial T}{\\partial t}\\,dA
        = \\int q\\,dA + \\oint (-k\\,\\nabla T \\cdot \\hat{n})\\,dS

    We approximate the left side using finite differences in time and check
    that the relative imbalance is below *rtol*.

    Args:
        solver: The :class:`HeatSolver2D` instance used to produce the trajectory.
        T_trajectory: Temperature history ``(n_steps+1, ny, nx)`` [K].
        q_source: Spatial source mask ``(ny, nx)`` (0/1 or float).
        q0: Base heat source magnitude [W/m^3].
        freq: Heat-source oscillation frequency [Hz].
        rtol: Relative tolerance for the energy balance check.

    Returns:
        Dictionary with:

        - ``"energy_stored"``: cumulative energy stored in the domain [J/m].
        - ``"energy_source"``: cumulative energy from heat source [J/m].
        - ``"energy_boundary"``: estimated boundary heat loss [J/m].
        - ``"relative_imbalance"``: ``|stored - source - boundary| / |source|``.
        - ``"passed"``: ``True`` if relative imbalance < *rtol*.
    """
    n_snaps = T_trajectory.shape[0]
    dt = solver.dt
    dx = solver.dx
    dy = solver.dy
    cell_area = dx * dy  # area of one grid cell [m^2]

    rho_cp = solver.rho * solver.cp  # (ny, nx)

    # --- Energy stored (change in internal energy) --------------------------
    dT_total = T_trajectory[-1] - T_trajectory[0]  # (ny, nx)
    energy_stored = float(np.sum(rho_cp * dT_total) * cell_area)

    # --- Energy from heat source -------------------------------------------
    energy_source = 0.0
    for i in range(n_snaps - 1):
        t = i * dt
        q_t = solver.heat_source(t, q0, freq)
        energy_source += float(np.sum(q_source * q_t) * cell_area * dt)

    # --- Boundary heat loss (approximate via boundary gradient) -------------
    # Sum convective losses: Q_bnd = h * (T_bnd - T_amb) * perimeter_element * dt
    h = solver.h_conv
    T_amb = solver.T_amb
    energy_boundary = 0.0

    for i in range(n_snaps - 1):
        T = T_trajectory[i]
        # Bottom and top rows.
        q_bot = h * (T[0, :] - T_amb) * dx
        q_top = h * (T[-1, :] - T_amb) * dx
        # Left and right columns.
        q_left = h * (T[:, 0] - T_amb) * dy
        q_right = h * (T[:, -1] - T_amb) * dy

        boundary_loss = float(np.sum(q_bot) + np.sum(q_top)
                              + np.sum(q_left) + np.sum(q_right))
        energy_boundary += boundary_loss * dt

    # energy_boundary is *loss* (positive means heat leaving the domain).
    # Energy balance: stored = source - boundary_loss
    imbalance = abs(energy_stored - energy_source + energy_boundary)
    denom = max(abs(energy_source), 1e-30)
    relative_imbalance = imbalance / denom

    passed = relative_imbalance < rtol

    result = {
        "energy_stored": energy_stored,
        "energy_source": energy_source,
        "energy_boundary": energy_boundary,
        "relative_imbalance": relative_imbalance,
        "passed": passed,
    }

    if passed:
        logger.info("Energy conservation check PASSED (rel. imbalance=%.4e)", relative_imbalance)
    else:
        logger.warning(
            "Energy conservation check FAILED (rel. imbalance=%.4e > rtol=%.2e)",
            relative_imbalance,
            rtol,
        )

    return result


# ---------------------------------------------------------------------------
# Method of Manufactured Solutions (MMS) convergence test
# ---------------------------------------------------------------------------

def manufactured_solution_test(
    solver: Optional[HeatSolver2D] = None,
    grid_sizes: Optional[List[int]] = None,
    k: float = 1.0,
    q: float = 1.0,
    n_time_steps: int = 50000,
    safety_factor: float = 0.4,
) -> Dict[str, Union[List[int], List[float]]]:
    """Run a spatial-convergence study using the Method of Manufactured Solutions.

    The test solves the heat equation to steady state on a unit square with
    uniform conductivity *k* and uniform source *q* using Dirichlet BCs
    (T = 0 on boundaries).  The exact steady-state solution is given by
    :func:`analytical_steady_state`.

    The solver is run at each resolution in *grid_sizes* and the L2 error
    is recorded.  The observed convergence rate should be approximately
    O(dx^2) for the second-order central-difference scheme.

    Note:
        The *solver* argument, if provided, is ignored in favour of freshly
        constructed solvers at each grid size (the function needs to control
        dx, dy, dt, etc.).  It is accepted for API compatibility with the
        package ``__init__.py``.

    Args:
        solver: Unused.  Accepted for interface compatibility.
        grid_sizes: List of grid resolutions to test.  Defaults to
            ``[16, 32, 64, 128]``.
        k: Thermal conductivity used for the manufactured solution.
        q: Volumetric heat source used for the manufactured solution.
        n_time_steps: Number of explicit Euler steps to approximate steady
            state.  Higher values give a more converged solution.
        safety_factor: Fraction of the maximum stable time step to use.
            Must be in (0, 1).

    Returns:
        Dictionary with keys:

        - ``"grid_sizes"`` (``List[int]``): The tested resolutions.
        - ``"L1"`` (``List[float]``): L1 errors at each resolution.
        - ``"L2"`` (``List[float]``): L2 errors at each resolution.
        - ``"Linf"`` (``List[float]``): Linf errors at each resolution.
        - ``"rates"`` (``List[float]``): Convergence rates from L2 errors.

    Example::

        results = manufactured_solution_test(grid_sizes=[16, 32, 64])
        for gs, err, rate in zip(results["grid_sizes"],
                                  results["L2"],
                                  results["rates"]):
            print(f"N={gs:>4d}  L2={err:.4e}  rate={rate:+.2f}")
    """
    if grid_sizes is None:
        grid_sizes = [16, 32, 64, 128]

    l1_errors: List[float] = []
    l2_errors: List[float] = []
    linf_errors: List[float] = []

    for n in grid_sizes:
        dx = 1.0 / n
        dy = 1.0 / n
        rho = 1.0
        cp = 1.0

        # Stable time step.
        dt_max = dx ** 2 * rho * cp / (4.0 * k)
        dt = safety_factor * dt_max

        # Build solver with Dirichlet BCs approximated via very large h.
        # With h -> inf the Robin BC becomes T_bnd = T_amb = 0.
        h_large = k / dx * 1e6
        test_solver = HeatSolver2D(
            nx=n, ny=n, dx=dx, dy=dy, dt=dt,
            k=k, rho=rho, cp=cp,
            T_amb=0.0, h_conv=h_large,
        )

        # Initial condition: T = 0 everywhere.
        T = np.zeros((n, n), dtype=np.float64)
        source_mask = np.ones((n, n), dtype=np.float64)

        # March to (approximate) steady state with constant source.
        # Override heat_source to return constant q (no oscillation).
        for step_i in range(n_time_steps):
            lap = test_solver.compute_laplacian(T)
            T_new = T + dt * (lap + q * source_mask) / (rho * cp)
            test_solver.apply_boundary_conditions(T_new, h=h_large, T_amb=0.0)
            T = T_new

        # Analytical solution on cell centres.
        xs = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n)
        ys = np.linspace(0.5 * dy, 1.0 - 0.5 * dy, n)
        X, Y = np.meshgrid(xs, ys)
        T_exact = analytical_steady_state(X, Y, k, q)

        # Compute errors (exclude boundary cells affected by Robin approx).
        interior = slice(1, -1)
        norms = compute_error_norms(
            T[interior, interior], T_exact[interior, interior]
        )
        l1_errors.append(norms["L1"])
        l2_errors.append(norms["L2"])
        linf_errors.append(norms["Linf"])

        logger.info(
            "MMS  N=%4d  L2=%.4e  Linf=%.4e",
            n, norms["L2"], norms["Linf"],
        )

    rates = convergence_rate(l2_errors, grid_sizes)

    return {
        "grid_sizes": grid_sizes,
        "L1": l1_errors,
        "L2": l2_errors,
        "Linf": linf_errors,
        "rates": rates,
    }


# ---------------------------------------------------------------------------
# Convenience: quick self-test
# ---------------------------------------------------------------------------

def run_quick_validation(verbose: bool = True) -> bool:
    """Run a fast validation with small grids and print results.

    This is intended as a smoke-test that can be called from CI or
    interactively.  It checks:

    1. Spatial convergence is at least O(dx^1.5) (conservative threshold).
    2. A short energy-conservation check passes.

    Args:
        verbose: If ``True``, print results to stdout.

    Returns:
        ``True`` if all checks pass.
    """
    import sys

    all_ok = True

    # 1. MMS convergence test (small grids, fewer steps for speed).
    results = manufactured_solution_test(
        grid_sizes=[8, 16, 32],
        n_time_steps=20000,
        safety_factor=0.4,
    )
    last_rate = results["rates"][-1]

    if verbose:
        print("=== MMS Spatial Convergence ===")
        for gs, err, rate in zip(
            results["grid_sizes"], results["L2"], results["rates"]
        ):
            rate_str = f"{rate:+.2f}" if not math.isnan(rate) else "  n/a"
            print(f"  N={gs:>4d}  L2={err:.4e}  rate={rate_str}")

    if math.isnan(last_rate) or last_rate < 1.5:
        if verbose:
            print(f"  ** FAIL: expected rate >= 1.5, got {last_rate:.2f}")
        all_ok = False
    else:
        if verbose:
            print(f"  PASS (rate={last_rate:.2f})")

    # 2. Energy conservation test.
    n = 32
    dx = 1e-3
    test_solver = HeatSolver2D(
        nx=n, ny=n, dx=dx, dy=dx,
        dt=0.5 * dx ** 2 * 2500 * 700 / (4.0 * 3.0),
        k=3.0, rho=2500.0, cp=700.0,
        T_amb=300.0, h_conv=10.0,
    )
    T0 = np.full((n, n), 300.0, dtype=np.float64)
    source_mask = np.ones((n, n), dtype=np.float64)
    n_steps = 200
    traj = test_solver.solve(T0, n_steps, q0=1e4, source_mask=source_mask)

    echeck = check_energy_conservation(
        test_solver, traj, source_mask, q0=1e4, freq=1.0, rtol=0.2,
    )

    if verbose:
        print("\n=== Energy Conservation ===")
        print(f"  Stored  : {echeck['energy_stored']:.4e} J/m")
        print(f"  Source  : {echeck['energy_source']:.4e} J/m")
        print(f"  Boundary: {echeck['energy_boundary']:.4e} J/m")
        print(f"  Rel. imbalance: {echeck['relative_imbalance']:.4e}")
        status = "PASS" if echeck["passed"] else "FAIL"
        print(f"  {status}")

    if not echeck["passed"]:
        all_ok = False

    if verbose:
        print(f"\nOverall: {'ALL PASSED' if all_ok else 'SOME CHECKS FAILED'}")

    return all_ok
