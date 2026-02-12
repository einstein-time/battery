"""Tests for the 2D heat equation solver."""

import numpy as np
import pytest

from src.physics.solver import HeatSolver2D
from src.physics.materials import create_material_mask


def test_solver_initialization():
    """Test basic solver initialization."""
    solver = HeatSolver2D(
        nx=32,
        ny=32,
        dx=0.001,
        dy=0.001,
        dt=0.0001,
        k=2.0,
        rho=2500.0,
        cp=700.0,
        T_amb=300.0,
        h_conv=10.0,
    )

    assert solver.nx == 32
    assert solver.ny == 32
    assert solver.k.shape == (32, 32)


def test_stability_check():
    """Test CFL stability condition."""
    solver = HeatSolver2D(
        nx=32,
        ny=32,
        dx=0.001,
        dy=0.001,
        dt=0.0001,
        k=2.0,
        rho=2500.0,
        cp=700.0,
    )

    # Check that a reasonable time step is stable
    assert solver.check_stability()

    # Create an unstable solver (very large dt)
    solver_unstable = HeatSolver2D(
        nx=32,
        ny=32,
        dx=0.001,
        dy=0.001,
        dt=1.0,  # Way too large
        k=2.0,
        rho=2500.0,
        cp=700.0,
    )

    assert not solver_unstable.check_stability()


def test_solver_step():
    """Test single time step produces reasonable output."""
    solver = HeatSolver2D(
        nx=32,
        ny=32,
        dx=0.001,
        dy=0.001,
        dt=0.0001,
        k=2.0,
        rho=2500.0,
        cp=700.0,
        T_amb=300.0,
        h_conv=10.0,
    )

    # Initial condition: uniform temperature
    T0 = np.full((32, 32), 300.0)

    # Add heat source in center
    q = np.zeros((32, 32))
    q[14:18, 14:18] = 1e5

    # Take one step
    T1 = solver.step(T0, q)

    # Check that temperature increased in the center
    assert T1[16, 16] > T0[16, 16]

    # Check that temperatures are still physical
    assert T1.min() >= 250.0
    assert T1.max() <= 400.0


def test_solver_from_material_fields():
    """Test solver construction from material mask."""
    mask = create_material_mask(grid_size=32, layout="grid")

    k_values = np.array([2.0, 0.6, 0.04])
    rho_values = np.array([2500.0, 998.0, 30.0])
    cp_values = np.array([700.0, 4182.0, 1400.0])

    solver = HeatSolver2D.from_material_fields(
        mask=mask,
        k_values=k_values,
        rho_values=rho_values,
        cp_values=cp_values,
        dx=0.001,
        dy=0.001,
        dt=0.0001,
        T_amb=300.0,
        h_conv=10.0,
    )

    assert solver.k.shape == (32, 32)
    assert solver.rho.shape == (32, 32)
    assert solver.cp.shape == (32, 32)

    # Check that different materials have different properties
    assert solver.k[mask == 0].mean() != solver.k[mask == 1].mean()


def test_solver_trajectory():
    """Test multi-step simulation."""
    mask = create_material_mask(grid_size=32, layout="single")
    source_mask = (mask == 0).astype(np.float64)

    k_values = np.array([2.0, 0.6, 0.04])
    rho_values = np.array([2500.0, 998.0, 30.0])
    cp_values = np.array([700.0, 4182.0, 1400.0])

    solver = HeatSolver2D.from_material_fields(
        mask=mask,
        k_values=k_values,
        rho_values=rho_values,
        cp_values=cp_values,
        dx=0.001,
        dy=0.001,
        dt=0.0001,
        T_amb=300.0,
        h_conv=10.0,
    )

    T0 = np.full((32, 32), 300.0)

    trajectory = solver.solve(
        T0=T0,
        n_steps=100,
        q0=1e5,
        source_mask=source_mask,
        save_every=10,
    )

    assert trajectory.shape[0] == 10  # Saved every 10 steps
    assert trajectory.shape[1:] == (32, 32)

    # Temperature should increase over time with heat source
    assert trajectory[-1].max() > trajectory[0].max()
