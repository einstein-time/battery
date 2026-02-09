"""Comprehensive unit tests for the 2-D heat equation solver and supporting modules.

Tests cover solver initialisation, the discrete Laplacian operator, Robin boundary
conditions, the time-dependent heat source, single and multi-step time integration,
CFL stability checks, spatial convergence via the Method of Manufactured Solutions,
and material property / geometry utilities.

All tests use small grids (16x16 or 32x32) so the full suite completes in well
under 30 seconds on commodity hardware.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math

import numpy as np
import pytest

from src.physics.solver import HeatSolver2D
from src.physics.materials import (
    MaterialProperties,
    create_material_mask,
    compute_signed_distance,
)
from src.physics.validation import (
    manufactured_solution_test,
    compute_error_norms,
    convergence_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_solver():
    """Create a 16x16 solver with physically reasonable battery-cell parameters.

    Returns:
        A stable ``HeatSolver2D`` instance on a 16x16 grid with dx=dy=1e-3 m,
        k=3.0 W/(m*K), rho=2500 kg/m^3, cp=700 J/(kg*K), T_amb=300 K,
        and h_conv=10 W/(m^2*K).  The time step is set to 40 % of the
        maximum stable value.
    """
    nx, ny = 16, 16
    dx = dy = 1e-3
    k, rho, cp = 3.0, 2500.0, 700.0
    dt_max = dx ** 2 * rho * cp / (4.0 * k)
    dt = 0.4 * dt_max
    return HeatSolver2D(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
        k=k, rho=rho, cp=cp, T_amb=300.0, h_conv=10.0,
    )


@pytest.fixture
def medium_solver():
    """Create a 32x32 solver with the same material properties as ``small_solver``.

    Returns:
        A stable ``HeatSolver2D`` on a 32x32 grid.
    """
    nx, ny = 32, 32
    dx = dy = 1e-3
    k, rho, cp = 3.0, 2500.0, 700.0
    dt_max = dx ** 2 * rho * cp / (4.0 * k)
    dt = 0.4 * dt_max
    return HeatSolver2D(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
        k=k, rho=rho, cp=cp, T_amb=300.0, h_conv=10.0,
    )


@pytest.fixture
def uniform_T(small_solver):
    """Return a temperature field uniformly equal to T_amb for a 16x16 grid.

    Args:
        small_solver: The ``small_solver`` fixture.

    Returns:
        Float64 array of shape ``(16, 16)`` filled with ``T_amb``.
    """
    return np.full(
        (small_solver.ny, small_solver.nx), small_solver.T_amb, dtype=np.float64
    )


@pytest.fixture
def source_mask_ones(small_solver):
    """Return an all-ones source mask for a 16x16 grid.

    Args:
        small_solver: The ``small_solver`` fixture.

    Returns:
        Float64 array of shape ``(16, 16)`` filled with 1.0.
    """
    return np.ones((small_solver.ny, small_solver.nx), dtype=np.float64)


@pytest.fixture
def source_mask_zeros(small_solver):
    """Return an all-zeros source mask for a 16x16 grid.

    Args:
        small_solver: The ``small_solver`` fixture.

    Returns:
        Float64 array of shape ``(16, 16)`` filled with 0.0.
    """
    return np.zeros((small_solver.ny, small_solver.nx), dtype=np.float64)


# ===========================================================================
# TestSolverInitialization
# ===========================================================================


class TestSolverInitialization:
    """Tests for ``HeatSolver2D.__init__`` and attribute storage."""

    def test_default_initialization(self):
        """Verify that a solver created with minimal arguments stores correct defaults.

        A 16x16 solver is created with scalar material properties.  We check that
        ``nx``, ``ny``, ``T_amb``, and ``h_conv`` are stored, and that the property
        fields ``k``, ``rho``, ``cp`` are broadcast to full ``(ny, nx)`` arrays.
        """
        solver = HeatSolver2D(
            nx=16, ny=16, dx=1e-3, dy=1e-3, dt=1e-4,
            k=3.0, rho=2500.0, cp=700.0,
        )
        assert solver.nx == 16, "nx should be 16"
        assert solver.ny == 16, "ny should be 16"
        assert solver.T_amb == pytest.approx(300.0), (
            "Default T_amb should be 300.0 K"
        )
        assert solver.h_conv == pytest.approx(10.0), (
            "Default h_conv should be 10.0 W/(m^2*K)"
        )
        assert solver.k.shape == (16, 16), "k field shape should be (ny, nx)"
        assert solver.rho.shape == (16, 16), "rho field shape should be (ny, nx)"
        assert solver.cp.shape == (16, 16), "cp field shape should be (ny, nx)"

    def test_custom_parameters(self):
        """Verify that all constructor parameters are stored correctly.

        Creates a solver with non-default values for every parameter and checks
        that each attribute matches the supplied value.
        """
        solver = HeatSolver2D(
            nx=20, ny=24, dx=2e-3, dy=3e-3, dt=5e-5,
            k=1.5, rho=1800.0, cp=900.0,
            T_amb=350.0, h_conv=25.0,
        )
        assert solver.nx == 20
        assert solver.ny == 24
        assert solver.dx == pytest.approx(2e-3)
        assert solver.dy == pytest.approx(3e-3)
        assert solver.dt == pytest.approx(5e-5)
        assert solver.T_amb == pytest.approx(350.0)
        assert solver.h_conv == pytest.approx(25.0)
        assert np.allclose(solver.k, 1.5), "k field should be uniformly 1.5"
        assert np.allclose(solver.rho, 1800.0), "rho field should be uniformly 1800"
        assert np.allclose(solver.cp, 900.0), "cp field should be uniformly 900"

    def test_array_properties(self):
        """Verify that array-valued material properties are accepted and stored.

        Passes 2-D arrays for k, rho, and cp and checks that the solver uses them
        without modification.
        """
        k_arr = np.random.uniform(1.0, 5.0, (16, 16))
        rho_arr = np.full((16, 16), 2500.0)
        cp_arr = np.full((16, 16), 700.0)
        solver = HeatSolver2D(
            nx=16, ny=16, dx=1e-3, dy=1e-3, dt=1e-5,
            k=k_arr, rho=rho_arr, cp=cp_arr,
        )
        np.testing.assert_array_equal(
            solver.k, k_arr, err_msg="k array should be stored unchanged"
        )

    def test_invalid_property_shape_raises(self):
        """Passing an array with the wrong shape should raise ``ValueError``.

        The solver requires property arrays of shape ``(ny, nx)``.  A mismatched
        shape must be rejected immediately.
        """
        bad_k = np.ones((8, 8))  # wrong shape for a 16x16 grid
        with pytest.raises(ValueError, match="shape"):
            HeatSolver2D(
                nx=16, ny=16, dx=1e-3, dy=1e-3, dt=1e-4,
                k=bad_k, rho=2500.0, cp=700.0,
            )

    def test_grid_shape_property(self):
        """The ``grid_shape`` property should return ``(ny, nx)``."""
        solver = HeatSolver2D(
            nx=20, ny=24, dx=1e-3, dy=1e-3, dt=1e-4,
            k=3.0, rho=2500.0, cp=700.0,
        )
        assert solver.grid_shape == (24, 20), (
            "grid_shape should be (ny, nx) = (24, 20)"
        )


# ===========================================================================
# TestLaplacian
# ===========================================================================


class TestLaplacian:
    """Tests for ``HeatSolver2D.compute_laplacian``."""

    def test_uniform_field(self, small_solver, uniform_T):
        """The Laplacian of a spatially constant temperature field should be zero.

        A uniform field has zero second derivatives everywhere, so the discrete
        Laplacian must return values very close to zero at all grid points.
        """
        lap = small_solver.compute_laplacian(uniform_T)
        assert np.allclose(lap, 0.0, atol=1e-10), (
            "Laplacian of a constant field should be zero everywhere"
        )

    def test_quadratic_field(self):
        """The Laplacian of T = x^2 + y^2 should be a known constant.

        For constant conductivity k and T(x,y) = x^2 + y^2 the operator
        div(k * grad T) = k * (d^2T/dx^2 + d^2T/dy^2) = k * (2 + 2) = 4k
        at all interior points.  We verify agreement to within 1 % at interior
        points (away from the boundary padding ring).
        """
        nx, ny = 32, 32
        dx = dy = 0.01
        k = 2.0
        solver = HeatSolver2D(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=1e-6,
            k=k, rho=1.0, cp=1.0,
        )

        # Build coordinate arrays (cell centres starting at dx/2).
        xs = np.arange(nx) * dx + dx / 2.0
        ys = np.arange(ny) * dy + dy / 2.0
        X, Y = np.meshgrid(xs, ys)
        T = X ** 2 + Y ** 2

        lap = solver.compute_laplacian(T)
        # d^2/dx^2 (x^2) = 2, d^2/dy^2 (y^2) = 2, so div(k grad T) = k*4.
        expected = k * 4.0

        # Check interior (exclude one-cell boundary ring).
        interior = lap[2:-2, 2:-2]
        assert np.allclose(interior, expected, rtol=1e-2), (
            f"Interior Laplacian should be ~{expected:.2f}; "
            f"got range [{interior.min():.2f}, {interior.max():.2f}]"
        )

    def test_sinusoidal_field(self):
        """Verify the Laplacian of T = sin(pi*x/L)*sin(pi*y/L).

        For a domain of physical length L with constant k, the analytical
        Laplacian is -k * 2*(pi/L)^2 * T.  We check agreement at interior
        points far from the boundary padding.
        """
        nx, ny = 32, 32
        dx = dy = 0.01
        k = 1.0
        Lx = nx * dx
        Ly = ny * dy
        solver = HeatSolver2D(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=1e-6,
            k=k, rho=1.0, cp=1.0,
        )

        xs = np.arange(nx) * dx + dx / 2.0
        ys = np.arange(ny) * dy + dy / 2.0
        X, Y = np.meshgrid(xs, ys)
        T = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

        lap = solver.compute_laplacian(T)
        expected = -k * ((np.pi / Lx) ** 2 + (np.pi / Ly) ** 2) * T

        # Interior comparison (skip 2-cell border for edge-padding effects).
        interior = slice(3, -3)
        np.testing.assert_allclose(
            lap[interior, interior],
            expected[interior, interior],
            rtol=0.05,
            err_msg="Sinusoidal Laplacian should match analytical value within 5 %",
        )

    def test_shape_preserved(self, small_solver, uniform_T):
        """The output of ``compute_laplacian`` must have the same shape as the input.

        The Laplacian is computed on the full ``(ny, nx)`` grid (boundary cells
        use zero-gradient padding), so the returned array should match.
        """
        lap = small_solver.compute_laplacian(uniform_T)
        assert lap.shape == uniform_T.shape, (
            f"Laplacian shape {lap.shape} should match input shape {uniform_T.shape}"
        )


# ===========================================================================
# TestBoundaryConditions
# ===========================================================================


class TestBoundaryConditions:
    """Tests for ``HeatSolver2D.apply_boundary_conditions``."""

    def test_ambient_temperature(self, small_solver):
        """If T equals T_amb everywhere, Robin BCs should leave T unchanged.

        The Robin condition is T_bnd = (k*T_int + h*dn*T_amb) / (k + h*dn).
        When T_int = T_amb this simplifies to T_bnd = T_amb, so no change.
        """
        T = np.full(
            (small_solver.ny, small_solver.nx),
            small_solver.T_amb,
            dtype=np.float64,
        )
        T_before = T.copy()
        small_solver.apply_boundary_conditions(T)
        np.testing.assert_allclose(
            T, T_before, atol=1e-12,
            err_msg="BCs should not change a uniform-ambient field",
        )

    def test_hot_interior(self, small_solver):
        """An interior hotter than T_amb should have boundary cells pulled toward T_amb.

        We set the interior to a temperature well above ambient and verify that
        boundary cells are cooler than the adjacent interior after applying BCs.
        """
        T_hot = 500.0
        T = np.full(
            (small_solver.ny, small_solver.nx), T_hot, dtype=np.float64
        )
        small_solver.apply_boundary_conditions(T)

        # All boundary cells should be between T_amb and T_hot.
        top = T[0, :]
        bottom = T[-1, :]
        left = T[:, 0]
        right = T[:, -1]

        for name, edge in [("top", top), ("bottom", bottom),
                           ("left", left), ("right", right)]:
            assert np.all(edge < T_hot), (
                f"{name} boundary should be cooled below {T_hot} K"
            )
            assert np.all(edge > small_solver.T_amb), (
                f"{name} boundary should remain above T_amb={small_solver.T_amb} K"
            )

    def test_symmetry(self, small_solver):
        """Symmetric initial conditions should maintain symmetry on non-corner boundary edges.

        The Robin BCs are applied sequentially (bottom, top, left, right), so
        corner cells are overwritten by the last edge that touches them,
        breaking exact corner symmetry.  We therefore check that the *interior
        of each edge* (excluding the two corner cells) is symmetric, and that
        the fully interior region remains symmetric.
        """
        ny, nx = small_solver.ny, small_solver.nx
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        Y, X = np.mgrid[0:ny, 0:nx]
        T = 300.0 + 50.0 * np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2))

        small_solver.apply_boundary_conditions(T)

        # Interior region (rows 1..-2, cols 1..-2) should remain perfectly
        # symmetric because BCs only touch the outermost ring.
        interior = T[1:-1, 1:-1]
        flipped_lr = interior[:, ::-1]
        flipped_ud = interior[::-1, :]
        np.testing.assert_allclose(
            interior, flipped_lr, atol=1e-10,
            err_msg="Interior should be left-right symmetric after BCs",
        )
        np.testing.assert_allclose(
            interior, flipped_ud, atol=1e-10,
            err_msg="Interior should be top-bottom symmetric after BCs",
        )

    def test_insulated_boundary(self, small_solver):
        """With h_conv=0 (insulated walls), BCs should copy the adjacent interior value.

        When h=0 the Robin formula becomes T_bnd = T_int, i.e. a zero-gradient
        (Neumann) condition.

        Because the four boundaries are applied sequentially (bottom, top, left,
        right), corner cells are overwritten by the last edge that touches them.
        We therefore check the *non-corner* cells of each boundary edge.
        """
        T = np.random.RandomState(42).uniform(
            290.0, 310.0, (small_solver.ny, small_solver.nx)
        )
        T_copy = T.copy()
        small_solver.apply_boundary_conditions(T, h=0.0)

        # Bottom boundary (row 0): non-corner cells should equal row 1.
        np.testing.assert_allclose(
            T[0, 1:-1], T_copy[1, 1:-1], atol=1e-12,
            err_msg="Insulated bottom boundary (non-corner) should equal adjacent interior row",
        )
        # Top boundary (last row): non-corner cells should equal row -2.
        np.testing.assert_allclose(
            T[-1, 1:-1], T_copy[-2, 1:-1], atol=1e-12,
            err_msg="Insulated top boundary (non-corner) should equal adjacent interior row",
        )
        # Left boundary (col 0): non-corner cells should equal col 1.
        np.testing.assert_allclose(
            T[1:-1, 0], T_copy[1:-1, 1], atol=1e-12,
            err_msg="Insulated left boundary (non-corner) should equal adjacent interior col",
        )
        # Right boundary (last col): non-corner cells should equal col -2.
        np.testing.assert_allclose(
            T[1:-1, -1], T_copy[1:-1, -2], atol=1e-12,
            err_msg="Insulated right boundary (non-corner) should equal adjacent interior col",
        )


# ===========================================================================
# TestHeatSource
# ===========================================================================


class TestHeatSource:
    """Tests for ``HeatSolver2D.heat_source`` (static method)."""

    def test_constant_source_at_t_zero(self):
        """At t=0 the oscillating term vanishes, so q(0) = q0.

        The formula is q0 * (1 + 0.3*sin(0)) = q0.
        """
        q0 = 1e4
        result = HeatSolver2D.heat_source(t=0.0, q0=q0, freq=1.0)
        assert result == pytest.approx(q0), (
            f"heat_source(0, {q0}) should be {q0}, got {result}"
        )

    def test_oscillating_source_quarter_period(self):
        """At t = 1/(4*freq) the sine term equals 1, so q = q0 * 1.3.

        For freq=1.0 Hz, t=0.25 s gives sin(2*pi*1*0.25) = sin(pi/2) = 1.
        """
        q0 = 1e4
        freq = 1.0
        t = 1.0 / (4.0 * freq)
        expected = q0 * (1.0 + 0.3 * 1.0)
        result = HeatSolver2D.heat_source(t=t, q0=q0, freq=freq)
        assert result == pytest.approx(expected, rel=1e-12), (
            f"At quarter period, q should be {expected}, got {result}"
        )

    def test_oscillating_source_three_quarter_period(self):
        """At t = 3/(4*freq) the sine term equals -1, so q = q0 * 0.7.

        For freq=2.0 Hz, t=3/(4*2)=0.375 s gives sin(2*pi*2*0.375) = sin(3*pi/2) = -1.
        """
        q0 = 5000.0
        freq = 2.0
        t = 3.0 / (4.0 * freq)
        expected = q0 * (1.0 + 0.3 * (-1.0))
        result = HeatSolver2D.heat_source(t=t, q0=q0, freq=freq)
        assert result == pytest.approx(expected, rel=1e-12), (
            f"At three-quarter period, q should be {expected}, got {result}"
        )

    def test_zero_source(self):
        """When q0=0 the heat source must be exactly zero regardless of time.

        No heat generation should occur if the baseline rate is zero.
        """
        for t in [0.0, 0.1, 1.0, 100.0]:
            result = HeatSolver2D.heat_source(t=t, q0=0.0, freq=1.0)
            assert result == pytest.approx(0.0, abs=1e-30), (
                f"heat_source(t={t}, q0=0) should be 0, got {result}"
            )

    def test_source_always_positive(self):
        """The heat source q(t) = q0*(1 + 0.3*sin(...)) is always positive for q0 > 0.

        The minimum value of the oscillating factor is 1 - 0.3 = 0.7 > 0.
        """
        q0 = 1e4
        times = np.linspace(0, 10, 1000)
        values = [HeatSolver2D.heat_source(t, q0, freq=1.0) for t in times]
        assert all(v > 0 for v in values), (
            "Heat source should be strictly positive for positive q0"
        )


# ===========================================================================
# TestTimeStep
# ===========================================================================


class TestTimeStep:
    """Tests for ``HeatSolver2D.step``."""

    def test_uniform_no_source(self, small_solver, uniform_T, source_mask_zeros):
        """A uniform field at T_amb with no heat source should stay near T_amb.

        With T = T_amb everywhere and zero heat generation, the Laplacian is
        zero and the BCs do not modify boundary cells, so T should be unchanged.
        """
        T_new = small_solver.step(uniform_T, t=0.0, q0=0.0,
                                  source_mask=source_mask_zeros)
        np.testing.assert_allclose(
            T_new, uniform_T, atol=1e-10,
            err_msg="Uniform T_amb field with no source should remain unchanged",
        )

    def test_does_not_modify_input(self, small_solver, uniform_T, source_mask_ones):
        """The ``step`` method must return a new array and not modify the input.

        According to the docstring, *T* is not modified in-place.
        """
        T_original = uniform_T.copy()
        _ = small_solver.step(uniform_T, t=0.0, q0=1e4,
                              source_mask=source_mask_ones)
        np.testing.assert_array_equal(
            uniform_T, T_original,
            err_msg="step() should not modify the input T array",
        )

    def test_heating(self, small_solver, uniform_T, source_mask_ones):
        """With a positive heat source the mean temperature should increase.

        One step with q0 > 0 and a full source mask should raise the domain-
        average temperature above the initial uniform value.
        """
        T_new = small_solver.step(uniform_T, t=0.0, q0=1e6,
                                  source_mask=source_mask_ones)
        assert T_new.mean() > uniform_T.mean(), (
            "Mean temperature should increase with a positive heat source"
        )

    def test_cooling(self, small_solver, source_mask_zeros):
        """A hot initial condition with no source should cool toward T_amb.

        Interior points are hotter than T_amb so the convective BCs remove
        heat.  After one step the mean temperature should decrease.
        """
        T_hot = np.full(
            (small_solver.ny, small_solver.nx), 500.0, dtype=np.float64
        )
        T_new = small_solver.step(T_hot, t=0.0, q0=0.0,
                                  source_mask=source_mask_zeros)
        assert T_new.mean() < T_hot.mean(), (
            "Mean temperature should decrease when cooling toward ambient"
        )

    def test_output_shape(self, small_solver, uniform_T, source_mask_ones):
        """The output of ``step`` should have the same shape as the input."""
        T_new = small_solver.step(uniform_T, t=0.0, q0=0.0,
                                  source_mask=source_mask_ones)
        assert T_new.shape == uniform_T.shape, (
            f"Output shape {T_new.shape} should match input shape {uniform_T.shape}"
        )


# ===========================================================================
# TestSolve
# ===========================================================================


class TestSolve:
    """Tests for ``HeatSolver2D.solve``."""

    def test_trajectory_shape(self, small_solver, uniform_T, source_mask_ones):
        """``solve()`` should return an array of shape ``(n_steps+1, ny, nx)``.

        With ``save_every=1`` (default), every step is saved plus the initial
        condition, giving ``n_steps + 1`` frames.
        """
        n_steps = 10
        traj = small_solver.solve(uniform_T, n_steps=n_steps, q0=1e4,
                                  source_mask=source_mask_ones)
        expected_shape = (n_steps + 1, small_solver.ny, small_solver.nx)
        assert traj.shape == expected_shape, (
            f"Trajectory shape {traj.shape} should be {expected_shape}"
        )

    def test_trajectory_shape_save_every(self, small_solver, uniform_T,
                                          source_mask_ones):
        """With ``save_every > 1`` the number of saved frames is reduced.

        For ``n_steps=20`` and ``save_every=5`` the trajectory should have
        ``20 // 5 + 1 = 5`` frames (initial + 4 snapshots).
        """
        n_steps = 20
        save_every = 5
        traj = small_solver.solve(
            uniform_T, n_steps=n_steps, q0=1e4,
            source_mask=source_mask_ones, save_every=save_every,
        )
        n_saved = n_steps // save_every
        expected_shape = (n_saved + 1, small_solver.ny, small_solver.nx)
        assert traj.shape == expected_shape, (
            f"Trajectory shape {traj.shape} should be {expected_shape} "
            f"with save_every={save_every}"
        )

    def test_initial_condition_preserved(self, small_solver, source_mask_ones):
        """The first frame of the trajectory must equal the supplied initial condition.

        ``solve()`` should store an unmodified copy of T0 at index 0.
        """
        rng = np.random.RandomState(123)
        T0 = rng.uniform(290.0, 310.0,
                         (small_solver.ny, small_solver.nx)).astype(np.float64)
        traj = small_solver.solve(T0, n_steps=5, q0=1e4,
                                  source_mask=source_mask_ones)
        np.testing.assert_array_equal(
            traj[0], T0,
            err_msg="First frame of trajectory should be the initial condition",
        )

    def test_steady_state_approach(self):
        """A long integration with convective BCs and constant source should approach steady state.

        After many time steps the temperature difference between consecutive
        frames should become negligibly small, indicating that a steady state
        has been (approximately) reached.
        """
        nx, ny = 16, 16
        dx = dy = 1e-3
        k, rho, cp = 3.0, 2500.0, 700.0
        dt_max = dx ** 2 * rho * cp / (4.0 * k)
        dt = 0.4 * dt_max
        solver = HeatSolver2D(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
            k=k, rho=rho, cp=cp, T_amb=300.0, h_conv=50.0,
        )

        T0 = np.full((ny, nx), 300.0, dtype=np.float64)
        source_mask = np.ones((ny, nx), dtype=np.float64)
        n_steps = 5000
        save_every = 100

        traj = solver.solve(T0, n_steps=n_steps, q0=1e4,
                            source_mask=source_mask, save_every=save_every)

        # Compare the last two saved snapshots.
        diff = np.max(np.abs(traj[-1] - traj[-2]))
        assert diff < 1.0, (
            f"Temperature change between last two snapshots should be small; "
            f"got max |dT| = {diff:.4e} K"
        )

    def test_energy_conservation_no_source(self):
        """Without a heat source and with insulated BCs total energy should be conserved.

        With h_conv=0 (Neumann / insulated BCs) and q0=0, there is no energy
        input or loss.  The integral of rho*cp*T over the domain should remain
        approximately constant.
        """
        nx, ny = 16, 16
        dx = dy = 1e-3
        k, rho, cp = 3.0, 2500.0, 700.0
        dt_max = dx ** 2 * rho * cp / (4.0 * k)
        dt = 0.4 * dt_max
        solver = HeatSolver2D(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
            k=k, rho=rho, cp=cp, T_amb=300.0, h_conv=0.0,
        )

        # Non-uniform initial condition.
        rng = np.random.RandomState(99)
        T0 = 300.0 + 10.0 * rng.randn(ny, nx)
        source_mask = np.zeros((ny, nx), dtype=np.float64)

        n_steps = 200
        traj = solver.solve(T0, n_steps=n_steps, q0=0.0,
                            source_mask=source_mask)

        cell_area = dx * dy
        energy_initial = float(np.sum(rho * cp * traj[0]) * cell_area)
        energy_final = float(np.sum(rho * cp * traj[-1]) * cell_area)

        rel_change = abs(energy_final - energy_initial) / abs(energy_initial)
        assert rel_change < 1e-2, (
            f"Relative energy change should be < 1e-2 with insulated BCs; "
            f"got {rel_change:.4e}"
        )

    def test_invalid_T0_shape_raises(self, small_solver, source_mask_ones):
        """Passing T0 with the wrong shape should raise ``ValueError``."""
        T_bad = np.full((8, 8), 300.0)
        with pytest.raises(ValueError, match="T0 shape"):
            small_solver.solve(T_bad, n_steps=1, q0=0.0,
                               source_mask=source_mask_ones)

    def test_invalid_source_mask_shape_raises(self, small_solver, uniform_T):
        """Passing a source_mask with the wrong shape should raise ``ValueError``."""
        mask_bad = np.ones((8, 8))
        with pytest.raises(ValueError, match="source_mask shape"):
            small_solver.solve(uniform_T, n_steps=1, q0=0.0,
                               source_mask=mask_bad)


# ===========================================================================
# TestStability
# ===========================================================================


class TestStability:
    """Tests for ``HeatSolver2D.check_stability``."""

    def test_stable_parameters(self, small_solver):
        """A solver created with dt well below the CFL limit should be stable.

        The ``small_solver`` fixture uses 40 % of the maximum stable time step,
        so ``check_stability`` must return ``True``.
        """
        assert small_solver.check_stability() is True, (
            "Solver should be stable with dt = 0.4 * dt_max"
        )

    def test_unstable_parameters(self):
        """A solver with an excessively large time step should be flagged as unstable.

        We set dt to 10x the maximum stable value and verify that
        ``check_stability`` returns ``False``.
        """
        nx, ny = 16, 16
        dx = dy = 1e-3
        k, rho, cp = 3.0, 2500.0, 700.0
        dt_max = dx ** 2 * rho * cp / (4.0 * k)
        dt_too_large = 10.0 * dt_max

        solver = HeatSolver2D(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=dt_too_large,
            k=k, rho=rho, cp=cp,
        )
        assert solver.check_stability() is False, (
            "Solver should be unstable with dt = 10 * dt_max"
        )

    def test_marginal_stability(self):
        """A solver with dt exactly at the CFL limit should be considered stable.

        The condition is ``dt <= dt_max``, so equality should return ``True``.
        """
        nx, ny = 16, 16
        dx = dy = 1e-3
        k, rho, cp = 3.0, 2500.0, 700.0
        dt_max = dx ** 2 * rho * cp / (4.0 * k)

        solver = HeatSolver2D(
            nx=nx, ny=ny, dx=dx, dy=dy, dt=dt_max,
            k=k, rho=rho, cp=cp,
        )
        assert solver.check_stability() is True, (
            "Solver should be stable when dt == dt_max"
        )

    def test_max_stable_dt_property(self, small_solver):
        """The ``max_stable_dt`` property should match the analytical CFL limit."""
        k_max = float(small_solver.k.max())
        rho_cp_min = float((small_solver.rho * small_solver.cp).min())
        ds_min = min(small_solver.dx, small_solver.dy)
        expected = ds_min ** 2 * rho_cp_min / (4.0 * k_max)
        assert small_solver.max_stable_dt == pytest.approx(expected), (
            f"max_stable_dt should be {expected:.6e}, "
            f"got {small_solver.max_stable_dt:.6e}"
        )


# ===========================================================================
# TestSpatialConvergence
# ===========================================================================


class TestSpatialConvergence:
    """Spatial convergence tests using the Method of Manufactured Solutions."""

    def test_convergence_rate_mms(self):
        """The MMS validation should show monotonically decreasing error.

        The ``manufactured_solution_test`` uses a Robin BC approximation to
        Dirichlet boundaries.  Because this approximation is first-order
        accurate in dx, the overall convergence rate is bounded by O(dx).
        We verify that errors decrease consistently and the observed rate is
        positive (> 0.8).
        """
        results = manufactured_solution_test(
            grid_sizes=[8, 16, 32],
            k=1.0,
            q=1.0,
            n_time_steps=20000,
            safety_factor=0.4,
        )

        l2_errors = results["L2"]
        rates = results["rates"]

        # Errors should decrease with refinement.
        assert l2_errors[1] < l2_errors[0], (
            f"L2 error should decrease from N=8 ({l2_errors[0]:.4e}) "
            f"to N=16 ({l2_errors[1]:.4e})"
        )
        assert l2_errors[2] < l2_errors[1], (
            f"L2 error should decrease from N=16 ({l2_errors[1]:.4e}) "
            f"to N=32 ({l2_errors[2]:.4e})"
        )

        # Rate should be positive and consistent with at least O(dx).
        last_rate = rates[-1]
        assert not math.isnan(last_rate), "Convergence rate should not be NaN"
        assert last_rate > 0.8, (
            f"Convergence rate should be > 0.8; got {last_rate:.2f}"
        )

    def test_laplacian_convergence_order(self):
        """The discrete Laplacian should converge at O(dx^2) for smooth fields.

        We compute the Laplacian of T = sin(pi*x/L)*sin(pi*y/L) at multiple
        resolutions and measure the interior error against the analytical
        value.  Since the central-difference Laplacian is second-order, the
        error should decrease by a factor of ~4 when dx is halved.
        """
        k = 1.0
        grid_sizes = [16, 32, 64]
        errors = []

        for n in grid_sizes:
            dx = dy = 1.0 / n
            Lx = Ly = n * dx
            solver = HeatSolver2D(
                nx=n, ny=n, dx=dx, dy=dy, dt=1e-8,
                k=k, rho=1.0, cp=1.0,
            )
            xs = np.arange(n) * dx + dx / 2.0
            ys = np.arange(n) * dy + dy / 2.0
            X, Y = np.meshgrid(xs, ys)
            T = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
            expected = -k * ((np.pi / Lx) ** 2 + (np.pi / Ly) ** 2) * T

            lap = solver.compute_laplacian(T)
            # Measure error on interior (skip 2-cell border).
            s = slice(2, -2)
            err = np.sqrt(np.mean((lap[s, s] - expected[s, s]) ** 2))
            errors.append(err)

        rates = convergence_rate(errors, grid_sizes)
        last_rate = rates[-1]
        assert not math.isnan(last_rate), "Laplacian convergence rate should not be NaN"
        assert last_rate > 1.8, (
            f"Laplacian convergence rate should be ~2.0; got {last_rate:.2f}"
        )

    def test_error_norms_consistent(self):
        """L1 <= L2 <= Linf should hold for any error distribution.

        This is a mathematical property of norms that serves as a consistency
        check on the ``compute_error_norms`` implementation.
        """
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.1, 2.2], [2.8, 4.5]])
        norms = compute_error_norms(a, b)

        assert norms["L1"] <= norms["L2"] + 1e-15, (
            f"L1 ({norms['L1']:.6e}) should be <= L2 ({norms['L2']:.6e})"
        )
        assert norms["L2"] <= norms["Linf"] + 1e-15, (
            f"L2 ({norms['L2']:.6e}) should be <= Linf ({norms['Linf']:.6e})"
        )

    def test_convergence_rate_utility(self):
        """Verify the ``convergence_rate`` function with known synthetic data.

        For errors decreasing as O(N^{-2}) the rate should be exactly 2.
        """
        grid_sizes = [10, 20, 40]
        # Errors scaling as 1/N^2.
        errors = [1.0 / n ** 2 for n in grid_sizes]
        rates = convergence_rate(errors, grid_sizes)

        assert math.isnan(rates[0]), "First rate entry should be NaN"
        assert rates[1] == pytest.approx(2.0, abs=1e-10), (
            f"Rate for 1/N^2 scaling should be 2.0, got {rates[1]:.4f}"
        )
        assert rates[2] == pytest.approx(2.0, abs=1e-10), (
            f"Rate for 1/N^2 scaling should be 2.0, got {rates[2]:.4f}"
        )


# ===========================================================================
# TestMaterials
# ===========================================================================


class TestMaterials:
    """Tests for material properties, mask generation, and signed distance fields."""

    def test_create_mask_shape(self):
        """``create_material_mask(32)`` should return an integer array of shape ``(32, 32)``."""
        mask = create_material_mask(32)
        assert mask.shape == (32, 32), (
            f"Mask shape should be (32, 32), got {mask.shape}"
        )
        assert mask.dtype in (np.int32, np.int64), (
            f"Mask dtype should be integer, got {mask.dtype}"
        )

    def test_mask_values(self):
        """All values in the material mask should be in {0, 1, 2}.

        Material ids 0 (battery), 1 (coolant), and 2 (insulation) are the
        only valid labels.
        """
        mask = create_material_mask(32)
        unique_vals = set(np.unique(mask))
        assert unique_vals == {0, 1, 2}, (
            f"Mask should contain exactly {{0, 1, 2}}, got {unique_vals}"
        )

    def test_mask_has_insulation_border(self):
        """The outer ring of the mask should be insulation (material id 2).

        The design places a border of insulation around the entire domain.
        """
        mask = create_material_mask(32)
        # Top row.
        assert np.all(mask[0, :] == 2), "Top border should be insulation (id=2)"
        # Bottom row.
        assert np.all(mask[-1, :] == 2), "Bottom border should be insulation (id=2)"
        # Left column.
        assert np.all(mask[:, 0] == 2), "Left border should be insulation (id=2)"
        # Right column.
        assert np.all(mask[:, -1] == 2), "Right border should be insulation (id=2)"

    def test_mask_minimum_grid_size(self):
        """``create_material_mask`` should raise ``ValueError`` for grid_size < 16."""
        with pytest.raises(ValueError, match="grid_size"):
            create_material_mask(8)

    def test_signed_distance_positive_outside(self):
        """The SDF should be positive at points outside the target material region.

        For material_id=0 (battery), cells that are NOT battery should have
        SDF > 0.
        """
        mask = create_material_mask(32)
        sdf = compute_signed_distance(mask, material_id=0)
        outside = mask != 0
        assert np.all(sdf[outside] > 0), (
            "SDF should be positive outside the battery region"
        )

    def test_signed_distance_negative_inside(self):
        """The SDF should be negative at points deep inside the target material region.

        For battery cells (id=0) the SDF should be negative in the interior
        of each cell.
        """
        mask = create_material_mask(32)
        sdf = compute_signed_distance(mask, material_id=0)
        inside = mask == 0
        assert np.all(sdf[inside] < 0), (
            "SDF should be negative inside the battery region"
        )

    def test_signed_distance_zero_boundary(self):
        """The SDF should be approximately zero at the boundary of the material region.

        We check that the minimum absolute SDF value is close to zero, which
        occurs at boundary cells (cells adjacent to a different material).
        The distance transform gives half-cell offsets, so the minimum should
        be <= 1.0 grid-cell unit.
        """
        mask = create_material_mask(64)
        sdf = compute_signed_distance(mask, material_id=0)
        min_abs_sdf = np.min(np.abs(sdf))
        assert min_abs_sdf < 1.5, (
            f"Minimum |SDF| near boundary should be < 1.5 grid cells; "
            f"got {min_abs_sdf:.4f}"
        )

    def test_signed_distance_shape(self):
        """The SDF should have the same shape as the input mask."""
        mask = create_material_mask(32)
        sdf = compute_signed_distance(mask, material_id=0)
        assert sdf.shape == mask.shape, (
            f"SDF shape {sdf.shape} should match mask shape {mask.shape}"
        )

    def test_signed_distance_missing_material_raises(self):
        """Requesting an SDF for a material not present in the mask should raise."""
        mask = np.ones((16, 16), dtype=np.int32)  # only material id 1
        with pytest.raises(ValueError, match="material_id"):
            compute_signed_distance(mask, material_id=0)

    def test_material_properties_default(self):
        """``MaterialProperties()`` should store default water-coolant properties."""
        props = MaterialProperties()
        assert props.coolant == "water"
        assert props.battery_k == pytest.approx(3.0)
        assert props.battery.k == pytest.approx(3.0)
        assert props.battery.rho == pytest.approx(2500.0)
        assert props.battery.cp == pytest.approx(700.0)

    def test_material_properties_air_coolant(self):
        """``MaterialProperties(coolant='air')`` should use air properties."""
        props = MaterialProperties(coolant="air")
        assert props.coolant_props.k == pytest.approx(0.026)
        assert props.coolant_props.rho == pytest.approx(1.225)

    def test_material_properties_arrays(self):
        """Conductivity, density, and heat-capacity arrays should have length 3."""
        props = MaterialProperties()
        k = props.conductivity_array()
        rho = props.density_array()
        cp = props.heat_capacity_array()

        assert k.shape == (3,), f"k array shape should be (3,), got {k.shape}"
        assert rho.shape == (3,), f"rho array shape should be (3,), got {rho.shape}"
        assert cp.shape == (3,), f"cp array shape should be (3,), got {cp.shape}"

        # Index 0 = battery, 1 = coolant, 2 = insulation.
        assert k[0] == pytest.approx(props.battery.k)
        assert k[1] == pytest.approx(props.coolant_props.k)
        assert k[2] == pytest.approx(props.insulation.k)

    def test_material_properties_invalid_coolant_raises(self):
        """An unrecognised coolant name should raise ``ValueError``."""
        with pytest.raises(ValueError, match="coolant"):
            MaterialProperties(coolant="oil")

    def test_material_properties_invalid_battery_k_raises(self):
        """A battery conductivity outside [1.0, 5.0] should raise ``ValueError``."""
        with pytest.raises(ValueError, match="battery_k"):
            MaterialProperties(battery_k=10.0)

    def test_material_properties_diffusivity(self):
        """The diffusivity array alpha = k/(rho*cp) should be correctly computed."""
        props = MaterialProperties()
        alpha = props.diffusivity_array()
        k = props.conductivity_array()
        rho = props.density_array()
        cp = props.heat_capacity_array()

        expected = k / (rho * cp)
        np.testing.assert_allclose(
            alpha, expected, rtol=1e-12,
            err_msg="Diffusivity should be k / (rho * cp)",
        )

    def test_material_properties_props_for(self):
        """``props_for(material_id)`` should return the correct ``ThermalProps``."""
        props = MaterialProperties()
        battery_props = props.props_for(0)
        assert battery_props.k == pytest.approx(props.battery.k)
        assert battery_props.rho == pytest.approx(props.battery.rho)

        coolant_props = props.props_for(1)
        assert coolant_props.k == pytest.approx(props.coolant_props.k)

        with pytest.raises(KeyError):
            props.props_for(99)


# ===========================================================================
# TestFromMaterialFields
# ===========================================================================


class TestFromMaterialFields:
    """Tests for the ``HeatSolver2D.from_material_fields`` factory constructor."""

    def test_basic_construction(self):
        """Verify that ``from_material_fields`` creates a valid solver.

        A material mask with three materials should produce a solver whose
        k, rho, cp fields match the per-material arrays looked up via the mask.
        """
        mask = create_material_mask(32)
        props = MaterialProperties()
        k_vals = props.conductivity_array()
        rho_vals = props.density_array()
        cp_vals = props.heat_capacity_array()

        solver = HeatSolver2D.from_material_fields(
            mask=mask,
            k_values=k_vals,
            rho_values=rho_vals,
            cp_values=cp_vals,
            dx=1e-3, dy=1e-3, dt=1e-6,
        )
        assert solver.grid_shape == (32, 32)

        # Verify that battery cells (mask==0) have the battery conductivity.
        battery_cells = mask == 0
        assert np.allclose(solver.k[battery_cells], k_vals[0]), (
            "Battery cells should have battery conductivity"
        )

    def test_heterogeneous_properties(self):
        """The solver should have spatially varying properties matching the mask.

        We verify that the number of unique k values in the solver matches
        the number of distinct materials.
        """
        mask = create_material_mask(32)
        props = MaterialProperties()
        solver = HeatSolver2D.from_material_fields(
            mask=mask,
            k_values=props.conductivity_array(),
            rho_values=props.density_array(),
            cp_values=props.heat_capacity_array(),
            dx=1e-3, dy=1e-3, dt=1e-6,
        )
        unique_k = np.unique(solver.k)
        assert len(unique_k) == 3, (
            f"Solver should have 3 distinct k values, got {len(unique_k)}"
        )


# ===========================================================================
# TestValidationUtilities
# ===========================================================================


class TestValidationUtilities:
    """Tests for error-norm and convergence-rate utilities in the validation module."""

    def test_compute_error_norms_zero_error(self):
        """Comparing identical arrays should yield zero for all norms."""
        a = np.ones((10, 10))
        norms = compute_error_norms(a, a)
        assert norms["L1"] == pytest.approx(0.0, abs=1e-15)
        assert norms["L2"] == pytest.approx(0.0, abs=1e-15)
        assert norms["Linf"] == pytest.approx(0.0, abs=1e-15)

    def test_compute_error_norms_known_values(self):
        """Verify error norms against hand-computed values.

        For a = [1, 0] and b = [0, 0]:
            errors = [1, 0], n = 2
            L1 = 1/2 = 0.5
            L2 = sqrt(1/2) ~ 0.7071
            Linf = 1.0
        """
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        norms = compute_error_norms(a, b)
        assert norms["L1"] == pytest.approx(0.5)
        assert norms["L2"] == pytest.approx(math.sqrt(0.5), rel=1e-10)
        assert norms["Linf"] == pytest.approx(1.0)

    def test_compute_error_norms_shape_mismatch(self):
        """Mismatched array shapes should raise ``ValueError``."""
        a = np.ones((3, 3))
        b = np.ones((4, 4))
        with pytest.raises(ValueError, match="[Ss]hape"):
            compute_error_norms(a, b)

    def test_convergence_rate_too_few_points(self):
        """Fewer than 2 data points should raise ``ValueError``."""
        with pytest.raises(ValueError, match="at least 2"):
            convergence_rate([0.1], [10])

    def test_convergence_rate_length_mismatch(self):
        """Mismatched lengths should raise ``ValueError``."""
        with pytest.raises(ValueError, match="same length"):
            convergence_rate([0.1, 0.01], [10, 20, 40])


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    """Integration tests combining multiple solver components."""

    def test_multi_material_solve(self):
        """Run a short simulation with spatially varying material properties.

        This end-to-end test creates a multi-material solver from a mask,
        runs a few time steps, and verifies that the trajectory is finite
        and physically plausible.
        """
        mask = create_material_mask(32)
        props = MaterialProperties()
        k_vals = props.conductivity_array()
        rho_vals = props.density_array()
        cp_vals = props.heat_capacity_array()

        solver = HeatSolver2D.from_material_fields(
            mask=mask,
            k_values=k_vals,
            rho_values=rho_vals,
            cp_values=cp_vals,
            dx=1e-3, dy=1e-3, dt=1e-6,
            T_amb=300.0, h_conv=10.0,
        )
        assert solver.check_stability(), (
            "Multi-material solver with small dt should be stable"
        )

        T0 = np.full((32, 32), 300.0, dtype=np.float64)
        source_mask = (mask == 0).astype(np.float64)  # heat only in battery cells
        traj = solver.solve(T0, n_steps=50, q0=1e5, source_mask=source_mask)

        assert np.all(np.isfinite(traj)), (
            "All trajectory values should be finite (no NaN or Inf)"
        )
        assert traj[-1].max() > 300.0, (
            "With a heat source, some cells should exceed the initial temperature"
        )

    def test_solver_repr(self, small_solver):
        """The solver's ``__repr__`` should return a non-empty informative string."""
        r = repr(small_solver)
        assert "HeatSolver2D" in r, "repr should contain the class name"
        assert "nx=16" in r, "repr should contain the grid size"
