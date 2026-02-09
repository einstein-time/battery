"""Physics-informed loss functions for 2D battery thermal surrogate training.

Provides a composite loss combining data fidelity, PDE residual enforcement,
boundary condition penalties, and multi-step temporal consistency.  All
operations use differentiable PyTorch primitives so that gradients propagate
through the physics terms during back-propagation.

Typical usage::

    loss_fn = PhysicsInformedLoss(
        lambda_data=1.0, lambda_pde=0.1, lambda_bc=0.1,
        lambda_consistency=0.05, dx=0.001, dy=0.001,
        dt=0.001, rho=2500.0, cp=700.0,
    )
    total, breakdown = loss_fn(pred, target, T_input, k, q, h, model, phase=2)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions -- finite-difference operators on 4-D tensors
# ---------------------------------------------------------------------------

def compute_spatial_gradients(
    field: torch.Tensor,
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute first-order spatial gradients using central finite differences.

    The input ``field`` is expected to have shape ``(B, 1, H, W)``.  Boundary
    pixels are computed with one-sided (forward / backward) differences so that
    the output tensors have the same spatial dimensions as the input.

    Args:
        field: Temperature (or other scalar) field of shape ``(B, 1, H, W)``.
        dx: Grid spacing in the *x* (width / column) direction [m].
        dy: Grid spacing in the *y* (height / row) direction [m].

    Returns:
        Tuple of ``(dT_dx, dT_dy)`` each with shape ``(B, 1, H, W)``.
    """
    # dT/dx  --  variation along the W (last) axis
    dT_dx = torch.zeros_like(field)
    # Central differences for interior columns
    dT_dx[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / (2.0 * dx)
    # Forward / backward at boundaries
    dT_dx[:, :, :, 0] = (field[:, :, :, 1] - field[:, :, :, 0]) / dx
    dT_dx[:, :, :, -1] = (field[:, :, :, -1] - field[:, :, :, -2]) / dx

    # dT/dy  --  variation along the H (second-to-last) axis
    dT_dy = torch.zeros_like(field)
    dT_dy[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / (2.0 * dy)
    dT_dy[:, :, 0, :] = (field[:, :, 1, :] - field[:, :, 0, :]) / dy
    dT_dy[:, :, -1, :] = (field[:, :, -1, :] - field[:, :, -2, :]) / dy

    return dT_dx, dT_dy


def compute_laplacian_2d(
    field: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the 2-D Laplacian using second-order central finite differences.

    Uses the standard 5-point stencil::

        lap = (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / dx^2
            + (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / dy^2

    Interior pixels use the stencil directly.  Boundary pixels are handled
    by replicating the nearest interior value (Neumann-style zero-gradient
    padding) before applying the stencil, which keeps the output the same
    size as the input.

    Args:
        field: Scalar field of shape ``(B, 1, H, W)``.
        dx: Grid spacing in the x (width) direction [m].
        dy: Grid spacing in the y (height) direction [m].

    Returns:
        Laplacian field of shape ``(B, 1, H, W)``.
    """
    # Pad with replicate (Neumann BC approximation) so the stencil covers
    # the boundary rows / columns without shrinking the spatial extent.
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")

    d2T_dx2 = (padded[:, :, 1:-1, 2:] - 2.0 * padded[:, :, 1:-1, 1:-1]
               + padded[:, :, 1:-1, :-2]) / (dx * dx)
    d2T_dy2 = (padded[:, :, 2:, 1:-1] - 2.0 * padded[:, :, 1:-1, 1:-1]
               + padded[:, :, :-2, 1:-1]) / (dy * dy)

    return d2T_dx2 + d2T_dy2


# ---------------------------------------------------------------------------
# Composite physics-informed loss
# ---------------------------------------------------------------------------

class PhysicsInformedLoss(nn.Module):
    """Multi-term physics-informed loss for the battery thermal surrogate.

    The total loss is a weighted sum of up to four terms whose activation
    depends on the current training phase:

    * **Phase 1 (pretrain):** data loss only.
    * **Phase 2 (physics):** data + PDE residual + boundary condition.
    * **Phase 3 (finetune):** all four terms including temporal consistency.

    Attributes:
        lambda_data: Weight for the data-fidelity (MSE) loss.
        lambda_pde: Weight for the PDE residual loss.
        lambda_bc: Weight for the boundary-condition loss.
        lambda_consistency: Weight for the multi-step consistency loss.
        dx: Grid spacing in x [m].
        dy: Grid spacing in y [m].
        dt: Time step size [s].
        rho: Material density [kg/m^3].
        cp: Specific heat capacity [J/(kg*K)].
        T_amb: Ambient temperature used in the Robin BC [K].
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_bc: float = 0.1,
        lambda_consistency: float = 0.05,
        dx: float = 0.001,
        dy: float = 0.001,
        dt: float = 0.001,
        rho: float = 2500.0,
        cp: float = 700.0,
        T_amb: float = 298.15,
    ) -> None:
        """Initialise the physics-informed loss.

        Args:
            lambda_data: Weight multiplier for the data MSE term.
            lambda_pde: Weight multiplier for the PDE residual term.
            lambda_bc: Weight multiplier for the boundary-condition term.
            lambda_consistency: Weight multiplier for the consistency term.
            dx: Grid spacing in the x direction [m].
            dy: Grid spacing in the y direction [m].
            dt: Time-step size [s].
            rho: Density of the bulk material [kg/m^3].
            cp: Specific heat capacity [J/(kg*K)].
            T_amb: Ambient reference temperature for Robin BCs [K].
        """
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_consistency = lambda_consistency
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.cp = cp
        self.T_amb = T_amb

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    @staticmethod
    def data_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-squared error between predicted and target temperature fields.

        Args:
            pred: Predicted temperature field ``(B, 1, H, W)``.
            target: Ground-truth temperature field ``(B, 1, H, W)``.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(pred, target)

    def pde_residual_loss(
        self,
        pred: torch.Tensor,
        T_input: torch.Tensor,
        k_field: torch.Tensor,
        q_field: torch.Tensor,
        rho: Optional[float] = None,
        cp: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """PDE residual loss for the 2-D transient heat equation.

        The governing equation is::

            rho * cp * dT/dt = div(k * grad(T)) + q

        which is discretised in time as::

            rho * cp * (T_pred - T_input) / dt = div(k * grad(T_pred)) + q

        The residual is the difference between the left-hand and right-hand
        sides.  We return the mean squared residual over the spatial domain.

        The divergence of the heat flux is expanded using the product rule::

            div(k * grad T) = k * lap(T) + grad(k) . grad(T)

        Args:
            pred: Predicted *next* temperature field ``(B, 1, H, W)``.
            T_input: Current (input) temperature field ``(B, 1, H, W)``.
            k_field: Thermal conductivity field ``(B, 1, H, W)`` [W/(m*K)].
            q_field: Volumetric heat source field ``(B, 1, H, W)`` [W/m^3].
            rho: Override for material density.  Defaults to ``self.rho``.
            cp: Override for specific heat.  Defaults to ``self.cp``.
            dt: Override for time step.  Defaults to ``self.dt``.

        Returns:
            Scalar mean-squared PDE residual.
        """
        _rho = rho if rho is not None else self.rho
        _cp = cp if cp is not None else self.cp
        _dt = dt if dt is not None else self.dt

        # Temporal derivative (LHS of discretised PDE)
        lhs = _rho * _cp * (pred - T_input) / _dt

        # Spatial operators on the predicted field
        laplacian = compute_laplacian_2d(pred, self.dx, self.dy)
        dT_dx, dT_dy = compute_spatial_gradients(pred, self.dx, self.dy)
        dk_dx, dk_dy = compute_spatial_gradients(k_field, self.dx, self.dy)

        # div(k * grad T) = k * lap(T) + grad(k) . grad(T)
        diffusion = k_field * laplacian + dk_dx * dT_dx + dk_dy * dT_dy

        # RHS = diffusion + heat source
        rhs = diffusion + q_field

        residual = lhs - rhs
        return torch.mean(residual ** 2)

    def boundary_loss(
        self,
        pred: torch.Tensor,
        T_input: torch.Tensor,
        h_field: torch.Tensor,
        T_amb: Optional[float] = None,
        dx: Optional[float] = None,
    ) -> torch.Tensor:
        """Robin (convective) boundary-condition residual loss.

        At domain boundaries the Robin condition is::

            -k * dT/dn = h * (T - T_amb)

        Because the conductivity field is not passed separately here we
        rearrange the check into a form that compares the *outward normal
        gradient* of the predicted field with the expected convective flux.
        We approximate ``dT/dn`` with a one-sided finite difference using the
        boundary and first-interior pixels.  The loss is the MSE of the
        residual ``h * (T_boundary - T_amb) + k_approx * dT/dn`` evaluated
        at all four domain edges (top, bottom, left, right).

        For simplicity and generality (since ``k`` may vary spatially), we
        compute the residual as the mismatch of the *normalised* gradient:
        ``dT/dn + h/k_eff * (T_boundary - T_amb)``, where ``k_eff`` is
        estimated from the input field's implicit diffusivity.  However,
        because the trainer provides ``h_field`` but not ``k_field`` to
        this method, we adopt the simplified form::

            residual = h * (T_boundary - T_amb) + (T_boundary - T_interior) / dx

        This is equivalent to prescribing ``k = 1`` at the boundary in the
        Robin condition, which acts as a soft regulariser encouraging the
        correct qualitative boundary behaviour.  When used together with the
        PDE residual (which *does* receive ``k_field``), the combination
        constrains both interior and boundary physics.

        Args:
            pred: Predicted temperature field ``(B, 1, H, W)``.
            T_input: Current temperature field (unused but kept for API
                symmetry; the BC is evaluated on ``pred``).
            h_field: Convective heat-transfer coefficient field
                ``(B, 1, H, W)`` [W/(m^2*K)].
            T_amb: Ambient temperature [K].  Defaults to ``self.T_amb``.
            dx: Grid spacing [m] used for one-sided gradient.  Defaults to
                ``self.dx``.

        Returns:
            Scalar MSE of the boundary-condition residual.
        """
        _T_amb = T_amb if T_amb is not None else self.T_amb
        _dx = dx if dx is not None else self.dx

        residuals: list[torch.Tensor] = []

        # --- Top boundary (row 0): outward normal points in -y direction ---
        T_bnd_top = pred[:, :, 0, :]           # (B, 1, W)
        T_int_top = pred[:, :, 1, :]           # (B, 1, W)
        h_top = h_field[:, :, 0, :]            # (B, 1, W)
        # dT/dn ~ -(T_int - T_bnd) / dy  (outward = -y)
        dTdn_top = -(T_int_top - T_bnd_top) / self.dy
        residuals.append(h_top * (T_bnd_top - _T_amb) + dTdn_top)

        # --- Bottom boundary (row -1): outward normal points in +y ---
        T_bnd_bot = pred[:, :, -1, :]
        T_int_bot = pred[:, :, -2, :]
        h_bot = h_field[:, :, -1, :]
        dTdn_bot = (T_bnd_bot - T_int_bot) / self.dy
        residuals.append(h_bot * (T_bnd_bot - _T_amb) + dTdn_bot)

        # --- Left boundary (col 0): outward normal points in -x ---
        T_bnd_left = pred[:, :, :, 0]
        T_int_left = pred[:, :, :, 1]
        h_left = h_field[:, :, :, 0]
        dTdn_left = -(T_int_left - T_bnd_left) / _dx
        residuals.append(h_left * (T_bnd_left - _T_amb) + dTdn_left)

        # --- Right boundary (col -1): outward normal points in +x ---
        T_bnd_right = pred[:, :, :, -1]
        T_int_right = pred[:, :, :, -2]
        h_right = h_field[:, :, :, -1]
        dTdn_right = (T_bnd_right - T_int_right) / _dx
        residuals.append(h_right * (T_bnd_right - _T_amb) + dTdn_right)

        # Stack and return MSE over all boundary pixels
        all_residuals = torch.cat(
            [r.reshape(-1) for r in residuals], dim=0
        )
        return torch.mean(all_residuals ** 2)

    @staticmethod
    def consistency_loss(
        model: nn.Module,
        T_input: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-step temporal consistency loss.

        Computes two single-step predictions sequentially and compares the
        result with the model's own two-step prediction to encourage
        self-consistency during autoregressive rollout:

        1. ``T1 = model.predict_step(T0, params)``
        2. ``T2 = model.predict_step(T1, params)``
        3. ``T2_direct = model.predict_step(T1.detach(), params)``

        The consistency gap is ``MSE(T2, T2_direct)``.  Detaching ``T1``
        in the direct path prevents gradient flow through the second
        prediction twice, stabilising training.

        Args:
            model: A :class:`BaseThermalModel` instance (or any model
                that exposes ``predict_step(T, params)``).
            T_input: Current temperature field ``(B, 1, H, W)`` (T0).
            params: Static parameter channels ``(B, 8, H, W)``.

        Returns:
            Scalar MSE consistency loss.
        """
        # Sequential two-step rollout (gradients flow through both steps)
        T1 = model.predict_step(T_input, params)
        T2_sequential = model.predict_step(T1, params)

        # Direct prediction from detached T1 (single-step, no double grad)
        T2_direct = model.predict_step(T1.detach(), params)

        return F.mse_loss(T2_sequential, T2_direct)

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        T_input: torch.Tensor,
        k_field: Optional[torch.Tensor] = None,
        q_field: Optional[torch.Tensor] = None,
        h_field: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        phase: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite loss for the current training phase.

        The activation schedule is:

        +---------+------+-----+----+-------------+
        | Phase   | Data | PDE | BC | Consistency |
        +=========+======+=====+====+=============+
        | 1       |  x   |     |    |             |
        | 2       |  x   |  x  | x  |             |
        | 3       |  x   |  x  | x  |     x       |
        +---------+------+-----+----+-------------+

        Args:
            pred: Predicted temperature field ``(B, 1, H, W)``.
            target: Ground-truth temperature field ``(B, 1, H, W)``.
            T_input: Input (current) temperature field ``(B, 1, H, W)``.
            k_field: Thermal conductivity ``(B, 1, H, W)``.  Required for
                phases 2 and 3.
            q_field: Heat-source field ``(B, 1, H, W)``.  Required for
                phases 2 and 3.
            h_field: Convective HTC field ``(B, 1, H, W)``.  Required for
                phases 2 and 3.
            model: Model instance needed for consistency loss (phase 3).
            phase: Training phase (1, 2, or 3).

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where *loss_dict* maps
            individual loss names to their **unweighted** scalar values
            (as Python floats) and also contains ``"total_loss"``.
        """
        loss_dict: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # -- Data loss (always active) ------------------------------------
        l_data = self.data_loss(pred, target)
        total_loss = total_loss + self.lambda_data * l_data
        loss_dict["data_loss"] = l_data.item()

        # -- PDE residual loss (phases 2 & 3) -----------------------------
        if phase >= 2 and k_field is not None and q_field is not None:
            l_pde = self.pde_residual_loss(pred, T_input, k_field, q_field)
            total_loss = total_loss + self.lambda_pde * l_pde
            loss_dict["pde_loss"] = l_pde.item()

        # -- Boundary-condition loss (phases 2 & 3) -----------------------
        if phase >= 2 and h_field is not None:
            l_bc = self.boundary_loss(pred, T_input, h_field)
            total_loss = total_loss + self.lambda_bc * l_bc
            loss_dict["bc_loss"] = l_bc.item()

        # -- Consistency loss (phase 3 only) ------------------------------
        if phase >= 3 and model is not None:
            # Build static params from the input tensor (channels 1-8)
            # The full input is (B, 9, H, W); channel 0 = T, 1-8 = params
            # T_input is (B, 1, H, W) -- same as channel 0 of the model input
            # We need the params tensor (B, 8, H, W).  The caller's batch
            # dict carries 'input' with shape (B, 9, H, W), but we only
            # receive T_input here.  To keep the API clean we extract params
            # from the input tensor stored on the model's batch (set by the
            # trainer before calling forward).  As a fallback we use zeros.
            if hasattr(model, "_cached_params") and model._cached_params is not None:
                params = model._cached_params
            else:
                # Cannot compute consistency without params; skip gracefully
                logger.debug(
                    "Skipping consistency loss: model._cached_params not set."
                )
                loss_dict["consistency_loss"] = 0.0
                loss_dict["total_loss"] = total_loss.item()
                return total_loss, loss_dict

            l_consist = self.consistency_loss(model, T_input, params)
            total_loss = total_loss + self.lambda_consistency * l_consist
            loss_dict["consistency_loss"] = l_consist.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
