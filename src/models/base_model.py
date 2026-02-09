"""Base model class for thermal surrogate models.

Provides an abstract base class that all thermal surrogate models must inherit
from. Defines the common interface for forward prediction, multi-step rollout,
parameter counting, and configuration export.
"""

import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseThermalModel(nn.Module, abc.ABC):
    """Abstract base class for all 2D battery thermal surrogate models.

    All models consume a 9-channel spatial input tensor and produce a
    single-channel spatial output representing the predicted temperature
    change (delta_T) over one time step.

    Input channels (C_in = 9):
        0. T_t           -- current temperature field [K]
        1. mask_cell     -- binary mask for battery cell regions
        2. mask_coolant  -- binary mask for coolant channel regions
        3. mask_insulation -- binary mask for insulation regions
        4. k_field       -- spatially varying thermal conductivity [W/(m*K)]
        5. q_field       -- volumetric heat generation rate [W/m^3]
        6. h_field       -- convective heat transfer coefficient [W/(m^2*K)]
        7. sdf_cell      -- signed distance field to cell boundaries [m]
        8. sdf_coolant   -- signed distance field to coolant boundaries [m]

    Output (C_out = 1):
        Predicted delta_T field [K] for a single time step.

    Attributes:
        delta_t_clip: Maximum absolute temperature change allowed per step.
            Used to clip predictions for numerical stability during rollout.
    """

    def __init__(self) -> None:
        """Initialise the base thermal model."""
        super().__init__()
        self.delta_t_clip: float = 50.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the temperature change for one time step.

        Args:
            x: Input tensor of shape ``(B, 9, H, W)``.

        Returns:
            Predicted delta_T tensor of shape ``(B, 1, H, W)``.
        """

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary describing the model.

        Returns:
            Dictionary containing architecture hyper-parameters, channel
            counts, and any other metadata needed to reconstruct the model.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def predict_step(
        self,
        t_current: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the temperature field by one time step.

        Constructs the 9-channel input by concatenating the current
        temperature field with the static parameter channels, runs the
        forward pass, clips the predicted delta_T, and returns the
        updated temperature field.

        Args:
            t_current: Current temperature field of shape ``(B, 1, H, W)``.
            params: Static parameter channels of shape ``(B, 8, H, W)``
                corresponding to channels 1-8 (masks, material fields,
                SDFs).

        Returns:
            Next temperature field of shape ``(B, 1, H, W)`` computed as
            ``T_current + clip(delta_T, -delta_t_clip, +delta_t_clip)``.
        """
        x = torch.cat([t_current, params], dim=1)  # (B, 9, H, W)
        delta_t = self.forward(x)  # (B, 1, H, W)
        delta_t = torch.clamp(delta_t, -self.delta_t_clip, self.delta_t_clip)
        t_next = t_current + delta_t
        return t_next

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Integer count of parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def rollout(
        self,
        t_init: torch.Tensor,
        params: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Autoregressively roll out the model for multiple time steps.

        Args:
            t_init: Initial temperature field of shape ``(B, 1, H, W)``.
            params: Static parameter channels of shape ``(B, 8, H, W)``.
            n_steps: Number of time steps to advance.

        Returns:
            Temperature trajectory tensor of shape ``(B, n_steps+1, H, W)``
            where index 0 is the initial state and index ``n_steps`` is the
            final predicted state.
        """
        trajectory = [t_init.squeeze(1)]  # list of (B, H, W)
        t_current = t_init
        for _ in range(n_steps):
            t_current = self.predict_step(t_current, params)
            trajectory.append(t_current.squeeze(1))
        return torch.stack(trajectory, dim=1)  # (B, n_steps+1, H, W)

    def extra_repr(self) -> str:
        """Return a string with extra model information for ``repr``.

        Returns:
            Human-readable string with parameter count and clip value.
        """
        return (
            f"trainable_params={self.count_parameters():,}, "
            f"delta_t_clip={self.delta_t_clip}"
        )
