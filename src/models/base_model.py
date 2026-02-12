"""Base model class for thermal surrogate models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseThermalModel(nn.Module, ABC):
    """Abstract base class for thermal surrogate models.

    All thermal models should inherit from this class and implement the
    forward pass. This ensures a consistent interface for training and
    evaluation.

    Args:
        in_channels: Number of input channels (e.g., 9 for full feature set).
        out_channels: Number of output channels (typically 1 for temperature).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, physics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.
            physics: Optional physics parameters of shape ``(B, P)`` for conditioning.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        pass

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and statistics."""
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "n_parameters": self.count_parameters(),
        }
