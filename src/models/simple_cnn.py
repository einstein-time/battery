"""Simple CNN baseline model without physics conditioning.

A straightforward convolutional neural network for comparison against the
physics-informed PC-U-Net. This model is trained purely on data without
any physics-based loss terms or conditioning.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_model import BaseThermalModel


class SimpleCNN(BaseThermalModel):
    """Simple CNN baseline for thermal surrogate modeling.

    A stack of convolutional layers with residual connections.
    No U-Net architecture, no physics conditioning.

    Args:
        in_channels: Number of input channels (default 9).
        out_channels: Number of output channels (default 1).
        hidden_channels: Number of hidden feature channels (default 64).
        num_layers: Number of convolutional layers (default 8).
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 8,
    ) -> None:
        super().__init__(in_channels, out_channels)

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                )
            )

        # Output projection
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.
            physics: Ignored (for interface compatibility).

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        # Input projection
        x = self.activation(self.input_conv(x))

        # Residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = self.activation(x + residual)

        # Output projection
        x = self.output_conv(x)

        return x
