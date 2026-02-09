"""Simple CNN baseline model for thermal surrogate prediction.

Provides a lightweight convolutional neural network with residual connections
as a baseline for comparison against more sophisticated architectures such
as the Physics-Conditioned U-Net.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.base_model import BaseThermalModel


class ResidualConvBlock(nn.Module):
    """Two-layer convolutional block with a residual (skip) connection.

    Applies two Conv2d -> GroupNorm -> GELU sequences and adds a learned
    shortcut when the input and output channel counts differ.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        n_groups = min(8, out_channels)
        while out_channels % n_groups != 0:
            n_groups -= 1

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)

        self.activation = nn.GELU()

        # Learned 1x1 projection for the shortcut when channel counts change.
        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolution block.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        identity = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.activation(out + identity)
        return out


class SimpleCNN(BaseThermalModel):
    """Simple CNN baseline for 2D battery thermal surrogate modelling.

    Eight convolutional layers organised into four ``ResidualConvBlock``
    pairs.  Every block preserves the spatial dimensions (same-padding)
    so no pooling or upsampling is required.

    Channel progression::

        9 -> 64 -> 64 -> 128 -> 128 -> 64 -> 64 -> 32 -> 1

    The first four layers progressively widen the receptive field and
    increase capacity; the last four contract back to a single output
    channel.  A final 1x1 convolution produces the delta_T prediction.

    Args:
        in_channels: Number of input channels (default 9).
        out_channels: Number of output channels (default 1).

    Input shape:
        ``(B, in_channels, H, W)``

    Output shape:
        ``(B, out_channels, H, W)`` -- predicted delta_T clipped to
        ``[-delta_t_clip, +delta_t_clip]``.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Channel progression with residual connections every 2 layers.
        # Each ResidualConvBlock contains exactly 2 conv layers.
        self.blocks = nn.Sequential(
            ResidualConvBlock(in_channels, 64),   # layers 1-2:  9 -> 64
            ResidualConvBlock(64, 128),            # layers 3-4: 64 -> 128
            ResidualConvBlock(128, 64),            # layers 5-6: 128 -> 64
            ResidualConvBlock(64, 32),             # layers 7-8: 64 -> 32
        )

        # 1x1 output projection.
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict delta_T for one time step.

        Args:
            x: Input tensor of shape ``(B, 9, H, W)``.

        Returns:
            Predicted delta_T of shape ``(B, 1, H, W)`` clipped to
            ``[-delta_t_clip, +delta_t_clip]``.
        """
        features = self.blocks(x)
        delta_t = self.head(features)
        delta_t = torch.clamp(delta_t, -self.delta_t_clip, self.delta_t_clip)
        return delta_t

    def get_config(self) -> Dict[str, Any]:
        """Return a serialisable configuration dictionary.

        Returns:
            Dictionary with constructor arguments and metadata.
        """
        return {
            "model_type": "SimpleCNN",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "delta_t_clip": self.delta_t_clip,
            "trainable_params": self.count_parameters(),
        }
