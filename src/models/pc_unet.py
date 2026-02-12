"""Physics-Conditioned U-Net with Monte Carlo dropout.

Implements a U-Net architecture enhanced with physics conditioning at every
encoder/decoder level and Monte Carlo dropout for epistemic uncertainty
estimation. Designed for 2D battery thermal management surrogate modelling.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseThermalModel


class ConvBlock(nn.Module):
    """Two successive convolution blocks with GroupNorm and GELU activation.

    Each sub-block applies: Conv2d -> GroupNorm -> GELU.
    An optional spatial dropout is applied after the second sub-block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout_rate: Dropout probability applied after the second
            convolution. Set to 0.0 to disable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # Determine a valid number of groups for GroupNorm.  We want 8
        # groups when the channel count is divisible by 8, otherwise we
        # fall back to fewer groups so that ``out_channels % n_groups == 0``.
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

        self.dropout: Optional[nn.Module] = None
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two conv-norm-activation blocks with optional dropout.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class PhysicsConditioningBlock(nn.Module):
    """Project global physics parameters to a spatial feature bias.

    A small MLP maps a vector of physics parameters (e.g. representative
    k, q, h values) to a bias vector of size ``feature_dim``.  The bias
    is then broadcast-added to every spatial location of the feature map.

    Args:
        physics_dim: Dimensionality of the input physics parameter vector.
        feature_dim: Number of feature channels to condition.
    """

    def __init__(self, physics_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        """Condition features with physics parameters.

        Args:
            features: Feature map of shape ``(B, C, H, W)``.
            physics: Physics parameters of shape ``(B, P)``.

        Returns:
            Conditioned features of shape ``(B, C, H, W)``.
        """
        bias = self.mlp(physics)  # (B, C)
        bias = bias[:, :, None, None]  # (B, C, 1, 1)
        return features + bias


class PCUNet(BaseThermalModel):
    """Physics-Conditioned U-Net for battery thermal surrogate modeling.

    Args:
        in_channels: Number of input channels (default 9).
        out_channels: Number of output channels (default 1).
        base_features: Number of features in the first encoder level (default 32).
        num_levels: Number of encoder/decoder levels (default 4).
        dropout_rate: Dropout probability for MC dropout (default 0.1).
        physics_dim: Dimensionality of physics parameter vector (default 3: k, q, h).
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        base_features: int = 32,
        num_levels: int = 4,
        dropout_rate: float = 0.1,
        physics_dim: int = 3,
    ) -> None:
        super().__init__(in_channels, out_channels)

        self.base_features = base_features
        self.num_levels = num_levels
        self.dropout_rate = dropout_rate
        self.physics_dim = physics_dim

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.physics_cond_enc = nn.ModuleList()

        in_ch = in_channels
        for level in range(num_levels):
            out_ch = base_features * (2 ** level)
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch, dropout_rate))
            self.physics_cond_enc.append(PhysicsConditioningBlock(physics_dim, out_ch))

            if level < num_levels - 1:
                self.downsample_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_ch = out_ch

        # Decoder path
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.physics_cond_dec = nn.ModuleList()

        for level in range(num_levels - 1, 0, -1):
            in_ch = base_features * (2 ** level)
            out_ch = base_features * (2 ** (level - 1))

            self.upsample_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(ConvBlock(in_ch, out_ch, dropout_rate))
            self.physics_cond_dec.append(PhysicsConditioningBlock(physics_dim, out_ch))

        # Final output layer
        self.output_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the PC-U-Net.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.
            physics: Optional physics parameters of shape ``(B, P)``.
                If None, physics conditioning is skipped.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        # If physics is not provided, create a zero tensor
        if physics is None:
            physics = torch.zeros(x.size(0), self.physics_dim, device=x.device)

        # Encoder path
        encoder_features = []
        for level in range(self.num_levels):
            x = self.encoder_blocks[level](x)
            x = self.physics_cond_enc[level](x, physics)
            encoder_features.append(x)

            if level < self.num_levels - 1:
                x = self.downsample_layers[level](x)

        # Decoder path
        for level in range(self.num_levels - 1):
            x = self.upsample_layers[level](x)

            # Skip connection
            skip = encoder_features[-(level + 2)]
            # Handle size mismatch due to odd dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[level](x)
            x = self.physics_cond_dec[level](x, physics)

        # Final output
        x = self.output_conv(x)

        return x

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        physics: Optional[torch.Tensor] = None,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with epistemic uncertainty via MC dropout.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.
            physics: Optional physics parameters of shape ``(B, P)``.
            n_samples: Number of MC dropout samples.

        Returns:
            Tuple of (mean_prediction, std_prediction), each of shape ``(B, C_out, H, W)``.
        """
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, physics)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (n_samples, B, C, H, W)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred
