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
        """Add physics-derived bias to a spatial feature map.

        Args:
            features: Feature map of shape ``(B, C, H, W)``.
            physics: Physics parameter vector of shape ``(B, physics_dim)``.

        Returns:
            Conditioned feature map of shape ``(B, C, H, W)``.
        """
        bias = self.mlp(physics)  # (B, C)
        return features + bias.unsqueeze(-1).unsqueeze(-1)


class PCUNet(BaseThermalModel):
    """Physics-Conditioned U-Net for thermal surrogate modelling.

    A 4-level encoder-decoder architecture with skip connections.  Physics
    parameters (thermal conductivity *k*, heat source *q*, convection
    coefficient *h*) are extracted from the input channels as spatially
    averaged scalars and injected at every level via learned
    ``PhysicsConditioningBlock`` modules.

    Monte Carlo dropout is applied at the bottleneck and every decoder
    level so that epistemic uncertainty can be estimated at inference
    time by running multiple stochastic forward passes.

    Args:
        in_channels: Number of input channels (default 9).
        out_channels: Number of output channels (default 1).
        base_features: Number of features at the first encoder level
            (default 32).  Subsequent levels double the channel count.
        num_levels: Number of encoder/decoder levels (default 4).
        dropout_rate: Dropout probability for MC dropout (default 0.1).
        physics_dim: Dimensionality of the extracted physics vector
            (default 3, corresponding to mean k, q, h).

    Input shape:
        ``(B, in_channels, H, W)`` where ``H`` and ``W`` must be
        divisible by ``2 ** num_levels``.

    Output shape:
        ``(B, out_channels, H, W)`` -- predicted delta_T clipped to
        ``[-delta_t_clip, +delta_t_clip]``.
    """

    # Indices of physics channels inside the 9-channel input tensor.
    _K_CHANNEL: int = 4
    _Q_CHANNEL: int = 5
    _H_CHANNEL: int = 6

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        base_features: int = 32,
        num_levels: int = 4,
        dropout_rate: float = 0.1,
        physics_dim: int = 3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.num_levels = num_levels
        self.dropout_rate = dropout_rate
        self.physics_dim = physics_dim

        # Feature widths at each level: [32, 64, 128, 256]
        level_features: List[int] = [
            base_features * (2 ** i) for i in range(num_levels)
        ]
        bottleneck_features: int = level_features[-1] * 2  # 512

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoder_physics = nn.ModuleList()

        prev_ch = in_channels
        for feat in level_features:
            self.encoders.append(ConvBlock(prev_ch, feat, dropout_rate=0.0))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_physics.append(
                PhysicsConditioningBlock(physics_dim, feat)
            )
            prev_ch = feat

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(
            level_features[-1], bottleneck_features, dropout_rate=dropout_rate,
        )
        self.bottleneck_physics = PhysicsConditioningBlock(
            physics_dim, bottleneck_features,
        )

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoder_physics = nn.ModuleList()

        prev_ch = bottleneck_features
        for feat in reversed(level_features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2)
            )
            # After concatenation with skip connection the channel count
            # doubles.
            self.decoders.append(
                ConvBlock(feat * 2, feat, dropout_rate=dropout_rate)
            )
            self.decoder_physics.append(
                PhysicsConditioningBlock(physics_dim, feat)
            )
            prev_ch = feat

        # ---- Output head ----
        self.head = nn.Conv2d(level_features[0], out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_physics(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a global physics parameter vector from the input.

        Computes the spatial mean of the k, q, and h channels.

        Args:
            x: Full input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Physics vector of shape ``(B, physics_dim)``.
        """
        k_mean = x[:, self._K_CHANNEL].mean(dim=(-2, -1))  # (B,)
        q_mean = x[:, self._Q_CHANNEL].mean(dim=(-2, -1))
        h_mean = x[:, self._H_CHANNEL].mean(dim=(-2, -1))
        return torch.stack([k_mean, q_mean, h_mean], dim=1)  # (B, 3)

    def encode(
        self,
        x: torch.Tensor,
        physics: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the encoder path and collect skip connections.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.
            physics: Physics vector of shape ``(B, physics_dim)``.

        Returns:
            Tuple of (bottleneck features, list of skip-connection tensors)
            ordered from the shallowest to the deepest level.
        """
        skips: List[torch.Tensor] = []
        for encoder, pool, phys_block in zip(
            self.encoders, self.pools, self.encoder_physics,
        ):
            x = encoder(x)
            x = phys_block(x, physics)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        x = self.bottleneck_physics(x, physics)
        return x, skips

    def decode(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        physics: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder path with skip connections.

        Args:
            x: Bottleneck feature tensor.
            skips: Skip-connection tensors from the encoder (shallow to deep).
            physics: Physics vector of shape ``(B, physics_dim)``.

        Returns:
            Decoded feature tensor ready for the output head.
        """
        for upconv, decoder, phys_block, skip in zip(
            self.upconvs,
            self.decoders,
            self.decoder_physics,
            reversed(skips),
        ):
            x = upconv(x)
            # Handle potential size mismatch caused by odd spatial dims.
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            x = phys_block(x, physics)
        return x

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict delta_T for one time step.

        Args:
            x: Input tensor of shape ``(B, 9, H, W)``.

        Returns:
            Predicted delta_T of shape ``(B, 1, H, W)`` clipped to
            ``[-delta_t_clip, +delta_t_clip]``.
        """
        physics = self._extract_physics(x)
        bottleneck, skips = self.encode(x, physics)
        decoded = self.decode(bottleneck, skips, physics)
        delta_t = self.head(decoded)
        delta_t = torch.clamp(delta_t, -self.delta_t_clip, self.delta_t_clip)
        return delta_t

    def mc_predict(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo dropout prediction for uncertainty estimation.

        Runs ``n_samples`` stochastic forward passes with dropout enabled
        and returns the per-pixel mean and standard deviation.

        Args:
            x: Input tensor of shape ``(B, 9, H, W)``.
            n_samples: Number of stochastic forward passes.

        Returns:
            Tuple ``(mean, std)`` each of shape ``(B, 1, H, W)``.
        """
        self.train()  # Enable dropout
        predictions: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        self.eval()

        stacked = torch.stack(predictions, dim=0)  # (n_samples, B, 1, H, W)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std

    def get_config(self) -> Dict[str, Any]:
        """Return a serialisable configuration dictionary.

        Returns:
            Dictionary with all constructor arguments and metadata.
        """
        return {
            "model_type": "PCUNet",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_features": self.base_features,
            "num_levels": self.num_levels,
            "dropout_rate": self.dropout_rate,
            "physics_dim": self.physics_dim,
            "delta_t_clip": self.delta_t_clip,
            "trainable_params": self.count_parameters(),
        }
