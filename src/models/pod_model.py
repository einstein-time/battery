"""Proper Orthogonal Decomposition (POD) baseline model.

Combines a data-driven POD basis (computed via truncated SVD on training
snapshots) with a small MLP that advances POD coefficients in time.  This
provides a physics-informed, reduced-order baseline for the thermal
surrogate task.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.base_model import BaseThermalModel


class PODModel(BaseThermalModel):
    """POD + MLP model for 2D battery thermal surrogate prediction.

    **Offline stage** (``fit_basis``):
        Given a collection of temperature-field snapshots, compute the
        truncated SVD to obtain a set of orthonormal spatial modes.

    **Online stage** (``forward``):
        1. Project the current 2D temperature field onto the POD basis
           to obtain a coefficient vector.
        2. Concatenate the coefficient vector with spatially averaged
           physics parameters (k, q, h).
        3. Pass through a small MLP to predict the POD coefficients at
           the next time step.
        4. Reconstruct the full-field delta_T from the predicted
           coefficient residual.

    Args:
        n_modes: Number of POD modes to retain (default 50).
        hidden_dim: Width of the hidden layers in the time-stepping MLP
            (default 256).
        field_height: Spatial height of the 2D temperature field.
        field_width: Spatial width of the 2D temperature field.
        in_channels: Number of input channels (default 9).  Only channel 0
            (temperature) is projected; channels 4-6 provide physics params.
        out_channels: Number of output channels (default 1).

    Attributes:
        basis: Registered buffer of shape ``(n_modes, H*W)`` holding the
            POD basis vectors.  Initialised to zeros and populated by
            ``fit_basis``.
        mean_field: Registered buffer of shape ``(H*W,)`` holding the
            training-set mean snapshot used for centring.
    """

    # Indices of physics channels in the 9-channel input.
    _K_CHANNEL: int = 4
    _Q_CHANNEL: int = 5
    _H_CHANNEL: int = 6

    def __init__(
        self,
        n_modes: int = 50,
        hidden_dim: int = 256,
        field_height: int = 64,
        field_width: int = 64,
        in_channels: int = 9,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.n_modes = n_modes
        self.hidden_dim = hidden_dim
        self.field_height = field_height
        self.field_width = field_width
        self.in_channels = in_channels
        self.out_channels = out_channels

        spatial_dim = field_height * field_width

        # POD basis vectors (populated by fit_basis).
        self.register_buffer(
            "basis", torch.zeros(n_modes, spatial_dim),
        )
        self.register_buffer(
            "mean_field", torch.zeros(spatial_dim),
        )
        self.register_buffer(
            "basis_fitted", torch.tensor(False),
        )

        # Physics parameter vector has 3 components (mean k, q, h).
        physics_dim: int = 3

        # MLP: maps (POD coefficients + physics) at time t to POD
        # coefficient *residuals* at time t + dt.
        self.mlp = nn.Sequential(
            nn.Linear(n_modes + physics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_modes),
        )

    # ------------------------------------------------------------------
    # Basis fitting
    # ------------------------------------------------------------------

    def fit_basis(self, snapshots: torch.Tensor) -> Dict[str, Any]:
        """Compute the POD basis from training snapshots via truncated SVD.

        Args:
            snapshots: Temperature field snapshots of shape ``(N, H, W)``
                where *N* is the number of snapshots collected from the
                training set.

        Returns:
            Dictionary containing:
                - ``singular_values``: The first ``n_modes`` singular values.
                - ``explained_variance_ratio``: Fraction of total variance
                  captured by each mode.

        Raises:
            ValueError: If the spatial dimensions of the snapshots do not
                match ``field_height`` and ``field_width``.
        """
        n_snap, h, w = snapshots.shape
        if h != self.field_height or w != self.field_width:
            raise ValueError(
                f"Snapshot spatial dims ({h}, {w}) do not match model "
                f"({self.field_height}, {self.field_width})."
            )

        # Flatten spatial dimensions: (N, H*W)
        flat = snapshots.reshape(n_snap, -1).float()

        # Centre the data.
        mean = flat.mean(dim=0)
        centred = flat - mean.unsqueeze(0)

        # Truncated SVD via torch.linalg.svd (full_matrices=False for
        # economy).  We only keep the first n_modes components.
        u, s, vh = torch.linalg.svd(centred, full_matrices=False)
        basis = vh[: self.n_modes]  # (n_modes, H*W)
        singular_values = s[: self.n_modes]

        # Explained variance ratio (based on squared singular values).
        total_var = (s ** 2).sum()
        explained = (singular_values ** 2) / total_var

        # Store as buffers so they travel with the model.
        self.basis.copy_(basis)
        self.mean_field.copy_(mean)
        self.basis_fitted.fill_(True)

        return {
            "singular_values": singular_values.detach().cpu().numpy(),
            "explained_variance_ratio": explained.detach().cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # Projection / reconstruction
    # ------------------------------------------------------------------

    def project(self, field: torch.Tensor) -> torch.Tensor:
        """Project a batch of 2D fields onto the POD basis.

        Args:
            field: Temperature field of shape ``(B, H, W)`` or
                ``(B, 1, H, W)``.

        Returns:
            POD coefficient vector of shape ``(B, n_modes)``.
        """
        if field.dim() == 4:
            field = field.squeeze(1)
        flat = field.reshape(field.shape[0], -1)  # (B, H*W)
        centred = flat - self.mean_field.unsqueeze(0)
        coeffs = centred @ self.basis.t()  # (B, n_modes)
        return coeffs

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full field from POD coefficients.

        Args:
            coeffs: Coefficient vector of shape ``(B, n_modes)``.

        Returns:
            Reconstructed temperature field of shape ``(B, 1, H, W)``.
        """
        flat = coeffs @ self.basis + self.mean_field.unsqueeze(0)  # (B, H*W)
        return flat.reshape(-1, 1, self.field_height, self.field_width)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_physics(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatially averaged physics parameters from input.

        Args:
            x: Full 9-channel input tensor of shape ``(B, 9, H, W)``.

        Returns:
            Physics vector of shape ``(B, 3)``.
        """
        k_mean = x[:, self._K_CHANNEL].mean(dim=(-2, -1))
        q_mean = x[:, self._Q_CHANNEL].mean(dim=(-2, -1))
        h_mean = x[:, self._H_CHANNEL].mean(dim=(-2, -1))
        return torch.stack([k_mean, q_mean, h_mean], dim=1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict delta_T for one time step using the POD-MLP model.

        Args:
            x: Input tensor of shape ``(B, 9, H, W)``.

        Returns:
            Predicted delta_T of shape ``(B, 1, H, W)`` clipped to
            ``[-delta_t_clip, +delta_t_clip]``.

        Raises:
            RuntimeError: If ``fit_basis`` has not been called before the
                first forward pass.
        """
        if not self.basis_fitted.item():
            raise RuntimeError(
                "POD basis has not been fitted. Call fit_basis() with "
                "training snapshots before running the forward pass."
            )

        # 1. Extract current temperature field and physics params.
        t_field = x[:, 0:1, :, :]  # (B, 1, H, W)
        physics = self._extract_physics(x)  # (B, 3)

        # 2. Project temperature onto POD basis.
        coeffs = self.project(t_field)  # (B, n_modes)

        # 3. Predict coefficient residual with MLP.
        mlp_input = torch.cat([coeffs, physics], dim=1)  # (B, n_modes+3)
        coeff_residual = self.mlp(mlp_input)  # (B, n_modes)

        # 4. Reconstruct the next temperature field.
        coeffs_next = coeffs + coeff_residual
        t_next = self.reconstruct(coeffs_next)  # (B, 1, H, W)

        # 5. delta_T = T_next - T_current
        delta_t = t_next - t_field
        delta_t = torch.clamp(delta_t, -self.delta_t_clip, self.delta_t_clip)
        return delta_t

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return a serialisable configuration dictionary.

        Returns:
            Dictionary with constructor arguments and metadata.
        """
        return {
            "model_type": "PODModel",
            "n_modes": self.n_modes,
            "hidden_dim": self.hidden_dim,
            "field_height": self.field_height,
            "field_width": self.field_width,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "delta_t_clip": self.delta_t_clip,
            "basis_fitted": bool(self.basis_fitted.item()),
            "trainable_params": self.count_parameters(),
        }
