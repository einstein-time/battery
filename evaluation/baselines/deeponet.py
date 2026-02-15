"""
Deep Operator Network (DeepONet) implementation.

Reference:
    Lu et al., "Learning nonlinear operators via DeepONet", Nature Machine Intelligence 2021
"""

import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """Deep Operator Network for PDE operator learning."""

    def __init__(self, branch_input_dim, trunk_input_dim=2, hidden_dim=128, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Branch network: encodes input function
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Trunk network: encodes query points
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params, T_curr):
        """
        Args:
            params: (batch, 6) physics parameters
            T_curr: (batch, 1, H, W) current temperature field
        Returns:
            T_next: (batch, 1, H, W) predicted next temperature
        """
        batch_size, _, H, W = T_curr.shape

        # Flatten and concatenate
        T_flat = T_curr.view(batch_size, -1)
        branch_input = torch.cat([T_flat, params], dim=1)

        # Branch encoding
        branch_output = self.branch_net(branch_input)

        # Create query grid
        y_coords = torch.linspace(-1, 1, H, device=T_curr.device)
        x_coords = torch.linspace(-1, 1, W, device=T_curr.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        trunk_input = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

        # Trunk encoding
        trunk_output = self.trunk_net(trunk_input)

        # Dot product
        output = torch.einsum('bl,pl->bp', branch_output, trunk_output) + self.bias
        output = output.view(batch_size, 1, H, W)

        return output
