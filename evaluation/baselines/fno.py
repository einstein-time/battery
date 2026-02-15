"""
Fourier Neural Operator (FNO) implementation.

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution in Fourier space."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

    def complex_mul2d(self, x, weights):
        """Complex multiplication in Fourier space."""
        real_part = torch.einsum('bixyr,ioxyr->boxyr', x[..., 0], weights[..., 0]) - \
                    torch.einsum('bixyr,ioxyr->boxyr', x[..., 1], weights[..., 1])
        imag_part = torch.einsum('bixyr,ioxyr->boxyr', x[..., 0], weights[..., 1]) + \
                    torch.einsum('bixyr,ioxyr->boxyr', x[..., 1], weights[..., 0])
        return torch.stack([real_part, imag_part], dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]

        # FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        # Multiply in Fourier space
        out_ft = torch.zeros(batch_size, self.out_channels, x.shape[-2], x.shape[-1]//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # IFFT
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.shape[-2], x.shape[-1]), norm='ortho')

        return x_out


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D PDEs."""

    def __init__(self, modes1=12, modes2=12, width=64, n_layers=4, in_channels=7, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)

        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W) - includes T_curr and params spatially broadcast
        Returns:
            (batch, out_channels, H, W) - predicted T_next
        """
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        return x
