"""
Helper classes and functions for evaluation notebooks.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Data-only baseline: U-Net without physics conditioning.
    Same architecture as PC-U-Net, but no physics loss.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, n_params=6, depth=4):
        super().__init__()

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # U-Net encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = [in_channels] + [base_channels * (2**i) for i in range(depth)]
        for i in range(depth):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU()
            ))
            self.pools.append(nn.MaxPool2d(2))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1]*2, 3, padding=1),
            nn.BatchNorm2d(channels[-1]*2),
            nn.ReLU(),
            nn.Conv2d(channels[-1]*2, channels[-1]*2, 3, padding=1),
            nn.BatchNorm2d(channels[-1]*2),
            nn.ReLU()
        )

        # U-Net decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth-1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i+1]*2, channels[i+1], 2, stride=2))
            self.decoders.append(nn.Sequential(
                nn.Conv2d(channels[i+1]*2, channels[i+1], 3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU()
            ))

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, T, params):
        # Encode parameters
        param_feat = self.param_encoder(params)  # (B, 128)

        # Encoder path
        skip_connections = []
        x = T
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            x = dec(x)

        # Output
        return self.output(x)
