"""U-Net (Ronneberger et al., 2015) for 512x512 single-channel panoramic X-rays.

Same topology as the tooth-segmentation U-Net of Helli & Hamamci (2022): four
down-sampling stages, double 3x3 convolutions, skip connections, dropout in the
bottleneck, and a 1x1 classifier with one channel per class.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 32, depth: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        cin = in_channels
        for c in chs[:-1]:
            self.downs.append(DoubleConv(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1], dropout)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c_hi, c_lo in zip(reversed(chs[1:]), reversed(chs[:-1])):
            self.ups.append(nn.ConvTranspose2d(c_hi, c_lo, 2, stride=2))
            self.up_convs.append(DoubleConv(2 * c_lo, c_lo))
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 ** self.depth or x.shape[-2] % 2 ** self.depth:
            raise ValueError(f"input size must be divisible by {2 ** self.depth}, got {tuple(x.shape[-2:])}")
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))
        return self.head(x)
