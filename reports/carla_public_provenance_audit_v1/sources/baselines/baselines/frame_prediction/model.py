"""U-Net for future frame prediction.

Takes T stacked RGB frames (3T input channels) and predicts the next
RGB frame (3 output channels).  Three encoder levels + bottleneck +
three decoder levels with skip connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetFFP(nn.Module):

    def __init__(self, context_frames: int = 4) -> None:
        super().__init__()
        in_ch = 3 * context_frames

        self.enc1 = _ConvBlock(in_ch, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.bottleneck = _ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.dec3 = _ConvBlock(512 + 256, 256)
        self.dec2 = _ConvBlock(256 + 128, 128)
        self.dec1 = _ConvBlock(128 + 64, 64)

        self.head = nn.Conv2d(64, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3*T, H, W) — stacked context frames.

        Returns
        -------
        pred : (B, 3, H, W) — predicted next frame.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
