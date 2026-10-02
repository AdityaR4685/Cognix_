"""PointNet for per-point semantic segmentation (no spatial transformers).

Architecture: shared MLP → global max-pool → concat local+global → seg MLP.
All operations are point-wise Conv1d (kernel_size=1) + one global max-pool,
so the model accepts arbitrary point counts at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetSeg(nn.Module):

    def __init__(self, num_classes: int, input_dim: int = 3, dropout_p: float = 0.3) -> None:
        super().__init__()

        self.feat = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )

        self.seg = nn.Sequential(
            nn.Conv1d(2048, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv1d(128, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, pts: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        pts : (B, N, input_dim)
        pad_mask : (B, N) bool, optional — True for real points, False for padding.

        Returns
        -------
        logits : (B, N, num_classes)
        """
        x = pts.permute(0, 2, 1)                          # (B, D, N)
        local_feat = self.feat(x)                          # (B, 1024, N)

        if pad_mask is not None:
            local_feat_for_pool = local_feat.masked_fill(~pad_mask.unsqueeze(1), float("-inf"))
        else:
            local_feat_for_pool = local_feat
        global_feat = local_feat_for_pool.max(dim=2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.expand_as(local_feat)    # (B, 1024, N)

        combined = torch.cat([local_feat, global_feat], dim=1)  # (B, 2048, N)
        out = self.seg(combined)                           # (B, 128, N)
        logits = self.head(out)                            # (B, C, N)
        return logits.permute(0, 2, 1)                     # (B, N, C)
