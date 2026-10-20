"""Cross-entropy + soft Dice: CE for per-pixel calibration, Dice against class imbalance
(teeth and especially lesions cover a small fraction of a panoramic X-ray)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 1.0, ignore_background: bool = True,
                 smooth: float = 1.0):
        super().__init__()
        self.num_classes, self.dice_weight = num_classes, dice_weight
        self.ignore_background, self.smooth = ignore_background, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        start = 1 if self.ignore_background else 0
        p, t = probs[:, start:], onehot[:, start:]
        inter = (p * t).sum(dim=(0, 2, 3))
        denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
        dice = ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        return {"loss": ce + self.dice_weight * (1 - dice), "ce": ce, "dice": dice}
