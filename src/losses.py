"""
Loss functions for binary change detection on heavily imbalanced data.

LEVIR-CD has ~4% changed pixels, so vanilla BCE would drive the model to
predict "no change" everywhere.  We combine:
  1. BCE with pos_weight  — scales changed-pixel loss by neg/pos ratio (~24×)
  2. Soft Dice loss       — optimises F1 directly, immune to class counts
  total = α·BCE + β·Dice
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Operates on sigmoid probabilities rather than hard predictions so it
    remains differentiable everywhere.  The smooth term prevents division
    by zero on empty masks.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        tgts = targets.view(targets.size(0), -1)
        intersection = (probs * tgts).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + tgts.sum(dim=1) + self.smooth
        )
        return (1.0 - dice).mean()


class CombinedLoss(nn.Module):
    """BCE-with-logits + Dice loss.

    Parameters
    ----------
    bce_weight  : scalar weight for BCE term
    dice_weight : scalar weight for Dice term
    pos_weight  : neg/pos pixel ratio passed to BCEWithLogitsLoss
    smooth      : Dice smoothing constant
    """

    def __init__(self,
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 pos_weight: Optional[float] = None,
                 smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

        pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
        if pw is not None:
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                  pos_weight=self.pos_weight)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss values for logging purposes."""
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                   pos_weight=self.pos_weight)
        dice = self.dice_loss(logits, targets)
        return {
            "bce": bce.item(),
            "dice": dice.item(),
            "total": (self.bce_weight * bce + self.dice_weight * dice).item(),
        }


def build_loss(cfg: dict, pos_weight: Optional[float] = None) -> CombinedLoss:
    """Build CombinedLoss from the project config dict."""
    loss_cfg = cfg.get("loss", {})
    pw = pos_weight if pos_weight is not None else loss_cfg.get("pos_weight")
    return CombinedLoss(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        pos_weight=pw,
    )
