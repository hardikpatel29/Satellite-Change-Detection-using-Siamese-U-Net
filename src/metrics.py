"""
Evaluation metrics for binary change detection.

Pixel accuracy is misleading on LEVIR-CD (~96% unchanged), so we primarily
report IoU and Dice which measure overlap on the minority (changed) class:

  IoU  = TP / (TP + FP + FN)
  Dice = 2·TP / (2·TP + FP + FN)

Precision and recall are also tracked to understand the precision/recall
trade-off when adjusting the decision threshold.
"""

from typing import Dict, Optional

import numpy as np
import torch


def compute_metrics(preds: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary segmentation metrics for a single batch.

    Parameters
    ----------
    preds    : (B, 1, H, W) logits or probabilities
    targets  : (B, 1, H, W) binary ground-truth masks {0, 1}
    threshold: decision threshold applied to sigmoid probabilities
    """
    probs = torch.sigmoid(preds) if (preds.min() < 0 or preds.max() > 1) else preds
    pred_flat = (probs >= threshold).float().view(-1)
    tgt_flat = targets.float().view(-1)

    tp = (pred_flat * tgt_flat).sum().item()
    fp = (pred_flat * (1 - tgt_flat)).sum().item()
    fn = ((1 - pred_flat) * tgt_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - tgt_flat)).sum().item()

    eps = 1e-7
    return {
        "iou":       tp / (tp + fp + fn + eps),
        "dice":      2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall":    tp / (tp + fn + eps),
        "accuracy":  (tp + tn) / (tp + tn + fp + fn + eps),
    }


class MetricAccumulator:
    """Accumulates TP/FP/FN/TN counts across batches for micro-averaged metrics.

    Micro-averaging is more reliable than per-image averaging when images vary
    widely in how much change they contain.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(preds) if (preds.min() < 0 or preds.max() > 1) else preds
        pred_flat = (probs >= self.threshold).float().view(-1)
        tgt_flat = targets.float().view(-1)

        self.tp += (pred_flat * tgt_flat).sum().item()
        self.fp += (pred_flat * (1 - tgt_flat)).sum().item()
        self.fn += ((1 - pred_flat) * tgt_flat).sum().item()
        self.tn += ((1 - pred_flat) * (1 - tgt_flat)).sum().item()

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-7
        return {
            "iou":       tp / (tp + fp + fn + eps),
            "dice":      2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall":    tp / (tp + fn + eps),
            "accuracy":  (tp + tn) / (tp + tn + fp + fn + eps),
        }


def find_optimal_threshold(probs_list: list,
                           targets_list: list,
                           thresholds: Optional[list] = None) -> Dict:
    """Find the threshold that maximises Dice over a validation set.

    Parameters
    ----------
    probs_list   : list of 1-D numpy probability arrays
    targets_list : list of 1-D numpy binary target arrays
    thresholds   : thresholds to search; defaults to 17 values in [0.1, 0.9]
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17).tolist()

    all_probs = np.concatenate(probs_list)
    all_targets = np.concatenate(targets_list)
    eps = 1e-7

    results = []
    for t in thresholds:
        pred = (all_probs >= t).astype(np.float32)
        tp = (pred * all_targets).sum()
        fp = (pred * (1 - all_targets)).sum()
        fn = ((1 - pred) * all_targets).sum()
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        results.append({"threshold": t, "dice": float(dice)})

    best = max(results, key=lambda r: r["dice"])
    return {"best_threshold": best["threshold"],
            "best_dice": best["dice"],
            "all_results": results}
