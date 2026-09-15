"""
Evaluate a trained Siamese U-Net on the LEVIR-CD test set.

Usage
-----
    python src/evaluate.py
    python src/evaluate.py --model models/final_model.pth
    python src/evaluate.py --threshold 0.4

Outputs
-------
  results/metrics.json              test-set evaluation metrics
  results/predictions/*_panel.png  5-panel visual (Before|After|GT|Prob|Pred)
  results/predictions/*_errors.png error-type map (TP/TN/FP/FN)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset import build_dataloaders
from metrics import MetricAccumulator
from model import build_model
from preprocessing import IMAGENET_MEAN, IMAGENET_STD


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 HxWx3."""
    img = tensor.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_prediction_panel(img_a, img_b, gt_mask, pred_prob, pred_mask,
                           filename: str, save_path: str):
    """Save a 5-panel figure: Before | After | Ground Truth | Prob | Prediction."""
    fig = plt.figure(figsize=(20, 4))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)
    panels = [
        (img_a,     "Before (T1)",  None),
        (img_b,     "After (T2)",   None),
        (gt_mask,   "Ground Truth", "binary_r"),
        (pred_prob, "Change Prob",  "RdYlGn_r"),
        (pred_mask, "Prediction",   "binary_r"),
    ]
    for i, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(data) if cmap is None else ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Change Detection: {filename}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def classify_errors(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Build an RGB error map: green=TP, white=TN, red=FP, blue=FN."""
    err = np.full((*gt_mask.shape, 3), 255, dtype=np.uint8)
    err[(pred_mask == 1) & (gt_mask == 1)] = [0,   200, 0]
    err[(pred_mask == 1) & (gt_mask == 0)] = [220, 50,  50]
    err[(pred_mask == 0) & (gt_mask == 1)] = [50,  50,  220]
    return err


def evaluate(cfg: dict, model_path: str = None, threshold: float = None,
             num_visuals: int = None) -> dict:
    """Evaluate on the test set; save metrics and visual panels.

    Parameters
    ----------
    cfg         : parsed config dict
    model_path  : checkpoint path (overrides config default)
    threshold   : decision threshold (overrides config default)
    num_visuals : number of visual panels to save
    """
    from PIL import Image as PILImage

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or cfg["paths"]["best_model"]
    threshold  = threshold  or cfg["evaluation"]["threshold"]
    num_vis    = num_visuals or cfg["evaluation"]["num_visual_samples"]

    os.makedirs(cfg["paths"]["predictions_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"],     exist_ok=True)

    print(f"[evaluate] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[evaluate] Epoch {ckpt.get('epoch', '?')}  "
          f"Val Dice {ckpt.get('val_dice', 0):.4f}")

    _, _, test_loader = build_dataloaders(cfg)

    acc = MetricAccumulator(threshold=threshold)
    vis_count = 0

    with torch.no_grad():
        for batch in test_loader:
            img_a = batch["img_a"].to(device)
            img_b = batch["img_b"].to(device)
            mask  = batch["mask"].to(device)

            logits   = model(img_a, img_b)
            acc.update(logits.detach(), mask)

            probs_np = torch.sigmoid(logits).cpu().numpy()
            preds_np = (probs_np >= threshold).astype(np.uint8)

            for i in range(img_a.size(0)):
                if vis_count >= num_vis:
                    break

                fname    = batch["filename"][i]
                prob_np  = probs_np[i, 0]
                pred_np  = preds_np[i, 0]
                gt_np    = mask[i, 0].cpu().numpy().astype(np.uint8)

                panel_path = os.path.join(
                    cfg["paths"]["predictions_dir"],
                    fname.replace(".png", "_panel.png")
                )
                save_prediction_panel(
                    denormalise(img_a[i].cpu()),
                    denormalise(img_b[i].cpu()),
                    gt_np, prob_np, pred_np,
                    fname, panel_path
                )

                err_path = panel_path.replace("_panel.png", "_errors.png")
                PILImage.fromarray(classify_errors(pred_np, gt_np)).save(err_path)
                vis_count += 1

    metrics = acc.compute()
    metrics.update({"threshold": threshold, "checkpoint": model_path})

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>15}: {v:.4f}" if isinstance(v, float) else f"  {k:>15}: {v}")
    print("\n  Note: IoU and Dice are the primary metrics; "
          "accuracy is inflated by the 96% unchanged background.")

    with open(cfg["paths"]["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[evaluate] Metrics → {cfg['paths']['metrics_file']}")
    print(f"[evaluate] {vis_count} panels → {cfg['paths']['predictions_dir']}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Siamese U-Net on LEVIR-CD test set"
    )
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--model",        default=None)
    p.add_argument("--threshold",    type=float, default=None)
    p.add_argument("--num_visuals",  type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_path = args.config
    for candidate in [cfg_path, os.path.join("..", cfg_path)]:
        if os.path.exists(candidate):
            cfg_path = candidate
            break

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, model_path=args.model, threshold=args.threshold,
             num_visuals=args.num_visuals)
