"""
Inference pipeline for Siamese U-Net change detection.

Usage
-----
    # Python API
    from src.inference import Predictor
    predictor = Predictor("models/best_model.pth", cfg)
    result = predictor.predict(img_before_array, img_after_array)

    # CLI
    python src/inference.py --before path/A.png --after path/B.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model import build_model
from preprocessing import load_image, normalise, to_tensor
from registration import register


class Predictor:
    """End-to-end inference pipeline.

    Parameters
    ----------
    model_path       : path to a .pth checkpoint saved by train.py
    cfg              : project config dict
    threshold        : sigmoid probability threshold for binary decision
    use_registration : whether to ECC-align T2 to T1 before inference
    device           : torch device (auto-detected if None)
    """

    def __init__(self,
                 model_path: str,
                 cfg: dict,
                 threshold: Optional[float] = None,
                 use_registration: Optional[bool] = None,
                 device: Optional[torch.device] = None):

        self.cfg        = cfg
        self.threshold  = threshold if threshold is not None \
                          else cfg["evaluation"]["threshold"]
        self.image_size = cfg["preprocessing"]["image_size"]
        self.use_registration = (cfg["registration"]["enabled"]
                                  if use_registration is None
                                  else use_registration)
        self.reg_cfg = cfg.get("registration", {})

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = build_model(cfg).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _prepare(self,
                 img_before: Union[str, np.ndarray],
                 img_after: Union[str, np.ndarray]
                 ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Load / normalise both images and return tensors + originals."""
        def _to_float(x):
            if isinstance(x, str):
                return load_image(x)
            arr = x.astype(np.float32)
            return arr / 255.0 if arr.max() > 1.0 else arr

        raw_a = _to_float(img_before)
        raw_b = _to_float(img_after)
        orig_a, orig_b = raw_a.copy(), raw_b.copy()

        if self.use_registration:
            try:
                raw_b = register(
                    raw_a, raw_b,
                    method=self.reg_cfg.get("method", "ecc"),
                    warp_mode=self.reg_cfg.get("warp_mode", "translation"),
                    max_iters=self.reg_cfg.get("ecc_max_iters", 100),
                    eps=self.reg_cfg.get("ecc_eps", 1e-5),
                )
            except Exception as e:
                print(f"[inference] Registration failed: {e}; skipping.")

        sz = self.image_size
        proc_a = to_tensor(normalise(cv2.resize(raw_a, (sz, sz)))).unsqueeze(0)
        proc_b = to_tensor(normalise(cv2.resize(raw_b, (sz, sz)))).unsqueeze(0)
        return proc_a, proc_b, orig_a, orig_b

    def predict(self,
                img_before: Union[str, np.ndarray],
                img_after: Union[str, np.ndarray]) -> Dict:
        """Run inference and return a results dict.

        Returns
        -------
        dict with:
          prob_map    : float32 HxW change probabilities [0,1]
          change_mask : uint8 HxW binary mask {0, 1}
          confidence  : float32 HxW — distance from 0.5 (0=uncertain, 0.5=certain)
          raw_before  : float32 HxWx3 original before image
          raw_after   : float32 HxWx3 original after image (pre-registration)
          changed_pct : float percentage of pixels predicted as changed
        """
        t_a, t_b, raw_a, raw_b = self._prepare(img_before, img_after)

        with torch.no_grad():
            prob = torch.sigmoid(
                self.model(t_a.to(self.device), t_b.to(self.device))
            )[0, 0].cpu().numpy()

        change_mask = (prob >= self.threshold).astype(np.uint8)
        return {
            "prob_map":    prob,
            "change_mask": change_mask,
            "confidence":  np.abs(prob - 0.5),
            "raw_before":  raw_a,
            "raw_after":   raw_b,
            "changed_pct": float(change_mask.mean() * 100),
        }

    def predict_and_save(self,
                         img_before: Union[str, np.ndarray],
                         img_after: Union[str, np.ndarray],
                         save_path: str) -> Dict:
        """Run predict() and save a 4-panel visualisation figure."""
        result = self.predict(img_before, img_after)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        data  = [result["raw_before"], result["raw_after"],
                 result["prob_map"],   result["change_mask"]]
        titles = ["Before (T1)", "After (T2)", "Change Probability", "Change Mask"]
        cmaps  = [None, None, "RdYlGn_r", "binary_r"]

        for ax, img, title, cmap in zip(axes, data, titles, cmaps):
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title, fontweight="bold")
            ax.axis("off")

        fig.suptitle(f"Changed pixels: {result['changed_pct']:.1f}%", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return result


def create_change_overlay(img_rgb: np.ndarray,
                           change_mask: np.ndarray,
                           color: Tuple[int, int, int] = (255, 0, 0),
                           alpha: float = 0.5) -> np.ndarray:
    """Overlay detected changes on the RGB image with alpha blending.

    Parameters
    ----------
    img_rgb     : float32 [0,1] or uint8 HxWx3
    change_mask : uint8 HxW binary {0, 1}
    color       : highlight colour (R,G,B)
    alpha       : blending weight for changed regions
    """
    base = (img_rgb * 255).astype(np.uint8).copy() \
           if img_rgb.max() <= 1.0 else img_rgb.astype(np.uint8).copy()
    mask_bool = change_mask.astype(bool)
    blended = base.copy()
    blended[mask_bool] = (
        alpha * np.array(color, dtype=np.float32)
        + (1 - alpha) * base[mask_bool].astype(np.float32)
    ).astype(np.uint8)
    return blended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run change detection inference on an image pair"
    )
    parser.add_argument("--before",    required=True)
    parser.add_argument("--after",     required=True)
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--model",     default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output",    default="results/inference_result.png")
    parser.add_argument("--no_reg",    action="store_true")
    args = parser.parse_args()

    cfg_path = args.config
    for candidate in [cfg_path, os.path.join("..", cfg_path)]:
        if os.path.exists(candidate):
            cfg_path = candidate
            break

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model or cfg["paths"]["best_model"]
    predictor  = Predictor(model_path, cfg,
                           threshold=args.threshold,
                           use_registration=not args.no_reg)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = predictor.predict_and_save(args.before, args.after, args.output)
    print(f"Changed pixel %: {result['changed_pct']:.2f}%")
    print(f"Saved to: {args.output}")
