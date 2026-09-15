"""
Full training pipeline for Siamese U-Net change detection.

Usage
-----
    python src/train.py                              # defaults from config.yaml
    python src/train.py --epochs 10 --batch_size 4
    python src/train.py --smoke_test                 # 1 batch × 2 epochs sanity check

Key features
------------
- AdamW + CosineAnnealingLR (configurable)
- pos_weight computed from training data automatically
- Best-model checkpoint saved by validation Dice
- Early stopping with configurable patience
- Training history written to results/training_history.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

sys.path.insert(0, str(Path(__file__).parent))

from dataset import build_dataloaders, compute_pos_weight
from losses import build_loss
from metrics import MetricAccumulator
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device,
              is_train: bool, threshold: float = 0.5) -> dict:
    """Run one full train or evaluation pass and return aggregated metrics."""
    model.train(is_train)
    total_loss = 0.0
    acc = MetricAccumulator(threshold=threshold)
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            img_a = batch["img_a"].to(device)
            img_b = batch["img_b"].to(device)
            mask  = batch["mask"].to(device)

            logits = model(img_a, img_b)
            loss   = criterion(logits, mask)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            acc.update(logits.detach(), mask)
            n_batches += 1

    metrics = acc.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def build_scheduler(optimizer, cfg: dict, train_loader_len: int):
    sched = cfg["training"].get("scheduler", "cosine").lower()
    if sched == "cosine":
        T_max = cfg["training"].get("scheduler_T_max", cfg["training"]["epochs"])
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    if sched == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    if sched == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=5,
                                 factor=0.5, verbose=True)
    return None


def train(cfg: dict, smoke_test: bool = False, override: dict = None) -> list:
    """Run a full training cycle.

    Parameters
    ----------
    cfg        : project config dict
    smoke_test : 1-batch × 2-epoch check to verify the entire pipeline
    override   : flat dict of config overrides, e.g. {"training.epochs": 10}
    """
    if override:
        for key, val in override.items():
            parts = key.split(".")
            node = cfg
            for p in parts[:-1]:
                node = node[p]
            node[parts[-1]] = val

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    for d in ("models_dir", "results_dir", "predictions_dir"):
        os.makedirs(cfg["paths"][d], exist_ok=True)

    print("[train] Building data loaders …")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    print("[train] Computing pos_weight …")
    pos_weight = compute_pos_weight(
        cfg["dataset"]["zip_path"],
        inner_root=cfg["dataset"]["zip_inner_root"],
        split="train",
        max_images=50,
    )
    print(f"[train] pos_weight = {pos_weight:.2f}  ({pos_weight:.0f}:1 imbalance)")

    model = build_model(cfg).to(device)
    print(f"[train] parameters: {model.count_parameters():,}")

    criterion = build_loss(cfg, pos_weight=pos_weight).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    epochs    = 2 if smoke_test else cfg["training"]["epochs"]
    patience  = cfg["training"].get("early_stopping_patience", 10)
    threshold = cfg["evaluation"]["threshold"]

    best_val_dice = -1.0
    no_improve    = 0
    history       = []
    last_val_dice = 0.0

    print(f"\n[train] Starting {'SMOKE TEST' if smoke_test else 'full training'} "
          f"for {epochs} epochs")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if smoke_test:
            model.train()
            batch  = next(iter(train_loader))
            logits = model(batch["img_a"].to(device), batch["img_b"].to(device))
            loss   = criterion(logits, batch["mask"].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_m = {"loss": loss.item(), "iou": 0, "dice": 0,
                       "precision": 0, "recall": 0, "accuracy": 0}
        else:
            train_m = run_epoch(model, train_loader, criterion,
                                optimizer, device, is_train=True,
                                threshold=threshold)

        val_m = run_epoch(model, val_loader, criterion, optimizer,
                          device, is_train=False, threshold=threshold)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_m["dice"])
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        last_val_dice = val_m["dice"]

        row = {
            "epoch":         epoch,
            "train_loss":    round(train_m["loss"],      4),
            "val_loss":      round(val_m["loss"],        4),
            "val_iou":       round(val_m["iou"],         4),
            "val_dice":      round(last_val_dice,        4),
            "val_precision": round(val_m["precision"],   4),
            "val_recall":    round(val_m["recall"],      4),
            "val_accuracy":  round(val_m["accuracy"],    4),
            "lr":            round(lr_now, 8),
            "elapsed_s":     round(time.time() - t0, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss {train_m['loss']:.4f}/{val_m['loss']:.4f} | "
            f"IoU {val_m['iou']:.4f} | Dice {last_val_dice:.4f} | "
            f"Prec {val_m['precision']:.4f} | Rec {val_m['recall']:.4f} | "
            f"LR {lr_now:.2e} | {row['elapsed_s']}s"
        )

        if last_val_dice > best_val_dice:
            best_val_dice = last_val_dice
            no_improve = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice":        best_val_dice,
                "cfg":             cfg,
            }, cfg["paths"]["best_model"])
            print(f"  ✔ Best model saved (Val Dice = {best_val_dice:.4f})")
        else:
            no_improve += 1

        if not smoke_test and no_improve >= patience:
            print(f"\n[train] Early stopping after {patience} epochs without improvement.")
            break

    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_dice":    last_val_dice,
        "cfg":         cfg,
    }, cfg["paths"]["final_model"])
    print(f"\n[train] Final model → {cfg['paths']['final_model']}")

    hist_path = cfg["paths"]["training_history_file"]
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] History    → {hist_path}")
    print(f"[train] Best Val Dice: {best_val_dice:.4f}")
    return history


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Siamese U-Net for satellite change detection"
    )
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--smoke_test", action="store_true")
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

    overrides = {}
    if args.epochs is not None:     overrides["training.epochs"]        = args.epochs
    if args.batch_size is not None: overrides["training.batch_size"]    = args.batch_size
    if args.lr is not None:         overrides["training.learning_rate"] = args.lr

    train(cfg, smoke_test=args.smoke_test, override=overrides or None)
