"""
PyTorch Dataset and DataLoader factory for LEVIR-CD change detection.

Images are read on-the-fly from inside the ZIP archive to avoid unpacking
~2.4 GB to disk.  The ZIP handle is cached globally so it isn't reopened
on every __getitem__ call.

Each sample returned by __getitem__:
  {
    "img_a":    Tensor (3, H, W)  before image, normalised
    "img_b":    Tensor (3, H, W)  after image, normalised (optionally registered)
    "mask":     Tensor (1, H, W)  binary change mask {0, 1}
    "filename": str               e.g. "train_1.png"
  }
"""

import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from preprocessing import load_image, load_mask, preprocess_train, preprocess_eval
from registration import register


_ZIP_CACHE: Dict[str, zipfile.ZipFile] = {}


def _get_zip(zip_path: str) -> zipfile.ZipFile:
    if zip_path not in _ZIP_CACHE:
        _ZIP_CACHE[zip_path] = zipfile.ZipFile(zip_path, "r")
    return _ZIP_CACHE[zip_path]


def _read_rgb_from_zip(zf: zipfile.ZipFile, inner_path: str) -> np.ndarray:
    """Return float32 [0,1] HxWx3 RGB array from a ZIP entry."""
    return np.array(
        Image.open(io.BytesIO(zf.read(inner_path))).convert("RGB"),
        dtype=np.float32
    ) / 255.0


def _read_mask_from_zip(zf: zipfile.ZipFile, inner_path: str) -> np.ndarray:
    """Return uint8 {0,1} HxW mask from a ZIP entry."""
    arr = np.array(Image.open(io.BytesIO(zf.read(inner_path))).convert("L"),
                   dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


class LEVIRCDDataset(Dataset):
    """LEVIR-CD change detection dataset.

    Parameters
    ----------
    zip_path         : path to the dataset ZIP
    split            : 'train' | 'val' | 'test'
    image_size       : spatial crop/resize target
    use_registration : run ECC alignment on each pair before training
    reg_cfg          : registration config dict
    inner_root       : top-level folder inside the ZIP
    """

    def __init__(self,
                 zip_path: str,
                 split: str,
                 image_size: int = 256,
                 use_registration: bool = True,
                 reg_cfg: Optional[dict] = None,
                 inner_root: str = "LEVIR CD"):

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")

        self.zip_path = zip_path
        self.split = split
        self.image_size = image_size
        self.use_registration = use_registration
        self.reg_cfg = reg_cfg or {}
        self.inner_root = inner_root

        zf = _get_zip(zip_path)
        names = zf.namelist()

        def _list(sub):
            prefix = f"{inner_root}/{split}/{sub}/"
            return sorted(n for n in names
                          if n.startswith(prefix) and n.endswith(".png"))

        self.files_a = _list("A")
        self.files_b = _list("B")
        self.files_label = _list("label")
        self._sanity_check()

    def _sanity_check(self):
        if not self.files_a:
            raise RuntimeError(f"No before-images found for split='{self.split}'")
        for files, label in [(self.files_b, "after"), (self.files_label, "label")]:
            if len(self.files_a) != len(files):
                raise RuntimeError(
                    f"File count mismatch: {len(self.files_a)} before "
                    f"vs {len(files)} {label}")
        for fa, fb, fl in zip(self.files_a, self.files_b, self.files_label):
            na, nb, nl = (os.path.basename(p) for p in (fa, fb, fl))
            if na != nb or na != nl:
                raise RuntimeError(f"Filename mismatch: {na} | {nb} | {nl}")

    def __len__(self) -> int:
        return len(self.files_a)

    def __getitem__(self, idx: int) -> Dict:
        zf = _get_zip(self.zip_path)
        img_a = _read_rgb_from_zip(zf, self.files_a[idx])
        img_b = _read_rgb_from_zip(zf, self.files_b[idx])
        mask  = _read_mask_from_zip(zf, self.files_label[idx])

        if self.use_registration:
            try:
                img_b = register(
                    img_a, img_b,
                    method=self.reg_cfg.get("method", "ecc"),
                    warp_mode=self.reg_cfg.get("warp_mode", "translation"),
                    max_iters=self.reg_cfg.get("ecc_max_iters", 100),
                    eps=self.reg_cfg.get("ecc_eps", 1e-5),
                )
            except Exception as e:
                print(f"[dataset] Registration error on {self.files_a[idx]}: {e}")

        if self.split == "train":
            t_a, t_b, t_mask = preprocess_train(img_a, img_b, mask, self.image_size)
        else:
            t_a, t_b, t_mask = preprocess_eval(img_a, img_b, mask, self.image_size)

        return {"img_a": t_a, "img_b": t_b, "mask": t_mask,
                "filename": os.path.basename(self.files_a[idx])}

    def get_image_info(self) -> Dict:
        return {"split": self.split, "num_samples": len(self),
                "image_size": self.image_size}


def build_dataloaders(cfg: dict,
                      use_registration: bool = True
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from the project config."""
    zip_path   = cfg["dataset"]["zip_path"]
    inner_root = cfg["dataset"]["zip_inner_root"]
    image_size = cfg["preprocessing"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    reg_cfg    = cfg.get("registration", {})
    reg_enabled = reg_cfg.get("enabled", True) and use_registration

    def _make(split: str, shuffle: bool) -> DataLoader:
        ds = LEVIRCDDataset(
            zip_path=zip_path, split=split, image_size=image_size,
            use_registration=reg_enabled, reg_cfg=reg_cfg,
            inner_root=inner_root,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False,
                          drop_last=(split == "train"))

    return _make("train", True), _make("val", False), _make("test", False)


def compute_pos_weight(zip_path: str,
                       inner_root: str = "LEVIR CD",
                       split: str = "train",
                       max_images: int = 100) -> float:
    """Estimate pos_weight = neg_pixels / pos_pixels for BCEWithLogitsLoss.

    Samples up to max_images from the given split to compute the ratio.
    """
    zf = _get_zip(zip_path)
    prefix = f"{inner_root}/{split}/label/"
    label_files = sorted(
        n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png")
    )[:max_images]

    total_pos = total_neg = 0
    for lf in label_files:
        mask = _read_mask_from_zip(zf, lf)
        total_pos += int(mask.sum())
        total_neg += int((1 - mask).sum())

    return total_neg / total_pos if total_pos > 0 else 1.0
