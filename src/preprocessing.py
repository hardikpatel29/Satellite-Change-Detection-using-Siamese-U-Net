"""
Image preprocessing utilities for satellite change detection.

Design notes:
- Geometric augmentations (crop, flip, rotate) are applied identically to
  the before image, after image, and mask to keep them geometrically consistent.
- Colour augmentations (jitter, blur) are applied *independently* to the before
  and after images to simulate realistic sensor and illumination differences.
- ImageNet normalisation statistics are used as a sensible prior for satellite
  RGB imagery; they also allow future encoder pre-training without changes.
"""

import random
from typing import Tuple, Optional

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str) -> np.ndarray:
    """Load an RGB image as a float32 [0, 1] numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    """Load a change mask and binarise to {0, 1} (input values are 0 or 255)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def normalise(img: np.ndarray,
              mean: np.ndarray = IMAGENET_MEAN,
              std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    return (img - mean) / std


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1))


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask).unsqueeze(0).float()


def random_crop(img_a: np.ndarray,
                img_b: np.ndarray,
                mask: np.ndarray,
                crop_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop of the same region from img_a, img_b, and mask."""
    h, w = img_a.shape[:2]
    if h < crop_size or w < crop_size:
        img_a = cv2.resize(img_a, (crop_size, crop_size))
        img_b = cv2.resize(img_b, (crop_size, crop_size))
        mask  = cv2.resize(mask,  (crop_size, crop_size),
                           interpolation=cv2.INTER_NEAREST)
        return img_a, img_b, mask
    top  = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (img_a[top:top+crop_size, left:left+crop_size],
            img_b[top:top+crop_size, left:left+crop_size],
            mask [top:top+crop_size, left:left+crop_size])


def center_crop(img_a: np.ndarray,
                img_b: np.ndarray,
                mask: np.ndarray,
                crop_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre crop of the same region from img_a, img_b, and mask."""
    h, w = img_a.shape[:2]
    top  = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)
    return (img_a[top:top+crop_size, left:left+crop_size],
            img_b[top:top+crop_size, left:left+crop_size],
            mask [top:top+crop_size, left:left+crop_size])


def random_horizontal_flip(img_a, img_b, mask, p=0.5):
    if random.random() < p:
        img_a = img_a[:, ::-1].copy()
        img_b = img_b[:, ::-1].copy()
        mask  = mask [:, ::-1].copy()
    return img_a, img_b, mask


def random_vertical_flip(img_a, img_b, mask, p=0.5):
    if random.random() < p:
        img_a = img_a[::-1].copy()
        img_b = img_b[::-1].copy()
        mask  = mask [::-1].copy()
    return img_a, img_b, mask


def random_rotation(img_a, img_b, mask,
                    angles: Tuple[int, ...] = (0, 90, 180, 270)):
    """Rotate all three arrays by a randomly chosen angle."""
    k = random.choice(angles) // 90
    if k == 0:
        return img_a, img_b, mask
    return (np.rot90(img_a, k).copy(),
            np.rot90(img_b, k).copy(),
            np.rot90(mask,  k).copy())


def random_color_jitter(img: np.ndarray,
                        brightness: float = 0.2,
                        contrast: float = 0.2,
                        saturation: float = 0.2,
                        p: float = 0.5) -> np.ndarray:
    """Brightness/contrast/saturation jitter on a [0,1] RGB image.

    Applied independently to before and after images to model sensor
    and illumination differences between acquisitions.
    """
    if random.random() > p:
        return img

    factor = 1.0 + random.uniform(-brightness, brightness)
    img = np.clip(img * factor, 0.0, 1.0)

    mean = img.mean()
    factor = 1.0 + random.uniform(-contrast, contrast)
    img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

    if saturation > 0:
        img_u8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * (1.0 + random.uniform(-saturation, saturation)),
            0, 255
        )
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return img


def random_gaussian_blur(img: np.ndarray, p: float = 0.2) -> np.ndarray:
    if random.random() > p:
        return img
    ksize = random.choice([3, 5])
    blurred = cv2.GaussianBlur((img * 255).astype(np.uint8), (ksize, ksize), 0)
    return blurred.astype(np.float32) / 255.0


def preprocess_train(img_a: np.ndarray,
                     img_b: np.ndarray,
                     mask: np.ndarray,
                     image_size: int = 256
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training pipeline: random crop → flips → rotation → colour jitter → normalise."""
    img_a, img_b, mask = random_crop(img_a, img_b, mask, image_size)
    img_a, img_b, mask = random_horizontal_flip(img_a, img_b, mask)
    img_a, img_b, mask = random_vertical_flip(img_a, img_b, mask)
    img_a, img_b, mask = random_rotation(img_a, img_b, mask)
    img_a = random_gaussian_blur(random_color_jitter(img_a, p=0.5), p=0.2)
    img_b = random_gaussian_blur(random_color_jitter(img_b, p=0.5), p=0.2)
    return to_tensor(normalise(img_a)), to_tensor(normalise(img_b)), mask_to_tensor(mask)


def preprocess_eval(img_a: np.ndarray,
                    img_b: np.ndarray,
                    mask: np.ndarray,
                    image_size: int = 256
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Val/test pipeline: centre crop → normalise (no augmentation)."""
    img_a, img_b, mask = center_crop(img_a, img_b, mask, image_size)
    return to_tensor(normalise(img_a)), to_tensor(normalise(img_b)), mask_to_tensor(mask)


def preprocess_inference(img_a: np.ndarray,
                         img_b: np.ndarray,
                         image_size: int = 256
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inference pipeline: resize → normalise (no mask required)."""
    img_a = cv2.resize(img_a, (image_size, image_size))
    img_b = cv2.resize(img_b, (image_size, image_size))
    return to_tensor(normalise(img_a)), to_tensor(normalise(img_b))
