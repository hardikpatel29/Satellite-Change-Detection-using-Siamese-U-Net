"""
Image registration for satellite change detection.

Before/after images may have sub-pixel offsets from GPS jitter, different
viewing angles, or atmospheric refraction.  Even a 2–3 px shift creates
spurious false detections along building edges.

LEVIR-CD is pre-registered (Google Earth crops), so only a translation-only
ECC warp is needed.  An ORB+homography fallback is provided for datasets
with larger misalignment.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


_WARP_MODES = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean":   cv2.MOTION_EUCLIDEAN,
    "affine":      cv2.MOTION_AFFINE,
    "homography":  cv2.MOTION_HOMOGRAPHY,
}


def register_ecc(img_before: np.ndarray,
                 img_after: np.ndarray,
                 warp_mode: str = "translation",
                 max_iters: int = 100,
                 eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Align img_after to img_before via ECC (Enhanced Correlation Coefficient).

    Parameters
    ----------
    img_before : float32 [0,1] RGB (H×W×3)
    img_after  : float32 [0,1] RGB (H×W×3)
    warp_mode  : 'translation' | 'euclidean' | 'affine' | 'homography'

    Returns
    -------
    (registered_img_after, warp_matrix)
    """
    if warp_mode not in _WARP_MODES:
        raise ValueError(f"Unknown warp_mode '{warp_mode}'. "
                         f"Choose from {list(_WARP_MODES)}")
    motion = _WARP_MODES[warp_mode]

    gray_before = cv2.cvtColor((img_before * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_after  = cv2.cvtColor((img_after  * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    warp_matrix = np.eye(3 if motion == cv2.MOTION_HOMOGRAPHY else 2, 3,
                         dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            gray_before, gray_after, warp_matrix, motion, criteria,
            inputMask=None, gaussFiltSize=5
        )
    except cv2.error as e:
        print(f"[registration] ECC failed ({e}); returning original.")
        return img_after, warp_matrix

    h, w = img_before.shape[:2]
    after_u8 = (img_after * 255).astype(np.uint8)

    if motion == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(after_u8, warp_matrix, (w, h),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(after_u8, warp_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned.astype(np.float32) / 255.0, warp_matrix


def register_orb(img_before: np.ndarray,
                 img_after: np.ndarray,
                 max_features: int = 5000,
                 match_ratio: float = 0.75) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Align img_after to img_before via ORB keypoints + RANSAC homography.

    More robust to large viewpoint changes, but can over-warp pre-registered
    data.  Use only when ECC is insufficient.
    """
    gray_before = cv2.cvtColor((img_before * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_after  = cv2.cvtColor((img_after  * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after,  None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("[registration] ORB: insufficient keypoints; skipping.")
        return img_after, None

    raw = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < match_ratio * n.distance]

    if len(good) < 4:
        print("[registration] ORB: insufficient good matches; skipping.")
        return img_after, None

    pts_before = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_after  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_after, pts_before, cv2.RANSAC, 5.0)

    if H is None:
        print("[registration] ORB: homography failed; skipping.")
        return img_after, None

    h, w = img_before.shape[:2]
    aligned = cv2.warpPerspective((img_after * 255).astype(np.uint8), H, (w, h))
    return aligned.astype(np.float32) / 255.0, H


def register(img_before: np.ndarray,
             img_after: np.ndarray,
             method: str = "ecc",
             **kwargs) -> np.ndarray:
    """Convenience wrapper: align img_after to img_before using 'ecc' or 'orb'."""
    if method == "ecc":
        return register_ecc(img_before, img_after, **kwargs)[0]
    if method == "orb":
        return register_orb(img_before, img_after, **kwargs)[0]
    raise ValueError(f"Unknown registration method: '{method}'. Use 'ecc' or 'orb'.")
