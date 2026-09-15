# DATASET.md — LEVIR-CD Change Detection Dataset

## Dataset Discovered

| Property | Value |
|---|---|
| **Dataset Name** | LEVIR-CD (Building Change Detection Dataset) |
| **Source ZIP** | `Satellite image data set.zip` (2.36 GB) |
| **Task** | Binary pixel-level building change detection |
| **Image Format** | PNG |
| **Image Resolution** | 1024 × 1024 pixels |
| **Image Channels** | RGB (3 channels) |
| **Mask Values** | 0 = no change, 255 = change |
| **Change Ratio** | ~4% changed pixels (strong class imbalance) |

---

## Dataset Split

The dataset is **pre-split** into train / validation / test partitions.
No additional splitting was performed.

| Split | Image Pairs | Approx % |
|---|---|---|
| Train | **445** | ~70% |
| Val | **64** | ~10% |
| Test | **128** | ~20% |
| **Total** | **637** | 100% |

---

## Directory Structure (inside ZIP)

```
Satellite image data set.zip
└── LEVIR CD/
    ├── train/
    │   ├── A/             ← Before images (T1) — 445 PNG files
    │   │   ├── train_1.png
    │   │   ├── train_2.png
    │   │   └── ...
    │   ├── B/             ← After images (T2) — 445 PNG files
    │   │   ├── train_1.png
    │   │   └── ...
    │   └── label/         ← Binary change masks — 445 PNG files
    │       ├── train_1.png
    │       └── ...
    ├── val/
    │   ├── A/             ← 64 files
    │   ├── B/             ← 64 files
    │   └── label/         ← 64 files
    └── test/
        ├── A/             ← 128 files
        ├── B/             ← 128 files
        └── label/         ← 128 files
```

### File Naming Convention

Files are named `{split}_{index}.png` where:
- `split` is one of `train`, `val`, `test`
- `index` is an integer from 1 onwards

Example: `train_1.png` in `A/`, `B/`, and `label/` all refer to the same geographic location.

---

## Image Properties

| Property | Before/After Images | Change Masks |
|---|---|---|
| Format | PNG | PNG |
| Mode | RGB | Grayscale (L) |
| Data type | uint8 | uint8 |
| Value range | 0–255 per channel | {0, 255} |
| Resolution | 1024 × 1024 px | 1024 × 1024 px |

---

## About LEVIR-CD

LEVIR-CD is a large-scale building change detection dataset focused on:
- **Domain**: Urban/suburban building footprint changes
- **Geography**: Various cities in China
- **Time span**: Changes captured over 5–14 years
- **Image source**: Google Earth high-resolution imagery
- **Change types**: Building construction, building demolition
- **Pre-registered**: Yes — images are aligned to the same geographic viewport

### Why LEVIR-CD for this project?

1. **Realistic**: Real high-resolution satellite imagery, not synthetic
2. **Well-labelled**: Pixel-precise binary change masks
3. **Benchmark**: Widely used in change detection research; results are comparable
4. **Pre-split**: Clean train/val/test split avoids data leakage
5. **Pre-registered**: Saves the need for aggressive geometric alignment

---

## Class Imbalance

The dataset has significant class imbalance:

```
Unchanged pixels: ~96%
Changed pixels:   ~4%
Imbalance ratio:  ~24:1
```

This is handled in training via:
1. **pos_weight** in `BCEWithLogitsLoss` (scales loss of changed pixels by ~24)
2. **Dice Loss** (inherently insensitive to class imbalance)

---

## Data Usage in this Project

The dataset ZIP is **never extracted to disk** — images are read directly from the ZIP
file on-the-fly using Python's `zipfile` module. This:
- Saves ~2.36 GB of disk space
- Keeps the dataset in its original form
- Slightly increases CPU overhead per batch (acceptable without GPU)

---

## Citation

```
@article{chen2021remote,
  title={Remote sensing image change detection with transformers},
  author={Chen, Hao and Qi, Zipeng and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--14},
  year={2021},
  publisher={IEEE}
}
```

Original dataset source: https://github.com/justchenhao/BIT_CD
