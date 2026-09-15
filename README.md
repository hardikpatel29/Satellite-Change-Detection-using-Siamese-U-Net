# 🛰️ Satellite Image Change Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

This project implements a complete **end-to-end satellite image change detection pipeline** using a
**Siamese U-Net** deep learning architecture. The system compares before/after satellite image pairs
and produces **pixel-level binary change maps** indicating which regions of the scene changed.

**Trained and evaluated on**: [LEVIR-CD](https://github.com/justchenhao/BIT_CD) —
a large-scale building change detection benchmark with 637 high-resolution (1024×1024) image pairs.

---

## 🎯 Problem Statement

Detecting changes in satellite imagery is critical for:
- 🌲 **Deforestation monitoring** — track illegal logging at planetary scale
- 🏗️ **Infrastructure damage** — post-disaster building damage assessment
- 🏙️ **Urban development** — monitor construction and land use change
- 🌊 **Disaster assessment** — flood extent and landslide mapping

Manual inspection at global scale is impossible. Automated change detection enables
near real-time monitoring with high spatial precision.

---

## ❓ Why Change Detection?

Traditional image differencing (simple pixel subtraction) fails because:
1. **Illumination differences** — same scene looks different at different times of day/year
2. **Sensor differences** — different satellites produce different colour responses
3. **Seasonal changes** — vegetation appearance varies with season
4. **Registration errors** — sub-pixel misalignment causes false edges

Deep learning change detection learns to ignore these irrelevant differences and focus on
**semantically meaningful** changes.

---

## 📊 Dataset: LEVIR-CD

| Property | Value |
|---|---|
| Dataset | LEVIR-CD |
| Total image pairs | 637 |
| Train / Val / Test | 445 / 64 / 128 |
| Image size | 1024 × 1024 px |
| Format | PNG (RGB) |
| Mask values | 0 = unchanged, 255 = changed |
| Change ratio | ~4% changed pixels |

See [DATASET.md](DATASET.md) for detailed dataset documentation.

### Dataset Structure

```
Satellite image data set.zip
└── LEVIR CD/
    ├── train/A/     ← before images (T1) — 445 files
    ├── train/B/     ← after  images (T2) — 445 files
    ├── train/label/ ← change masks       — 445 files
    ├── val/  ...
    └── test/ ...
```

---

## 🔧 Preprocessing

The preprocessing pipeline handles:

1. **Image loading**: Direct from ZIP (no extraction needed)
2. **Normalisation**: ImageNet statistics (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
3. **Spatial augmentation**: Random crop / flip / 90° rotation — **same transform** applied to before, after, and mask
4. **Intensity augmentation**: Color jitter, Gaussian blur — **independently** applied to before and after
5. **Tensor conversion**: HxWxC numpy → CxHxW torch tensor

Model input size: **256 × 256** (random crop from 1024 × 1024)

---

## 📐 Image Registration

Registration aligns the after-image (T2) to the before-image (T1) to correct spatial misalignment.

**Method**: ECC (Enhanced Correlation Coefficient) with translation-only warp model.

**Rationale**: LEVIR-CD is pre-aligned (Google Earth crops), so only small sub-pixel
GPS jitter needs correction — not full homography warps which would distort building edges.

Configurable via `config.yaml`:
```yaml
registration:
  enabled: true
  method: "ecc"          # or "orb"
  warp_mode: "translation"
```

---

## 🧠 Siamese U-Net Architecture

```
  BEFORE IMAGE (3×256×256)     AFTER IMAGE (3×256×256)
         │                             │
  ┌──────┴──────┐               ┌──────┴──────┐
  │  Shared     │               │  Shared     │   ← IDENTICAL WEIGHTS
  │  Encoder    │               │  Encoder    │
  └──────┬──────┘               └──────┬──────┘
         │           |A−B|             │
         └──────►  Feature Diff ◄──────┘   (per-scale)
                       │
                  U-Net Decoder
                  + Skip Connections
                       │
                  1×1 Conv → Logit
                       │
                  σ → Probability Map
                       │
                  Threshold @ 0.5
                       │
                  Change Mask
```

### Why Shared Weights?

Both images are encoded by the **same encoder with the same weights**, ensuring features
from T1 and T2 live in the same embedding space. Without shared weights, the difference
`|f_A - f_B|` would be meaningless.

### Why U-Net?

U-Net's **skip connections** preserve fine spatial detail through the decoder, enabling
**pixel-precise** change masks rather than coarse blobs.

### Feature Comparison

`|feat_A - feat_B|` is used because:
- **Symmetric**: order of T1/T2 doesn't matter
- **Interpretable**: naturally near-zero for unchanged regions
- **Compact**: one representation instead of concatenated pairs

---

## 📉 Loss Function

```
Total Loss = BCE_weight × BCE(pos_weight) + Dice_weight × Dice
```

### BCE with pos_weight

LEVIR-CD has a **~24:1 class imbalance** (unchanged:changed). `pos_weight ≈ 24` scales
the loss contribution of changed pixels proportionally, preventing the model from
just predicting "no change" everywhere.

### Dice Loss

Dice = 2·TP / (2·TP + FP + FN)

Directly optimises the F1/Dice overlap. Naturally insensitive to class imbalance
because it does not involve TN counts.

---

## 🚀 Training Procedure

| Hyperparameter | Default Value |
|---|---|
| Epochs | 50 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR |
| Early stopping | Patience = 10 |
| Random seed | 42 |
| Device | CUDA if available, else CPU |

**Best model** is selected by validation Dice score.

---

## 📈 Evaluation Metrics

| Metric | Definition | Why important |
|---|---|---|
| **IoU** | TP / (TP + FP + FN) | Measures change region overlap |
| **Dice / F1** | 2·TP / (2·TP + FP + FN) | Harmonic mean of precision & recall |
| **Precision** | TP / (TP + FP) | How reliable are the detections? |
| **Recall** | TP / (TP + FN) | What fraction of changes are found? |
| **Pixel Accuracy** | (TP+TN) / All | Misleading alone (high due to imbalance) |

### Why accuracy is insufficient

A model predicting "no change" everywhere achieves ~96% accuracy on LEVIR-CD but 0% IoU.
**Always report IoU and Dice** for change detection tasks.

---

## 📊 Results

> ⚠️ Run the training pipeline to generate actual results. Results will be populated after training completes.

Metrics are saved to `results/metrics.json` after running:
```bash
python src/evaluate.py
```

---

## 🔬 Failure Cases

Common model failure modes:
- **Shadows**: Cast shadows from buildings appear as "change" due to seasonal sun angle differences
- **Seasonal vegetation**: Tree canopy density changes between seasons create false positives
- **Illumination**: Haze, cloud cover, or sun angle differences affect pixel values
- **Small objects**: Buildings < 10 px are hard to detect after 4× encoder downsampling
- **Registration errors**: Even 1–2 px offset creates spurious edge artifacts

The Streamlit app's **Failure Analysis** tab allows interactive inspection of test predictions.

---

## 🏃 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Smoke Test (verify pipeline works, ~2 min)

```bash
python src/train.py --smoke_test
```

### 2. Full Training

```bash
python src/train.py
```

With custom parameters:
```bash
python src/train.py --epochs 50 --batch_size 8 --lr 0.0001
```

### 3. Evaluate on Test Set

```bash
python src/evaluate.py
```

With custom threshold:
```bash
python src/evaluate.py --threshold 0.4
```

### 4. Single-Pair Inference

```bash
python src/inference.py --before path/to/before.png --after path/to/after.png
```

### 5. Launch Streamlit App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
satellite-change-detection/
│
├── Satellite image data set.zip  ← Dataset (original, unmodified)
│
├── data/
│   ├── raw/                      ← Reserved for extracted data
│   └── processed/                ← Processed outputs
│
├── src/
│   ├── dataset.py                ← PyTorch Dataset + DataLoader factory
│   ├── preprocessing.py          ← Augmentation & normalisation pipeline
│   ├── registration.py           ← ECC & ORB image registration
│   ├── model.py                  ← Siamese U-Net implementation
│   ├── losses.py                 ← BCE + Dice combined loss
│   ├── metrics.py                ← IoU, Dice, Precision, Recall, Accuracy
│   ├── train.py                  ← Full training pipeline (CLI)
│   ├── evaluate.py               ← Test-set evaluation (CLI)
│   └── inference.py              ← Single-pair inference (CLI + library)
│
├── models/
│   ├── best_model.pth            ← Best checkpoint (by val Dice)
│   └── final_model.pth           ← Final epoch checkpoint
│
├── results/
│   ├── metrics.json              ← Test-set evaluation results
│   ├── training_history.json     ← Per-epoch training log
│   └── predictions/              ← Visual prediction panels
│
├── notebooks/                    ← Jupyter notebooks (optional)
│
├── app.py                        ← Streamlit application (8 tabs)
├── config.yaml                   ← All hyperparameters & paths
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
└── DATASET.md                    ← Dataset documentation
```

---

## 🔮 Future Improvements

1. **Pretrained encoder**: Use ResNet-18/34 ImageNet weights as encoder initialisation
2. **Attention mechanisms**: Add channel/spatial attention to focus on changed regions
3. **Transformer encoder**: Replace CNN encoder with a ViT-based encoder (e.g. BIT, ChangeFormer)
4. **Test-time augmentation (TTA)**: Average predictions over flips/rotations for better accuracy
5. **Multi-scale inference**: Slide over tiles of the full 1024×1024 image for better coverage
6. **Uncertainty estimation**: Monte Carlo dropout for per-pixel confidence maps
7. **Class-weighted Focal Loss**: Further address extreme imbalance cases
8. **Multi-temporal change detection**: Extend to sequences of more than 2 images

---

## 📦 Dependencies

```
torch==2.4.1
torchvision==0.19.1
opencv-python==4.10.0.84
numpy==1.26.4
pandas==2.3.3
matplotlib==3.11.0
scikit-learn==1.9.0
Pillow==10.4.0
streamlit==1.38.0
PyYAML>=6.0
tqdm>=4.66
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
