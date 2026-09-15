"""
Streamlit dashboard for Satellite Image Change Detection.

Tabs
----
  1. Project Overview    — motivation, architecture summary, use-cases
  2. Dataset             — LEVIR-CD stats, sample viewer, class distribution
  3. Training            — configure and launch training, view curves
  4. Model Evaluation    — test-set metrics, radar chart
  5. Change Detection    — live inference on test pairs or uploaded images
  6. Registration        — ECC alignment demo
  7. Model Architecture  — component walkthrough
  8. Failure Analysis    — browse prediction panels, error-type breakdown

Run
---
    streamlit run app.py
"""

import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Satellite Change Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


cfg = load_config()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="metric-container"] label {
    color: #a78bfa !important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; font-weight: 500; }
.stTabs [aria-selected="true"] { background-color: #7c3aed !important; color: white !important; }

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 10px 24px; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; color: white; }

.section-header {
    font-size: 1.5rem; font-weight: 700; color: #a78bfa;
    margin-bottom: 0.4rem; border-left: 4px solid #7c3aed; padding-left: 10px;
}
.badge {
    display: inline-block; background: rgba(124,58,237,0.2); color: #a78bfa;
    border: 1px solid #7c3aed; border-radius: 20px; padding: 2px 12px;
    font-size: 0.8rem; font-weight: 600; margin: 2px;
}
.info-box {
    background: rgba(79,70,229,0.1); border-left: 3px solid #4f46e5;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🛰️ SatChange")
    st.markdown("### Satellite Image Change Detection")
    st.divider()

    tab_choice = st.radio(
        "Navigation",
        [
            "🌍 Project Overview",
            "📊 Dataset",
            "🚀 Training",
            "📈 Model Evaluation",
            "🔍 Change Detection",
            "📐 Registration",
            "🧠 Model Architecture",
            "🔬 Failure Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    model_ready = os.path.exists(cfg["paths"]["best_model"])
    if model_ready:
        st.success("✅ Trained model found")
        ckpt = torch.load(cfg["paths"]["best_model"], map_location="cpu", weights_only=False)
        st.markdown(f"**Epoch:** {ckpt.get('epoch', 'N/A')}")
        st.markdown(f"**Val Dice:** {ckpt.get('val_dice', 0):.4f}")
    else:
        st.warning("⚠️ No trained model yet\nGo to 🚀 Training tab")

    if os.path.exists(cfg["paths"]["metrics_file"]):
        st.success("✅ Test metrics available")

    st.divider()
    st.caption("Built with PyTorch • OpenCV • Streamlit")


@st.cache_resource
def get_predictor(threshold=None):
    try:
        from inference import Predictor
        model_path = cfg["paths"]["best_model"]
        if not os.path.exists(model_path):
            return None
        return Predictor(model_path, cfg, threshold=threshold)
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        return None


@st.cache_data
def load_metrics():
    if not os.path.exists(cfg["paths"]["metrics_file"]):
        return None
    with open(cfg["paths"]["metrics_file"]) as f:
        return json.load(f)


@st.cache_data
def load_history():
    if not os.path.exists(cfg["paths"]["training_history_file"]):
        return None
    with open(cfg["paths"]["training_history_file"]) as f:
        return json.load(f)


@st.cache_data
def get_dataset_info():
    with zipfile.ZipFile(cfg["dataset"]["zip_path"]) as zf:
        names = zf.namelist()
    root = cfg["dataset"]["zip_inner_root"]
    info = {}
    for split in ("train", "val", "test"):
        prefix = f"{root}/{split}/A/"
        info[split] = sorted(n for n in names
                             if n.startswith(prefix) and n.endswith(".png"))
    return info, names


@st.cache_data
def read_zip_image_rgb(zip_path, inner_path):
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(inner_path)
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


@st.cache_data
def read_zip_mask(zip_path, inner_path):
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(inner_path)
    return np.array(Image.open(io.BytesIO(data)).convert("L"))


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def dark_axes(ax):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("gray")


if tab_choice == "🌍 Project Overview":
    st.markdown('<div class="section-header">🛰️ Satellite Image Change Detection</div>',
                unsafe_allow_html=True)
    st.markdown("""
    > **Compare before/after satellite images and identify regions where meaningful changes occurred.**
    > Uses a Siamese U-Net deep learning architecture for pixel-level binary change detection.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "LEVIR-CD")
    col2.metric("Image Pairs", "637")
    col3.metric("Image Size", "1024×1024 px")
    col4.metric("Architecture", "Siamese U-Net")

    st.divider()

    col_prob, col_arch = st.columns(2)

    with col_prob:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        Satellite imagery provides a powerful, scalable view of Earth's surface.
        **Change detection** — automatically identifying *what changed* and *where* between
        two time points — is critical for:

        - 🌲 **Deforestation monitoring** — track illegal logging in real-time
        - 🏗️ **Infrastructure damage** — assess post-disaster destruction
        - 🏙️ **Urban development** — monitor construction growth
        - 🌊 **Disaster assessment** — flood extent, landslide mapping

        Manual inspection at planetary scale is impossible; deep learning automates it.
        """)

    with col_arch:
        st.markdown("### 🔬 Why Siamese U-Net?")
        st.markdown("""
        **Siamese Networks** use *shared weights* to encode both images identically —
        ensuring features are comparable in the same embedding space.

        **U-Net decoder** produces full-resolution, pixel-accurate predictions through
        skip connections that preserve spatial detail lost in downsampling.

        **Feature difference** `|f(A) - f(B)|` focuses the decoder only on *what changed*.

        The combination gives:
        - ✅ Spatially precise pixel-level masks
        - ✅ Strong performance with limited labelled data
        - ✅ Interpretable intermediate representations
        """)

    st.divider()
    st.markdown("### 🛠️ Technology Stack")
    cols = st.columns(6)
    stack = [
        ("🐍", "Python 3.11"), ("🔥", "PyTorch 2.4"), ("📷", "OpenCV 4.10"),
        ("🔢", "NumPy"),       ("⚡", "Streamlit"),    ("📊", "scikit-learn"),
    ]
    for col, (icon, badge) in zip(cols, stack):
        col.markdown(f"""
        <div style="text-align:center; background:rgba(124,58,237,0.1);
                    border:1px solid #7c3aed; border-radius:10px; padding:12px">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="font-weight:600; color:#a78bfa; font-size:0.85rem">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📋 Use Cases")
    use_cases = [
        ("🌲", "Deforestation Monitoring",
         "Track tree cover loss from illegal logging, agriculture expansion, or wildfires."),
        ("🏗️", "Infrastructure Damage",
         "Assess extent of damage after earthquakes, floods, or conflict."),
        ("🏙️", "Urban Development",
         "Monitor construction projects, urban sprawl, and land-use change."),
        ("🌊", "Disaster Assessment",
         "Map flooded areas, landslides, and post-disaster recovery."),
    ]
    for col, (icon, title, desc) in zip(st.columns(4), use_cases):
        with col:
            st.markdown(f"""
            <div class="info-box">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:700; color:#a78bfa; margin:6px 0">{title}</div>
                <div style="font-size:0.85rem; color:#ccc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


elif tab_choice == "📊 Dataset":
    st.markdown('<div class="section-header">📊 Dataset: LEVIR-CD</div>',
                unsafe_allow_html=True)

    dataset_info, _ = get_dataset_info()
    root = cfg["dataset"]["zip_inner_root"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train Pairs",      len(dataset_info["train"]))
    c2.metric("Val Pairs",        len(dataset_info["val"]))
    c3.metric("Test Pairs",       len(dataset_info["test"]))
    c4.metric("Total Pairs",      sum(len(v) for v in dataset_info.values()))
    c5.metric("Image Resolution", "1024×1024 px")

    st.divider()

    col_info, col_struct = st.columns(2)
    with col_info:
        st.markdown("### ℹ️ Dataset Details")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Name** | LEVIR-CD |
        | **Task** | Building Change Detection |
        | **Format** | PNG |
        | **Channels** | RGB (3-channel) |
        | **Mask values** | 0 = no change · 255 = change |
        | **Change ratio** | ~4% changed pixels |
        | **Pre-registration** | ✅ Yes (Google Earth crops) |
        | **Pre-split** | ✅ train / val / test |
        """)

    with col_struct:
        st.markdown("### 📁 Dataset Structure")
        st.code("""
LEVIR CD/
├── train/
│   ├── A/         ← before images (T1) — 445 files
│   ├── B/         ← after  images (T2) — 445 files
│   └── label/     ← change masks       — 445 files
├── val/
│   ├── A/  B/  label/  ← 64 pairs
└── test/
    ├── A/  B/  label/  ← 128 pairs
        """)

    st.divider()
    st.markdown("### 🖼️ Sample Image Pairs")

    split_choice = st.selectbox("Select split", ["train", "val", "test"])
    files_a = dataset_info[split_choice]
    idx = st.slider("Sample index", 0, len(files_a) - 1, 0)

    fa = files_a[idx]
    fb = fa.replace(f"/{split_choice}/A/", f"/{split_choice}/B/")
    fl = fa.replace(f"/{split_choice}/A/", f"/{split_choice}/label/")

    img_a = read_zip_image_rgb(cfg["dataset"]["zip_path"], fa)
    img_b = read_zip_image_rgb(cfg["dataset"]["zip_path"], fb)
    mask  = read_zip_mask(cfg["dataset"]["zip_path"], fl)

    c1, c2, c3 = st.columns(3)
    c1.image(img_a, caption="Before (T1)",              use_container_width=True)
    c2.image(img_b, caption="After (T2)",               use_container_width=True)
    c3.image(mask,  caption="Change Mask (255=changed)", use_container_width=True)

    st.info(f"**{os.path.basename(fa)}** — Changed pixels: {(mask > 127).mean() * 100:.2f}%")

    st.divider()
    st.markdown("### ⚖️ Class Imbalance")
    st.markdown("""
    LEVIR-CD has ~**4% changed pixels** vs **96% unchanged** — a 24:1 imbalance.
    We address this with:
    - **pos_weight** in BCE loss (≈ neg/pos pixel ratio)
    - **Dice Loss** — inherently insensitive to class imbalance
    - **IoU and Dice** reported instead of pixel accuracy
    """)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Unchanged (~96%)", "Changed (~4%)"], [96, 4],
           color=["#4f46e5", "#f43f5e"], edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Pixel %")
    ax.set_title("Class Distribution in LEVIR-CD", fontweight="bold", color="white")
    ax.set_facecolor("#111"); fig.patch.set_facecolor("#111")
    ax.tick_params(colors="white"); ax.yaxis.label.set_color("white")
    for s in ax.spines.values(): s.set_edgecolor("gray")
    st.pyplot(fig); plt.close()


elif tab_choice == "🚀 Training":
    st.markdown('<div class="section-header">🚀 Model Training</div>',
                unsafe_allow_html=True)

    col_cfg, col_status = st.columns(2)

    with col_cfg:
        st.markdown("### ⚙️ Training Configuration")
        epochs     = st.slider("Epochs",      5, 100, cfg["training"]["epochs"])
        batch_size = st.select_slider("Batch Size",    [2, 4, 8, 16],
                                       value=cfg["training"]["batch_size"])
        lr         = st.select_slider("Learning Rate", [1e-5, 5e-5, 1e-4, 3e-4, 1e-3],
                                       value=cfg["training"]["learning_rate"])
        smoke_test = st.checkbox("🧪 Smoke Test (1 batch / 2 epochs)", value=False)

    with col_status:
        st.markdown("### 📋 Current Status")
        if os.path.exists(cfg["paths"]["best_model"]):
            st.success("✅ A trained model exists")
            ckpt = torch.load(cfg["paths"]["best_model"], map_location="cpu", weights_only=False)
            st.metric("Best Val Dice",   f"{ckpt.get('val_dice', 0):.4f}")
            st.metric("Trained Epochs",  ckpt.get("epoch", "?"))
        else:
            st.warning("No trained model found. Click 'Start Training'.")

        history = load_history()
        if history:
            last = history[-1]
            st.metric("Last Train Loss", f"{last.get('train_loss', 0):.4f}")
            st.metric("Last Val Dice",   f"{last.get('val_dice', 0):.4f}")

    st.divider()

    if st.button("🚀 Start Training", type="primary"):
        cmd = [sys.executable, "src/train.py",
               "--epochs",     str(epochs),
               "--batch_size", str(batch_size),
               "--lr",         str(lr)]
        if smoke_test:
            cmd.append("--smoke_test")

        st.info("⚙️ Training started — check terminal for live output.")
        st.code(" ".join(cmd), language="bash")

        with st.spinner("Training … (this may take a while on CPU)"):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("✅ Training completed!")
            st.text_area("Output", result.stdout[-3000:], height=300)
            st.cache_resource.clear(); st.cache_data.clear()
        else:
            st.error("❌ Training failed.")
            st.text_area("Error", result.stderr[-3000:], height=300)

    st.divider()
    st.markdown("### 📈 Training Curves")

    history = load_history()
    if history:
        epochs_l = [r["epoch"]         for r in history]
        tr_loss  = [r["train_loss"]    for r in history]
        vl_loss  = [r["val_loss"]      for r in history]
        vl_dice  = [r["val_dice"]      for r in history]
        vl_iou   = [r["val_iou"]       for r in history]
        vl_prec  = [r["val_precision"] for r in history]
        vl_rec   = [r["val_recall"]    for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor("#111")

        ax1.plot(epochs_l, tr_loss, label="Train Loss", color="#7c3aed", lw=2)
        ax1.plot(epochs_l, vl_loss, label="Val Loss",   color="#f43f5e", lw=2)
        ax1.set_title("Loss Curves",        color="white", fontweight="bold")
        ax1.set_xlabel("Epoch", color="white"); ax1.set_ylabel("Loss", color="white")
        ax1.legend(facecolor="#222", edgecolor="#555", labelcolor="white")
        dark_axes(ax1)

        ax2.plot(epochs_l, vl_dice, label="Dice / F1",  color="#10b981", lw=2)
        ax2.plot(epochs_l, vl_iou,  label="IoU",        color="#f59e0b", lw=2)
        ax2.plot(epochs_l, vl_prec, label="Precision",  color="#60a5fa", lw=2, ls="--")
        ax2.plot(epochs_l, vl_rec,  label="Recall",     color="#fb7185", lw=2, ls="--")
        ax2.set_title("Validation Metrics", color="white", fontweight="bold")
        ax2.set_xlabel("Epoch", color="white"); ax2.set_ylabel("Score", color="white")
        ax2.legend(facecolor="#222", edgecolor="#555", labelcolor="white")
        dark_axes(ax2)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        with st.expander("📋 Full Training Log"):
            import pandas as pd
            st.dataframe(pd.DataFrame(history).set_index("epoch"),
                         use_container_width=True)
    else:
        st.info("No training history found. Train the model first.")

    st.divider()
    st.markdown("### 💻 Manual Training Commands")
    st.code("python src/train.py --epochs 50 --batch_size 8", language="bash")
    st.code("python src/train.py --smoke_test", language="bash")


elif tab_choice == "📈 Model Evaluation":
    st.markdown('<div class="section-header">📈 Test Set Evaluation</div>',
                unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics is None:
        st.warning("⚠️ No metrics found. Run evaluation first:")
        st.code("python src/evaluate.py", language="bash")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("IoU",            f"{metrics.get('iou', 0):.4f}")
    c2.metric("Dice / F1",      f"{metrics.get('dice', 0):.4f}")
    c3.metric("Precision",      f"{metrics.get('precision', 0):.4f}")
    c4.metric("Recall",         f"{metrics.get('recall', 0):.4f}")
    c5.metric("Pixel Accuracy", f"{metrics.get('accuracy', 0):.4f}")

    st.divider()

    col_chart, col_note = st.columns(2)

    with col_chart:
        names  = ["IoU", "Dice", "Precision", "Recall"]
        values = [metrics.get(k, 0) for k in ["iou", "dice", "precision", "recall"]]
        v_cl   = values + [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
        a_cl   = angles + [angles[0]]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(a_cl, v_cl, alpha=0.25, color="#7c3aed")
        ax.plot(a_cl, v_cl, color="#a78bfa", lw=2)
        ax.set_xticks(angles); ax.set_xticklabels(names, color="white", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", fontsize=8)
        ax.set_facecolor("#111"); fig.patch.set_facecolor("#111")
        ax.grid(color="gray", alpha=0.3); ax.spines["polar"].set_color("gray")
        ax.set_title("Model Performance (Test Set)", color="white",
                     fontweight="bold", pad=15)
        st.pyplot(fig); plt.close()

    with col_note:
        st.markdown("### 📌 Why Not Just Report Accuracy?")
        st.markdown("""
        LEVIR-CD has **~96% unchanged pixels**.

        A model predicting **"no change" everywhere** achieves:
        - ✅ ~96% pixel accuracy
        - ❌ 0% IoU
        - ❌ 0% Dice
        - ❌ 0% Recall

        **IoU and Dice** measure overlap between predicted and ground-truth change
        regions, regardless of true-negative count — they are the canonical metrics
        for change detection.

        **Precision** = of all predicted changes, how many are real?
        **Recall** = of all real changes, how many did we find?
        """)

    st.divider()
    with st.expander("📋 View full metrics.json"):
        st.json(metrics)

    st.divider()
    if st.button("🔁 Re-run Evaluation on Test Set"):
        with st.spinner("Running evaluation …"):
            result = subprocess.run(
                [sys.executable, "src/evaluate.py"], capture_output=True, text=True
            )
        if result.returncode == 0:
            st.success("Evaluation complete! Refresh to see updated metrics.")
            st.cache_data.clear()
        else:
            st.error(result.stderr[-2000:])


elif tab_choice == "🔍 Change Detection":
    st.markdown('<div class="section-header">🔍 Change Detection — Live Inference</div>',
                unsafe_allow_html=True)

    if not os.path.exists(cfg["paths"]["best_model"]):
        st.error("❌ No trained model found. Please train first (🚀 Training tab).")
        st.stop()

    threshold_val = st.slider(
        "🎚️ Change threshold", 0.1, 0.9,
        float(cfg["evaluation"]["threshold"]), step=0.05,
        help="Pixels with probability ≥ threshold are predicted as changed",
    )

    tab_sel, tab_upload = st.tabs(["📂 Select Test Pair", "📤 Upload Images"])

    with tab_sel:
        dataset_info, _ = get_dataset_info()
        test_files = dataset_info["test"]

        if test_files:
            idx = st.selectbox(
                "Test pair", range(len(test_files)),
                format_func=lambda i: os.path.basename(test_files[i])
            )
            fa = test_files[idx]
            fb = fa.replace("/test/A/", "/test/B/")
            fl = fa.replace("/test/A/", "/test/label/")

            raw_a = read_zip_image_rgb(cfg["dataset"]["zip_path"], fa)
            raw_b = read_zip_image_rgb(cfg["dataset"]["zip_path"], fb)
            gt    = read_zip_mask(cfg["dataset"]["zip_path"], fl)

            if st.button("▶️ Run Change Detection", key="run_sel"):
                with st.spinner("Running inference …"):
                    predictor = get_predictor(threshold_val)
                    if predictor is None:
                        st.error("Failed to load predictor."); st.stop()
                    result = predictor.predict(
                        raw_a.astype(np.float32) / 255.0,
                        raw_b.astype(np.float32) / 255.0,
                    )

                from inference import create_change_overlay

                c1, c2, c3, c4 = st.columns(4)
                c1.image(raw_a, caption="Before (T1)", use_container_width=True)
                c2.image(raw_b, caption="After (T2)",  use_container_width=True)
                c3.image((gt > 0).astype(np.uint8) * 255,
                         caption="Ground Truth", use_container_width=True, clamp=True)
                c4.image(result["change_mask"] * 255,
                         caption="Prediction", use_container_width=True, clamp=True)

                st.divider()
                c_prob, c_overlay, c_info = st.columns(3)

                with c_prob:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(result["prob_map"], cmap="RdYlGn_r", vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title("Change Probability Map", color="white")
                    ax.axis("off"); fig.patch.set_facecolor("#111"); ax.set_facecolor("#111")
                    st.pyplot(fig); plt.close()

                with c_overlay:
                    overlay = create_change_overlay(
                        raw_b.astype(np.float32) / 255.0,
                        result["change_mask"], color=(255, 0, 80), alpha=0.5
                    )
                    st.image(overlay, caption="Overlay on After Image",
                             use_container_width=True)

                with c_info:
                    st.metric("Changed Pixel %",    f"{result['changed_pct']:.2f}%")
                    st.metric("GT Changed Pixel %", f"{float((gt > 127).mean() * 100):.2f}%")
                    st.metric("Threshold Used",     f"{threshold_val:.2f}")

    with tab_upload:
        st.info("Upload your own before/after satellite image pair (RGB, any size).")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            before_file = st.file_uploader("Before image (T1)", type=["png","jpg","jpeg","tif"])
        with col_up2:
            after_file  = st.file_uploader("After image (T2)",  type=["png","jpg","jpeg","tif"])

        if before_file and after_file:
            img_a_up = np.array(Image.open(before_file).convert("RGB"))
            img_b_up = np.array(Image.open(after_file).convert("RGB"))

            c1, c2 = st.columns(2)
            c1.image(img_a_up, caption="Uploaded: Before", use_container_width=True)
            c2.image(img_b_up, caption="Uploaded: After",  use_container_width=True)

            if st.button("▶️ Run Change Detection", key="run_upload"):
                with st.spinner("Running inference …"):
                    predictor = get_predictor(threshold_val)
                    if predictor is None:
                        st.error("Failed to load predictor."); st.stop()
                    result = predictor.predict(
                        img_a_up.astype(np.float32) / 255.0,
                        img_b_up.astype(np.float32) / 255.0,
                    )

                from inference import create_change_overlay

                orig_h, orig_w = img_b_up.shape[:2]
                mask_orig = cv2.resize(
                    result["change_mask"].astype(np.uint8),
                    (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                c3, c4 = st.columns(2)
                c3.image(mask_orig * 255, caption="Prediction (original size)",
                         use_container_width=True, clamp=True)
                c4.image(
                    create_change_overlay(img_b_up.astype(np.float32) / 255.0,
                                          mask_orig, color=(255, 0, 80), alpha=0.5),
                    caption="Overlay on After Image", use_container_width=True
                )
                st.metric("Changed Pixel %", f"{result['changed_pct']:.2f}%")


elif tab_choice == "📐 Registration":
    st.markdown('<div class="section-header">📐 Image Registration</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Why Registration Matters

    Satellite images taken at different times may have **sub-pixel spatial offsets** from
    different viewing angles, GPS jitter, atmospheric refraction, or different sensor platforms.

    Even a **2–3 pixel shift** creates spurious false changes along all building edges
    and roads — even if nothing actually changed.

    **ECC (Enhanced Correlation Coefficient)** with a *translation-only* warp corrects
    this without distorting image content.

    ℹ️ LEVIR-CD is pre-registered, so offsets are minimal. Registration is kept enabled
    as a safeguard.
    """)

    st.divider()

    dataset_info, _ = get_dataset_info()
    test_files = dataset_info["test"]

    idx_reg = st.selectbox(
        "Select test pair for registration demo",
        range(len(test_files)),
        format_func=lambda i: os.path.basename(test_files[i]),
        key="reg_select",
    )

    fa = test_files[idx_reg]
    raw_a = read_zip_image_rgb(cfg["dataset"]["zip_path"], fa).astype(np.float32) / 255.0
    raw_b = read_zip_image_rgb(cfg["dataset"]["zip_path"],
                               fa.replace("/test/A/", "/test/B/")).astype(np.float32) / 255.0

    if st.button("🔄 Run Registration Demo"):
        with st.spinner("Running ECC registration …"):
            from registration import register_ecc
            registered, warp = register_ecc(raw_a, raw_b,
                                            warp_mode=cfg["registration"]["warp_mode"])
            tx = warp[0, 2] if warp is not None else 0.0
            ty = warp[1, 2] if warp is not None else 0.0

        c1, c2, c3 = st.columns(3)
        c1.image(raw_a,       caption="Before (T1) — Reference",          use_container_width=True)
        c2.image(raw_b,       caption="After (T2) — Before Registration",  use_container_width=True)
        c3.image(registered,  caption="After (T2) — After Registration",   use_container_width=True)

        st.info(f"**Estimated displacement:** tx = {tx:.3f} px, ty = {ty:.3f} px")

        st.divider()
        st.markdown("#### Pixel Difference (Highlights Misalignment)")

        diff_before = np.abs(raw_a - raw_b).mean(axis=2)
        diff_after  = np.abs(raw_a - registered).mean(axis=2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#111")
        for ax, diff, title in zip(axes,
                                    [diff_before, diff_after],
                                    ["Difference BEFORE Registration",
                                     "Difference AFTER Registration"]):
            ax.imshow(diff, cmap="hot", vmin=0, vmax=0.3)
            ax.set_title(title, color="white", fontweight="bold")
            ax.axis("off"); ax.set_facecolor("#111")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


elif tab_choice == "🧠 Model Architecture":
    st.markdown('<div class="section-header">🧠 Siamese U-Net Architecture</div>',
                unsafe_allow_html=True)

    col_diag, col_desc = st.columns(2)

    with col_diag:
        st.markdown("### Architecture Diagram")
        st.code("""
  BEFORE IMAGE (3×256×256)    AFTER IMAGE (3×256×256)
         │                           │
   ┌─────┴─────┐               ┌─────┴─────┐
   │  Encoder  │               │  Encoder  │  ← SHARED WEIGHTS
   │  Block 1  │               │  Block 1  │    (64 ch)
   └─────┬─────┘               └─────┬─────┘
         │         |A−B|             │
         └──────►  64ch  ◄───────────┘  → skip 0
   ...repeat for blocks 2,3,4 at 128/256/512 ch...
                    │
               Bottleneck (256ch)
                    │
           ┌────────┘
           │  Decoder Block 1 ← skip 2 (256ch diff)
           │  Decoder Block 2 ← skip 1 (128ch diff)
           │  Decoder Block 3 ← skip 0  (64ch diff)
           │  Bilinear Upsample → 256×256
           │  Dropout(0.1)
           │  1×1 Conv → 1 channel
           ↓
      CHANGE LOGIT → sigmoid → PROBABILITY MAP → threshold → BINARY MASK
        """, language="text")

    with col_desc:
        st.markdown("### Component Explanations")

        with st.expander("🔗 Siamese Network & Shared Weights", expanded=True):
            st.markdown("""
            Both T1 and T2 images pass through **the same encoder** (shared weights).

            This guarantees features from both images live in the same embedding space,
            making `|f(A) - f(B)|` a meaningful change signal.  Without shared weights
            the two encoders could learn completely different representations, and the
            difference would be noise.
            """)

        with st.expander("📉 U-Net Encoder-Decoder"):
            st.markdown("""
            **Encoder** progressively downsamples to extract high-level features.
            **Decoder** upsamples back to full resolution.
            **Skip connections** pass encoder feature differences directly to the decoder,
            preserving fine spatial detail that pooling discards.
            """)

        with st.expander("➖ Feature Difference"):
            st.markdown("""
            `diff = |encoder(T1) - encoder(T2)|` at every encoder scale.

            Absolute difference is used (not concatenation) because:
            - Symmetric — order of T1/T2 doesn't change the result
            - Near-zero for unchanged regions, large for changed regions
            - Half the channels vs concatenation
            """)

        with st.expander("📐 Loss Function"):
            st.markdown("""
            `Loss = BCE_weight × BCE + Dice_weight × Dice`

            **BCE with pos_weight ≈ 24** upweights changed-pixel gradients to
            counteract the 96/4 background/foreground imbalance.

            **Dice Loss** optimises F1 directly — it is inherently insensitive to
            class imbalance because it only involves TP, FP, FN counts.
            """)

    st.divider()
    st.markdown("### 📊 Model Parameters")
    try:
        from model import build_model
        m = build_model(cfg)
        n = m.count_parameters()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Parameters",  f"{n:,}")
        c2.metric("Encoder Channels",  str(cfg["model"]["encoder_channels"]))
        c3.metric("Decoder Channels",  str(cfg["model"]["decoder_channels"]))
        st.info(f"The model has **{n / 1e6:.2f}M parameters** — "
                "lightweight enough to train on CPU in a reasonable timeframe.")
    except Exception as e:
        st.warning(f"Could not load model for parameter count: {e}")


elif tab_choice == "🔬 Failure Analysis":
    st.markdown('<div class="section-header">🔬 Failure Case Analysis</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Failure case analysis reveals **where and why** the model makes mistakes.

    **Error categories:**
    - 🟢 **True Positive (TP)** — correctly detected change
    - ⬜ **True Negative (TN)** — correctly identified no-change
    - 🔴 **False Positive (FP)** — predicted change where none exists
    - 🔵 **False Negative (FN)** — missed actual change
    """)

    pred_dir = cfg["paths"]["predictions_dir"]
    panel_files = sorted(f for f in os.listdir(pred_dir) if f.endswith("_panel.png")) \
        if os.path.exists(pred_dir) else []

    if not panel_files:
        st.warning("No predictions found. Run evaluation first:")
        st.code("python src/evaluate.py", language="bash")
    else:
        selected = st.selectbox("Select prediction panel",
                                range(len(panel_files)),
                                format_func=lambda i: panel_files[i])
        st.image(os.path.join(pred_dir, panel_files[selected]),
                 use_container_width=True)

        err_path = os.path.join(pred_dir,
                                panel_files[selected].replace("_panel.png", "_errors.png"))
        if os.path.exists(err_path):
            st.image(err_path, caption="Error Map: 🟢TP | ⬜TN | 🔴FP | 🔵FN",
                     use_container_width=True)

        st.divider()
        st.markdown("### 🔍 Common Failure Causes")
        causes = {
            "🌑 Shadows": (
                "Seasonal or time-of-day differences cast shadows that "
                "create apparent intensity changes even where nothing moved."
            ),
            "🍂 Seasonal Differences": (
                "Vegetation colour and density vary between seasons — "
                "summer-to-winter transitions produce widespread false positives."
            ),
            "💡 Illumination": (
                "Different sun angles, cloud cover, or haze cause "
                "intensity differences unrelated to actual scene changes."
            ),
            "📍 Small Objects": (
                "Buildings or sites smaller than ~10 px are hard to detect "
                "after the encoder's 4× spatial downsampling."
            ),
            "🌿 Vegetation Changes": (
                "Tree growth, seasonal foliage, and crop cycles produce "
                "false positives without strong regularisation."
            ),
            "📡 Sensor Noise": (
                "Different satellites or imaging conditions produce noise patterns "
                "that don't represent true scene changes."
            ),
        }
        cols = st.columns(2)
        for i, (cause, explanation) in enumerate(causes.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="info-box">
                    <div style="font-weight:700; color:#a78bfa">{cause}</div>
                    <div style="font-size:0.85rem; color:#ccc; margin-top:6px">{explanation}</div>
                </div>
                """, unsafe_allow_html=True)


st.divider()
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8rem; padding:8px">
    Satellite Image Change Detection · LEVIR-CD · Siamese U-Net · PyTorch 2.4 · Streamlit
</div>
""", unsafe_allow_html=True)
