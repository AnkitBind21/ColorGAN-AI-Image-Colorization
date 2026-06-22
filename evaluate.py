"""
Phase 9: Evaluation — PSNR, SSIM + Training Plots
----------------------------------------------------
Run after training to:
  1. Evaluate trained model on test set (PSNR, SSIM)
  2. Plot training loss curves
  3. Save a grid of colorized examples
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from dataset    import ColorizationDataset, lab_to_rgb
from inference  import load_generator, colorize_image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity   as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    print("[Warning] scikit-image not installed. Install with: pip install scikit-image")
    HAS_SKIMAGE = False


# ─────────────────────────────────────────────────────────────
# 1. Evaluate on test set
# ─────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, test_dir: str, n_samples: int = 50, device: str = "cpu"):
    """
    Compute PSNR and SSIM on the test set.

    Returns:
        dict with mean/std of PSNR and SSIM
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Eval] ⚠ No checkpoint at {checkpoint_path}. Skipping evaluation.")
        return None

    G = load_generator(checkpoint_path, device=device)
    ds = ColorizationDataset(test_dir, split="test")
    n = min(n_samples, len(ds))

    psnr_list, ssim_list = [], []

    print(f"\n[Eval] Evaluating on {n} test images...")

    for i in range(n):
        batch = ds[i]
        L_t  = batch["L"]   # (1, 256, 256)
        AB_t = batch["AB"]  # (2, 256, 256)
        path = batch["path"]

        # Real RGB
        real_rgb = lab_to_rgb(L_t, AB_t)

        # Predicted RGB
        L_in = L_t.unsqueeze(0).to(device)  # (1, 1, 256, 256)
        with torch.no_grad():
            fake_AB = G(L_in)                # (1, 2, 256, 256)
        fake_rgb = lab_to_rgb(L_in[0], fake_AB[0])

        if HAS_SKIMAGE:
            p = psnr_fn(real_rgb, fake_rgb, data_range=255)
            s = ssim_fn(real_rgb, fake_rgb, data_range=255, channel_axis=2)
            psnr_list.append(p)
            ssim_list.append(s)

            if i % 10 == 0:
                print(f"  [{i+1:3d}/{n}] PSNR: {p:.2f} dB | SSIM: {s:.4f}")

    if not psnr_list:
        print("[Eval] scikit-image required for metrics. Install: pip install scikit-image")
        return {}

    results = {
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std" : float(np.std(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std" : float(np.std(ssim_list)),
        "n_samples": n,
    }

    print(f"\n[Eval] Results over {n} samples:")
    print(f"  PSNR : {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM : {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"\n  Reference ranges (rough):")
    print(f"    PSNR > 30 dB  → good  |  > 35 dB → excellent")
    print(f"    SSIM > 0.85   → good  |  > 0.92  → excellent")

    return results


# ─────────────────────────────────────────────────────────────
# 2. Plot training curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(history_path: str, output_path: str):
    """Plot G_loss, D_loss, PSNR, SSIM over epochs."""
    if not os.path.exists(history_path):
        print(f"[Plot] No history file at {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["G_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#0d0d0d")

    plot_cfg = dict(facecolor="#161616", labelcolor="#e8e8e8", titlecolor="#f4a300")

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161616")
        ax.set_title(title, color="#f4a300", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, color="#666", fontsize=9)
        ax.set_ylabel(ylabel, color="#666", fontsize=9)
        ax.tick_params(colors="#666")
        ax.spines["bottom"].set_color("#2a2a2a")
        ax.spines["left"].set_color("#2a2a2a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="#666")

    # G Loss
    axes[0,0].plot(epochs, history["G_loss"], color="#f4a300", linewidth=1.5)
    style_ax(axes[0,0], "Generator Loss", "Epoch", "Loss")

    # D Loss
    axes[0,1].plot(epochs, history["D_loss"], color="#6699ff", linewidth=1.5)
    style_ax(axes[0,1], "Discriminator Loss", "Epoch", "Loss")

    # PSNR
    if history.get("psnr") and any(v > 0 for v in history["psnr"]):
        axes[1,0].plot(epochs, history["psnr"], color="#2ecc71", linewidth=1.5)
    style_ax(axes[1,0], "PSNR (dB)", "Epoch", "dB")

    # SSIM
    if history.get("ssim") and any(v > 0 for v in history["ssim"]):
        axes[1,1].plot(epochs, history["ssim"], color="#e05c00", linewidth=1.5)
    style_ax(axes[1,1], "SSIM", "Epoch", "SSIM")

    fig.suptitle("colorGAN — Training Curves", color="#e8e8e8", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[Plot] Training curves saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# 3. Visual comparison grid
# ─────────────────────────────────────────────────────────────

def save_comparison_grid(
    checkpoint_path: str,
    test_dir: str,
    output_path: str,
    n: int = 6,
    device: str = "cpu",
):
    """Save a side-by-side grid: [Grayscale | Colorized | Ground Truth]."""
    if not os.path.exists(checkpoint_path):
        print("[Grid] No checkpoint — skipping comparison grid.")
        return

    G  = load_generator(checkpoint_path, device=device)
    ds = ColorizationDataset(test_dir, split="test")
    n  = min(n, len(ds))

    fig = plt.figure(figsize=(12, n * 4))
    fig.patch.set_facecolor("#0d0d0d")

    for i in range(n):
        batch = ds[i]
        L_t  = batch["L"]
        AB_t = batch["AB"]

        # Ground truth RGB
        gt_rgb = lab_to_rgb(L_t, AB_t)

        # Grayscale (just L channel repeated to RGB)
        L_np   = ((L_t.squeeze().numpy() + 1.0) * 127.5).astype(np.uint8)
        gray_rgb = np.stack([L_np, L_np, L_np], axis=2)

        # Model prediction
        L_in = L_t.unsqueeze(0).to(device)
        with torch.no_grad():
            fake_AB = G(L_in)
        pred_rgb = lab_to_rgb(L_in[0], fake_AB[0])

        # Plot row
        for col, (img, title) in enumerate(zip(
            [gray_rgb, pred_rgb, gt_rgb],
            ["Grayscale Input", "Model Output", "Ground Truth"]
        )):
            ax = fig.add_subplot(n, 3, i * 3 + col + 1)
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(title, color="#f4a300", fontsize=10, pad=8)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[Grid] Comparison grid saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CHECKPOINT   = "checkpoints/generator_latest.pt"
    TEST_DIR     = "dataset/test"
    RESULTS_DIR  = "results"
    HISTORY_PATH = os.path.join(RESULTS_DIR, "train_history.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Evaluate
    results = evaluate(CHECKPOINT, TEST_DIR, n_samples=50)
    if results:
        with open(os.path.join(RESULTS_DIR, "eval_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    # 2. Training curves
    plot_training_curves(
        HISTORY_PATH,
        os.path.join(RESULTS_DIR, "training_curves.png")
    )

    # 3. Comparison grid
    save_comparison_grid(
        CHECKPOINT,
        TEST_DIR,
        os.path.join(RESULTS_DIR, "comparison_grid.png"),
        n=6,
    )
