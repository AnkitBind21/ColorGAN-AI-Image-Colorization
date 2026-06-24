"""
Phase 6a: Inference Script
---------------------------
Loads a trained generator checkpoint and colorizes a grayscale image.
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models.generator import UNetGenerator
from models.dataset import lab_to_rgb

def load_generator(checkpoint_path: str, device: str = "cpu") -> UNetGenerator:
    """Load generator from checkpoint."""
    G = UNetGenerator(in_channels=1, out_channels=2)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle both raw state dict and full checkpoint
    if "G_state" in state:
        state = state["G_state"]
    G.load_state_dict(state)
    G.eval()
    G.to(device)
    print(f"[Inference] Generator loaded from {checkpoint_path}")
    return G


def colorize_image(
    G: UNetGenerator,
    image_input,  # PIL Image, numpy array, or file path
    image_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Colorize a grayscale or RGB image using the generator.

    Args:
        G           : Trained UNetGenerator
        image_input : PIL Image, np.ndarray (H,W,3 or H,W), or file path string
        image_size  : Resize target (must match training size)
        device      : "cpu" or "cuda"

    Returns:
        colorized   : np.ndarray (H, W, 3) uint8 RGB
    """
    # ── Load image ──────────────────────────────────────────
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    # ── Resize ──────────────────────────────────────────────
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img_np = np.array(img, dtype=np.uint8)

    # ── RGB → LAB → L channel ───────────────────────────────
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = img_lab[:, :, 0]

    # ── Normalize L → [-1, 1] ───────────────────────────────
    L_norm = (L / 127.5) - 1.0

    # ── To tensor ───────────────────────────────────────────
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    # Shape: (1, 1, 256, 256)

    # ── Generate AB ─────────────────────────────────────────
    with torch.no_grad():
        fake_AB = G(L_tensor)  # (1, 2, 256, 256)

    # ── Combine L + AB → RGB ────────────────────────────────
    colorized = lab_to_rgb(L_tensor[0], fake_AB[0])  # (H, W, 3) uint8

    return colorized


def colorize_file(
    checkpoint_path: str,
    input_path: str,
    output_path: str,
    device: str = "cpu",
) -> str:
    """Convenience wrapper: load model, colorize file, save result."""
    G = load_generator(checkpoint_path, device)
    result = colorize_image(G, input_path, device=device)
    img_out = Image.fromarray(result)
    img_out.save(output_path)
    print(f"[Inference] Colorized image saved → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    parser.add_argument("--input",      required=True, help="Path to input image")
    parser.add_argument("--output",     default="colorized.jpg", help="Output path")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    colorize_file(args.checkpoint, args.input, args.output, args.device)
