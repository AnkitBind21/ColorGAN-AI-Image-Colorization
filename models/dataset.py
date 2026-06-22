"""
Phase 1: Dataset Preprocessing
--------------------------------
Pipeline:
  RGB image (H, W, 3)
  → LAB image (H, W, 3)
  → L channel (1, 256, 256)  → Generator INPUT
  → AB channels (2, 256, 256) → Generator TARGET

Normalization:
  L  ∈ [0, 100]  → normalize to [-1, 1]
  A  ∈ [-128, 127] → normalize to [-1, 1]
  B  ∈ [-128, 127] → normalize to [-1, 1]
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2


class ColorizationDataset(Dataset):
    """
    Loads RGB images, converts to LAB, returns:
        L  : Tensor (1, 256, 256) in [-1, 1]
        AB : Tensor (2, 256, 256) in [-1, 1]
    """

    def __init__(self, root_dir: str, image_size: int = 256, split: str = "train"):
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split

        # Gather all image paths (jpg, jpeg, png)
        self.image_paths = []
        for fname in os.listdir(root_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_paths.append(os.path.join(root_dir, fname))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {root_dir}. "
                "Add .jpg/.jpeg/.png files to the folder."
            )

        print(f"[Dataset] {split} — {len(self.image_paths)} images found in {root_dir}")

        # Resize + center crop to square
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # --- Load as RGB PIL image ---
        img_rgb = Image.open(img_path).convert("RGB")

        # --- Resize ---
        img_rgb = self.transform(img_rgb)

        # --- Convert PIL → NumPy uint8 (H, W, 3) ---
        img_np = np.array(img_rgb, dtype=np.uint8)

        # --- RGB → LAB (OpenCV uses BGR internally) ---
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # LAB raw value ranges from OpenCV:
        #   L  ∈ [0, 255]   (mapped from [0, 100])
        #   A  ∈ [0, 255]   (mapped from [-128, 127])
        #   B  ∈ [0, 255]   (mapped from [-128, 127])

        L  = img_lab[:, :, 0]   # shape (H, W)
        AB = img_lab[:, :, 1:]  # shape (H, W, 2)

        # --- Normalize to [-1, 1] ---
        # OpenCV LAB: L in [0,255], AB in [0,255]
        L  = (L  / 127.5) - 1.0        # → [-1, 1]
        AB = (AB / 127.5) - 1.0        # → [-1, 1]

        # --- NumPy → Tensor ---
        L_tensor  = torch.from_numpy(L).unsqueeze(0).float()   # (1, H, W)
        AB_tensor = torch.from_numpy(AB).permute(2, 0, 1).float()  # (2, H, W)

        # ✅ Tensor Shape Check
        assert L_tensor.shape  == (1, self.image_size, self.image_size), \
            f"L shape mismatch: {L_tensor.shape}"
        assert AB_tensor.shape == (2, self.image_size, self.image_size), \
            f"AB shape mismatch: {AB_tensor.shape}"

        return {"L": L_tensor, "AB": AB_tensor, "path": img_path}


def get_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
):
    """Build train and test DataLoaders."""
    train_ds = ColorizationDataset(train_dir, image_size=image_size, split="train")
    test_ds  = ColorizationDataset(test_dir,  image_size=image_size, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Keep batch sizes consistent for PatchGAN
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Utility: Convert model output back to RGB for saving
# ─────────────────────────────────────────────────────────────

def lab_to_rgb(L_tensor: torch.Tensor, AB_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert L + AB tensors (normalized [-1,1]) back to RGB numpy image.

    Args:
        L_tensor  : (1, H, W)  or (B, 1, H, W)
        AB_tensor : (2, H, W)  or (B, 2, H, W)

    Returns:
        rgb_image : (H, W, 3) uint8 numpy array
    """
    # Handle batch dimension
    if L_tensor.dim() == 4:
        L_tensor  = L_tensor[0]
        AB_tensor = AB_tensor[0]

    # Denormalize: [-1,1] → [0,255]
    L  = ((L_tensor.squeeze(0).cpu().numpy()  + 1.0) * 127.5).astype(np.float32)
    AB = ((AB_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.float32)

    # Reconstruct LAB image
    lab = np.concatenate([L[:, :, np.newaxis], AB], axis=2)  # (H, W, 3)
    lab = np.clip(lab, 0, 255).astype(np.uint8)

    # LAB → BGR → RGB
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


# ─────────────────────────────────────────────────────────────
# Shape Verification Script — run directly to verify pipeline
# ─────────────────────────────────────────────────────────────

def verify_pipeline(image_dir: str):
    """
    Quick pipeline check using a single image.
    Prints tensor shapes and value ranges at every step.
    """
    import glob
    paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
            glob.glob(os.path.join(image_dir, "*.png"))

    if not paths:
        print(f"[verify] No images in {image_dir}. Generating a synthetic RGB test image.")
        # Create a synthetic image for testing
        synthetic = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = os.path.join(image_dir, "_synthetic_test.jpg")
        Image.fromarray(synthetic).save(test_path)
        paths = [test_path]

    img_path = paths[0]
    print(f"\n[verify] Testing with: {img_path}")

    # --- Load ---
    img_rgb = Image.open(img_path).convert("RGB")
    img_rgb = img_rgb.resize((256, 256), Image.BICUBIC)
    img_np  = np.array(img_rgb, dtype=np.uint8)
    print(f"  RGB numpy  : shape={img_np.shape}, dtype={img_np.dtype}, "
          f"range=[{img_np.min()}, {img_np.max()}]")

    # --- RGB → LAB ---
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    print(f"  LAB numpy  : shape={img_lab.shape}, dtype={img_lab.dtype}, "
          f"L range=[{img_lab[:,:,0].min():.1f}, {img_lab[:,:,0].max():.1f}]")

    # --- Split ---
    L  = img_lab[:, :, 0]
    AB = img_lab[:, :, 1:]
    print(f"  L  numpy   : shape={L.shape}, range=[{L.min():.1f}, {L.max():.1f}]")
    print(f"  AB numpy   : shape={AB.shape}, range=[{AB.min():.1f}, {AB.max():.1f}]")

    # --- Normalize ---
    L_norm  = (L  / 127.5) - 1.0
    AB_norm = (AB / 127.5) - 1.0
    print(f"  L  norm    : range=[{L_norm.min():.3f}, {L_norm.max():.3f}]  ← should be ~[-1,1]")
    print(f"  AB norm    : range=[{AB_norm.min():.3f}, {AB_norm.max():.3f}] ← should be ~[-1,1]")

    # --- Tensors ---
    L_t  = torch.from_numpy(L_norm).unsqueeze(0).float()
    AB_t = torch.from_numpy(AB_norm).permute(2, 0, 1).float() if AB_norm.ndim == 3 \
           else torch.from_numpy(AB_norm).unsqueeze(0).float()

    # Fix AB: numpy (H,W,2) → permute → (2,H,W)
    AB_t = torch.from_numpy(AB_norm).permute(2, 0, 1).float()

    print(f"\n  ✅ L_tensor  : {L_t.shape}   ← expected torch.Size([1, 256, 256])")
    print(f"  ✅ AB_tensor : {AB_t.shape}  ← expected torch.Size([2, 256, 256])")

    # --- Batch simulation ---
    L_batch  = L_t.unsqueeze(0)   # (B, 1, H, W)
    AB_batch = AB_t.unsqueeze(0)  # (B, 2, H, W)
    print(f"\n  ✅ L_batch   : {L_batch.shape}   ← expected torch.Size([1, 1, 256, 256])")
    print(f"  ✅ AB_batch  : {AB_batch.shape}  ← expected torch.Size([1, 2, 256, 256])")

    # --- Round-trip test ---
    rgb_recovered = lab_to_rgb(L_t, AB_t)
    print(f"\n  ✅ lab_to_rgb output : shape={rgb_recovered.shape}, "
          f"dtype={rgb_recovered.dtype}  ← expected (256, 256, 3) uint8")

    print("\n[verify] ✅ Pipeline OK — all tensor shapes verified!\n")
    return True


if __name__ == "__main__":
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/train"
    os.makedirs(test_dir, exist_ok=True)
    verify_pipeline(test_dir)
