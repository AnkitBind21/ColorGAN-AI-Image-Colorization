"""
Phase 4 & 5: GAN Training Loop
--------------------------------
Trains Generator (U-Net) + Discriminator (PatchGAN) together.

Training step for each batch:
  1. Generator forward:  fake_AB = G(L)
  2. Discriminator step:
       D_real_loss = BCE(D(L, real_AB), ones)
       D_fake_loss = BCE(D(L, fake_AB.detach()), zeros)
       D_loss = (D_real + D_fake) / 2
  3. Generator step:
       G_gan_loss  = BCE(D(L, fake_AB), ones)   ← fool discriminator
       G_l1_loss   = L1(fake_AB, real_AB) * 100 ← pixel fidelity
       G_loss = G_gan_loss + G_l1_loss

Key implementation notes:
  - fake_AB.detach() in D step — stops gradients flowing into G
  - GradScaler for mixed precision (fp16) to save VRAM
  - Checkpoint every 5 epochs
  - Log losses every N steps for monitoring
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add models dir to path
sys.path.insert(0, os.path.dirname(__file__))
from dataset       import ColorizationDataset, lab_to_rgb
from generator     import UNetGenerator
from discriminator import PatchGANDiscriminator

# Optional: metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity   as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

class TrainConfig:
    image_size   : int   = 256
    batch_size   : int   = 4      # Lower default for CPU/low VRAM
    epochs       : int   = 100
    lr           : float = 0.0002
    betas        : tuple = (0.5, 0.999)
    lambda_l1    : float = 100.0
    num_workers  : int   = 2
    save_every   : int   = 5      # Save checkpoint every N epochs
    log_every    : int   = 10     # Log losses every N steps
    device       : str   = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    train_dir    : str   = "dataset/train"
    test_dir     : str   = "dataset/test"
    checkpoint_dir: str  = "checkpoints"
    results_dir  : str   = "results"


cfg = TrainConfig()


# ─────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Wraps BCEWithLogitsLoss with automatic label creation.
    Uses ones_like/zeros_like — safe regardless of patch output shape.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target = torch.ones_like(pred)   # (B, 1, 30, 30) real label
        else:
            target = torch.zeros_like(pred)  # (B, 1, 30, 30) fake label
        return self.criterion(pred, target)


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(real_rgb: np.ndarray, fake_rgb: np.ndarray):
    """
    Compute PSNR and SSIM between real and fake RGB images.
    Both inputs: (H, W, 3) uint8 numpy arrays.
    """
    if not HAS_SKIMAGE:
        return {"psnr": None, "ssim": None}

    p = psnr_fn(real_rgb, fake_rgb, data_range=255)
    s = ssim_fn(real_rgb, fake_rgb, data_range=255, channel_axis=2)
    return {"psnr": float(p), "ssim": float(s)}


# ─────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def save_checkpoint(epoch, G, D, opt_G, opt_D, history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"    : epoch,
        "G_state"  : G.state_dict(),
        "D_state"  : D.state_dict(),
        "opt_G"    : opt_G.state_dict(),
        "opt_D"    : opt_D.state_dict(),
        "history"  : history,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    return ckpt["epoch"], ckpt.get("history", {})


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────

def train(cfg: TrainConfig = None, resume_from: str = None):
    if cfg is None:
        cfg = TrainConfig()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir,    exist_ok=True)
    os.makedirs(cfg.test_dir,       exist_ok=True)

    device = torch.device(cfg.device)
    print(f"\n[Train] Device: {device}")
    if device.type == "cuda":
        print(f"        GPU   : {torch.cuda.get_device_name(0)}")

    # ── Dataset ────────────────────────────────────────────────
    print("\n[Train] Loading datasets...")
    train_ds = ColorizationDataset(cfg.train_dir, image_size=cfg.image_size, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"  Train batches per epoch: {len(train_loader)}")

    # ── Models ─────────────────────────────────────────────────
    G = UNetGenerator(in_channels=1, out_channels=2).to(device)
    D = PatchGANDiscriminator(in_channels=3).to(device)

    G_params = sum(p.numel() for p in G.parameters())
    D_params = sum(p.numel() for p in D.parameters())
    print(f"\n[Train] Generator params     : {G_params:,}")
    print(f"        Discriminator params  : {D_params:,}")

    # ── Optimizers ─────────────────────────────────────────────
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    # ── Loss ───────────────────────────────────────────────────
    gan_loss = GANLoss().to(device)
    l1_loss  = nn.L1Loss()

    # ── Mixed Precision ────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)
    print(f"\n[Train] Mixed precision (AMP) : {use_amp}")

    # ── Resume ─────────────────────────────────────────────────
    start_epoch = 0
    history = {"G_loss": [], "D_loss": [], "G_gan": [], "G_l1": [], "psnr": [], "ssim": []}

    if resume_from and os.path.exists(resume_from):
        print(f"\n[Train] Resuming from {resume_from}")
        start_epoch, history = load_checkpoint(resume_from, G, D, opt_G, opt_D, device)
        start_epoch += 1

    # ─────────────────────────────────────────────────────────
    # Main epoch loop
    # ─────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training — epochs {start_epoch} → {cfg.epochs}")
    print("=" * 60)

    for epoch in range(start_epoch, cfg.epochs):
        G.train(); D.train()
        epoch_G, epoch_D, epoch_psnr, epoch_ssim = [], [], [], []
        t_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{cfg.epochs}", leave=False)

        for step, batch in enumerate(pbar):
            L_real  = batch["L"].to(device)   # (B, 1, 256, 256)
            AB_real = batch["AB"].to(device)   # (B, 2, 256, 256)

            # ══ Discriminator Step ══════════════════════════════
            # Detach fake_AB so G gradients don't flow through D
            with autocast(enabled=use_amp):
                fake_AB   = G(L_real)                         # (B, 2, 256, 256)
                D_real    = D(L_real, AB_real)                # (B, 1, 30, 30) real
                D_fake    = D(L_real, fake_AB.detach())       # (B, 1, 30, 30) fake
                D_real_loss = gan_loss(D_real, target_is_real=True)
                D_fake_loss = gan_loss(D_fake, target_is_real=False)
                D_loss    = (D_real_loss + D_fake_loss) * 0.5

            opt_D.zero_grad()
            scaler.scale(D_loss).backward()
            scaler.step(opt_D)

            # ══ Generator Step ══════════════════════════════════
            with autocast(enabled=use_amp):
                D_fake_for_G  = D(L_real, fake_AB)           # (B, 1, 30, 30)
                G_gan_loss    = gan_loss(D_fake_for_G, target_is_real=True)
                G_l1          = l1_loss(fake_AB, AB_real) * cfg.lambda_l1
                G_loss        = G_gan_loss + G_l1

            opt_G.zero_grad()
            scaler.scale(G_loss).backward()
            scaler.step(opt_G)
            scaler.update()

            # ── Track losses ──────────────────────────────────
            epoch_G.append(G_loss.item())
            epoch_D.append(D_loss.item())

            # ── PSNR/SSIM on first sample of batch ────────────
            if HAS_SKIMAGE and step % cfg.log_every == 0:
                with torch.no_grad():
                    real_rgb = lab_to_rgb(L_real[0:1], AB_real[0:1])
                    fake_rgb = lab_to_rgb(L_real[0:1], fake_AB[0:1])
                m = compute_metrics(real_rgb, fake_rgb)
                if m["psnr"]: epoch_psnr.append(m["psnr"])
                if m["ssim"]: epoch_ssim.append(m["ssim"])

            if step % cfg.log_every == 0:
                pbar.set_postfix({
                    "D": f"{D_loss.item():.3f}",
                    "G": f"{G_loss.item():.3f}",
                })

        # ── Epoch summary ─────────────────────────────────────
        mean_G    = np.mean(epoch_G)
        mean_D    = np.mean(epoch_D)
        mean_psnr = np.mean(epoch_psnr) if epoch_psnr else 0.0
        mean_ssim = np.mean(epoch_ssim) if epoch_ssim else 0.0
        elapsed   = time.time() - t_start

        history["G_loss"].append(mean_G)
        history["D_loss"].append(mean_D)
        history["psnr"].append(mean_psnr)
        history["ssim"].append(mean_ssim)

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | "
              f"G: {mean_G:.4f} | D: {mean_D:.4f} | "
              f"PSNR: {mean_psnr:.2f} | SSIM: {mean_ssim:.4f} | "
              f"Time: {elapsed:.1f}s")

        # ── Checkpoint ────────────────────────────────────────
        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            save_checkpoint(epoch, G, D, opt_G, opt_D, history, ckpt_path)
            # Also save best-format for inference
            torch.save(G.state_dict(), os.path.join(cfg.checkpoint_dir, "generator_latest.pt"))

    # ── Save history ──────────────────────────────────────────
    hist_path = os.path.join(cfg.results_dir, "train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved → {hist_path}")
    print("[Train] ✅ Training complete!")

    return G, D, history


# ─────────────────────────────────────────────────────────────
# Dry-run check (verifies full forward pass without real data)
# ─────────────────────────────────────────────────────────────

def verify_train_step():
    """
    Run one synthetic training step to verify the entire
    G + D + loss computation works correctly before using real data.
    """
    print("\n[Training Step Verification — Synthetic Data]")
    print("=" * 55)

    device = torch.device("cpu")
    B = 2

    G = UNetGenerator(1, 2).to(device)
    D = PatchGANDiscriminator(3).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gan_loss = GANLoss()
    l1_loss  = nn.L1Loss()

    L_real  = torch.randn(B, 1, 256, 256)
    AB_real = torch.randn(B, 2, 256, 256).clamp(-1, 1)

    # ── D Step ──────────────────────────────────────────
    fake_AB     = G(L_real)
    D_real      = D(L_real, AB_real)
    D_fake      = D(L_real, fake_AB.detach())
    print(f"  D_real output : {D_real.shape}  ← (B, 1, 30, 30)")
    print(f"  D_fake output : {D_fake.shape}  ← (B, 1, 30, 30)")

    D_loss = (gan_loss(D_real, True) + gan_loss(D_fake, False)) * 0.5
    opt_D.zero_grad()
    D_loss.backward()
    opt_D.step()
    print(f"  D_loss        : {D_loss.item():.4f}  ← should be ~0.69 (ln2) at init")

    # ── G Step ──────────────────────────────────────────
    fake_AB    = G(L_real)
    G_gan_loss = gan_loss(D(L_real, fake_AB), True)
    G_l1       = l1_loss(fake_AB, AB_real) * 100.0
    G_loss     = G_gan_loss + G_l1
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    print(f"  G_gan_loss    : {G_gan_loss.item():.4f}")
    print(f"  G_l1_loss     : {G_l1.item():.4f}  (L1 × 100)")
    print(f"  G_total_loss  : {G_loss.item():.4f}")
    print(f"  fake_AB range : [{fake_AB.min():.3f}, {fake_AB.max():.3f}]")

    assert fake_AB.shape == (B, 2, 256, 256)
    print(f"\n  ✅ Full training step verified — no shape errors ✓")


if __name__ == "__main__":
    import argparse

    # First always verify the training step synthetically
    verify_train_step()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=4)
    args = parser.parse_args()

    if args.train:
        cfg.epochs     = args.epochs
        cfg.batch_size = args.batch
        train(cfg, resume_from=args.resume)
