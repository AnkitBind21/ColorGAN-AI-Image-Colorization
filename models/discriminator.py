"""
Phase 3: Discriminator — PatchGAN Architecture
------------------------------------------------
PatchGAN judges PATCHES of the image as real/fake rather than
the whole image. This forces the model to get local textures right.

Input: Concatenation of L + AB → (B, 3, 256, 256)
Output: Patch probability map → (B, 1, 30, 30)

Each value in the 30×30 output corresponds to a 70×70 patch in the input.
This is called a "70×70 PatchGAN".

CRITICAL: Never hardcode label shapes as (B, 1, 30, 30).
Always use: torch.ones_like(pred) and torch.zeros_like(pred)
because the output shape depends on input resolution.
"""

import torch
import torch.nn as nn


class PatchDiscBlock(nn.Module):
    """Conv → [BatchNorm] → LeakyReLU block for PatchGAN."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, apply_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not apply_bn)
        ]
        if apply_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN Discriminator.

    Input  : concat(L, AB) → (B, 3, 256, 256)
    Output : patch map     → (B, 1, 30, 30)

    Each output pixel is the "realness" score of a 70×70 receptive field patch.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # No BN on first layer
        self.block1 = PatchDiscBlock(in_channels, 64,  stride=2, apply_bn=False) # (B,64,128,128)
        self.block2 = PatchDiscBlock(64,  128, stride=2)  # (B,128,64,64)
        self.block3 = PatchDiscBlock(128, 256, stride=2)  # (B,256,32,32)
        self.block4 = PatchDiscBlock(256, 512, stride=1)  # (B,512,31,31)  ← stride=1 here

        # Final conv to produce 1-channel patch output
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        # (B,1,30,30) — NO sigmoid here; BCEWithLogitsLoss handles it

    def forward(self, L, AB):
        """
        Args:
            L  : (B, 1, 256, 256) — input L channel
            AB : (B, 2, 256, 256) — real or generated AB channels

        Returns:
            patch_map : (B, 1, 30, 30)
        """
        # Concatenate along channel dim → (B, 3, 256, 256)
        x = torch.cat([L, AB], dim=1)

        x = self.block1(x)   # (B,64,128,128)
        x = self.block2(x)   # (B,128,64,64)
        x = self.block3(x)   # (B,256,32,32)
        x = self.block4(x)   # (B,512,31,31)
        x = self.final(x)    # (B,1,30,30) — raw logits

        return x


# ─────────────────────────────────────────────────────────────
# Shape Verification
# ─────────────────────────────────────────────────────────────

def verify_discriminator():
    print("\n[Discriminator Shape Verification]")
    print("=" * 50)

    disc = PatchGANDiscriminator(in_channels=3)
    disc.eval()

    total_params = sum(p.numel() for p in disc.parameters())
    print(f"  Total parameters: {total_params:,}")

    B = 2
    dummy_L  = torch.randn(B, 1, 256, 256)
    dummy_AB = torch.randn(B, 2, 256, 256)

    print(f"\n  Input  L    : {dummy_L.shape}")
    print(f"  Input  AB   : {dummy_AB.shape}")
    print(f"  Concat      : (B, 3, 256, 256)  ← L + AB")

    with torch.no_grad():
        patch_out = disc(dummy_L, dummy_AB)

    print(f"  Output      : {patch_out.shape}  ← raw logits, no sigmoid")
    print(f"  Output range: [{patch_out.min():.3f}, {patch_out.max():.3f}]")

    assert patch_out.shape == (B, 1, 30, 30), \
        f"Discriminator output shape WRONG: {patch_out.shape}"

    # ── Label creation (CRITICAL: use ones_like/zeros_like, never hardcode) ──
    real_labels = torch.ones_like(patch_out)   # (B, 1, 30, 30) — 1s
    fake_labels = torch.zeros_like(patch_out)  # (B, 1, 30, 30) — 0s
    print(f"\n  Real labels : {real_labels.shape}  (ones_like — never hardcoded)")
    print(f"  Fake labels : {fake_labels.shape}  (zeros_like — never hardcoded)")

    # ── Loss test ──
    criterion = nn.BCEWithLogitsLoss()
    real_loss = criterion(patch_out, real_labels)
    fake_loss = criterion(patch_out, fake_labels)
    print(f"\n  BCEWithLogitsLoss(real): {real_loss.item():.4f}")
    print(f"  BCEWithLogitsLoss(fake): {fake_loss.item():.4f}")

    print("\n  ✅ Discriminator shape verified — (B, 1, 30, 30) ✓")
    print("  ✅ Labels created with ones_like/zeros_like ✓")
    print("  ✅ BCEWithLogitsLoss (no sigmoid needed) ✓")


if __name__ == "__main__":
    verify_discriminator()
