"""
Phase 2: Generator — U-Net Architecture
-----------------------------------------
Input  : L channel   → (B, 1, 256, 256)
Output : AB channels → (B, 2, 256, 256)

U-Net uses skip connections from encoder → decoder to preserve
spatial detail, which is crucial for colorization.

Encoder path (downsampling):
  (B,1,256,256) → (B,64,128,128) → (B,128,64,64) → (B,256,32,32)
  → (B,512,16,16) → (B,512,8,8) → (B,512,4,4) → (B,512,2,2)

Bottleneck:
  (B,512,2,2) → (B,512,1,1)

Decoder path (upsampling) with skip-concat:
  (B,512,1,1) → (B,512,2,2) → concat with enc6 → (B,1024,2,2) → (B,512,2,2)
  ...
  → Final: (B,2,256,256) with Tanh → values in [-1,1] (normalized AB)
"""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Encoder block: Conv → BatchNorm → LeakyReLU
    LeakyReLU instead of ReLU to prevent dead neurons in discriminator-style blocks.
    First block skips BN (standard pix2pix practice).
    """
    def __init__(self, in_ch: int, out_ch: int, apply_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not apply_bn)
        ]
        if apply_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block: ConvTranspose → BatchNorm → ReLU (+ optional Dropout)
    Dropout on first 3 decoder blocks adds stochasticity (helps avoid mode collapse).
    """
    def __init__(self, in_ch: int, out_ch: int, apply_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for image colorization.

    Input  : (B, 1, 256, 256)  — L channel, normalized to [-1, 1]
    Output : (B, 2, 256, 256)  — AB channels, normalized to [-1, 1]

    Encoder produces 8 feature maps (e1 … e8).
    Decoder upsamples and concatenates skip connections from encoder.
    Skip connections double the channel count at each decoder step.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────
        # No BN on first layer (pix2pix convention)
        self.e1 = EncoderBlock(in_channels, 64,   apply_bn=False)  # → (B,64,128,128)
        self.e2 = EncoderBlock(64,  128)                            # → (B,128,64,64)
        self.e3 = EncoderBlock(128, 256)                            # → (B,256,32,32)
        self.e4 = EncoderBlock(256, 512)                            # → (B,512,16,16)
        self.e5 = EncoderBlock(512, 512)                            # → (B,512,8,8)
        self.e6 = EncoderBlock(512, 512)                            # → (B,512,4,4)
        self.e7 = EncoderBlock(512, 512)                            # → (B,512,2,2)

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # → (B,512,1,1)
            nn.ReLU(inplace=True),
        )

        # ── Decoder ─────────────────────────────────────────────
        # After upsample + skip concat, in_ch doubles
        self.d1 = DecoderBlock(512,       512, apply_dropout=True)  # → (B,512,2,2)
        self.d2 = DecoderBlock(512 + 512, 512, apply_dropout=True)  # → (B,512,4,4)
        self.d3 = DecoderBlock(512 + 512, 512, apply_dropout=True)  # → (B,512,8,8)
        self.d4 = DecoderBlock(512 + 512, 512)                       # → (B,512,16,16)
        self.d5 = DecoderBlock(512 + 512, 256)                       # → (B,256,32,32)
        self.d6 = DecoderBlock(256 + 256, 128)                       # → (B,128,64,64)
        self.d7 = DecoderBlock(128 + 128, 64)                        # → (B,64,128,128)

        # ── Final Layer ──────────────────────────────────────────
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1] matching normalized AB range
        )                                                             # → (B,2,256,256)

    def forward(self, x):
        # ── Encode ──────────────────────────────────────
        e1 = self.e1(x)          # (B,64,128,128)
        e2 = self.e2(e1)         # (B,128,64,64)
        e3 = self.e3(e2)         # (B,256,32,32)
        e4 = self.e4(e3)         # (B,512,16,16)
        e5 = self.e5(e4)         # (B,512,8,8)
        e6 = self.e6(e5)         # (B,512,4,4)
        e7 = self.e7(e6)         # (B,512,2,2)
        bn = self.bottleneck(e7) # (B,512,1,1)

        # ── Decode (with skip connections) ──────────────
        d1 = self.d1(bn)                         # (B,512,2,2)
        d2 = self.d2(torch.cat([d1, e7], dim=1)) # (B,1024,2,2) → (B,512,4,4)
        d3 = self.d3(torch.cat([d2, e6], dim=1)) # (B,1024,4,4) → (B,512,8,8)
        d4 = self.d4(torch.cat([d3, e5], dim=1)) # (B,1024,8,8) → (B,512,16,16)
        d5 = self.d5(torch.cat([d4, e4], dim=1)) # (B,1024,16,16) → (B,256,32,32)
        d6 = self.d6(torch.cat([d5, e3], dim=1)) # (B,512,32,32) → (B,128,64,64)
        d7 = self.d7(torch.cat([d6, e2], dim=1)) # (B,256,64,64) → (B,64,128,128)
        out = self.final(torch.cat([d7, e1], dim=1))  # (B,128,128,128) → (B,2,256,256)

        return out  # AB channels in [-1, 1]


# ─────────────────────────────────────────────────────────────
# Shape Verification
# ─────────────────────────────────────────────────────────────

def verify_generator():
    print("\n[Generator Shape Verification]")
    print("=" * 50)

    model = UNetGenerator(in_channels=1, out_channels=2)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Forward pass
    dummy_L = torch.randn(2, 1, 256, 256)  # Batch of 2
    print(f"\n  Input  L    : {dummy_L.shape}  (B, 1, H, W)")

    with torch.no_grad():
        dummy_AB = model(dummy_L)

    print(f"  Output AB   : {dummy_AB.shape}  (B, 2, H, W)")
    print(f"  AB range    : [{dummy_AB.min():.3f}, {dummy_AB.max():.3f}]  ← should be in (-1, 1)")

    assert dummy_AB.shape == (2, 2, 256, 256), \
        f"Generator output shape WRONG: {dummy_AB.shape}"
    assert dummy_AB.min() >= -1.0 and dummy_AB.max() <= 1.0, \
        "Generator output out of [-1, 1] range!"

    print("\n  ✅ Generator shape verified — (B, 2, 256, 256) ✓")
    print("  ✅ Tanh output in [-1, 1] ✓")

    # Check intermediate shapes manually
    print("\n  [Intermediate Shape Trace]")
    model_trace = UNetGenerator(in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 256, 256)
    e1 = model_trace.e1(x);          print(f"    e1: {e1.shape}")
    e2 = model_trace.e2(e1);         print(f"    e2: {e2.shape}")
    e3 = model_trace.e3(e2);         print(f"    e3: {e3.shape}")
    e4 = model_trace.e4(e3);         print(f"    e4: {e4.shape}")
    e5 = model_trace.e5(e4);         print(f"    e5: {e5.shape}")
    e6 = model_trace.e6(e5);         print(f"    e6: {e6.shape}")
    e7 = model_trace.e7(e6);         print(f"    e7: {e7.shape}")
    bn = model_trace.bottleneck(e7); print(f"    bn: {bn.shape}")
    d1 = model_trace.d1(bn);         print(f"    d1: {d1.shape}")
    d2 = model_trace.d2(torch.cat([d1, e7], dim=1)); print(f"    d2: {d2.shape}")
    d3 = model_trace.d3(torch.cat([d2, e6], dim=1)); print(f"    d3: {d3.shape}")
    d4 = model_trace.d4(torch.cat([d3, e5], dim=1)); print(f"    d4: {d4.shape}")
    d5 = model_trace.d5(torch.cat([d4, e4], dim=1)); print(f"    d5: {d5.shape}")
    d6 = model_trace.d6(torch.cat([d5, e3], dim=1)); print(f"    d6: {d6.shape}")
    d7 = model_trace.d7(torch.cat([d6, e2], dim=1)); print(f"    d7: {d7.shape}")
    out = model_trace.final(torch.cat([d7, e1], dim=1)); print(f"    out: {out.shape}")

    return model


if __name__ == "__main__":
    verify_generator()
