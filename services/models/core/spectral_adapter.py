"""
Spectral Channel Adapter for InternVL2-2B.

Problem:
  InternVL2's InternViT takes 3-channel RGB (448×448).
  Sentinel-2 has 13 bands at 3 resolutions. Sentinel-1 has 2 SAR bands.
  Total: 15 channels → need to project to 3 for the frozen ViT.

Solution:
  A tiny learned adapter (Conv2d) before the frozen vision encoder.
  This is standard in remote sensing VLM literature (see RSPrompter, GeoChat).

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Raw input: (B, 15, H, W) — 13 S2 bands + 2 S1 bands           │
  │                                                                  │
  │  SpectralAdapter (learned, ~2K params):                         │
  │    ├── Conv2d(15, 32, 1×1) + GELU                              │
  │    ├── Conv2d(32, 3, 1×1)                                       │
  │    └── LayerNorm                                                 │
  │                                                                  │
  │  Output: (B, 3, 448, 448) — pseudo-RGB for InternViT            │
  │                                                                  │
  │  InternViT-300M (FROZEN):                                       │
  │    └── 1024-dim visual features                                  │
  │                                                                  │
  │  MLP Projector → InternLM2-1.8B (LoRA on q/k/v/o)             │
  └──────────────────────────────────────────────────────────────────┘

The adapter is trained alongside the LoRA weights.
Only ~2K parameters — negligible overhead.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralChannelAdapter(nn.Module):
    """Projects multi-band satellite imagery to 3-channel pseudo-RGB.

    Supports flexible input channels:
      - 3 channels: RGB passthrough (for benchmark datasets like VRSBench)
      - 10 channels: S2 10m+20m bands only
      - 13 channels: All S2 bands
      - 15 channels: S2 (13) + S1 (2) — the full BigEarthNet setup
      - 2 channels: S1 SAR only (VV, VH)

    Architecture: Two 1×1 conv layers with GELU and LayerNorm.
    """

    # Canonical band ordering for each input configuration
    BAND_ORDERS = {
        2:  ["VV", "VH"],  # S1 SAR only
        3:  ["B04", "B03", "B02"],  # RGB passthrough
        10: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],  # 10m+20m
        13: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "B10"],  # All S2
        15: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "B10",
             "VV", "VH"],  # S2 + S1
    }

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 3,
        hidden_dim: int = 32,
        target_size: int = 448,
        init_rgb_passthrough: bool = True,
    ):
        """
        Args:
            in_channels: Number of input spectral bands.
            out_channels: Must be 3 (RGB for ViT).
            hidden_dim: Intermediate feature dimension.
            target_size: Output spatial size (448 for InternVL2).
            init_rgb_passthrough: If True, initialize so that B04/B03/B02
                                  roughly pass through (warm start from RGB).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size

        # Projection: in_channels → hidden → 3
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True),
        )

        # LayerNorm over channel dimension
        self.norm = nn.LayerNorm(out_channels)

        # Initialize
        self._init_weights(init_rgb_passthrough)

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SpectralChannelAdapter: {n_params:,} params "
              f"({in_channels}ch -> {out_channels}ch via {hidden_dim}d)")

    def _init_weights(self, init_rgb_passthrough: bool):
        """Initialize weights, optionally with RGB passthrough bias."""
        for m in self.proj:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if init_rgb_passthrough and self.in_channels >= 4:
            # Warm-start: make the first layer pay attention to R/G/B channels
            # In our canonical ordering for 15ch: B04=idx3 (R), B03=idx2 (G), B02=idx1 (B)
            band_order = self.BAND_ORDERS.get(self.in_channels)
            if band_order:
                with torch.no_grad():
                    conv1 = self.proj[0]
                    # Boost the weights for the RGB band indices
                    rgb_names = ["B04", "B03", "B02"]
                    for i, name in enumerate(rgb_names):
                        if name in band_order:
                            idx = band_order.index(name)
                            # Give extra weight to RGB channels in the first 3 hidden dims
                            conv1.weight[i, idx] += 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) where C = in_channels. Values should be
               normalized using official BigEarthNet v2 statistics.

        Returns:
            (B, 3, target_size, target_size) pseudo-RGB for InternViT.
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # If input is already RGB (3ch), still pass through adapter
        # (lets us keep the same pipeline for benchmarks)

        # Resize to target_size if needed
        if H != self.target_size or W != self.target_size:
            x = F.interpolate(
                x, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False,
            )

        # Project channels: C → 3
        out = self.proj(x)  # (B, 3, H, W)

        # LayerNorm over channels (need to permute)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, 3)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # (B, 3, H, W)

        return out


class DualModalityAdapter(nn.Module):
    """Separate adapters for optical (S2) and SAR (S1), then fuse.

    This is more principled than concatenating all 15 bands:
    S2 (reflectance, 0-10000) and S1 (dB, -25 to +5) have very
    different value ranges and physics.

    Architecture:
      S2 (13ch) → SpectralAdapter → 3ch pseudo-RGB₁
      S1 (2ch)  → SpectralAdapter → 3ch pseudo-RGB₂
      Fusion gate (learned α): output = α·RGB₁ + (1-α)·RGB₂
    """

    def __init__(
        self,
        s2_channels: int = 10,  # 10 bands (no 60m by default)
        s1_channels: int = 2,
        target_size: int = 448,
    ):
        super().__init__()

        self.s2_adapter = SpectralChannelAdapter(
            in_channels=s2_channels,
            out_channels=3,
            hidden_dim=32,
            target_size=target_size,
            init_rgb_passthrough=True,
        )

        self.s1_adapter = SpectralChannelAdapter(
            in_channels=s1_channels,
            out_channels=3,
            hidden_dim=16,
            target_size=target_size,
            init_rgb_passthrough=False,
        )

        # Learned fusion weight (initialized to favor optical)
        self.fusion_gate = nn.Parameter(torch.tensor(0.8))  # 80% optical

    def forward(
        self,
        s2_bands: torch.Tensor,
        s1_bands: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s2_bands: (B, s2_channels, H, W) — normalized S2 bands
            s1_bands: (B, s1_channels, H, W) — normalized S1 bands (optional)

        Returns:
            (B, 3, 448, 448) — fused pseudo-RGB for InternViT
        """
        optical_rgb = self.s2_adapter(s2_bands)

        if s1_bands is not None:
            sar_rgb = self.s1_adapter(s1_bands)
            alpha = torch.sigmoid(self.fusion_gate)
            fused = alpha * optical_rgb + (1 - alpha) * sar_rgb
            return fused
        else:
            return optical_rgb

    @property
    def modality_weight(self) -> dict:
        """Return the learned fusion weights (for §6.4 modality attribution)."""
        alpha = torch.sigmoid(self.fusion_gate).item()
        return {
            "optical_weight": alpha,
            "sar_weight": 1 - alpha,
        }


# =============================================================================
# Convenience: wrap InternVL2 + adapter
# =============================================================================

def create_adapted_model(
    model_path: str = "models/InternVL2-2B",
    s2_channels: int = 10,
    s1_channels: int = 2,
    device: str = "cuda",
):
    """Create InternVL2-2B with spectral channel adapter.

    Returns (model, adapter, tokenizer).
    The adapter is placed BEFORE the frozen ViT.
    """
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Freeze vision encoder
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # Create dual-modality adapter
    adapter = DualModalityAdapter(
        s2_channels=s2_channels,
        s1_channels=s1_channels,
        target_size=448,
    ).to(device, dtype=torch.bfloat16)

    return model, adapter, tokenizer
