"""
Multi-Band Dataset for InternVL2-2B LoRA Training.

Loads BigEarthNet.txt instruction-tuning entries and pairs them with
multi-band satellite imagery from the imagery cache.

Pipeline:
  1. Load JSONL entry (question + answer in InternVL2 chat format)
  2. Load .npz file for the patch (10 S2 bands + 2 S1 bands)
  3. Normalize using official BigEarthNet v2 statistics
  4. Pass through SpectralChannelAdapter → 3ch pseudo-RGB
  5. Tokenize conversation
  6. Return tensors for training

Handles missing imagery gracefully:
  - If .npz exists → use real bands
  - If .npz missing → use synthetic placeholder
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

# Band ordering (must match SpectralChannelAdapter.BAND_ORDERS[10])
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

S2_STATS = {
    "B02": {"mean": 438.372, "std": 607.027},
    "B03": {"mean": 614.056, "std": 603.297},
    "B04": {"mean": 588.410, "std": 684.569},
    "B05": {"mean": 942.843, "std": 738.433},
    "B06": {"mean": 1769.932, "std": 1100.456},
    "B07": {"mean": 2049.552, "std": 1275.805},
    "B08": {"mean": 2193.292, "std": 1369.372},
    "B8A": {"mean": 2235.557, "std": 1356.544},
    "B11": {"mean": 1568.227, "std": 1070.161},
    "B12": {"mean": 997.732, "std": 813.528},
}

S1_STATS = {
    "VV": {"mean": -12.644, "std": 5.133},
    "VH": {"mean": -19.353, "std": 5.591},
}


class BigEarthNetMultiBandDataset(Dataset):
    """Multi-band dataset for InternVL2-2B instruction tuning.

    Each item returns:
      - s2_bands: (10, 120, 120) normalized S2 bands
      - s1_bands: (2, 120, 120) normalized S1 bands (or zeros)
      - input_ids: tokenized conversation
      - attention_mask: for the tokenizer
      - labels: for computing loss
      - has_real_image: bool, whether this sample has real imagery
    """

    def __init__(
        self,
        jsonl_path: str,
        imagery_cache_dir: str = "data/imagery_cache",
        tokenizer=None,
        max_length: int = 1024,
        image_size: int = 448,
        require_real_images: bool = False,
    ):
        """
        Args:
            jsonl_path: Path to train.jsonl or val.jsonl
            imagery_cache_dir: Directory with .npz patch files
            tokenizer: InternLM2 tokenizer
            max_length: Max sequence length
            image_size: Target image size (448 for InternVL2)
            require_real_images: If True, skip entries without cached imagery
        """
        self.imagery_cache = Path(imagery_cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size

        # Load JSONL entries
        self.entries = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                if require_real_images:
                    npz_path = self.imagery_cache / f"{entry['image']}.npz"
                    if not npz_path.exists():
                        skipped += 1
                        continue
                self.entries.append(entry)

        logger.info(
            f"Loaded {len(self.entries)} entries from {jsonl_path}"
            f"{f' (skipped {skipped} without imagery)' if skipped else ''}"
        )

        # Build normalization arrays
        self.s2_mean = np.array([S2_STATS[b]["mean"] for b in S2_BANDS], dtype=np.float32).reshape(-1, 1, 1)
        self.s2_std = np.array([S2_STATS[b]["std"] for b in S2_BANDS], dtype=np.float32).reshape(-1, 1, 1)
        self.s1_mean = np.array([S1_STATS["VV"]["mean"], S1_STATS["VH"]["mean"]], dtype=np.float32).reshape(-1, 1, 1)
        self.s1_std = np.array([S1_STATS["VV"]["std"], S1_STATS["VH"]["std"]], dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        patch_id = entry["image"]
        s1_name = entry.get("s1_name", "")

        # --- Load imagery ---
        s2_bands, s1_bands, has_real = self._load_imagery(patch_id)

        # --- Normalize ---
        s2_norm = (s2_bands - self.s2_mean) / (self.s2_std + 1e-8)
        s1_norm = (s1_bands - self.s1_mean) / (self.s1_std + 1e-8)

        # --- Build conversation text ---
        conversations = entry["conversations"]
        assert len(conversations) == 2, f"Expected 2 turns, got {len(conversations)}"

        human_msg = conversations[0]["value"]
        gpt_msg = conversations[1]["value"]

        # Format as InternVL2 chat template
        # The <image> token tells the model where to insert visual features
        if "<image>" not in human_msg:
            human_msg = "<image>\n" + human_msg

        # Tokenize
        if self.tokenizer is not None:
            result = self._tokenize_conversation(human_msg, gpt_msg)
        else:
            result = {}

        result.update({
            "s2_bands": torch.from_numpy(s2_norm).float(),
            "s1_bands": torch.from_numpy(s1_norm).float(),
            "has_real_image": has_real,
            "patch_id": patch_id,
            # Keep text attached to its image sample. The DataLoader shuffles
            # training samples, so indexing self.entries in the training loop
            # can silently pair an image with a different patch's label.
            "human_message": human_msg,
            "assistant_message": gpt_msg,
        })

        return result

    def _load_imagery(self, patch_id: str) -> tuple:
        """Load imagery from cache, or generate synthetic."""
        npz_path = self.imagery_cache / f"{patch_id}.npz"

        if npz_path.exists():
            data = np.load(str(npz_path))
            s2 = data["s2"].astype(np.float32)
            s1 = data.get("s1", np.zeros((2, 120, 120), dtype=np.float32)).astype(np.float32)

            # Ensure correct shape
            if s2.shape[0] != 10:
                s2 = np.zeros((10, 120, 120), dtype=np.float32)
            if s1.shape[0] != 2:
                s1 = np.zeros((2, 120, 120), dtype=np.float32)

            return s2, s1, True

        else:
            # Generate deterministic synthetic imagery
            s2, s1 = self._generate_synthetic(patch_id)
            return s2, s1, False

    def _generate_synthetic(self, patch_id: str) -> tuple:
        """Generate synthetic imagery from band statistics.

        Deterministic: same patch_id always produces the same image.
        """
        seed = int(hashlib.md5(patch_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        # S2 bands
        s2 = np.zeros((10, 120, 120), dtype=np.float32)
        for i, band in enumerate(S2_BANDS):
            stats = S2_STATS[band]
            base = rng.normal(stats["mean"], stats["std"] * 0.4, (120, 120))
            # Add spatial structure
            from scipy.ndimage import gaussian_filter
            smooth = gaussian_filter(base, sigma=8.0)
            s2[i] = np.clip(0.3 * base + 0.7 * smooth, 0, 10000).astype(np.float32)

        # S1 bands
        s1 = np.zeros((2, 120, 120), dtype=np.float32)
        for i, pol in enumerate(["VV", "VH"]):
            stats = S1_STATS[pol]
            base = rng.normal(stats["mean"], stats["std"] * 0.5, (120, 120))
            from scipy.ndimage import gaussian_filter
            smooth = gaussian_filter(base, sigma=6.0)
            s1[i] = (0.4 * base + 0.6 * smooth).astype(np.float32)

        return s2, s1

    def _tokenize_conversation(self, human_msg: str, gpt_msg: str) -> dict:
        """Tokenize a single conversation turn for InternVL2 training.

        Returns input_ids, attention_mask, and labels with the human
        portion masked out (labels=-100) so we only compute loss on
        the model's response.
        """
        # Build the full conversation string
        # InternVL2 uses internlm2-chat template:
        #   <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
        full_text = (
            f"<|im_start|>user\n{human_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{gpt_msg}<|im_end|>"
        )

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Create labels: mask the human part, only compute loss on assistant response
        labels = input_ids.clone()

        # Find where the assistant response starts
        assistant_start_text = f"<|im_start|>assistant\n"
        assistant_tokens = self.tokenizer.encode(assistant_start_text, add_special_tokens=False)

        # Find the position of assistant start in input_ids
        for pos in range(len(input_ids) - len(assistant_tokens)):
            if input_ids[pos:pos + len(assistant_tokens)].tolist() == assistant_tokens:
                # Mask everything before and including the assistant header
                labels[:pos + len(assistant_tokens)] = -100
                break
        else:
            # Couldn't find assistant start — mask first half as fallback
            labels[:len(labels) // 2] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_real_image_stats(self) -> dict:
        """Report how many entries have real vs synthetic imagery."""
        real = 0
        synthetic = 0
        for entry in self.entries:
            npz_path = self.imagery_cache / f"{entry['image']}.npz"
            if npz_path.exists():
                real += 1
            else:
                synthetic += 1
        return {
            "total": len(self.entries),
            "real_imagery": real,
            "synthetic_imagery": synthetic,
            "real_pct": real / len(self.entries) * 100 if self.entries else 0,
        }


def create_dataloaders(
    train_jsonl: str = "data/training/train.jsonl",
    val_jsonl: str = "data/training/val.jsonl",
    imagery_cache: str = "data/imagery_cache",
    tokenizer=None,
    batch_size: int = 1,
    max_length: int = 1024,
    require_real: bool = False,
    num_workers: int = 0,
):
    """Create train and validation DataLoaders."""
    from torch.utils.data import DataLoader

    train_ds = BigEarthNetMultiBandDataset(
        jsonl_path=train_jsonl,
        imagery_cache_dir=imagery_cache,
        tokenizer=tokenizer,
        max_length=max_length,
        require_real_images=require_real,
    )

    val_ds = BigEarthNetMultiBandDataset(
        jsonl_path=val_jsonl,
        imagery_cache_dir=imagery_cache,
        tokenizer=tokenizer,
        max_length=max_length,
        require_real_images=require_real,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Log imagery stats
    train_stats = train_ds.get_real_image_stats()
    val_stats = val_ds.get_real_image_stats()
    logger.info(f"Train: {train_stats}")
    logger.info(f"Val:   {val_stats}")

    return train_loader, val_loader
