"""
BigEarthNet.txt Data Pipeline for SatQuery AI — REVISED.

Now based on the ACTUAL parquet schema after data exploration:
    Columns: ID, s1_name, patch_id, input, output, type, category,
             split, latitude, longitude, country, season, climate_zone

    Types: binary (3.6M), mcq (3.3M), bounding box (2.2M), captioning (464K)
    Categories: presence, area, count, adjacency, point, reference,
                season, climate zone, country, relative pos

    Train split: 4,674,281 annotations across 464,044 unique patches.
    Bench split: 15,029 annotations across 1,082 patches.

This script:
1. Loads the parquet file
2. Samples a stratified subset of patches from the TRAIN split
3. Formats annotations into InternVL2-2B instruction tuning JSONL
4. Creates train/val splits
5. Generates §5.1a-compliant documentation

IMPORTANT: Images are loaded separately via the LMDB format using
BigEarthNet's official BENImageReader. We generate the JSONL with
image paths (patch_ids) that will be resolved at training time.
The actual image download + LMDB conversion is a separate step
(see download_imagery.py).
"""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# Configuration
# =============================================================================

class PipelineConfig:
    """Configuration for BigEarthNet.txt data pipeline."""

    def __init__(
        self,
        parquet_path: str = "data/bigearthnet_txt/BigEarthNet.txt.parquet",
        output_dir: str = "data/training",

        # Subset sampling
        target_patch_count: int = 30_000,  # Number of unique patches to sample
        random_seed: int = 42,

        # Train/val split (from the sampled subset)
        val_fraction: float = 0.1,

        # Task balancing for instruction tuning
        # Max annotations per task type (prevents VQA binary from dominating)
        max_per_type: Optional[dict] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.output_dir = Path(output_dir)
        self.target_patch_count = target_patch_count
        self.random_seed = random_seed
        self.val_fraction = val_fraction

        # Balance the training set — without this, binary VQA would be 38%
        self.max_per_type = max_per_type or {
            "captioning": None,        # Use all (~464K total, ~228K in train)
            "binary": 100_000,         # Cap binary VQA (3.6M is too many)
            "mcq": 80_000,             # Cap MCQ
            "bounding box": 60_000,    # Cap grounding
        }


# =============================================================================
# InternVL2-2B Prompt Templates
# =============================================================================

# These templates wrap the raw BigEarthNet.txt input/output into the
# conversation format expected by InternVL2-2B for instruction tuning.

PROMPT_TEMPLATES = {
    "binary": (
        "<image>\nYou are a remote sensing expert analyzing a satellite image. "
        "Answer the following yes/no question based on what you observe.\n"
        "Question: {input}\n"
        "Answer with 'yes' or 'no'."
    ),
    "mcq": (
        "<image>\nYou are a remote sensing expert analyzing a satellite image. "
        "Answer the following multiple-choice question.\n"
        "{input}\n"
        "Respond with only the letter of the correct answer."
    ),
    "captioning": (
        "<image>\nYou are a remote sensing expert. "
        "Describe this satellite image in detail, including the land cover types, "
        "their spatial arrangement, and notable features."
    ),
    "bounding box": (
        "<image>\nYou are a remote sensing expert. "
        "Locate the following in the satellite image and provide bounding box coordinates.\n"
        "Instruction: {input}"
    ),
}


# =============================================================================
# Pipeline
# =============================================================================

class BigEarthNetTxtPipeline:
    """Process BigEarthNet.txt parquet into InternVL2-2B training format."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.df = None
        self.train_entries = []
        self.val_entries = []
        self.stats = {}

    def run(self):
        """Execute the full pipeline."""
        self.load()
        self.sample()
        self.format()
        self.save()
        self.save_adaptation_log()
        return self

    def load(self):
        """Load the parquet file and filter to train split."""
        logger.info(f"Loading {self.config.parquet_path}...")
        self.df = pd.read_parquet(self.config.parquet_path)
        total = len(self.df)
        logger.info(f"Total annotations: {total:,}")

        # Use ONLY the official train split for fine-tuning
        self.df = self.df[self.df["split"] == "train"].copy()
        logger.info(f"Train split: {len(self.df):,} annotations")

        self.stats["total_annotations"] = total
        self.stats["train_split_annotations"] = len(self.df)
        self.stats["total_unique_patches"] = self.df["patch_id"].nunique()
        self.stats["type_distribution_full"] = dict(self.df["type"].value_counts())
        self.stats["category_distribution_full"] = dict(self.df["category"].value_counts())
        return self

    def sample(self):
        """Stratified sampling of patches."""
        rng = np.random.RandomState(self.config.random_seed)
        n_target = self.config.target_patch_count

        all_patches = self.df["patch_id"].unique()
        n_total = len(all_patches)
        logger.info(f"Unique patches in train: {n_total:,}")

        if n_target >= n_total:
            logger.info(f"Using all {n_total:,} patches.")
            selected_patches = set(all_patches)
        else:
            # Stratified by country for geographic diversity
            patch_country = (
                self.df.groupby("patch_id")["country"]
                .first()
                .reset_index()
            )

            selected = []
            country_counts = patch_country["country"].value_counts()
            for country, count in country_counts.items():
                n_from_country = max(1, int(n_target * count / n_total))
                country_patches = patch_country[
                    patch_country["country"] == country
                ]["patch_id"].values
                n_sample = min(n_from_country, len(country_patches))
                selected.extend(
                    rng.choice(country_patches, n_sample, replace=False).tolist()
                )

            # Trim or pad to exact target
            if len(selected) > n_target:
                selected = rng.choice(selected, n_target, replace=False).tolist()
            selected_patches = set(selected)

        # Filter dataframe to selected patches
        self.df = self.df[self.df["patch_id"].isin(selected_patches)].copy()
        logger.info(
            f"After sampling: {len(selected_patches):,} patches, "
            f"{len(self.df):,} annotations"
        )

        self.stats["sampled_patches"] = len(selected_patches)
        self.stats["sampled_annotations"] = len(self.df)
        self.stats["sampling_strategy"] = "stratified by country"
        return self

    def format(self):
        """Convert to InternVL2-2B instruction tuning JSONL format."""
        rng = random.Random(self.config.random_seed)

        # Apply per-type caps
        type_groups = self.df.groupby("type")
        capped_frames = []
        for task_type, group in type_groups:
            max_n = self.config.max_per_type.get(task_type)
            if max_n is not None and len(group) > max_n:
                capped_frames.append(group.sample(n=max_n, random_state=self.config.random_seed))
                logger.info(f"Capped {task_type}: {len(group):,} → {max_n:,}")
            else:
                capped_frames.append(group)
                logger.info(f"Using all {task_type}: {len(group):,}")

        df_capped = pd.concat(capped_frames, ignore_index=True)

        # Format into JSONL entries
        entries = []
        for idx, row in df_capped.iterrows():
            entry = self._format_row(row)
            if entry:
                entries.append(entry)

        self.stats["formatted_total"] = len(entries)
        self.stats["formatted_by_type"] = dict(Counter(
            e.get("_task_type", "unknown") for e in entries
        ))

        # Split by patch, never by individual instruction row. Otherwise the
        # same satellite image appears in both train and validation data.
        patch_ids = sorted({entry["image"] for entry in entries})
        rng.shuffle(patch_ids)
        val_patch_count = max(1, int(len(patch_ids) * self.config.val_fraction))
        val_patch_ids = set(patch_ids[:val_patch_count])
        self.val_entries = [entry for entry in entries if entry["image"] in val_patch_ids]
        self.train_entries = [entry for entry in entries if entry["image"] not in val_patch_ids]
        rng.shuffle(self.val_entries)
        rng.shuffle(self.train_entries)
        self.stats["train_patches"] = len(patch_ids) - len(val_patch_ids)
        self.stats["val_patches"] = len(val_patch_ids)
        self.stats["split_method"] = "patch-level random split (seeded)"

        # Remove internal metadata before saving
        for e in self.train_entries + self.val_entries:
            e.pop("_task_type", None)

        self.stats["train_size"] = len(self.train_entries)
        self.stats["val_size"] = len(self.val_entries)

        logger.info(
            f"Formatted: {len(entries):,} total → "
            f"Train: {len(self.train_entries):,}, Val: {len(self.val_entries):,}"
        )
        return self

    def _format_row(self, row) -> Optional[dict]:
        """Format one parquet row into InternVL2 instruction format."""
        task_type = row["type"]
        text_input = row["input"]
        text_output = str(row["output"])
        patch_id = row["patch_id"]

        # Get the prompt template
        template = PROMPT_TEMPLATES.get(task_type)
        if template is None:
            return None

        # Build the human prompt
        if task_type == "captioning":
            human_msg = template  # Caption prompt doesn't use {input}
        else:
            human_msg = template.format(input=text_input)

        return {
            "id": f"{task_type}_{patch_id}_{row['ID']}",
            "image": patch_id,  # Resolved to actual path at training time
            "s1_name": row["s1_name"],
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": text_output},
            ],
            "_task_type": task_type,
        }

    def save(self):
        """Write JSONL files."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.config.output_dir / "train.jsonl"
        val_path = self.config.output_dir / "val.jsonl"
        meta_path = self.config.output_dir / "dataset_meta.json"
        manifest_path = self.config.output_dir / "subset_manifest.json"

        with open(train_path, "w", encoding="utf-8") as f:
            for entry in self.train_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.train_entries):,} train entries → {train_path}")

        with open(val_path, "w", encoding="utf-8") as f:
            for entry in self.val_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.val_entries):,} val entries → {val_path}")

        with open(meta_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        manifest = {}
        for entry in self.train_entries + self.val_entries:
            manifest[entry["image"]] = {
                "patch_id": entry["image"],
                "s1_name": entry.get("s1_name", ""),
            }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(list(manifest.values()), f, indent=2, ensure_ascii=False)
        self.stats["manifest_patches"] = len(manifest)
        logger.info(f"Saved metadata → {meta_path}")

        return self

    def save_adaptation_log(self, docs_dir: str = "docs"):
        """Generate and save §5.1a-compliant adaptation log."""
        docs_path = Path(docs_dir)
        docs_path.mkdir(parents=True, exist_ok=True)

        log = f"""# BigEarthNet.txt Adaptation Log (§5.1a Compliance)

## Component Adapted
- **Model:** InternVL2-2B (OpenGVLab/InternVL2-2B)
- **Adaptation method:** QLoRA (4-bit NF4 quantization, LoRA rank=8, alpha=16)
- **Target modules:** wqkv, wo (InternLM2 fused QKV + output projection)
- **Vision encoder:** Frozen (not fine-tuned -- critical for 6GB VRAM constraint)

## Dataset: BigEarthNet.txt
- **Source:** https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt
- **Total available:** {self.stats.get('total_annotations', 'N/A'):,} annotations across 464,044 co-registered S1+S2 image pairs
- **License:** CDLA Permissive 1.0

## Portion Used
- **Split used:** Official `train` split only ({self.stats.get('train_split_annotations', 'N/A'):,} annotations)
- **Patches sampled:** {self.stats.get('sampled_patches', 'N/A'):,} of {self.stats.get('total_unique_patches', 'N/A'):,}
- **Annotations after sampling:** {self.stats.get('sampled_annotations', 'N/A'):,}
- **After task capping:** {self.stats.get('formatted_total', 'N/A'):,} instruction entries

## Sampling Strategy
- **Method:** Stratified by acquisition country for geographic diversity
- **Random seed:** {self.config.random_seed}
- **Task type capping:** Applied to prevent binary VQA from dominating
  - captioning: uncapped (all available)
  - binary: capped to {self.config.max_per_type.get('binary', 'N/A'):,}
  - mcq: capped to {self.config.max_per_type.get('mcq', 'N/A'):,}
  - bounding box: capped to {self.config.max_per_type.get('bounding box', 'N/A'):,}

## Adaptation Objective
Multi-task instruction tuning:
1. **Captioning** — Scene description from satellite imagery
2. **Binary VQA** — Yes/no questions about land cover (presence, area, adjacency, etc.)
3. **MCQ VQA** — Multiple-choice questions (presence, season, climate zone, country)
4. **Bounding Box** — Referring expression detection / LULC localization

## Training Configuration
- Batch size: 1 (gradient accumulation: 8, effective batch: 8)
- Learning rate: 2e-4 (cosine schedule with warmup)
- Epochs: 3
- Max sequence length: 1024 tokens
- Optimizer: paged_adamw_8bit
- Gradient checkpointing: enabled

## Train/Validation Split
- **Train:** {self.stats.get('train_size', 'N/A'):,} entries ({1 - self.config.val_fraction:.0%})
- **Val:** {self.stats.get('val_size', 'N/A'):,} entries ({self.config.val_fraction:.0%})
- **Split method:** Patch-level split (all annotations for one image stay together) with seed {self.config.random_seed}

## Task Distribution After Formatting
```
{json.dumps(self.stats.get('formatted_by_type', {}), indent=2)}
```

## Checkpoint
- **Version:** [TO BE FILLED AFTER TRAINING]
- **Path:** checkpoints/satquery-lora/final/
- **Referenced in provenance (D4):** Yes — every inference response includes checkpoint version
"""
        log_path = docs_path / "adaptation_log.md"
        log_path.write_text(log)
        logger.info(f"Saved adaptation log → {log_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BigEarthNet.txt → InternVL2 training data")
    parser.add_argument("--parquet", default="data/bigearthnet_txt/BigEarthNet.txt.parquet")
    parser.add_argument("--output", default="data/training")
    parser.add_argument("--patches", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig(
        parquet_path=args.parquet,
        output_dir=args.output,
        target_patch_count=args.patches,
        random_seed=args.seed,
    )

    pipeline = BigEarthNetTxtPipeline(config)
    pipeline.run()
