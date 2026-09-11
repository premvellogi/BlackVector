"""
SatQuery AI -- Create Experiment 2 Training Data.

Scales from Exp 1 (494 patches, ~10K samples) to Exp 2 (1,500 patches, ~30K samples).
Uses the same pipeline but with 3x more patches.

Key change: We only sample patches that have cached imagery in data/imagery_cache/.
"""

import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_exp2_data(
    parquet_path="data/bigearthnet_txt/BigEarthNet.txt.parquet",
    imagery_cache="data/imagery_cache",
    output_dir="data/training_exp2",
    target_patches=1500,
    val_fraction=0.1,
    seed=42,
):
    """Create Exp 2 training data by sampling 1,500 patches."""

    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load parquet ──
    logger.info(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"  Total annotations: {len(df):,}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Filter to train split only
    df = df[df["split"] == "train"].copy()
    logger.info(f"  Train split: {len(df):,} annotations")

    # ── 2. Find patches with cached imagery ──
    cached_patches = set(f.stem for f in Path(imagery_cache).glob("*.npz"))
    logger.info(f"  Cached patches: {len(cached_patches)}")

    # Match parquet patch_ids to cached
    df["has_imagery"] = df["patch_id"].isin(cached_patches)
    available = df[df["has_imagery"]].copy()
    available_patches = sorted(available["patch_id"].unique())
    logger.info(f"  Patches with imagery in parquet: {len(available_patches)}")

    if len(available_patches) < target_patches:
        logger.warning(f"  Only {len(available_patches)} available, using all")
        target_patches = len(available_patches)

    # ── 3. Stratified sampling by country ──
    random.seed(seed)

    # Get country distribution
    patch_country = available.drop_duplicates("patch_id")[["patch_id", "country"]].set_index("patch_id")["country"]

    country_patches = defaultdict(list)
    for pid in available_patches:
        country = patch_country.get(pid, "unknown")
        country_patches[country].append(pid)

    # Sample proportionally from each country
    selected_patches = set()
    total_available = len(available_patches)

    for country, pids in sorted(country_patches.items()):
        n_from_country = max(1, int(len(pids) / total_available * target_patches))
        sampled = random.sample(pids, min(n_from_country, len(pids)))
        selected_patches.update(sampled)

    # Top up if needed
    remaining = [p for p in available_patches if p not in selected_patches]
    random.shuffle(remaining)
    while len(selected_patches) < target_patches and remaining:
        selected_patches.add(remaining.pop())

    logger.info(f"  Selected patches: {len(selected_patches)}")

    # ── 4. Filter annotations to selected patches ──
    subset = available[available["patch_id"].isin(selected_patches)].copy()
    logger.info(f"  Annotations for selected patches: {len(subset):,}")

    # Type distribution
    type_dist = subset["type"].value_counts()
    logger.info(f"  Type distribution:\n{type_dist.to_string()}")

    # ── 5. Format into InternVL2 JSONL ──
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

    entries = []
    for _, row in subset.iterrows():
        prompt_type = row["type"]
        template = PROMPT_TEMPLATES.get(prompt_type)
        if template is None:
            continue

        if prompt_type == "captioning":
            user_text = template
        else:
            user_text = template.format(input=row["input"])

        entry = {
            "image": row["patch_id"],
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": str(row["output"])},
            ],
        }
        entries.append(entry)

    logger.info(f"  Formatted entries: {len(entries):,}")

    # ── 6. Split train/val (patch-level) ──
    patch_list = sorted(selected_patches)
    random.seed(seed)
    random.shuffle(patch_list)

    val_size = max(10, int(len(patch_list) * val_fraction))
    val_patches = set(patch_list[:val_size])
    train_patches = set(patch_list[val_size:])

    train_entries = [e for e in entries if e["image"] in train_patches]
    val_entries = [e for e in entries if e["image"] in val_patches]

    random.shuffle(train_entries)
    random.shuffle(val_entries)

    logger.info(f"  Train: {len(train_entries):,} samples from {len(train_patches)} patches")
    logger.info(f"  Val:   {len(val_entries):,} samples from {len(val_patches)} patches")

    # ── 7. Save ──
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    with open(val_path, "w") as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + "\n")

    # ── 8. Save metadata ──
    meta = {
        "experiment": "exp2",
        "target_patches": target_patches,
        "actual_patches": len(selected_patches),
        "train_patches": len(train_patches),
        "val_patches": len(val_patches),
        "train_size": len(train_entries),
        "val_size": len(val_entries),
        "total_formatted": len(entries),
        "sampling_strategy": "stratified by country",
        "split_method": "patch-level random split (seeded)",
        "seed": seed,
        "formatted_by_type": dict(Counter(
            json.loads(open(train_path).readline())  # Just count from entries
        ) if False else {}),
    }

    # Count types properly
    type_counts = Counter()
    for e in entries:
        # Extract type from prompt template
        user_msg = e["conversations"][0]["value"]
        if "yes/no question" in user_msg:
            type_counts["binary"] += 1
        elif "multiple-choice" in user_msg:
            type_counts["mcq"] += 1
        elif "Describe this satellite" in user_msg:
            type_counts["captioning"] += 1
        elif "bounding box" in user_msg:
            type_counts["bounding box"] += 1
        else:
            type_counts["other"] += 1

    meta["formatted_by_type"] = dict(type_counts)

    with open(os.path.join(output_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n  Saved to {output_dir}/")
    logger.info(f"  Types: {dict(type_counts)}")

    # ── 9. Compare with Exp 1 ──
    exp1_meta_path = "data/training_exp1/dataset_meta.json"
    if os.path.exists(exp1_meta_path):
        with open(exp1_meta_path) as f:
            exp1 = json.load(f)
        logger.info(f"\n  === Exp 1 vs Exp 2 ===")
        logger.info(f"  Patches:  {exp1.get('sampled_patches', exp1.get('train_patches', '?'))} -> {len(selected_patches)}")
        logger.info(f"  Train:    {exp1.get('train_size', '?')} -> {len(train_entries)}")
        logger.info(f"  Val:      {exp1.get('val_size', '?')} -> {len(val_entries)}")

    return output_dir


if __name__ == "__main__":
    create_exp2_data()
