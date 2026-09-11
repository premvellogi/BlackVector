"""
SatQuery AI -- Create Exp 3 Dataset.

Combines three categories with controlled sampling:
  Category A (70%): General satellite understanding (from Exp 2)
  Category B (20%): SAR-specific QA (oversampled 2x)
  Category C (10%): Fusion reasoning tasks (newly generated)

Also creates separate validation sets for tracking per-category improvement.

Output:
  data/training_exp3/train.jsonl     (combined training set)
  data/training_exp3/val.jsonl       (combined val)
  data/training_exp3/val_general.jsonl
  data/training_exp3/val_sar.jsonl
  data/training_exp3/val_fusion.jsonl
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 2026
OUTPUT_DIR = "data/training_exp3"

# ── Category C: Fusion Reasoning QA Templates ────────────────────────────

FUSION_TEMPLATES = {
    "compare_modalities": [
        {
            "input": "Compare what the optical and SAR observations reveal about this area.",
            "output": (
                "The optical imagery provides spectral information about surface "
                "materials, vegetation health through reflectance bands, and color-based "
                "land cover discrimination. The SAR radar data complements this with "
                "information about surface structure, roughness, and moisture content "
                "through backscatter intensity. Together, the fused observations provide "
                "a more complete understanding than either modality alone."
            ),
        },
        {
            "input": "What are the differences between the optical and radar observations?",
            "output": (
                "The optical bands capture reflected sunlight, showing vegetation "
                "greenness, water color, and surface albedo. The SAR radar measures "
                "microwave backscatter, which reveals surface texture, structural "
                "features like buildings, and moisture levels independent of cloud "
                "cover or lighting conditions. These complementary views allow for "
                "more robust land cover classification."
            ),
        },
    ],
    "sar_advantage": [
        {
            "input": "What additional information does the SAR radar provide beyond optical imagery?",
            "output": (
                "SAR provides structural information that optical imagery cannot: "
                "surface roughness through backscatter texture, building detection "
                "through double-bounce scattering, water surface detection through "
                "specular reflection, and soil moisture estimation. SAR also works "
                "regardless of cloud cover, illumination, and time of day, making "
                "it essential for all-weather monitoring."
            ),
        },
        {
            "input": "In what situations would SAR radar be more useful than optical imagery?",
            "output": (
                "SAR is more useful than optical in several situations: detecting "
                "water bodies (very low backscatter from specular reflection), "
                "identifying urban structures (double-bounce from buildings), "
                "monitoring soil moisture, mapping flood extent under vegetation, "
                "and any scenario with cloud cover or nighttime conditions where "
                "optical sensors cannot acquire useful data."
            ),
        },
    ],
    "optical_advantage": [
        {
            "input": "When does optical imagery provide stronger evidence than SAR?",
            "output": (
                "Optical imagery is stronger for vegetation species discrimination "
                "(through narrow spectral bands), crop type identification, water "
                "quality assessment, mineral/soil composition mapping, and any task "
                "requiring fine color or spectral differentiation. The multispectral "
                "bands (visible, near-infrared, shortwave infrared) each highlight "
                "different surface properties that SAR's two polarizations cannot "
                "distinguish."
            ),
        },
    ],
    "fusion_benefit": [
        {
            "input": "How does combining optical and SAR observations improve accuracy?",
            "output": (
                "Multimodal fusion improves accuracy by providing complementary "
                "information: optical contributes spectral discrimination while SAR "
                "adds structural and textural context. For example, a bare field "
                "and a young crop field may look similar in SAR but differ "
                "spectrally in optical; conversely, shadows in optical imagery can "
                "be resolved using SAR backscatter. The combined information reduces "
                "ambiguity and improves classification confidence."
            ),
        },
        {
            "input": "Why is multi-sensor fusion important for Earth observation?",
            "output": (
                "Multi-sensor fusion combines the strengths of different sensors: "
                "optical provides rich spectral information for material and "
                "vegetation identification, while SAR provides all-weather, day-night "
                "structural and moisture information. By fusing both, we achieve "
                "more robust, accurate, and temporally complete Earth observation "
                "than any single sensor can provide."
            ),
        },
    ],
}

# Land-cover-specific fusion QA
LANDCOVER_FUSION = {
    "water": [
        {
            "input": "Which modality provides stronger evidence for water detection in this image?",
            "output": (
                "Both modalities contribute to water detection. SAR shows very low "
                "backscatter from specular reflection off calm water surfaces, making "
                "water bodies appear dark. Optical imagery shows characteristic blue/dark "
                "spectral signatures. The combination is especially valuable for "
                "detecting water under vegetation or distinguishing water from shadows."
            ),
        },
    ],
    "urban": [
        {
            "input": "How do the modalities differ in urban area detection?",
            "output": (
                "Urban areas show distinctive signatures in both modalities. In SAR, "
                "buildings cause strong double-bounce scattering, appearing as bright "
                "spots with high backscatter. In optical imagery, built-up areas show "
                "characteristic spectral signatures from concrete, asphalt, and metal. "
                "Combining both provides robust urban mapping that handles shadows "
                "and cloud cover."
            ),
        },
    ],
    "forest": [
        {
            "input": "What do the different sensors reveal about the forested areas?",
            "output": (
                "Optical imagery reveals forest type through spectral signatures: "
                "broad-leaved versus coniferous forests show different near-infrared "
                "reflectance patterns. SAR adds information about canopy structure "
                "through volume scattering, tree height estimation, and biomass. "
                "The fusion provides both species-level classification from optical "
                "and structural characterization from radar."
            ),
        },
    ],
    "agricultural": [
        {
            "input": "How do optical and SAR complement each other for agricultural monitoring?",
            "output": (
                "Optical bands enable crop type identification through vegetation "
                "indices like NDVI, while SAR tracks crop growth stages through "
                "changing backscatter patterns (bare soil to mature crop). SAR is "
                "particularly valuable during cloudy seasons when optical data is "
                "unavailable. Together they enable continuous, accurate crop monitoring."
            ),
        },
    ],
}


def generate_fusion_qa(patch_id: str, landcover: str) -> list:
    """Generate fusion reasoning QA pairs for a patch."""
    pairs = []

    # General fusion questions (pick 1-2 random)
    for category, templates in FUSION_TEMPLATES.items():
        t = random.choice(templates)
        pairs.append({
            "image": patch_id,
            "input": t["input"],
            "output": t["output"],
            "type": "descriptive",
            "category": f"fusion_{category}",
        })

    # Land-cover-specific fusion (if available)
    if landcover in LANDCOVER_FUSION:
        for t in LANDCOVER_FUSION[landcover]:
            pairs.append({
                "image": patch_id,
                "input": t["input"],
                "output": t["output"],
                "type": "descriptive",
                "category": f"fusion_{landcover}",
            })

    return pairs


def to_conversations(item: dict) -> dict:
    """Convert input/output format to conversations format for training."""
    if "conversations" in item:
        return item  # Already in correct format

    question = item.get("input", "")
    answer = item.get("output", "")
    patch_id = item.get("image", "")

    return {
        "image": patch_id,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
        "category": item.get("category", "unknown"),
    }


def main():
    print("=" * 65)
    print("  SatQuery AI -- Exp 3 Dataset Builder")
    print("=" * 65)

    random.seed(SEED)

    # ── Load Category A: General (Exp 2) ──────────────────────────────
    print("\n  Loading Category A (General - Exp 2)...")
    cat_a_train = []
    for line in open("data/training_exp2/train.jsonl"):
        item = json.loads(line)
        item["category"] = "general"
        cat_a_train.append(item)

    cat_a_val = []
    for line in open("data/training_exp2/val.jsonl"):
        item = json.loads(line)
        item["category"] = "general"
        cat_a_val.append(item)

    print(f"    Train: {len(cat_a_train)}, Val: {len(cat_a_val)}")

    # ── Load Category B: SAR (2x oversampled) ─────────────────────────
    print("  Loading Category B (SAR - 2x oversampled)...")
    cat_b_train_raw = []
    for line in open("data/training_sar/train.jsonl"):
        item = json.loads(line)
        item["category"] = "sar"
        cat_b_train_raw.append(to_conversations(item))

    cat_b_val = []
    for line in open("data/training_sar/val.jsonl"):
        item = json.loads(line)
        item["category"] = "sar"
        cat_b_val.append(to_conversations(item))

    # 2x oversample SAR training data
    cat_b_train = cat_b_train_raw * 2
    random.shuffle(cat_b_train)

    print(f"    Raw: {len(cat_b_train_raw)}, Oversampled: {len(cat_b_train)}, Val: {len(cat_b_val)}")

    # ── Generate Category C: Fusion Reasoning ─────────────────────────
    print("  Generating Category C (Fusion reasoning)...")

    # Use cached patches that are in the Exp 2 training set
    cached = set(p.stem for p in Path("data/imagery_cache").glob("*.npz"))

    # Get land cover classification
    df = pd.read_parquet("data/bigearthnet_txt/BigEarthNet.txt.parquet")
    from evaluation.create_benchmark_manifest import classify_landcover

    binary_yes = df[
        (df["type"] == "binary") &
        (df["output"].str.strip().str.lower() == "yes")
    ]
    patch_labels = defaultdict(list)
    for _, row in binary_yes.iterrows():
        if row["patch_id"] in cached:
            patch_labels[row["patch_id"]].append(row["input"])

    # Generate fusion QA for diverse patches
    exp2_patches = set()
    for item in cat_a_train:
        exp2_patches.add(item["image"])

    # Select diverse patches for fusion QA
    by_lc = defaultdict(list)
    for pid in exp2_patches:
        if pid in patch_labels:
            lc = classify_landcover(patch_labels[pid])
            by_lc[lc].append(pid)

    cat_c_train = []
    cat_c_val = []
    target_fusion = 800  # ~800 fusion training samples

    patches_per_lc = target_fusion // max(len(by_lc), 1) // 5  # ~5 QA per patch
    for lc, pids in by_lc.items():
        selected = random.sample(pids, min(patches_per_lc, len(pids)))
        for pid in selected:
            pairs = generate_fusion_qa(pid, lc)
            cat_c_train.extend([to_conversations({**p, "category": "fusion"}) for p in pairs])

    # Split 85/15 for val
    random.shuffle(cat_c_train)
    val_size = max(20, int(len(cat_c_train) * 0.15))
    cat_c_val = cat_c_train[:val_size]
    cat_c_train = cat_c_train[val_size:]

    print(f"    Train: {len(cat_c_train)}, Val: {len(cat_c_val)}")

    # ── Combine with controlled proportions ───────────────────────────
    print("\n  Combining with controlled proportions...")

    # Target: 70% general, 20% SAR, 10% fusion
    total_target = len(cat_a_train) + len(cat_b_train) + len(cat_c_train)
    actual_a = len(cat_a_train)
    actual_b = len(cat_b_train)
    actual_c = len(cat_c_train)

    print(f"    Category A (General): {actual_a} ({actual_a/total_target*100:.1f}%)")
    print(f"    Category B (SAR 2x):  {actual_b} ({actual_b/total_target*100:.1f}%)")
    print(f"    Category C (Fusion):  {actual_c} ({actual_c/total_target*100:.1f}%)")
    print(f"    Total:                {total_target}")

    # Combine and shuffle
    train_combined = cat_a_train + cat_b_train + cat_c_train
    random.shuffle(train_combined)

    val_combined = cat_a_val + cat_b_val + cat_c_val
    random.shuffle(val_combined)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Combined
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w") as f:
        for item in train_combined:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(OUTPUT_DIR, "val.jsonl"), "w") as f:
        for item in val_combined:
            f.write(json.dumps(item) + "\n")

    # Separate validation sets
    with open(os.path.join(OUTPUT_DIR, "val_general.jsonl"), "w") as f:
        for item in cat_a_val:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(OUTPUT_DIR, "val_sar.jsonl"), "w") as f:
        for item in cat_b_val:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(OUTPUT_DIR, "val_fusion.jsonl"), "w") as f:
        for item in cat_c_val:
            f.write(json.dumps(item) + "\n")

    # Summary
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  train.jsonl:       {len(train_combined)} samples")
    print(f"  val.jsonl:         {len(val_combined)} samples")
    print(f"  val_general.jsonl: {len(cat_a_val)} samples")
    print(f"  val_sar.jsonl:     {len(cat_b_val)} samples")
    print(f"  val_fusion.jsonl:  {len(cat_c_val)} samples")

    # Category distribution in train
    cat_counts = Counter(item["category"] for item in train_combined)
    print(f"\n  Training category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"    {cat}: {count} ({count/len(train_combined)*100:.1f}%)")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
