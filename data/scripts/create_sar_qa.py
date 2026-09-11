"""
SatQuery AI -- Generate SAR-Aware QA Training Data.

Creates SAR-specific question-answer pairs by mapping BigEarthNet land-cover
labels to SAR-observable properties (backscatter, texture, moisture).

This teaches the model what SAR data reveals, fixing the current weakness
where SAR-only inference produces generic/empty responses.

SAR Physics Mapping:
  - Urban/Built-up     → High backscatter, strong double-bounce scattering
  - Water bodies       → Very low backscatter (specular reflection)
  - Forest/Vegetation  → Volume scattering, moderate-high backscatter
  - Agricultural       → Variable backscatter depending on crop stage
  - Bare soil          → Low-moderate backscatter, surface roughness dependent
  - Wetlands           → Mixed signal, high moisture indicators

Usage:
    python data/scripts/create_sar_qa.py
"""

import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PARQUET = "data/bigearthnet_txt/BigEarthNet.txt.parquet"
IMAGERY_CACHE = "data/imagery_cache"
OUTPUT_DIR = "data/training_sar"
SEED = 42

# ── SAR Physics Knowledge Base ───────────────────────────────────────────

# Maps BigEarthNet label keywords to SAR-observable properties
SAR_PROPERTIES = {
    "urban": {
        "backscatter": "high",
        "mechanism": "double-bounce scattering from buildings and vertical structures",
        "texture": "heterogeneous with bright spots from corner reflectors",
        "moisture": "low (impervious surfaces)",
        "keywords": ["urban", "built-up", "residential", "industrial", "commercial"],
    },
    "water": {
        "backscatter": "very low",
        "mechanism": "specular reflection away from sensor",
        "texture": "smooth and dark with uniform low returns",
        "moisture": "high (water surface)",
        "keywords": ["water", "river", "lake", "sea", "pond", "reservoir"],
    },
    "forest": {
        "backscatter": "moderate to high",
        "mechanism": "volume scattering from canopy structure and branches",
        "texture": "rough texture with moderate variability",
        "moisture": "moderate (canopy interception)",
        "keywords": ["forest", "woodland", "broad-leaved", "coniferous", "mixed forest"],
    },
    "crop": {
        "backscatter": "variable (depends on growth stage)",
        "mechanism": "surface scattering when bare, volume scattering when mature",
        "texture": "regular geometric patterns from field boundaries",
        "moisture": "variable (irrigation dependent)",
        "keywords": ["crop", "agricultural", "arable", "permanently irrigated", "rice"],
    },
    "grassland": {
        "backscatter": "low to moderate",
        "mechanism": "surface scattering from short vegetation",
        "texture": "relatively smooth and homogeneous",
        "moisture": "moderate",
        "keywords": ["grass", "pasture", "meadow", "natural grassland"],
    },
    "bare": {
        "backscatter": "low to moderate (roughness dependent)",
        "mechanism": "surface scattering controlled by soil roughness",
        "texture": "smooth to moderately rough",
        "moisture": "variable (exposed soil)",
        "keywords": ["bare", "rock", "sand", "sparse vegetation", "burnt"],
    },
    "wetland": {
        "backscatter": "mixed (high from emergent vegetation over water)",
        "mechanism": "double-bounce between water surface and vegetation stems",
        "texture": "heterogeneous with bright patches over flooded areas",
        "moisture": "very high (saturated soil or standing water)",
        "keywords": ["wetland", "marsh", "bog", "peat", "inland marsh"],
    },
    "shrub": {
        "backscatter": "moderate",
        "mechanism": "mixed surface and volume scattering",
        "texture": "moderately rough with scattered bright returns",
        "moisture": "low to moderate",
        "keywords": ["shrub", "moor", "heathland", "sclerophyllous", "transitional"],
    },
}


def classify_patch_sar(labels: list) -> list:
    """Map BigEarthNet labels to SAR property categories."""
    matched = []
    labels_lower = " ".join(labels).lower()
    for category, props in SAR_PROPERTIES.items():
        for kw in props["keywords"]:
            if kw in labels_lower:
                matched.append(category)
                break
    return matched if matched else ["unknown"]


# ── QA Template Generation ───────────────────────────────────────────────

def generate_sar_qa_pairs(patch_id: str, sar_categories: list) -> list:
    """Generate SAR-specific QA pairs for a patch."""
    pairs = []

    for cat in sar_categories:
        if cat == "unknown":
            continue
        props = SAR_PROPERTIES[cat]

        # Binary SAR questions
        pairs.append({
            "image": patch_id,
            "input": f"Does the SAR radar data show {props['backscatter']} backscatter levels in this area?",
            "output": "Yes",
            "type": "binary",
            "category": f"sar_{cat}",
        })

        pairs.append({
            "image": patch_id,
            "input": f"Based on the radar signal, is there evidence of {props['mechanism']}?",
            "output": "Yes",
            "type": "binary",
            "category": f"sar_{cat}",
        })

        # Negative binary (ask about wrong SAR property)
        wrong_cats = [c for c in SAR_PROPERTIES if c != cat and c != "unknown"]
        if wrong_cats:
            wrong = random.choice(wrong_cats)
            wrong_props = SAR_PROPERTIES[wrong]
            pairs.append({
                "image": patch_id,
                "input": f"Does the SAR data indicate {wrong_props['mechanism']}?",
                "output": "No",
                "type": "binary",
                "category": f"sar_{cat}_negative",
            })

        # Descriptive SAR questions
        pairs.append({
            "image": patch_id,
            "input": "What does the SAR radar backscatter pattern reveal about the surface in this image?",
            "output": (
                f"The SAR data shows {props['backscatter']} backscatter intensity, "
                f"consistent with {props['mechanism']}. "
                f"The radar texture appears {props['texture']}. "
                f"Moisture indicators suggest {props['moisture']} conditions."
            ),
            "type": "descriptive",
            "category": f"sar_{cat}",
        })

        pairs.append({
            "image": patch_id,
            "input": "Describe the surface characteristics visible in the radar data.",
            "output": (
                f"The radar imagery reveals {props['texture']}. "
                f"The backscatter level is {props['backscatter']}, "
                f"indicating {props['mechanism']}. "
                f"Surface moisture appears {props['moisture']}."
            ),
            "type": "descriptive",
            "category": f"sar_{cat}",
        })

        # Fusion-aware questions (teaching the model to reason about both)
        pairs.append({
            "image": patch_id,
            "input": (
                "How do the optical and SAR observations complement each other "
                "for this area?"
            ),
            "output": (
                f"The optical data provides spectral information about surface "
                f"materials and vegetation health through reflectance bands. "
                f"The SAR data complements this with {props['backscatter']} "
                f"backscatter revealing {props['mechanism']}. "
                f"Together, they give a more complete picture: optical shows "
                f"what the surface looks like, while SAR reveals its physical "
                f"structure and moisture state ({props['moisture']})."
            ),
            "type": "descriptive",
            "category": f"fusion_{cat}",
        })

    return pairs


def main():
    print("=" * 65)
    print("  SatQuery AI -- SAR QA Data Generation")
    print("=" * 65)

    random.seed(SEED)

    # Load parquet
    df = pd.read_parquet(PARQUET)
    df_train = df[df["split"] == "train"]

    # Get cached patches
    cached = set(p.stem for p in Path(IMAGERY_CACHE).glob("*.npz"))
    print(f"  Cached patches: {len(cached)}")

    # Get binary labels for each patch (to determine land cover)
    binary_df = df_train[df_train["type"] == "binary"]
    patch_labels = {}
    for _, row in binary_df.iterrows():
        pid = row["patch_id"]
        if pid in cached:
            if pid not in patch_labels:
                patch_labels[pid] = []
            if row["output"].strip().lower() == "yes":
                patch_labels[pid].append(row["input"])

    print(f"  Patches with labels: {len(patch_labels)}")

    # Generate SAR QA pairs
    all_pairs = []
    category_counts = {}

    for pid, labels in patch_labels.items():
        sar_cats = classify_patch_sar(labels)
        pairs = generate_sar_qa_pairs(pid, sar_cats)
        all_pairs.extend(pairs)

        for cat in sar_cats:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"  Total SAR QA pairs generated: {len(all_pairs)}")
    print(f"  Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count} patches")

    # Shuffle and split
    random.shuffle(all_pairs)

    # Limit to ~2000 pairs (balanced across categories)
    if len(all_pairs) > 2000:
        # Sample proportionally from each category
        by_cat = {}
        for p in all_pairs:
            c = p["category"]
            if c not in by_cat:
                by_cat[c] = []
            by_cat[c].append(p)

        target_per_cat = 2000 // len(by_cat)
        sampled = []
        for cat, items in by_cat.items():
            sampled.extend(random.sample(items, min(target_per_cat, len(items))))
        random.shuffle(sampled)
        all_pairs = sampled[:2000]

    # Split 80/20
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")

    with open(os.path.join(OUTPUT_DIR, "val.jsonl"), "w") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n  Train: {len(train_pairs)} pairs -> {OUTPUT_DIR}/train.jsonl")
    print(f"  Val:   {len(val_pairs)} pairs -> {OUTPUT_DIR}/val.jsonl")

    # Show examples
    print("\n  Example SAR QA pairs:")
    for pair in train_pairs[:4]:
        print(f"    [{pair['type']}] Q: {pair['input'][:80]}...")
        print(f"            A: {pair['output'][:80]}...")
        print()

    print("=" * 65)


if __name__ == "__main__":
    main()
