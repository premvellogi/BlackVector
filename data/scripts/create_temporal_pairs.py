"""
SatQuery AI -- Build Real Temporal Change Detection Dataset.

Finds co-located patches captured at different dates in BigEarthNet,
compares their land-cover labels to determine ground truth change,
and creates a structured dataset for training the ChangeHead.

Key findings from data analysis:
  - 134,717 geographic locations have multiple observations
  - 86 temporal pairs are already in our imagery cache
  - ALL 86 show label changes (seasonal/land-use change)

We supplement with "no change" pairs from same-patch self-comparisons
to create a balanced dataset.

Usage:
    python data/scripts/create_temporal_pairs.py
"""

import json
import os
import re
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PARQUET = "data/bigearthnet_txt/BigEarthNet.txt.parquet"
IMAGERY_CACHE = "data/imagery_cache"
OUTPUT_DIR = "data/temporal_pairs"
SEED = 42


def parse_patch_metadata(patch_id: str) -> dict:
    """Extract date, tile, and pixel coordinates from BigEarthNet patch ID.

    Format: S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_86_13
                       YYYYMMDD                    TILE   X  Y
    """
    parts = patch_id.split("_")
    metadata = {"patch_id": patch_id}

    # Date
    date_match = re.search(r"(\d{8})T\d{6}", patch_id)
    if date_match:
        metadata["date_str"] = date_match.group(1)
        metadata["date"] = datetime.strptime(date_match.group(1), "%Y%m%d")
    else:
        metadata["date_str"] = "unknown"
        metadata["date"] = None

    # Tile and pixel
    if len(parts) >= 8:
        metadata["tile"] = parts[5]         # e.g. T29UPU
        metadata["pixel_x"] = parts[6]      # e.g. 86
        metadata["pixel_y"] = parts[7]      # e.g. 13
        metadata["location_key"] = f"{parts[5]}_{parts[6]}_{parts[7]}"
    else:
        metadata["tile"] = "unknown"
        metadata["pixel_x"] = "0"
        metadata["pixel_y"] = "0"
        metadata["location_key"] = patch_id

    return metadata


def get_patch_labels(df: pd.DataFrame, patch_id: str) -> set:
    """Get the set of 'yes' binary labels for a patch."""
    subset = df[(df["patch_id"] == patch_id) & (df["type"] == "binary")]
    yes_labels = set()
    for _, row in subset.iterrows():
        if row["output"].strip().lower() == "yes":
            yes_labels.add(row["input"].strip())
    return yes_labels


def compute_label_change(labels_t1: set, labels_t2: set) -> dict:
    """Compare labels between two time steps."""
    added = labels_t2 - labels_t1
    removed = labels_t1 - labels_t2
    common = labels_t1 & labels_t2

    changed = len(added) + len(removed) > 0
    jaccard = len(common) / max(len(labels_t1 | labels_t2), 1)

    # Classify change type
    if not changed:
        change_type = "no_change"
    elif len(added) > 0 and len(removed) > 0:
        change_type = "land_use_transition"
    elif len(removed) > 0:
        change_type = "land_cover_loss"
    else:
        change_type = "land_cover_gain"

    return {
        "changed": changed,
        "change_type": change_type,
        "labels_added": sorted(added),
        "labels_removed": sorted(removed),
        "labels_common": sorted(common),
        "jaccard_similarity": round(jaccard, 4),
        "n_changes": len(added) + len(removed),
    }


def main():
    print("=" * 65)
    print("  SatQuery AI -- Temporal Change Detection Dataset")
    print("=" * 65)

    random.seed(SEED)

    # Load data
    df = pd.read_parquet(PARQUET)
    cached = set(p.stem for p in Path(IMAGERY_CACHE).glob("*.npz"))
    print(f"  Cached patches: {len(cached)}")

    # Parse all cached patches
    patch_meta = {}
    for pid in cached:
        meta = parse_patch_metadata(pid)
        patch_meta[pid] = meta

    # Group by geographic location
    locations = {}
    for pid, meta in patch_meta.items():
        key = meta["location_key"]
        if key not in locations:
            locations[key] = []
        locations[key].append(meta)

    # Find temporal pairs (same location, different date)
    changed_pairs = []
    unchanged_pairs = []

    for key, patches in locations.items():
        if len(patches) < 2:
            continue

        # Sort by date
        patches.sort(key=lambda x: x["date_str"])

        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                p1 = patches[i]
                p2 = patches[j]

                if p1["date_str"] == p2["date_str"]:
                    continue  # Same date, skip

                # Get labels
                labels_t1 = get_patch_labels(df, p1["patch_id"])
                labels_t2 = get_patch_labels(df, p2["patch_id"])

                # Compute change
                change = compute_label_change(labels_t1, labels_t2)

                # Compute temporal gap
                if p1["date"] and p2["date"]:
                    gap_days = abs((p2["date"] - p1["date"]).days)
                else:
                    gap_days = 0

                pair = {
                    "patch_t1": p1["patch_id"],
                    "patch_t2": p2["patch_id"],
                    "date_t1": p1["date_str"],
                    "date_t2": p2["date_str"],
                    "temporal_gap_days": gap_days,
                    "tile": p1["tile"],
                    "pixel_x": p1["pixel_x"],
                    "pixel_y": p1["pixel_y"],
                    "location_key": key,
                    "change_label": 1 if change["changed"] else 0,
                    "change_type": change["change_type"],
                    "labels_added": change["labels_added"],
                    "labels_removed": change["labels_removed"],
                    "jaccard_similarity": change["jaccard_similarity"],
                    "n_changes": change["n_changes"],
                }

                if change["changed"]:
                    changed_pairs.append(pair)
                else:
                    unchanged_pairs.append(pair)

    print(f"  Real changed pairs: {len(changed_pairs)}")
    print(f"  Real unchanged pairs: {len(unchanged_pairs)}")

    # Supplement no-change with self-comparison pairs
    # (same patch compared to itself = guaranteed no change)
    all_cached_list = sorted(cached)
    random.shuffle(all_cached_list)
    n_self_needed = max(0, len(changed_pairs) - len(unchanged_pairs))

    for pid in all_cached_list[:n_self_needed]:
        meta = patch_meta[pid]
        unchanged_pairs.append({
            "patch_t1": pid,
            "patch_t2": pid,  # Self-comparison
            "date_t1": meta["date_str"],
            "date_t2": meta["date_str"],
            "temporal_gap_days": 0,
            "tile": meta["tile"],
            "pixel_x": meta["pixel_x"],
            "pixel_y": meta["pixel_y"],
            "location_key": meta["location_key"],
            "change_label": 0,
            "change_type": "no_change_self",
            "labels_added": [],
            "labels_removed": [],
            "jaccard_similarity": 1.0,
            "n_changes": 0,
        })

    print(f"  After balancing: {len(changed_pairs)} changed, {len(unchanged_pairs)} unchanged")

    # Combine and shuffle
    all_pairs = changed_pairs + unchanged_pairs
    random.shuffle(all_pairs)

    # Split: 70% train, 15% val, 15% test
    n = len(all_pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train = all_pairs[:n_train]
    val = all_pairs[n_train:n_train + n_val]
    test = all_pairs[n_train + n_val:]

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)
        n_changed = sum(1 for p in split_data if p["change_label"] == 1)
        n_unchanged = len(split_data) - n_changed
        print(f"  {split_name}: {len(split_data)} pairs "
              f"({n_changed} changed, {n_unchanged} unchanged) -> {path}")

    # Statistics
    if changed_pairs:
        gaps = [p["temporal_gap_days"] for p in changed_pairs]
        print(f"\n  Temporal gap stats (changed pairs):")
        print(f"    Min: {min(gaps)} days")
        print(f"    Max: {max(gaps)} days")
        print(f"    Mean: {sum(gaps)/len(gaps):.0f} days")

        # Change type distribution
        types = {}
        for p in changed_pairs:
            t = p["change_type"]
            types[t] = types.get(t, 0) + 1
        print(f"\n  Change type distribution:")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c}")

    # Show examples
    print("\n  Example temporal pairs:")
    for pair in all_pairs[:3]:
        status = "CHANGED" if pair["change_label"] == 1 else "UNCHANGED"
        print(f"    [{status}] {pair['date_t1']} -> {pair['date_t2']} "
              f"(gap: {pair['temporal_gap_days']}d)")
        print(f"      T1: {pair['patch_t1'][-30:]}")
        print(f"      T2: {pair['patch_t2'][-30:]}")
        if pair["labels_added"]:
            print(f"      Added: {pair['labels_added'][:3]}")
        if pair["labels_removed"]:
            print(f"      Removed: {pair['labels_removed'][:3]}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
