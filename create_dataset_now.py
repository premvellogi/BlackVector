import json
import os
from pathlib import Path
import sys

sys.path.insert(0, r"d:\Netra")

from services.models.training.dataset.sample_bigearthnet import (
    scan_patch_labels,
    compute_class_statistics,
    stratified_sample,
    split_train_val,
)

def main():
    data_root = Path(r"D:\BigEarthNet\BigEarthNet-S2")
    output_path = Path(r"d:\Netra\data\sampled_patches.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning BigEarthNet-S2 at {data_root}...")
    patch_labels, patch_rel_paths = scan_patch_labels(data_root)
    class_stats = compute_class_statistics(patch_labels)
    
    print("Performing 20K multi-label stratified sampling...")
    selected = stratified_sample(patch_labels, class_stats, total_samples=20000, seed=42)
    train_set, val_set = split_train_val(selected, val_ratio=0.2, seed=42)
    
    output = {
        "total_patches": len(selected),
        "train_patches": len(train_set),
        "val_patches": len(val_set),
        "seed": 42,
        "data_root": str(data_root),
        "train": train_set,
        "val": val_set,
        "class_stats": class_stats,
        "patch_labels": {p: patch_labels[p] for p in selected},
        "patch_paths": {p: patch_rel_paths[p] for p in selected},
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print("SUCCESSFULLY CREATED DATASET FILE:")
    print("  Path:", output_path.resolve())
    print("  Exists:", output_path.exists())
    print("  Size:", output_path.stat().st_size)

if __name__ == "__main__":
    main()
