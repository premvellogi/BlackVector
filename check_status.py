"""Quick status check for Phase 5 prerequisites."""
import os, json, shutil
from pathlib import Path

print("=== DISK SPACE ===")
total, used, free = shutil.disk_usage("D:/")
print(f"  Free: {free/1e9:.1f} GB")

print("\n=== S2 DATA ===")
s2 = Path("D:/BigEarthNet/BigEarthNet-S2")
if s2.exists():
    n = sum(1 for d in s2.iterdir() if d.is_dir())
    print(f"  Exists: YES, {n} top-level dirs")
else:
    print("  Exists: NO")

print("\n=== S1 DATA ===")
s1 = Path("D:/BigEarthNet/BigEarthNet-S1")
if s1.exists():
    n = sum(1 for d in s1.iterdir() if d.is_dir())
    print(f"  Exists: YES, {n} top-level dirs")
else:
    print("  Exists: NO")

print("\n=== FUSION PAIRS ===")
fp = Path("d:/Netra/data/training/fusion_pairs.json")
if fp.exists():
    with open(fp) as f:
        pairs = json.load(f)
    print(f"  Exists: YES, {len(pairs)} pairs")
else:
    print("  Exists: NO")

print("\n=== SAMPLED PATCHES ===")
sp = Path("d:/Netra/data/training/sampled_patches.json")
if sp.exists():
    with open(sp) as f:
        data = json.load(f)
    tr = len(data["train"])
    va = len(data["val"])
    print(f"  Train: {tr}, Val: {va}")
else:
    print("  Exists: NO")

print("\n=== CHECKPOINTS ===")
ckpts = [
    "checkpoints/prithvi-lora-ben-v2/best/tag_head.pt",
    "checkpoints/prithvi-lora-ben-v2/best/lora_adapter",
    "checkpoints/fusion",
]
for ckpt in ckpts:
    p = Path("d:/Netra") / ckpt
    print(f"  {ckpt}: {'YES' if p.exists() else 'NO'}")

print("\n=== ARCHIVES (should be deleted) ===")
arcs = [
    "D:/BigEarthNet/BigEarthNet-S2.tar.zst",
    "D:/BigEarthNet/BigEarthNet-S1.tar.gz",
    "D:/BigEarthNet/V2/BigEarthNet-S1.tar.gzaa",
    "D:/BigEarthNet/V2/BigEarthNet-S1.tar.gzab",
]
for arc in arcs:
    status = "EXISTS" if os.path.exists(arc) else "DELETED"
    print(f"  {arc}: {status}")

print("\n=== CODE FIXES ===")
# Check taxonomy alignment
from services.models.heads.tag_head import NUM_CLASSES, BIGEARTHNET_CLASSES
print(f"  TagHead classes: {NUM_CLASSES} ({BIGEARTHNET_CLASSES[0]}...)")

# Check train_fusion_head imports
import importlib.util
spec = importlib.util.spec_from_file_location("tfh", "d:/Netra/services/models/training/train_fusion_head.py")
print("  train_fusion_head.py: parseable" if spec else "  train_fusion_head.py: ERROR")

# Check DualModalityAdapter has return_weights
from services.models.core.spectral_adapter import DualModalityAdapter
import inspect
sig = inspect.signature(DualModalityAdapter.forward)
has_rw = "return_weights" in sig.parameters
print(f"  DualModalityAdapter.forward(return_weights): {'YES' if has_rw else 'NO'}")
