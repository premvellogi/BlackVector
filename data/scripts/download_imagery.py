"""
BigEarthNet v2 Imagery Download and LMDB Conversion Helper.

The BigEarthNet.txt annotations reference image patches by patch_id
(e.g., S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57).

The actual satellite imagery must be downloaded separately from:
    https://bigearth.net/

This script provides:
1. Download instructions
2. LMDB conversion commands (using rico-hdl)
3. Verification of the download

Full pipeline:
1. Download S1 + S2 images from bigearth.net (~66GB + ~55GB compressed)
2. Convert to LMDB format using rico-hdl
3. Verify against our training JSONL

NOTE: For initial development without the full imagery, the training
script uses placeholder images. You can train on text-only first,
then upgrade to image+text once imagery is available.
"""

import json
import logging
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def print_download_instructions():
    """Print step-by-step download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  BigEarthNet v2 Imagery Download Instructions               ║
╚══════════════════════════════════════════════════════════════╝

STEP 1: Download from https://bigearth.net/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Sentinel-2 (BigEarthNet-S2-v2.0): ~55 GB compressed, ~90 GB uncompressed
  - Sentinel-1 (BigEarthNet-S1-v2.0): ~66 GB compressed, ~110 GB uncompressed
  
  Download to:
    d:\\Netra\\data\\BigEarthNet-S2-v2.0\\
    d:\\Netra\\data\\BigEarthNet-S1-v2.0\\

STEP 2: Install rico-hdl for LMDB conversion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install rico-hdl
  
  # Or from source:
  git clone https://github.com/rsim-tu-berlin/rico-hdl
  cd rico-hdl && pip install .

STEP 3: Convert to LMDB (required by BigEarthNet.txt's BENImageReader)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  rico-hdl bigearthnet \\
    --bigearthnet-s1-dir data/BigEarthNet-S1-v2.0 \\
    --bigearthnet-s2-dir data/BigEarthNet-S2-v2.0 \\
    --target-dir data/Encoded-BigEarthNet

  This creates an LMDB database at data/Encoded-BigEarthNet/

STEP 4: Verify
━━━━━━━━━━━━━━
  python data/scripts/download_imagery.py --verify

ALTERNATIVE (for development without full imagery):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The training script uses placeholder images when imagery
  is not available. You can start training on text-only mode
  immediately:
  
    python services/models/training/train_lora.py
  
  The model will still learn the instruction format and remote
  sensing vocabulary, even without actual satellite images.
  You can add imagery later and resume training.
""")


def verify_lmdb(lmdb_dir: str = "data/Encoded-BigEarthNet",
                training_jsonl: str = "data/training/train.jsonl"):
    """Verify LMDB contains the patches referenced in training data."""
    lmdb_path = Path(lmdb_dir)
    jsonl_path = Path(training_jsonl)

    if not lmdb_path.exists():
        logger.error(f"LMDB directory not found: {lmdb_path}")
        logger.info("Run the download + conversion steps first.")
        return False

    # Get patch IDs from training data
    if not jsonl_path.exists():
        logger.error(f"Training JSONL not found: {jsonl_path}")
        return False

    required_patches = set()
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            required_patches.add(entry["image"])

    logger.info(f"Training data requires {len(required_patches)} unique patches")

    # Check LMDB
    try:
        import lmdb
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            found = 0
            missing = []
            for patch_id in required_patches:
                if txn.get(patch_id.encode()) is not None:
                    found += 1
                else:
                    missing.append(patch_id)

        coverage = found / len(required_patches) * 100
        logger.info(f"Found: {found}/{len(required_patches)} patches ({coverage:.1f}%)")

        if missing:
            logger.warning(f"Missing {len(missing)} patches. First 5: {missing[:5]}")

        return len(missing) == 0

    except ImportError:
        logger.error("lmdb not installed. Run: pip install lmdb")
        return False


def check_disk_space(required_gb: float = 200.0):
    """Check if there's enough disk space for imagery."""
    import shutil
    usage = shutil.disk_usage("d:\\")
    free_gb = usage.free / 1e9
    logger.info(f"Disk space: {free_gb:.1f} GB free")
    if free_gb < required_gb:
        logger.warning(
            f"Need ~{required_gb:.0f} GB but only {free_gb:.1f} GB available. "
            f"Consider using a subset or external drive."
        )
        return False
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BigEarthNet v2 Imagery Helper")
    parser.add_argument("--verify", action="store_true", help="Verify LMDB coverage")
    parser.add_argument("--check-space", action="store_true", help="Check disk space")
    args = parser.parse_args()

    if args.verify:
        verify_lmdb()
    elif args.check_space:
        check_disk_space()
    else:
        print_download_instructions()
