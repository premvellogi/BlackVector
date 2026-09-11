"""
LEVIR-CD Automatic Downloader and Extractor.

Downloads official LEVIR-CD train, val, and test archives from Hugging Face
(satellite-image-deep-learning/LEVIR-CD) and extracts them into data/LEVIR-CD/.

Directory Structure created:
  data/LEVIR-CD/
    train/
      A/
      B/
      label/
    val/
      A/, B/, label/
    test/
      A/, B/, label/

Usage:
    python data/scripts/download_levir_cd.py
    python data/scripts/download_levir_cd.py --target-dir data/LEVIR-CD
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

HF_REPO = "satellite-image-deep-learning/LEVIR-CD"
ARCHIVES = ["train.zip", "val.zip", "test.zip"]


def main():
    parser = argparse.ArgumentParser(description="LEVIR-CD Dataset Downloader")
    parser.add_argument(
        "--target-dir",
        default="data/LEVIR-CD",
        help="Target directory to extract LEVIR-CD into",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LEVIR-CD Dataset Downloader")
    logger.info(f"Target Directory: {target_dir.resolve()}")
    logger.info("=" * 60)

    for archive_name in ARCHIVES:
        logger.info(f"\n[1/2] Downloading {archive_name} from {HF_REPO}...")
        zip_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=archive_name,
            repo_type="dataset",
        )
        logger.info(f"Downloaded to cache: {zip_path}")

        logger.info(f"[2/2] Extracting {archive_name} to {target_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info(f"Successfully extracted {archive_name}")

    # Verify extracted structure
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION REPORT")
    logger.info("=" * 60)

    for split in ["train", "val", "test"]:
        split_dir = target_dir / split
        t1_count = len(list((split_dir / "A").glob("*"))) if (split_dir / "A").exists() else 0
        t2_count = len(list((split_dir / "B").glob("*"))) if (split_dir / "B").exists() else 0
        lbl_count = len(list((split_dir / "label").glob("*"))) if (split_dir / "label").exists() else 0
        logger.info(f"  {split.upper()}: A={t1_count} images, B={t2_count} images, label={lbl_count} masks")

    logger.info("\nLEVIR-CD download and extraction complete!")


if __name__ == "__main__":
    main()
