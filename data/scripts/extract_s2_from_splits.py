"""
Extract BigEarthNet-S2 patches from split .tar.gz parts.

Streams the split archive parts (aa, ab) sequentially through gzip
decompression + tar extraction, pulling only the 4,994 patches listed
in subset_manifest.json.

Each patch's 10 GeoTIFF bands are read with rasterio and saved as a
compressed .npz file (uint16) to data/imagery_cache/.

Usage:
    python data/scripts/extract_s2_from_splits.py
    python data/scripts/extract_s2_from_splits.py --parts D:\\BigEarthNet\\V2\\BigEarthNet-S2.tar.gzaa D:\\BigEarthNet\\V2\\BigEarthNet-S2.tar.gzab
"""

import argparse
import gzip
import io
import json
import logging
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 10 S2 bands we use (skip 60m: B01, B09)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


class MultiFileReader:
    """Read multiple files sequentially as a single stream.

    This lets us feed split archive parts (aa, ab, ...) into tarfile
    without concatenating them to disk first.
    """

    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self._file_idx = 0
        self._current_fh = None
        self._total_read = 0
        self._open_next()

    def _open_next(self):
        if self._current_fh is not None:
            self._current_fh.close()
        if self._file_idx < len(self.file_paths):
            path = self.file_paths[self._file_idx]
            logger.info(f"Opening part {self._file_idx + 1}/{len(self.file_paths)}: {path}")
            self._current_fh = open(path, "rb")
            self._file_idx += 1
        else:
            self._current_fh = None

    def read(self, size=-1):
        if self._current_fh is None:
            return b""

        chunks = []
        remaining = size

        while remaining != 0:
            if remaining > 0:
                data = self._current_fh.read(remaining)
            else:
                data = self._current_fh.read()

            if data:
                chunks.append(data)
                self._total_read += len(data)
                if remaining > 0:
                    remaining -= len(data)
            else:
                # End of current file, move to next
                self._open_next()
                if self._current_fh is None:
                    break

        return b"".join(chunks)

    def close(self):
        if self._current_fh:
            self._current_fh.close()
            self._current_fh = None

    @property
    def bytes_read(self):
        return self._total_read


def parse_s2_patch_id(tar_path: str) -> str:
    """Extract patch_id from a tar member path like:
    BigEarthNet-S2/<tile>/<patch_id>/<patch_id>_B02.tif

    The patch_id is the directory containing the .tif file.
    """
    parts = tar_path.replace("\\", "/").split("/")
    # For a .tif file, patch_id is the parent directory (second-to-last component)
    if len(parts) >= 4 and parts[-1].endswith(".tif"):
        return parts[-2]  # The directory name IS the patch_id
    return ""


def main():
    parser = argparse.ArgumentParser(description="Extract S2 patches from split tar.gz")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=[
            r"D:\BigEarthNet\V2\BigEarthNet-S2.tar.gzaa",
            r"D:\BigEarthNet\V2\BigEarthNet-S2.tar.gzab",
        ],
        help="Paths to split archive parts in order",
    )
    parser.add_argument("--manifest", default="data/training/subset_manifest.json")
    parser.add_argument("--cache-dir", default="data/imagery_cache")
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        entries = json.load(f)
    wanted = {e["patch_id"] for e in entries}
    logger.info(f"Manifest: {len(wanted)} patches to extract")

    # Check all parts exist
    total_archive_size = 0
    for p in args.parts:
        if not os.path.exists(p):
            logger.error(f"Part not found: {p}")
            sys.exit(1)
        size = os.path.getsize(p)
        total_archive_size += size
        logger.info(f"  Part: {p} ({size / 1e9:.2f} GB)")
    logger.info(f"  Total archive: {total_archive_size / 1e9:.2f} GB")

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Check how many are already cached
    already_cached = sum(1 for pid in wanted if (cache / f"{pid}.npz").exists())
    if already_cached > 0:
        logger.info(f"Already cached: {already_cached}/{len(wanted)} patches")
    if already_cached == len(wanted):
        logger.info("All patches already cached! Nothing to do.")
        return

    import rasterio

    # Stream through the split archive
    logger.info(f"\nStreaming through {total_archive_size / 1e9:.2f} GB archive...")
    logger.info(f"Extracting {len(wanted) - already_cached} remaining patches\n")

    reader = MultiFileReader(args.parts)
    patch_bands = defaultdict(dict)
    extracted = 0
    total_members = 0
    start_time = time.time()

    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                total_members += 1

                if total_members % 50_000 == 0:
                    elapsed = time.time() - start_time
                    gb_read = reader.bytes_read / 1e9
                    speed = gb_read / elapsed if elapsed > 0 else 0
                    pct = reader.bytes_read / total_archive_size * 100
                    logger.info(
                        f"  Scanned {total_members:,} entries | "
                        f"{gb_read:.1f}/{total_archive_size/1e9:.1f} GB ({pct:.1f}%) | "
                        f"{speed:.2f} GB/s | "
                        f"Extracted {extracted}/{len(wanted)} patches"
                    )

                    # Early exit if we found all patches
                    if extracted >= len(wanted):
                        logger.info("All patches extracted! Stopping early.")
                        break

                if not member.isfile() or not member.name.endswith(".tif"):
                    continue

                patch_id = parse_s2_patch_id(member.name)
                if not patch_id or patch_id not in wanted:
                    continue

                # Skip if already cached
                if (cache / f"{patch_id}.npz").exists():
                    continue

                # Check if this is a band we want
                # Filename is like: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B02.tif
                # Band name is the suffix after the last underscore
                basename = os.path.basename(member.name).replace(".tif", "")
                band_name = basename.split("_")[-1]  # e.g. "B02", "B8A"
                if band_name not in S2_BANDS:
                    continue

                # Read the GeoTIFF into memory
                f_obj = tar.extractfile(member)
                if f_obj is None:
                    continue

                tif_bytes = f_obj.read()
                with rasterio.open(io.BytesIO(tif_bytes)) as src:
                    data = src.read(1).astype(np.uint16)
                    patch_bands[patch_id][band_name] = data

                # Save when all 10 bands collected
                if len(patch_bands[patch_id]) == len(S2_BANDS):
                    # BigEarthNet bands have different resolutions:
                    #   10m bands (B02,B03,B04,B08): 120x120
                    #   20m bands (B05,B06,B07,B8A,B11,B12): 60x60
                    # Resample all to 120x120
                    target_size = 120
                    resampled = []
                    for b in S2_BANDS:
                        arr = patch_bands[patch_id][b]
                        if arr.shape[0] != target_size or arr.shape[1] != target_size:
                            # Simple nearest-neighbor resize
                            from PIL import Image
                            img = Image.fromarray(arr)
                            img = img.resize((target_size, target_size), Image.NEAREST)
                            arr = np.array(img, dtype=np.uint16)
                        resampled.append(arr)

                    s2_stack = np.stack(resampled, axis=0)  # (10, 120, 120)

                    npz_path = cache / f"{patch_id}.npz"
                    np.savez_compressed(str(npz_path), s2=s2_stack)

                    extracted += 1
                    del patch_bands[patch_id]

                    if extracted % 100 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"  Extracted {extracted}/{len(wanted)} patches | "
                            f"{elapsed/60:.1f} min elapsed"
                        )

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.close()

    elapsed = time.time() - start_time

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total archive entries scanned: {total_members:,}")
    logger.info(f"Patches extracted this run: {extracted}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")

    # Verify
    found = 0
    missing = []
    for pid in wanted:
        npz_path = cache / f"{pid}.npz"
        if npz_path.exists():
            found += 1
        else:
            missing.append(pid)

    cache_size = sum(f.stat().st_size for f in cache.glob("*.npz")) / 1e9

    logger.info(f"\nVerification:")
    logger.info(f"  Expected:  {len(wanted)}")
    logger.info(f"  Found:     {found}")
    logger.info(f"  Missing:   {len(missing)}")
    logger.info(f"  Cache size: {cache_size:.2f} GB")

    if missing:
        logger.info(f"  First 10 missing: {missing[:10]}")

    # Save report
    report = {
        "total_expected": len(wanted),
        "found": found,
        "missing_count": len(missing),
        "missing_patches": missing,
        "cache_size_gb": round(cache_size, 2),
        "extraction_time_min": round(elapsed / 60, 1),
    }
    report_path = cache / "s2_extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
