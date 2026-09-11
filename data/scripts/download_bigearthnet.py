"""
BigEarthNet v2.0 Selective Archive Downloader.

Downloads official BigEarthNet archives from Zenodo and extracts ONLY the
patches listed in subset_manifest.json. Archives are processed SEQUENTIALLY
(S2 first, then S1) to minimize peak disk usage.

Workflow per archive:
  1. Stream-download the .tar.zst file
  2. Decompress zstd on-the-fly via streaming
  3. Walk the tar entries, extracting only matching patch directories
  4. Convert extracted GeoTIFF bands into compact .npz files
  5. Delete the archive after verification

Per SatQuery_Data_Engineering_Instructions.md:
  - S2 bands stored as uint16 (native)
  - S1 bands stored as float16 (from float32 dB values)
  - Final .npz files go to data/imagery_cache/

Usage:
    python data/scripts/download_bigearthnet.py
    python data/scripts/download_bigearthnet.py --s2-only
    python data/scripts/download_bigearthnet.py --s1-only
    python data/scripts/download_bigearthnet.py --skip-download  # extract from existing archive
"""

import argparse
import io
import json
import logging
import os
import struct
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ZENODO_RECORD = "10891137"
S2_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/BigEarthNet-S2.tar.zst"
S1_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/BigEarthNet-S1.tar.zst"

# BigEarthNet v2.0 S2 bands (10m: B02,B03,B04,B08; 20m: B05,B06,B07,B8A,B11,B12)
# We use 10 bands (skip 60m: B01, B09)
S2_BANDS_WANTED = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# S1 polarizations
S1_BANDS_WANTED = ["VV", "VH"]


# =============================================================================
# Helpers
# =============================================================================

def load_manifest(manifest_path: str) -> dict:
    """Load subset_manifest.json and return {patch_id: s1_name} mapping."""
    with open(manifest_path) as f:
        entries = json.load(f)
    mapping = {}
    for entry in entries:
        mapping[entry["patch_id"]] = entry.get("s1_name", "")
    logger.info(f"Manifest loaded: {len(mapping)} patches to extract")
    return mapping


def parse_s2_patch_id_from_path(tar_path: str) -> str:
    """Extract BigEarthNet S2 patch_id from a tar member path.

    Archive structure:
      BigEarthNet-S2/<tile_dir>/<patch_id>/B02.tif
      BigEarthNet-S2/<tile_dir>/<patch_id>/B03.tif
      ...

    The patch_id is the full directory name like:
      S2B_MSIL2A_20180326T112109_N9999_R037_T29SNB_03_36
    """
    parts = tar_path.replace("\\", "/").split("/")
    # Typically: BigEarthNet-S2 / <tile> / <patch_id> / <band>.tif
    # or:       BigEarthNet-S2 / <patch_id> / <band>.tif
    for i, part in enumerate(parts):
        if part.startswith("S2") and "_MSIL2A_" in part and part.count("_") >= 6:
            return part
    return ""


def parse_s1_patch_id_from_path(tar_path: str) -> str:
    """Extract BigEarthNet S1 patch name from a tar member path."""
    parts = tar_path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("S1") and part.count("_") >= 4:
            return part
    return ""


def download_with_resume(url: str, dest_path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Download a file with resume support and progress reporting."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest.stat().st_size if dest.exists() else 0

    # Get total size
    head = requests.head(url, allow_redirects=True, timeout=30)
    total_size = int(head.headers.get("content-length", 0))

    if existing_size >= total_size and total_size > 0:
        logger.info(f"Already downloaded: {dest_path} ({total_size / 1e9:.2f} GB)")
        return str(dest)

    logger.info(
        f"Downloading: {url}\n"
        f"  Size: {total_size / 1e9:.2f} GB\n"
        f"  Resume from: {existing_size / 1e9:.2f} GB\n"
        f"  Destination: {dest_path}"
    )

    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    mode = "ab" if existing_size > 0 else "wb"
    downloaded = existing_size
    start_time = time.time()
    last_report = start_time

    with open(dest, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_report >= 30:  # Report every 30s
                elapsed = now - start_time
                speed = (downloaded - existing_size) / elapsed / 1e6
                pct = downloaded / total_size * 100 if total_size > 0 else 0
                eta = (total_size - downloaded) / (speed * 1e6) if speed > 0 else 0
                logger.info(
                    f"  Progress: {pct:.1f}% ({downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB) "
                    f"| {speed:.1f} MB/s | ETA: {eta / 60:.0f} min"
                )
                last_report = now

    elapsed = time.time() - start_time
    speed = (downloaded - existing_size) / elapsed / 1e6 if elapsed > 0 else 0
    logger.info(f"Download complete: {downloaded / 1e9:.2f} GB in {elapsed / 60:.1f} min ({speed:.1f} MB/s)")

    return str(dest)


def extract_s2_patches(
    archive_path: str,
    wanted_patches: set,
    cache_dir: str,
    bands: list = S2_BANDS_WANTED,
) -> dict:
    """Stream through S2 tar.zst archive and extract only wanted patches.

    Reads the archive as a zstd-compressed tar stream, collecting band TIFFs
    for each wanted patch, then saves as compressed .npz.

    Returns dict of {patch_id: npz_path} for successfully extracted patches.
    """
    import rasterio

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    extracted = {}
    patch_bands = defaultdict(dict)  # {patch_id: {band_name: numpy_array}}
    total_members = 0
    matched_members = 0

    logger.info(f"Streaming S2 archive: {archive_path}")
    logger.info(f"Looking for {len(wanted_patches)} patches, {len(bands)} bands each")

    dctx = zstd.ZstdDecompressor()

    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    total_members += 1

                    if total_members % 100_000 == 0:
                        logger.info(
                            f"  Scanned {total_members:,} entries, "
                            f"matched {matched_members:,}, "
                            f"completed {len(extracted):,} patches"
                        )

                    if not member.isfile() or not member.name.endswith(".tif"):
                        continue

                    patch_id = parse_s2_patch_id_from_path(member.name)
                    if not patch_id or patch_id not in wanted_patches:
                        continue

                    # Check if this is a band we want
                    basename = os.path.basename(member.name)
                    band_name = basename.replace(".tif", "")
                    if band_name not in bands:
                        continue

                    matched_members += 1

                    # Extract the TIF to memory and read with rasterio
                    f_obj = tar.extractfile(member)
                    if f_obj is None:
                        continue

                    tif_bytes = f_obj.read()
                    with rasterio.open(io.BytesIO(tif_bytes)) as src:
                        data = src.read(1)  # Single band GeoTIFF -> (H, W)
                        patch_bands[patch_id][band_name] = data.astype(np.uint16)

                    # If we have all bands for this patch, save the .npz
                    if len(patch_bands[patch_id]) == len(bands):
                        npz_path = cache / f"{patch_id}.npz"

                        # Stack bands in canonical order
                        s2_stack = np.stack(
                            [patch_bands[patch_id][b] for b in bands],
                            axis=0,
                        )  # (10, H, W)

                        # Check if there's already S1 data
                        if npz_path.exists():
                            existing = np.load(str(npz_path))
                            if "s1" in existing:
                                np.savez_compressed(
                                    str(npz_path),
                                    s2=s2_stack,
                                    s1=existing["s1"],
                                )
                            else:
                                np.savez_compressed(str(npz_path), s2=s2_stack)
                        else:
                            np.savez_compressed(str(npz_path), s2=s2_stack)

                        extracted[patch_id] = str(npz_path)
                        del patch_bands[patch_id]  # Free memory

                        if len(extracted) % 500 == 0:
                            logger.info(f"  Extracted {len(extracted):,}/{len(wanted_patches):,} S2 patches")

    logger.info(
        f"S2 extraction complete: {len(extracted):,}/{len(wanted_patches):,} patches "
        f"from {total_members:,} archive entries"
    )

    return extracted


def extract_s1_patches(
    archive_path: str,
    wanted_s1_names: dict,  # {s1_name: patch_id}
    cache_dir: str,
) -> dict:
    """Stream through S1 tar.zst archive and extract only wanted patches.

    Similar to extract_s2_patches but for Sentinel-1 VV/VH bands.
    S1 values are dB (float32), stored as float16 for compactness.
    """
    import rasterio

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    extracted = {}
    patch_bands = defaultdict(dict)
    total_members = 0
    matched_members = 0

    logger.info(f"Streaming S1 archive: {archive_path}")
    logger.info(f"Looking for {len(wanted_s1_names)} S1 patches")

    dctx = zstd.ZstdDecompressor()

    with open(archive_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    total_members += 1

                    if total_members % 100_000 == 0:
                        logger.info(
                            f"  Scanned {total_members:,} entries, "
                            f"matched {matched_members:,}, "
                            f"completed {len(extracted):,} patches"
                        )

                    if not member.isfile() or not member.name.endswith(".tif"):
                        continue

                    s1_name = parse_s1_patch_id_from_path(member.name)
                    if not s1_name or s1_name not in wanted_s1_names:
                        continue

                    basename = os.path.basename(member.name)
                    band_name = basename.replace(".tif", "")  # "VV" or "VH"
                    if band_name not in S1_BANDS_WANTED:
                        continue

                    matched_members += 1

                    f_obj = tar.extractfile(member)
                    if f_obj is None:
                        continue

                    tif_bytes = f_obj.read()
                    with rasterio.open(io.BytesIO(tif_bytes)) as src:
                        data = src.read(1).astype(np.float16)  # dB values
                        patch_bands[s1_name][band_name] = data

                    if len(patch_bands[s1_name]) == len(S1_BANDS_WANTED):
                        patch_id = wanted_s1_names[s1_name]
                        npz_path = cache / f"{patch_id}.npz"

                        s1_stack = np.stack(
                            [patch_bands[s1_name][b] for b in S1_BANDS_WANTED],
                            axis=0,
                        )  # (2, H, W)

                        # Merge with existing S2 data if present
                        if npz_path.exists():
                            existing = np.load(str(npz_path))
                            save_dict = {"s1": s1_stack}
                            if "s2" in existing:
                                save_dict["s2"] = existing["s2"]
                            np.savez_compressed(str(npz_path), **save_dict)
                        else:
                            np.savez_compressed(str(npz_path), s1=s1_stack)

                        extracted[s1_name] = str(npz_path)
                        del patch_bands[s1_name]

                        if len(extracted) % 500 == 0:
                            logger.info(f"  Extracted {len(extracted):,}/{len(wanted_s1_names):,} S1 patches")

    logger.info(
        f"S1 extraction complete: {len(extracted):,}/{len(wanted_s1_names):,} patches "
        f"from {total_members:,} archive entries"
    )

    return extracted


def verify_cache(cache_dir: str, manifest: dict) -> dict:
    """Verify extracted patches against manifest."""
    cache = Path(cache_dir)
    results = {
        "total_expected": len(manifest),
        "found": 0,
        "missing": 0,
        "has_s2": 0,
        "has_s1": 0,
        "has_both": 0,
        "corrupt": 0,
        "missing_patches": [],
    }

    for patch_id, s1_name in manifest.items():
        npz_path = cache / f"{patch_id}.npz"
        if not npz_path.exists():
            results["missing"] += 1
            results["missing_patches"].append(patch_id)
            continue

        results["found"] += 1
        try:
            data = np.load(str(npz_path))
            has_s2 = "s2" in data and data["s2"].shape[0] == 10
            has_s1 = "s1" in data and data["s1"].shape[0] == 2

            if has_s2:
                results["has_s2"] += 1
            if has_s1:
                results["has_s1"] += 1
            if has_s2 and has_s1:
                results["has_both"] += 1
        except Exception as e:
            results["corrupt"] += 1
            logger.warning(f"Corrupt: {npz_path}: {e}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BigEarthNet v2.0 Selective Downloader")
    parser.add_argument("--manifest", default="data/training/subset_manifest.json")
    parser.add_argument("--cache-dir", default="data/imagery_cache")
    parser.add_argument("--archive-dir", default="data/archives")
    parser.add_argument("--s2-only", action="store_true", help="Download and extract S2 only")
    parser.add_argument("--s1-only", action="store_true", help="Download and extract S1 only")
    parser.add_argument("--skip-download", action="store_true", help="Use existing archive files")
    parser.add_argument("--keep-archives", action="store_true", help="Don't delete archives after extraction")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing cache")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    if args.verify_only:
        results = verify_cache(args.cache_dir, manifest)
        logger.info(f"\nVerification Results:")
        for k, v in results.items():
            if k != "missing_patches":
                logger.info(f"  {k}: {v}")
        if results["missing_patches"]:
            logger.info(f"  First 10 missing: {results['missing_patches'][:10]}")
        return

    do_s2 = not args.s1_only
    do_s1 = not args.s2_only

    # --- Phase 1: S2 ---
    if do_s2:
        logger.info("=" * 60)
        logger.info("PHASE 1: Sentinel-2 Archive")
        logger.info("=" * 60)

        if args.skip_download:
            s2_archive = os.path.join(args.archive_dir, "BigEarthNet-S2.tar.zst")
        else:
            s2_archive = download_with_resume(
                S2_URL,
                os.path.join(args.archive_dir, "BigEarthNet-S2.tar.zst"),
            )

        wanted_s2 = set(manifest.keys())
        s2_result = extract_s2_patches(s2_archive, wanted_s2, args.cache_dir)

        # Verify S2
        results = verify_cache(args.cache_dir, manifest)
        logger.info(f"After S2: {results['has_s2']}/{len(manifest)} patches have S2 data")

        # Delete S2 archive
        if not args.keep_archives and os.path.exists(s2_archive):
            s2_size = os.path.getsize(s2_archive) / 1e9
            logger.info(f"Deleting S2 archive ({s2_size:.2f} GB) after successful extraction...")
            os.remove(s2_archive)
            logger.info("S2 archive deleted.")

    # --- Phase 2: S1 ---
    if do_s1:
        logger.info("=" * 60)
        logger.info("PHASE 2: Sentinel-1 Archive")
        logger.info("=" * 60)

        # Build reverse mapping: s1_name -> patch_id
        wanted_s1 = {}
        for patch_id, s1_name in manifest.items():
            if s1_name:
                wanted_s1[s1_name] = patch_id

        if not wanted_s1:
            logger.warning("No S1 names in manifest -- skipping S1 download")
        else:
            if args.skip_download:
                s1_archive = os.path.join(args.archive_dir, "BigEarthNet-S1.tar.zst")
            else:
                s1_archive = download_with_resume(
                    S1_URL,
                    os.path.join(args.archive_dir, "BigEarthNet-S1.tar.zst"),
                )

            s1_result = extract_s1_patches(s1_archive, wanted_s1, args.cache_dir)

            # Verify S1
            results = verify_cache(args.cache_dir, manifest)
            logger.info(f"After S1: {results['has_s1']}/{len(manifest)} patches have S1 data")
            logger.info(f"Complete pairs (S1+S2): {results['has_both']}/{len(manifest)}")

            # Delete S1 archive
            if not args.keep_archives and os.path.exists(s1_archive):
                s1_size = os.path.getsize(s1_archive) / 1e9
                logger.info(f"Deleting S1 archive ({s1_size:.2f} GB) after successful extraction...")
                os.remove(s1_archive)
                logger.info("S1 archive deleted.")

    # Final verification
    logger.info("=" * 60)
    logger.info("FINAL VERIFICATION")
    logger.info("=" * 60)
    results = verify_cache(args.cache_dir, manifest)
    cache_size = sum(
        f.stat().st_size for f in Path(args.cache_dir).glob("*.npz")
    ) / 1e9

    logger.info(f"Expected patches:  {results['total_expected']}")
    logger.info(f"Found patches:     {results['found']}")
    logger.info(f"Missing patches:   {results['missing']}")
    logger.info(f"With S2 data:      {results['has_s2']}")
    logger.info(f"With S1 data:      {results['has_s1']}")
    logger.info(f"Complete S1+S2:    {results['has_both']}")
    logger.info(f"Corrupt files:     {results['corrupt']}")
    logger.info(f"Cache size:        {cache_size:.2f} GB")

    # Save verification report
    report_path = Path(args.cache_dir) / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
