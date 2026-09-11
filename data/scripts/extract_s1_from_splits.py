"""
Extract BigEarthNet-S1 patches from split .tar.gz parts.

Streams the split archive parts (aa, ab) sequentially through gzip
decompression + tar extraction, pulling only the S1 patches that
correspond to our 4,994 S2 patches via the subset_manifest.json mapping.

Each patch's VV/VH GeoTIFF bands are read with rasterio and merged
into the existing S2 .npz file as float16 (dB values).

Usage:
    python data/scripts/extract_s1_from_splits.py
"""

import argparse
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

S1_BANDS = ["VV", "VH"]


class MultiFileReader:
    """Read multiple files sequentially as a single stream."""

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


def main():
    parser = argparse.ArgumentParser(description="Extract S1 patches from split tar.gz")
    parser.add_argument(
        "--parts",
        nargs="+",
        default=[
            r"D:\BigEarthNet\V2\BigEarthNet-S1.tar.gzaa",
            r"D:\BigEarthNet\V2\BigEarthNet-S1.tar.gzab",
        ],
    )
    parser.add_argument("--manifest", default="data/training/subset_manifest.json")
    parser.add_argument("--cache-dir", default="data/imagery_cache")
    args = parser.parse_args()

    # Load manifest: build {s1_name -> s2_patch_id} reverse mapping
    with open(args.manifest) as f:
        entries = json.load(f)

    s1_to_s2 = {}
    for e in entries:
        s1_name = e.get("s1_name", "")
        if s1_name:
            s1_to_s2[s1_name] = e["patch_id"]

    logger.info(f"Manifest: {len(s1_to_s2)} S1 patches to extract")

    # Check parts exist
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

    # Check how many already have S1 data
    already_have_s1 = 0
    for s1_name, s2_pid in s1_to_s2.items():
        npz_path = cache / f"{s2_pid}.npz"
        if npz_path.exists():
            try:
                data = np.load(str(npz_path))
                if "s1" in data:
                    already_have_s1 += 1
            except Exception:
                pass

    if already_have_s1 > 0:
        logger.info(f"Already have S1: {already_have_s1}/{len(s1_to_s2)} patches")
    if already_have_s1 == len(s1_to_s2):
        logger.info("All patches already have S1 data! Nothing to do.")
        return

    import rasterio

    logger.info(f"\nStreaming through {total_archive_size / 1e9:.2f} GB S1 archive...")
    logger.info(f"Extracting {len(s1_to_s2) - already_have_s1} remaining S1 patches\n")

    reader = MultiFileReader(args.parts)
    patch_bands = defaultdict(dict)  # {s1_name: {band: array}}
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
                        f"Extracted {extracted}/{len(s1_to_s2)} S1 patches"
                    )

                    if extracted >= len(s1_to_s2):
                        logger.info("All S1 patches extracted! Stopping early.")
                        break

                if not member.isfile() or not member.name.endswith(".tif"):
                    continue

                # Parse s1_name from path: BigEarthNet-S1/<scene>/<s1_name>/<s1_name>_VV.tif
                parts = member.name.replace("\\", "/").split("/")
                if len(parts) < 4:
                    continue
                s1_name = parts[-2]  # directory name = s1_name

                if s1_name not in s1_to_s2:
                    continue

                # Check if already has S1
                s2_pid = s1_to_s2[s1_name]
                npz_path = cache / f"{s2_pid}.npz"

                # Parse band from filename suffix
                basename = os.path.basename(member.name).replace(".tif", "")
                band = basename.split("_")[-1]  # "VV" or "VH"
                if band not in S1_BANDS:
                    continue

                f_obj = tar.extractfile(member)
                if f_obj is None:
                    continue

                tif_bytes = f_obj.read()
                with rasterio.open(io.BytesIO(tif_bytes)) as src:
                    data = src.read(1).astype(np.float16)  # dB values as float16
                    patch_bands[s1_name][band] = data

                # Save when both VV and VH collected
                if len(patch_bands[s1_name]) == len(S1_BANDS):
                    s1_stack = np.stack(
                        [patch_bands[s1_name][b] for b in S1_BANDS], axis=0
                    )  # (2, H, W)

                    # Merge with existing S2 data
                    save_dict = {"s1": s1_stack}
                    if npz_path.exists():
                        try:
                            existing = np.load(str(npz_path))
                            if "s2" in existing:
                                save_dict["s2"] = existing["s2"]
                        except Exception:
                            pass

                    np.savez_compressed(str(npz_path), **save_dict)
                    extracted += 1
                    del patch_bands[s1_name]

                    if extracted % 100 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"  Extracted {extracted}/{len(s1_to_s2)} S1 patches | "
                            f"{elapsed/60:.1f} min elapsed"
                        )

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.close()

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"S1 EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Archive entries scanned: {total_members:,}")
    logger.info(f"S1 patches extracted: {extracted}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")

    # Verify: count patches with both S2 and S1
    has_s2 = 0
    has_s1 = 0
    has_both = 0
    missing_s1 = []
    for s1_name, s2_pid in s1_to_s2.items():
        npz_path = cache / f"{s2_pid}.npz"
        if npz_path.exists():
            try:
                data = np.load(str(npz_path))
                got_s2 = "s2" in data
                got_s1 = "s1" in data
                if got_s2:
                    has_s2 += 1
                if got_s1:
                    has_s1 += 1
                if got_s2 and got_s1:
                    has_both += 1
                else:
                    if not got_s1:
                        missing_s1.append(s1_name)
            except Exception:
                missing_s1.append(s1_name)
        else:
            missing_s1.append(s1_name)

    cache_size = sum(f.stat().st_size for f in cache.glob("*.npz")) / 1e9

    logger.info(f"\nVerification:")
    logger.info(f"  Total patches:     {len(s1_to_s2)}")
    logger.info(f"  With S2 data:      {has_s2}")
    logger.info(f"  With S1 data:      {has_s1}")
    logger.info(f"  Complete (S1+S2):  {has_both}")
    logger.info(f"  Missing S1:        {len(missing_s1)}")
    logger.info(f"  Cache size:        {cache_size:.2f} GB")

    if missing_s1:
        logger.info(f"  First 10 missing S1: {missing_s1[:10]}")

    report = {
        "total": len(s1_to_s2),
        "has_s2": has_s2,
        "has_s1": has_s1,
        "has_both": has_both,
        "missing_s1_count": len(missing_s1),
        "missing_s1": missing_s1,
        "cache_size_gb": round(cache_size, 2),
        "extraction_time_min": round(elapsed / 60, 1),
    }
    report_path = cache / "s1_extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
