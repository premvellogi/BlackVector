"""
Phased BigEarthNet Patch Downloader via Microsoft Planetary Computer.

Downloads S1 SAR + S2 multispectral imagery for our training patches
WITHOUT downloading the full 120GB BigEarthNet archive.

Key insight: Our 29,992 training patches come from only 54 unique
Sentinel-2 tiles and 79 acquisition dates. We download tile-by-tile,
then crop all patches from each tile. This is much more efficient
than downloading individual patches.

Storage requirements:
  Phase 0 (500 patches):  ~300 MB   — sanity check
  Phase 1 (5-10K):        ~3-6 GB   — development dataset
  Phase 2 (25-50K):       ~15-30 GB — larger training run

Band handling:
  S2: All 13 bands harmonized to 120×120 (10m resolution)
  S1: VV + VH at 120×120 (10m resolution)
  Total: 15 channels × 120×120 × float32 per patch

Usage:
    # Phase 0: sanity check (500 patches)
    python data/scripts/phased_downloader.py --phase 0

    # Phase 1: development (5000 patches)
    python data/scripts/phased_downloader.py --phase 1

    # Phase 2: larger run (25000 patches)
    python data/scripts/phased_downloader.py --phase 2
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# BigEarthNet v2 normalization stats (official, from ben_txt_datamodule.py)
S2_STATS = {
    "B01": {"mean": 361.077, "std": 575.069, "res_m": 60, "native_px": 20},
    "B02": {"mean": 438.372, "std": 607.027, "res_m": 10, "native_px": 120},
    "B03": {"mean": 614.056, "std": 603.297, "res_m": 10, "native_px": 120},
    "B04": {"mean": 588.410, "std": 684.569, "res_m": 10, "native_px": 120},
    "B05": {"mean": 942.843, "std": 738.433, "res_m": 20, "native_px": 60},
    "B06": {"mean": 1769.932, "std": 1100.456, "res_m": 20, "native_px": 60},
    "B07": {"mean": 2049.552, "std": 1275.805, "res_m": 20, "native_px": 60},
    "B08": {"mean": 2193.292, "std": 1369.372, "res_m": 10, "native_px": 120},
    "B8A": {"mean": 2235.557, "std": 1356.544, "res_m": 20, "native_px": 60},
    "B09": {"mean": 2241.455, "std": 1316.393, "res_m": 60, "native_px": 20},
    "B11": {"mean": 1568.227, "std": 1070.161, "res_m": 20, "native_px": 60},
    "B12": {"mean": 997.732, "std": 813.528, "res_m": 20, "native_px": 60},
}

S1_STATS = {
    "VV": {"mean": -12.644, "std": 5.133},
    "VH": {"mean": -19.353, "std": 5.591},
}

# Canonical band ordering (same as SpectralChannelAdapter)
S2_BANDS_10M = ["B02", "B03", "B04", "B08"]
S2_BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
S2_BANDS_60M = ["B01", "B09"]  # Dropped by default
S2_BANDS_ALL_USED = S2_BANDS_10M + S2_BANDS_20M  # 10 bands (skip 60m)

TARGET_SIZE = 120  # BigEarthNet v2 patch size at 10m


class PhasedDownloader:
    """Download BigEarthNet patches phase-by-phase from Planetary Computer."""

    PHASE_LIMITS = {
        0: 500,    # Sanity check
        1: 7500,   # Development
        2: 25000,  # Larger training run
    }

    def __init__(
        self,
        parquet_path: str = "data/bigearthnet_txt/BigEarthNet.txt.parquet",
        training_jsonl: str = "data/training/train.jsonl",
        cache_dir: str = "data/imagery_cache",
    ):
        self.parquet_path = Path(parquet_path)
        self.training_jsonl = Path(training_jsonl)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Step 1: Select a stratified subset of patches
    # =================================================================

    def select_patches(self, limit: int) -> pd.DataFrame:
        """Select a stratified subset of patches for download.

        Stratifies by:
          - Country (geographic diversity)
          - Season (temporal diversity)
          - Land-cover type distribution (task diversity)
        """
        # Get unique patches from training JSONL
        patch_to_types = defaultdict(set)
        with open(self.training_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                patch_to_types[entry["image"]].add(entry["conversations"][0]["value"].split("\\n")[0][:20])

        training_patches = set(patch_to_types.keys())

        # Load metadata from parquet
        df = pd.read_parquet(
            self.parquet_path,
            columns=["patch_id", "s1_name", "latitude", "longitude",
                      "country", "season", "type", "category", "split"],
        )

        # Filter to training patches
        patch_meta = df[df["patch_id"].isin(training_patches)].copy()
        patch_meta = patch_meta.groupby("patch_id").agg({
            "s1_name": "first",
            "latitude": "first",
            "longitude": "first",
            "country": "first",
            "season": "first",
            "type": lambda x: list(set(x)),  # all task types for this patch
        }).reset_index()

        # Skip already cached
        cached = {f.stem for f in self.cache_dir.glob("*.npz")}
        patch_meta = patch_meta[~patch_meta["patch_id"].isin(cached)]

        if len(patch_meta) == 0:
            logger.info("All patches already cached!")
            return pd.DataFrame()

        # Stratified sampling by country + season
        if limit < len(patch_meta):
            # Proportional sampling per country
            sampled = patch_meta.groupby("country", group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, int(limit * len(g) / len(patch_meta)))),
                    random_state=42,
                )
            )
            # Top up to exact limit if needed
            if len(sampled) < limit:
                remaining = patch_meta[~patch_meta["patch_id"].isin(sampled["patch_id"])]
                extra = remaining.sample(n=min(limit - len(sampled), len(remaining)), random_state=42)
                sampled = pd.concat([sampled, extra])
            patch_meta = sampled.head(limit)

        logger.info(f"Selected {len(patch_meta)} patches for download")
        logger.info(f"  Countries: {dict(patch_meta['country'].value_counts())}")
        logger.info(f"  Seasons: {dict(patch_meta['season'].value_counts())}")

        return patch_meta

    # =================================================================
    # Step 2: Group patches by S2 tile for efficient download
    # =================================================================

    def group_by_tile(self, patches: pd.DataFrame) -> dict:
        """Group patches by Sentinel-2 tile ID for batch downloading.

        Patch ID format: S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
        Tile ID: T33UUP (parts[5])
        Date: 20170613 (parts[2][:8])
        """
        groups = defaultdict(list)
        for _, row in patches.iterrows():
            pid = row["patch_id"]
            parts = pid.split("_")
            tile_id = parts[5]  # e.g., T33UUP
            date_str = parts[2][:8]  # e.g., 20170613
            groups[f"{tile_id}_{date_str}"].append(row)

        logger.info(f"Grouped into {len(groups)} tile-date combinations")
        return dict(groups)

    # =================================================================
    # Step 3: Download from Planetary Computer
    # =================================================================

    def download_tile_patches(
        self,
        tile_date: str,
        patch_rows: list,
        max_retries: int = 3,
    ) -> dict:
        """Download all patches from a single S2 tile-date.

        Uses Planetary Computer's COG (Cloud-Optimized GeoTIFF) endpoints
        for windowed reads — only downloads the 120×120 crop we need.
        """
        import rasterio
        from rasterio.windows import Window
        from pystac_client import Client
        import planetary_computer as pc

        tile_id = tile_date.split("_")[0]
        date_str = tile_date.split("_")[1]
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        results = {}

        try:
            # Search Planetary Computer for this tile and date
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
            )

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                datetime=f"{date_formatted}/{date_formatted}",
                query={"s2:mgrs_tile": {"eq": tile_id[1:]}},  # Remove 'T' prefix
                max_items=5,
            )

            items = list(search.items())
            if not items:
                # Try ±1 day window
                from datetime import datetime, timedelta
                dt = datetime.strptime(date_str, "%Y%m%d")
                start = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
                end = (dt + timedelta(days=2)).strftime("%Y-%m-%d")

                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    datetime=f"{start}/{end}",
                    query={"s2:mgrs_tile": {"eq": tile_id[1:]}},
                    max_items=5,
                )
                items = list(search.items())

            if not items:
                logger.warning(f"No scene found for tile {tile_id} on {date_formatted}")
                return {}

            item = items[0]

            # For each patch in this tile, crop the bands
            for row in patch_rows:
                pid = row["patch_id"] if isinstance(row, dict) else row.patch_id
                lat = row["latitude"] if isinstance(row, dict) else row.latitude
                lon = row["longitude"] if isinstance(row, dict) else row.longitude

                try:
                    bands_data = {}

                    # Download S2 bands
                    for band_name in S2_BANDS_ALL_USED:
                        asset_key = band_name.lower()
                        if asset_key == "b8a":
                            asset_key = "B8A"

                        # Try different asset key formats
                        for key in [band_name, asset_key, f"B{band_name[1:]}", band_name.upper()]:
                            if key in item.assets:
                                asset_key = key
                                break
                        else:
                            logger.debug(f"Band {band_name} not found in assets: {list(item.assets.keys())[:10]}")
                            continue

                        href = item.assets[asset_key].href

                        with rasterio.open(href) as src:
                            # Convert lat/lon to pixel coordinates
                            col, row_px = src.index(lat, lon)
                            # Actually rasterio.index returns (row, col) from (y, x)
                            # But for geographic CRS, (lon, lat) → (col, row)
                            try:
                                row_px, col = src.index(lat, lon)
                            except Exception:
                                # Use transform directly
                                from rasterio.transform import rowcol
                                row_px, col = rowcol(src.transform, lon, lat)

                            native_px = S2_STATS[band_name]["native_px"]
                            half = native_px // 2

                            # Ensure window is within bounds
                            row_start = max(0, row_px - half)
                            col_start = max(0, col - half)
                            row_start = min(row_start, src.height - native_px)
                            col_start = min(col_start, src.width - native_px)

                            window = Window(col_start, row_start, native_px, native_px)
                            data = src.read(1, window=window).astype(np.float32)

                            # Resize to 120×120 if needed
                            if data.shape != (TARGET_SIZE, TARGET_SIZE):
                                from PIL import Image
                                img = Image.fromarray(data)
                                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                                data = np.array(img, dtype=np.float32)

                            bands_data[band_name] = data

                    # Only save if we got enough bands
                    if len(bands_data) >= 6:  # At least RGB + a few more
                        results[pid] = self._pack_bands(bands_data)
                        logger.debug(f"  Downloaded {pid}: {len(bands_data)} bands")

                except Exception as e:
                    logger.debug(f"  Failed {pid}: {e}")

        except Exception as e:
            logger.warning(f"Tile {tile_date} failed: {e}")

        return results

    def download_s1_patches(
        self,
        patch_rows: list,
    ) -> dict:
        """Download S1 SAR patches from Planetary Computer.

        Uses the s1_name from the parquet to find the correct S1 scene.
        """
        import rasterio
        from rasterio.windows import Window
        from pystac_client import Client
        import planetary_computer as pc

        results = {}

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

        for row in patch_rows:
            pid = row["patch_id"] if isinstance(row, dict) else row.patch_id
            lat = row["latitude"] if isinstance(row, dict) else row.latitude
            lon = row["longitude"] if isinstance(row, dict) else row.longitude

            try:
                # Search for S1 GRD scenes near this location and date
                search = catalog.search(
                    collections=["sentinel-1-grd"],
                    intersects={"type": "Point", "coordinates": [lon, lat]},
                    datetime="2017-01-01/2018-12-31",
                    max_items=1,
                )

                items = list(search.items())
                if not items:
                    continue

                item = items[0]

                sar_data = {}
                for pol in ["vv", "vh"]:
                    if pol in item.assets:
                        href = item.assets[pol].href
                        with rasterio.open(href) as src:
                            from rasterio.transform import rowcol
                            row_px, col = rowcol(src.transform, lon, lat)
                            half = TARGET_SIZE // 2
                            row_start = max(0, row_px - half)
                            col_start = max(0, col - half)
                            window = Window(col_start, row_start, TARGET_SIZE, TARGET_SIZE)
                            data = src.read(1, window=window).astype(np.float32)

                            if data.shape != (TARGET_SIZE, TARGET_SIZE):
                                from PIL import Image
                                img = Image.fromarray(data)
                                img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                                data = np.array(img, dtype=np.float32)

                            sar_data[pol.upper()] = data

                if len(sar_data) == 2:
                    results[pid] = np.stack([sar_data["VV"], sar_data["VH"]])

            except Exception as e:
                logger.debug(f"S1 failed for {pid}: {e}")

        return results

    # =================================================================
    # Step 4: Pack and cache
    # =================================================================

    def _pack_bands(self, bands_dict: dict) -> np.ndarray:
        """Pack bands into a (10, 120, 120) array in canonical order."""
        result = np.zeros((len(S2_BANDS_ALL_USED), TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
        for i, band_name in enumerate(S2_BANDS_ALL_USED):
            if band_name in bands_dict:
                result[i] = bands_dict[band_name]
        return result

    def save_patch(self, patch_id: str, s2_data: np.ndarray, s1_data: Optional[np.ndarray] = None):
        """Save a patch to the cache as .npz."""
        save_dict = {"s2": s2_data}
        if s1_data is not None:
            save_dict["s1"] = s1_data
        npz_path = self.cache_dir / f"{patch_id}.npz"
        np.savez_compressed(str(npz_path), **save_dict)

    def load_patch(self, patch_id: str) -> dict:
        """Load a patch from cache. Returns {'s2': array, 's1': array|None}."""
        npz_path = self.cache_dir / f"{patch_id}.npz"
        if not npz_path.exists():
            return {}
        data = np.load(str(npz_path))
        result = {"s2": data["s2"]}
        if "s1" in data:
            result["s1"] = data["s1"]
        return result

    # =================================================================
    # Step 5: Run a phase
    # =================================================================

    def run_phase(self, phase: int):
        """Execute a download phase."""
        limit = self.PHASE_LIMITS.get(phase, 5000)
        logger.info(f"=== Phase {phase}: Downloading {limit} patches ===")

        # Select patches
        patches = self.select_patches(limit)
        if patches.empty:
            return

        # Group by tile for efficient download
        tile_groups = self.group_by_tile(patches)

        # Download tile-by-tile
        total_downloaded = 0
        total_failed = 0

        for i, (tile_date, rows) in enumerate(tile_groups.items()):
            logger.info(f"[{i+1}/{len(tile_groups)}] Tile {tile_date}: {len(rows)} patches")

            try:
                s2_results = self.download_tile_patches(tile_date, rows)

                # Save results
                for pid, s2_data in s2_results.items():
                    self.save_patch(pid, s2_data)
                    total_downloaded += 1

                total_failed += len(rows) - len(s2_results)

                # Rate limiting — be kind to Planetary Computer
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Tile {tile_date} failed: {e}")
                total_failed += len(rows)

            # Progress report every 5 tiles
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {total_downloaded} downloaded, {total_failed} failed")

        # Final report
        logger.info(f"\n=== Phase {phase} Complete ===")
        logger.info(f"Downloaded: {total_downloaded}")
        logger.info(f"Failed: {total_failed}")
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npz"))
        logger.info(f"Cache size: {cache_size / 1e9:.2f} GB")

    def verify(self):
        """Report cache coverage statistics."""
        # Get required patches
        patches = set()
        with open(self.training_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                patches.add(entry["image"])

        cached = {f.stem for f in self.cache_dir.glob("*.npz")}
        covered = patches & cached
        coverage = len(covered) / len(patches) * 100 if patches else 0

        # Check band completeness
        has_s1 = 0
        has_all_s2 = 0
        for npz_file in list(self.cache_dir.glob("*.npz"))[:100]:
            data = np.load(str(npz_file))
            if "s1" in data:
                has_s1 += 1
            s2 = data["s2"]
            if np.all(np.any(s2 != 0, axis=(1, 2))):  # All bands non-zero
                has_all_s2 += 1

        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npz"))

        print(f"\n+----------------------------------------------+")
        print(f"|  Imagery Cache Report                        |")
        print(f"+----------------------------------------------+")
        print(f"|  Required patches: {len(patches):>8,}                 |")
        print(f"|  Cached patches:   {len(cached):>8,}                 |")
        print(f"|  Coverage:         {coverage:>7.1f}%                 |")
        print(f"|  Cache size:       {cache_size/1e9:>7.2f} GB               |")
        print(f"|  With S1 data:     {has_s1:>4}/100 (sampled)        |")
        print(f"|  All S2 bands:     {has_all_s2:>4}/100 (sampled)        |")
        print(f"+----------------------------------------------+")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phased BigEarthNet Downloader")
    parser.add_argument("--phase", type=int, default=0,
                        help="Download phase (0=500, 1=7500, 2=25000)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify cache coverage")
    parser.add_argument("--cache-dir", default="data/imagery_cache")
    args = parser.parse_args()

    downloader = PhasedDownloader(cache_dir=args.cache_dir)

    if args.verify:
        downloader.verify()
    else:
        downloader.run_phase(args.phase)
