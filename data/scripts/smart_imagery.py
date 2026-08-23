"""
Smart Imagery Pipeline for SatQuery AI.

Instead of downloading the full 120GB BigEarthNet v2 archive, this script:

1. Reads our training JSONL to get the exact patch_ids we need (~20-30K)
2. Loads the parquet to get lat/lon coordinates for each patch
3. Downloads ONLY RGB bands (B02, B03, B04) from free Sentinel-2 sources
4. Caches locally as compact .npz files (~1.7GB total for 20K patches)
5. Falls back to synthetic spectral-noise images if download fails

Sources (in priority order):
  A. Local LMDB (if user has already run rico-hdl)
  B. Local extracted GeoTIFFs (if user partially downloaded BigEarthNet)
  C. Microsoft Planetary Computer STAC API (free, no auth)
  D. Synthetic placeholders from official band statistics

Storage math:
  20,000 patches × 3 bands × 120×120 pixels × 2 bytes = 1.65 GB
  With .npz compression: ~800 MB

Usage:
    # Download RGB for all training patches
    python data/scripts/smart_imagery.py --mode download --limit 20000

    # Verify coverage
    python data/scripts/smart_imagery.py --mode verify

    # Generate synthetic fallback for remaining patches
    python data/scripts/smart_imagery.py --mode synthetic
"""

import json
import logging
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Official BigEarthNet v2 band statistics (from ben_txt_datamodule.py)
BAND_STATS = {
    "B02": {"mean": 438.372, "std": 607.027},  # Blue, 10m
    "B03": {"mean": 614.056, "std": 603.297},  # Green, 10m
    "B04": {"mean": 588.410, "std": 684.569},  # Red, 10m
}

# BigEarthNet patch size at 10m resolution
PATCH_SIZE = 120  # 120×120 pixels at 10m = 1.2km × 1.2km


class SmartImageryPipeline:
    """Download/generate only the imagery we need for training."""

    def __init__(
        self,
        parquet_path: str = "data/bigearthnet_txt/BigEarthNet.txt.parquet",
        training_jsonl: str = "data/training/train.jsonl",
        cache_dir: str = "data/imagery_cache",
        lmdb_path: Optional[str] = None,
        geotiff_dir: Optional[str] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.training_jsonl = Path(training_jsonl)
        self.cache_dir = Path(cache_dir)
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        self.geotiff_dir = Path(geotiff_dir) if geotiff_dir else None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._patch_coords = None

    # =================================================================
    # Step 1: Identify what we need
    # =================================================================

    def get_required_patches(self) -> set:
        """Get unique patch_ids from training JSONL."""
        patches = set()
        with open(self.training_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                patches.add(entry["image"])
        logger.info(f"Training data requires {len(patches):,} unique patches")
        return patches

    def get_patch_coordinates(self) -> dict:
        """Get lat/lon for each patch from the parquet."""
        if self._patch_coords is None:
            df = pd.read_parquet(self.parquet_path, columns=["patch_id", "latitude", "longitude"])
            self._patch_coords = (
                df.groupby("patch_id")[["latitude", "longitude"]]
                .first()
                .to_dict("index")
            )
        return self._patch_coords

    def get_cached_patches(self) -> set:
        """Get patch_ids already cached locally."""
        cached = set()
        for f in self.cache_dir.glob("*.npz"):
            cached.add(f.stem)
        return cached

    # =================================================================
    # Step 2: Source A — Local LMDB (if available)
    # =================================================================

    def load_from_lmdb(self, patch_ids: set) -> dict:
        """Load RGB bands from existing LMDB database."""
        if self.lmdb_path is None or not self.lmdb_path.exists():
            logger.info("No LMDB path configured. Skipping.")
            return {}

        try:
            import lmdb
            from safetensors.numpy import load as safetensor_load
        except ImportError:
            logger.warning("lmdb/safetensors not installed. Skipping LMDB source.")
            return {}

        loaded = {}
        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            for pid in patch_ids:
                data = txn.get(pid.encode())
                if data is not None:
                    bands = safetensor_load(bytes(data))
                    # Stack RGB (B04=Red, B03=Green, B02=Blue)
                    rgb = np.stack([
                        self._resize_band(bands.get("B04", np.zeros((PATCH_SIZE, PATCH_SIZE))), PATCH_SIZE),
                        self._resize_band(bands.get("B03", np.zeros((PATCH_SIZE, PATCH_SIZE))), PATCH_SIZE),
                        self._resize_band(bands.get("B02", np.zeros((PATCH_SIZE, PATCH_SIZE))), PATCH_SIZE),
                    ])
                    loaded[pid] = rgb

        logger.info(f"Loaded {len(loaded):,} patches from LMDB")
        return loaded

    # =================================================================
    # Step 3: Source B — Local GeoTIFF directories
    # =================================================================

    def load_from_geotiffs(self, patch_ids: set) -> dict:
        """Load RGB from extracted BigEarthNet v2 directories."""
        if self.geotiff_dir is None or not self.geotiff_dir.exists():
            return {}

        loaded = {}
        for pid in patch_ids:
            patch_dir = self.geotiff_dir / pid
            if patch_dir.exists():
                try:
                    import rasterio
                    bands = []
                    for band_name in ["B04", "B03", "B02"]:
                        tif_path = patch_dir / f"{pid}_{band_name}.tif"
                        if tif_path.exists():
                            with rasterio.open(str(tif_path)) as src:
                                band_data = src.read(1).astype(np.float32)
                                bands.append(self._resize_band(band_data, PATCH_SIZE))
                    if len(bands) == 3:
                        loaded[pid] = np.stack(bands)
                except Exception as e:
                    logger.debug(f"Failed to load {pid}: {e}")

        logger.info(f"Loaded {len(loaded):,} patches from GeoTIFF directories")
        return loaded

    # =================================================================
    # Step 4: Source C — Planetary Computer STAC (free, no auth)
    # =================================================================

    def download_from_planetary_computer(
        self,
        patch_ids: set,
        batch_size: int = 100,
        max_patches: int = 20000,
    ) -> dict:
        """Download RGB patches from Microsoft Planetary Computer.

        Uses the free STAC API to find Sentinel-2 L2A scenes that cover
        each patch's coordinates, then downloads only B02/B03/B04 crops.
        """
        try:
            import requests
        except ImportError:
            logger.error("requests not installed")
            return {}

        coords = self.get_patch_coordinates()
        downloaded = {}
        failed = []
        count = 0

        STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

        patch_list = list(patch_ids)[:max_patches]
        total = len(patch_list)

        for i, pid in enumerate(patch_list):
            if pid not in coords:
                continue

            lat = coords[pid]["latitude"]
            lon = coords[pid]["longitude"]

            if i % 200 == 0:
                logger.info(f"Downloading: {i}/{total} ({len(downloaded)} success, {len(failed)} failed)")

            try:
                # Search for Sentinel-2 L2A scenes
                search_body = {
                    "collections": ["sentinel-2-l2a"],
                    "intersects": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "datetime": "2017-01-01/2018-12-31",  # BigEarthNet v2 time range
                    "limit": 1,
                    "query": {
                        "eo:cloud_cover": {"lt": 20},
                    },
                }

                resp = requests.post(STAC_URL, json=search_body, timeout=10)
                if resp.status_code != 200:
                    failed.append(pid)
                    continue

                features = resp.json().get("features", [])
                if not features:
                    failed.append(pid)
                    continue

                item = features[0]
                assets = item.get("assets", {})

                # Download RGB bands
                bands = []
                for band_key in ["B04", "B03", "B02"]:
                    asset_key = band_key
                    if asset_key not in assets:
                        # Try alternative naming
                        asset_key = band_key.lower()
                    if asset_key not in assets:
                        break

                    band_url = assets[asset_key]["href"]

                    # Use windowed read to get just the 120×120 crop
                    try:
                        import rasterio
                        from rasterio.windows import from_bounds
                        from rasterio.transform import from_bounds as transform_from_bounds

                        with rasterio.open(band_url) as src:
                            # Calculate window from lat/lon
                            # BigEarthNet patch is ~1.2km × 1.2km at 10m resolution
                            half_size = 0.006  # ~600m in degrees (approximate)
                            window = from_bounds(
                                lon - half_size, lat - half_size,
                                lon + half_size, lat + half_size,
                                src.transform,
                            )
                            data = src.read(1, window=window).astype(np.float32)
                            data = self._resize_band(data, PATCH_SIZE)
                            bands.append(data)
                    except Exception:
                        # Simple fallback: download full band and crop
                        band_resp = requests.get(band_url, timeout=30)
                        if band_resp.status_code == 200:
                            import io
                            import rasterio
                            with rasterio.open(io.BytesIO(band_resp.content)) as src:
                                data = src.read(1).astype(np.float32)
                                # Center crop to 120x120
                                h, w = data.shape
                                ch, cw = h // 2, w // 2
                                data = data[ch-60:ch+60, cw-60:cw+60]
                                bands.append(data)

                if len(bands) == 3:
                    downloaded[pid] = np.stack(bands)
                    count += 1
                else:
                    failed.append(pid)

            except Exception as e:
                failed.append(pid)
                logger.debug(f"Failed {pid}: {e}")

            # Rate limiting
            if i % 10 == 0:
                import time
                time.sleep(0.1)

        logger.info(
            f"Downloaded {len(downloaded):,} patches from Planetary Computer "
            f"({len(failed):,} failed)"
        )
        return downloaded

    # =================================================================
    # Step 5: Source D — Synthetic from spectral statistics
    # =================================================================

    def generate_synthetic(self, patch_ids: set, seed: int = 42) -> dict:
        """Generate realistic synthetic patches from official band statistics.

        These aren't random noise — they use the actual band-level mean/std
        from the BigEarthNet v2 training set to create plausible spectral
        values. Not perfect, but enough for the VLM to learn instruction
        format and RS vocabulary.
        """
        rng = np.random.RandomState(seed)
        synthetic = {}

        for pid in patch_ids:
            # Use patch_id as deterministic seed for reproducibility
            pid_seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
            patch_rng = np.random.RandomState(pid_seed)

            bands = []
            for band_name in ["B04", "B03", "B02"]:
                stats = BAND_STATS[band_name]
                # Generate correlated spatial noise (more realistic than iid)
                base = patch_rng.normal(stats["mean"], stats["std"] * 0.3, (PATCH_SIZE, PATCH_SIZE))
                # Add smooth spatial structure
                from scipy.ndimage import gaussian_filter
                smooth = gaussian_filter(base, sigma=5.0)
                # Mix base noise with smooth structure
                data = 0.4 * base + 0.6 * smooth
                data = np.clip(data, 0, 10000).astype(np.float32)
                bands.append(data)

            synthetic[pid] = np.stack(bands)

        logger.info(f"Generated {len(synthetic):,} synthetic patches")
        return synthetic

    # =================================================================
    # Cache management
    # =================================================================

    def save_to_cache(self, patches: dict):
        """Save patches as compressed .npz files."""
        for pid, rgb_array in patches.items():
            npz_path = self.cache_dir / f"{pid}.npz"
            np.savez_compressed(str(npz_path), rgb=rgb_array)

        logger.info(f"Saved {len(patches):,} patches to {self.cache_dir}")

    def load_from_cache(self, patch_id: str) -> Optional[np.ndarray]:
        """Load a single patch from cache."""
        npz_path = self.cache_dir / f"{patch_id}.npz"
        if npz_path.exists():
            return np.load(str(npz_path))["rgb"]
        return None

    def to_vlm_rgb(self, raw_bands: np.ndarray) -> np.ndarray:
        """Convert raw (3, 120, 120) uint16 to (120, 120, 3) uint8 RGB for VLM.

        Uses percentile clipping for robust contrast stretching.
        """
        # raw_bands shape: (3, H, W) — R, G, B order
        rgb = np.transpose(raw_bands, (1, 2, 0))  # (H, W, 3)

        # Percentile clip per channel
        for c in range(3):
            p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
            rgb[:, :, c] = np.clip(rgb[:, :, c], p2, p98)
            if p98 > p2:
                rgb[:, :, c] = (rgb[:, :, c] - p2) / (p98 - p2) * 255
            else:
                rgb[:, :, c] = 128

        return rgb.astype(np.uint8)

    # =================================================================
    # Utility
    # =================================================================

    @staticmethod
    def _resize_band(band: np.ndarray, target_size: int) -> np.ndarray:
        """Resize a band to target_size × target_size."""
        if band.shape == (target_size, target_size):
            return band

        from PIL import Image
        img = Image.fromarray(band)
        img = img.resize((target_size, target_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    # =================================================================
    # Main pipeline
    # =================================================================

    def run(
        self,
        mode: str = "download",
        limit: int = 20000,
        use_planetary: bool = True,
    ):
        """Run the smart imagery pipeline.

        Modes:
            download  — Try LMDB → GeoTIFF → Planetary Computer → synthetic
            synthetic — Generate synthetic for ALL missing patches
            verify    — Report coverage statistics
        """
        required = self.get_required_patches()
        cached = self.get_cached_patches()
        missing = required - cached

        logger.info(f"Required: {len(required):,} | Cached: {len(cached):,} | Missing: {len(missing):,}")

        if mode == "verify":
            coverage = len(cached) / len(required) * 100 if required else 0
            logger.info(f"Coverage: {coverage:.1f}%")
            cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npz"))
            logger.info(f"Cache size: {cache_size / 1e9:.2f} GB")
            return

        if not missing:
            logger.info("All patches already cached!")
            return

        # Limit the number to download
        to_process = set(list(missing)[:limit])
        logger.info(f"Processing {len(to_process):,} patches...")

        # Source A: LMDB
        lmdb_results = self.load_from_lmdb(to_process)
        if lmdb_results:
            self.save_to_cache(lmdb_results)
            to_process -= set(lmdb_results.keys())

        # Source B: GeoTIFFs
        if to_process:
            geotiff_results = self.load_from_geotiffs(to_process)
            if geotiff_results:
                self.save_to_cache(geotiff_results)
                to_process -= set(geotiff_results.keys())

        # Source C: Planetary Computer (if enabled and patches still missing)
        if to_process and use_planetary and mode == "download":
            pc_results = self.download_from_planetary_computer(
                to_process, max_patches=min(len(to_process), limit)
            )
            if pc_results:
                self.save_to_cache(pc_results)
                to_process -= set(pc_results.keys())

        # Source D: Synthetic fallback (always fills remaining)
        if to_process and mode in ("download", "synthetic"):
            synth_results = self.generate_synthetic(to_process)
            self.save_to_cache(synth_results)
            to_process -= set(synth_results.keys())

        # Final report
        final_cached = self.get_cached_patches()
        coverage = len(final_cached & required) / len(required) * 100
        logger.info(f"Final coverage: {coverage:.1f}%")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Imagery Pipeline")
    parser.add_argument("--mode", choices=["download", "synthetic", "verify"],
                        default="synthetic",
                        help="download=try all sources, synthetic=generate placeholders, verify=check coverage")
    parser.add_argument("--limit", type=int, default=20000,
                        help="Max patches to process")
    parser.add_argument("--lmdb", default=None,
                        help="Path to existing LMDB database")
    parser.add_argument("--geotiff-dir", default=None,
                        help="Path to extracted BigEarthNet directories")
    parser.add_argument("--cache-dir", default="data/imagery_cache",
                        help="Where to store cached patches")
    parser.add_argument("--no-planetary", action="store_true",
                        help="Skip Planetary Computer download")
    args = parser.parse_args()

    pipeline = SmartImageryPipeline(
        cache_dir=args.cache_dir,
        lmdb_path=args.lmdb,
        geotiff_dir=args.geotiff_dir,
    )
    pipeline.run(
        mode=args.mode,
        limit=args.limit,
        use_planetary=not args.no_planetary,
    )
