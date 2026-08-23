"""
Sentinel-2 Multi-Resolution Band Handling for SatQuery AI.

PROBLEM:
    Sentinel-2 has 13 spectral bands at THREE different spatial resolutions:
    - 10m:  B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)           → 4 bands
    - 20m:  B05, B06, B07 (Red Edge), B8A (Narrow NIR), B11, B12    → 6 bands
    - 60m:  B01 (Coastal Aerosol), B09 (Water Vapor), B10 (Cirrus)  → 3 bands

    For a 1.2km x 1.2km BigEarthNet patch:
    - 10m bands → 120 x 120 pixels
    - 20m bands →  60 x  60 pixels
    - 60m bands →  20 x  20 pixels

SOLUTION (Multi-Resolution Harmonization Strategy):
    We implement a THREE-TIER approach:

    Tier 1 — VLM Input (RGB + NIR):
        Select B04 (R), B03 (G), B02 (B), B08 (NIR) at native 10m → 120x120.
        Compose a false-color or true-color RGB image for the VLM.
        The VLM (InternVL2-2B) expects standard image input, so we normalize
        to [0, 255] uint8 or [0, 1] float and resize to 448x448 (VLM tile size).

    Tier 2 — Auxiliary Spectral Features:
        Upsample 20m bands to 10m resolution (60x60 → 120x120) using bilinear
        interpolation. These provide Red Edge, SWIR info critical for LULC.
        Drop 60m bands (B01, B09, B10) — atmospheric correction bands, not
        useful for LULC classification and too coarse for meaningful upsampling.
        This matches BigEarthNet v2's standard 12-band selection (no B10).

    Tier 3 — Spectral Index Computation:
        Compute NDVI, NDWI, NDBI from harmonized bands as additional features
        for the change detection and fusion heads.

NORMALIZATION:
    - Sentinel-2 L2A reflectance values: typically 0–10000 (scaled by 10000).
    - Per-band statistics from BigEarthNet v2 are used for standardization.
    - SAR (Sentinel-1) has completely different value distributions (dB scale)
      and is handled separately in sentinel1_preprocessing.py.

PROVENANCE:
    Every preprocessing step is logged for auditability (§5.1b, D4).
    The band selection, resampling method, and normalization stats are
    recorded in the output metadata.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Sentinel-2 Band Specification
# =============================================================================

class S2Resolution(Enum):
    """Sentinel-2 native spatial resolutions."""
    R10M = 10
    R20M = 20
    R60M = 60


@dataclass(frozen=True)
class S2BandSpec:
    """Specification for a single Sentinel-2 spectral band."""
    name: str               # e.g., "B02"
    description: str         # e.g., "Blue"
    wavelength_nm: int       # Central wavelength in nm
    resolution: S2Resolution # Native spatial resolution
    index_in_l2a: int        # 0-based index in standard L2A product


# Complete Sentinel-2 L2A band catalog (13 bands)
S2_BANDS = [
    S2BandSpec("B01", "Coastal Aerosol",    443,  S2Resolution.R60M, 0),
    S2BandSpec("B02", "Blue",               490,  S2Resolution.R10M, 1),
    S2BandSpec("B03", "Green",              560,  S2Resolution.R10M, 2),
    S2BandSpec("B04", "Red",                665,  S2Resolution.R10M, 3),
    S2BandSpec("B05", "Red Edge 1",         705,  S2Resolution.R20M, 4),
    S2BandSpec("B06", "Red Edge 2",         740,  S2Resolution.R20M, 5),
    S2BandSpec("B07", "Red Edge 3",         783,  S2Resolution.R20M, 6),
    S2BandSpec("B08", "NIR",                842,  S2Resolution.R10M, 7),
    S2BandSpec("B8A", "Narrow NIR",         865,  S2Resolution.R20M, 8),
    S2BandSpec("B09", "Water Vapor",        945,  S2Resolution.R60M, 9),
    S2BandSpec("B10", "Cirrus",             1375, S2Resolution.R60M, 10),
    S2BandSpec("B11", "SWIR 1",             1610, S2Resolution.R20M, 11),
    S2BandSpec("B12", "SWIR 2",             2190, S2Resolution.R20M, 12),
]

S2_BAND_MAP = {b.name: b for b in S2_BANDS}

# BigEarthNet v2 uses 12 bands (drops B10/Cirrus), matching sen2cor L2A output
BIGEARTH_S2_BANDS = [b for b in S2_BANDS if b.name != "B10"]

# The 4 bands at native 10m — used for VLM RGB input
S2_10M_BANDS = [b for b in S2_BANDS if b.resolution == S2Resolution.R10M]

# The 6 bands at 20m — upsampled for auxiliary spectral features
S2_20M_BANDS = [b for b in S2_BANDS if b.resolution == S2Resolution.R20M]

# The 3 bands at 60m — typically dropped for LULC tasks
S2_60M_BANDS = [b for b in S2_BANDS if b.resolution == S2Resolution.R60M]


# =============================================================================
# Per-Band Normalization Statistics (BigEarthNet v2 training set)
# =============================================================================

# These are approximate per-band mean/std values for Sentinel-2 L2A reflectance
# computed over the BigEarthNet v2 training split. Values are in the raw
# reflectance scale (0–10000). They should be refined once we download and
# compute stats on our actual subset.
#
# Source: Approximated from published BigEarthNet statistics and
# common Sentinel-2 normalization values used in RS literature.

S2_NORM_STATS = {
    # band_name: (mean, std)
    # Source: Official BigEarthNet v2 training split, interpolated to 120x120 (nearest)
    # From: data/bigearthnet_txt/ben_txt_datamodule.py
    "B01": (361.0767822265625, 575.0687255859375),
    "B02": (438.3720703125, 607.02685546875),
    "B03": (614.0556640625, 603.2968139648438),
    "B04": (588.4096069335938, 684.56884765625),
    "B05": (942.8433227539062, 738.4326782226562),
    "B06": (1769.931640625, 1100.4560546875),
    "B07": (2049.551513671875, 1275.805419921875),
    "B08": (2193.2919921875, 1369.3717041015625),
    "B8A": (2235.556640625, 1356.5440673828125),
    "B09": (2241.455322265625, 1316.393310546875),
    "B10": (0.0, 1.0),    # Dropped by default (not in BigEarthNet v2)
    "B11": (1568.226806640625, 1070.1612548828125),
    "B12": (997.7324829101562, 813.5276489257812),
}


# =============================================================================
# Preprocessing Configuration
# =============================================================================

@dataclass
class S2PreprocessConfig:
    """Configuration for Sentinel-2 preprocessing pipeline.

    This config controls how multi-resolution bands are handled,
    what output format is produced, and which normalization is applied.
    """

    # --- Band selection ---
    # Which bands to include in the harmonized output
    use_10m_bands: bool = True       # B02, B03, B04, B08 (always True for VLM)
    use_20m_bands: bool = True       # B05, B06, B07, B8A, B11, B12
    use_60m_bands: bool = False      # B01, B09, B10 — dropped by default

    # --- Resolution harmonization ---
    target_resolution_m: int = 10    # Upsample everything to this resolution
    resample_method: str = "bilinear"  # "bilinear", "nearest", "cubic"
    # For a 1.2km BigEarthNet patch at 10m → 120x120 pixels
    # For arbitrary GeoTIFF, spatial size depends on the image

    # --- VLM input preparation ---
    vlm_rgb_bands: tuple[str, ...] = ("B04", "B03", "B02")  # True color: R, G, B
    vlm_tile_size: int = 448         # InternVL2 tile size
    vlm_normalize_to_uint8: bool = True  # Scale to [0, 255] for VLM

    # RGB percentile clipping for visualization (handles outlier reflectance)
    rgb_clip_percentile_low: float = 2.0
    rgb_clip_percentile_high: float = 98.0

    # --- Spectral normalization ---
    normalize_method: str = "standardize"  # "standardize" (z-score) or "minmax"

    # --- Spectral indices ---
    compute_ndvi: bool = True   # (B08 - B04) / (B08 + B04)
    compute_ndwi: bool = True   # (B03 - B08) / (B03 + B08)
    compute_ndbi: bool = False  # (B11 - B08) / (B11 + B08) — needs 20m bands


# =============================================================================
# Multi-Resolution Preprocessor
# =============================================================================

@dataclass
class PreprocessingProvenance:
    """Audit trail for preprocessing decisions (§5.1b, D4)."""
    bands_used: list[str] = field(default_factory=list)
    bands_dropped: list[str] = field(default_factory=list)
    drop_reasons: dict[str, str] = field(default_factory=dict)
    resample_method: str = ""
    target_resolution_m: int = 0
    normalization: str = ""
    norm_stats_source: str = ""
    spectral_indices: list[str] = field(default_factory=list)
    original_crs: str = ""
    original_shape: tuple = ()
    output_shape: tuple = ()


class Sentinel2Preprocessor:
    """Multi-resolution Sentinel-2 preprocessing pipeline.

    Solves the fundamental problem: Sentinel-2's 13 bands are at 3 different
    spatial resolutions (10m, 20m, 60m). This preprocessor harmonizes them
    to a common resolution, prepares VLM-ready RGB images, and computes
    auxiliary spectral features.

    Usage:
        config = S2PreprocessConfig()
        preprocessor = Sentinel2Preprocessor(config)

        # From a GeoTIFF file
        result = preprocessor.process_geotiff("/path/to/sentinel2.tif")

        # From raw band arrays (BigEarthNet format)
        result = preprocessor.process_bands(bands_dict)
    """

    def __init__(self, config: Optional[S2PreprocessConfig] = None):
        self.config = config or S2PreprocessConfig()
        self._validate_config()

    def _validate_config(self):
        """Ensure config is consistent."""
        if self.config.compute_ndbi and not self.config.use_20m_bands:
            raise ValueError(
                "NDBI requires B11 (20m band). Set use_20m_bands=True "
                "or compute_ndbi=False."
            )
        for band_name in self.config.vlm_rgb_bands:
            if band_name not in S2_BAND_MAP:
                raise ValueError(f"Unknown band in vlm_rgb_bands: {band_name}")

    def process_geotiff(
        self,
        filepath: Union[str, Path],
        band_names: Optional[list[str]] = None,
    ) -> dict:
        """Process a Sentinel-2 GeoTIFF file.

        Args:
            filepath: Path to the GeoTIFF file.
            band_names: Ordered list of band names in the file. If None,
                        auto-detect from metadata or assume BigEarthNet order.

        Returns:
            Dict with keys:
                - 'vlm_rgb': np.ndarray (H, W, 3) uint8 — ready for VLM
                - 'spectral_harmonized': np.ndarray (C, H, W) float32 —
                    all selected bands at target resolution, normalized
                - 'spectral_indices': dict of np.ndarray — NDVI, NDWI, etc.
                - 'provenance': PreprocessingProvenance
                - 'metadata': dict — CRS, bounds, resolution, etc.
        """
        import rasterio
        from rasterio.enums import Resampling

        filepath = Path(filepath)
        provenance = PreprocessingProvenance()

        with rasterio.open(filepath) as src:
            # Extract metadata
            provenance.original_crs = str(src.crs)
            provenance.original_shape = (src.count, src.height, src.width)

            metadata = {
                "crs": str(src.crs),
                "bounds": dict(zip(["left", "bottom", "right", "top"], src.bounds)),
                "resolution": src.res,
                "band_count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "transform": list(src.transform),
            }

            # Read all bands
            all_data = src.read()  # shape: (num_bands, H, W)
            num_bands = all_data.shape[0]

            # Auto-detect band names if not provided
            if band_names is None:
                band_names = self._infer_band_names(src, num_bands)

            logger.info(
                f"Read {filepath.name}: {num_bands} bands, "
                f"shape={all_data.shape}, CRS={src.crs}"
            )

        # Build band dict: band_name -> 2D array
        bands_dict = {}
        for i, name in enumerate(band_names):
            if i < num_bands:
                bands_dict[name] = all_data[i].astype(np.float32)

        return self.process_bands(bands_dict, provenance, metadata)

    def process_bands(
        self,
        bands_dict: dict[str, np.ndarray],
        provenance: Optional[PreprocessingProvenance] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Process raw band arrays (e.g., from BigEarthNet patches).

        Args:
            bands_dict: Mapping of band_name -> 2D np.ndarray.
                        Arrays may have different spatial dimensions
                        (the whole point of this preprocessor).

        Returns:
            Same dict structure as process_geotiff.
        """
        if provenance is None:
            provenance = PreprocessingProvenance()
        if metadata is None:
            metadata = {}

        provenance.resample_method = self.config.resample_method
        provenance.target_resolution_m = self.config.target_resolution_m

        # Step 1: Determine target spatial size from 10m bands
        target_h, target_w = self._get_target_shape(bands_dict)
        logger.info(f"Target shape after harmonization: ({target_h}, {target_w})")

        # Step 2: Select and harmonize bands
        harmonized = self._harmonize_bands(
            bands_dict, target_h, target_w, provenance
        )

        # Step 3: Prepare VLM RGB input
        vlm_rgb = self._prepare_vlm_rgb(bands_dict, target_h, target_w)

        # Step 4: Normalize spectral bands
        spectral_normalized = self._normalize_spectral(harmonized, provenance)

        # Step 5: Compute spectral indices
        indices = self._compute_spectral_indices(bands_dict, target_h, target_w)
        provenance.spectral_indices = list(indices.keys())

        provenance.output_shape = spectral_normalized.shape
        provenance.normalization = self.config.normalize_method
        provenance.norm_stats_source = "BigEarthNet v2 training set (approximate)"

        return {
            "vlm_rgb": vlm_rgb,
            "spectral_harmonized": spectral_normalized,
            "spectral_indices": indices,
            "provenance": provenance,
            "metadata": metadata,
        }

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _infer_band_names(self, src, num_bands: int) -> list[str]:
        """Try to infer band names from GeoTIFF metadata.

        Falls back to BigEarthNet v2 ordering (12 bands, no B10)
        or full 13-band L2A ordering.
        """
        # Try rasterio band descriptions
        descriptions = src.descriptions
        if descriptions and all(d is not None for d in descriptions):
            # Some GeoTIFFs have band names in descriptions
            inferred = []
            for desc in descriptions:
                desc_upper = desc.upper().strip()
                for band in S2_BANDS:
                    if band.name in desc_upper or band.description.upper() in desc_upper:
                        inferred.append(band.name)
                        break
            if len(inferred) == num_bands:
                return inferred

        # Fallback: assume standard ordering
        if num_bands == 12:
            # BigEarthNet v2 (no B10)
            return [b.name for b in BIGEARTH_S2_BANDS]
        elif num_bands == 13:
            # Full L2A
            return [b.name for b in S2_BANDS]
        elif num_bands <= 4:
            # Likely RGB or RGBNIR
            if num_bands == 3:
                return ["B04", "B03", "B02"]  # R, G, B
            elif num_bands == 4:
                return ["B04", "B03", "B02", "B08"]  # R, G, B, NIR
        
        # Last resort: generic names
        logger.warning(
            f"Could not infer band names for {num_bands}-band image. "
            f"Using generic B01..B{num_bands:02d} naming."
        )
        return [f"B{i+1:02d}" for i in range(num_bands)]

    def _get_target_shape(self, bands_dict: dict[str, np.ndarray]) -> tuple[int, int]:
        """Determine the target spatial shape from 10m bands.

        The highest-resolution bands define the target grid.
        All other bands will be resampled to match.
        """
        # Look for 10m bands first
        for band_name in ["B02", "B03", "B04", "B08"]:
            if band_name in bands_dict:
                return bands_dict[band_name].shape

        # If no 10m bands, use the largest array
        max_size = 0
        max_shape = (120, 120)  # default BigEarthNet
        for arr in bands_dict.values():
            size = arr.shape[0] * arr.shape[1]
            if size > max_size:
                max_size = size
                max_shape = arr.shape
        return max_shape

    def _harmonize_bands(
        self,
        bands_dict: dict[str, np.ndarray],
        target_h: int,
        target_w: int,
        provenance: PreprocessingProvenance,
    ) -> np.ndarray:
        """Select bands per config, upsample to target resolution.

        Returns:
            np.ndarray of shape (num_selected_bands, target_h, target_w)
        """
        selected_bands = []
        selected_names = []
        dropped_names = []

        for band_spec in S2_BANDS:
            name = band_spec.name
            if name not in bands_dict:
                continue

            # Apply band selection rules
            if band_spec.resolution == S2Resolution.R10M and self.config.use_10m_bands:
                pass  # Include
            elif band_spec.resolution == S2Resolution.R20M and self.config.use_20m_bands:
                pass  # Include (will be upsampled)
            elif band_spec.resolution == S2Resolution.R60M and self.config.use_60m_bands:
                pass  # Include (will be upsampled)
            else:
                dropped_names.append(name)
                reason = f"Excluded by config (resolution={band_spec.resolution.value}m)"
                provenance.drop_reasons[name] = reason
                continue

            arr = bands_dict[name]

            # Resample if needed
            if arr.shape != (target_h, target_w):
                arr = self._resample_band(
                    arr, target_h, target_w, self.config.resample_method
                )
                logger.debug(
                    f"Resampled {name} from {bands_dict[name].shape} "
                    f"to ({target_h}, {target_w}) via {self.config.resample_method}"
                )

            selected_bands.append(arr)
            selected_names.append(name)

        provenance.bands_used = selected_names
        provenance.bands_dropped = dropped_names

        if not selected_bands:
            raise ValueError("No bands selected after filtering. Check config.")

        return np.stack(selected_bands, axis=0)  # (C, H, W)

    def _resample_band(
        self,
        band: np.ndarray,
        target_h: int,
        target_w: int,
        method: str = "bilinear",
    ) -> np.ndarray:
        """Resample a single band to target dimensions.

        This handles the core multi-resolution problem:
        - 20m bands (60x60 for BigEarthNet) → 10m (120x120)
        - 60m bands (20x20 for BigEarthNet) → 10m (120x120)

        Uses scipy/skimage for resampling to avoid heavy GDAL dependency
        during inference (rasterio is used only for file I/O).
        """
        from scipy.ndimage import zoom

        if band.shape == (target_h, target_w):
            return band

        zoom_h = target_h / band.shape[0]
        zoom_w = target_w / band.shape[1]

        if method == "bilinear":
            order = 1
        elif method == "cubic":
            order = 3
        elif method == "nearest":
            order = 0
        else:
            raise ValueError(f"Unknown resample method: {method}")

        return zoom(band, (zoom_h, zoom_w), order=order).astype(np.float32)

    def _prepare_vlm_rgb(
        self,
        bands_dict: dict[str, np.ndarray],
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Prepare a VLM-ready RGB image from Sentinel-2 bands.

        InternVL2-2B expects standard image input (RGB, uint8 or float).
        We select the configured RGB bands (default: B04=Red, B03=Green, B02=Blue),
        harmonize resolution, apply percentile clipping, and normalize to uint8.

        Returns:
            np.ndarray of shape (H, W, 3) dtype uint8
        """
        rgb_channels = []
        for band_name in self.config.vlm_rgb_bands:
            if band_name not in bands_dict:
                raise ValueError(
                    f"VLM RGB band {band_name} not found in input. "
                    f"Available: {list(bands_dict.keys())}"
                )
            arr = bands_dict[band_name].astype(np.float32)
            # Resample if needed (e.g., if someone passes a 20m band for RGB)
            if arr.shape != (target_h, target_w):
                arr = self._resample_band(arr, target_h, target_w, "bilinear")
            rgb_channels.append(arr)

        rgb = np.stack(rgb_channels, axis=-1)  # (H, W, 3)

        if self.config.vlm_normalize_to_uint8:
            rgb = self._percentile_clip_and_scale(rgb)

        return rgb

    def _percentile_clip_and_scale(self, rgb: np.ndarray) -> np.ndarray:
        """Clip reflectance outliers and scale to uint8 [0, 255].

        Sentinel-2 reflectance values can have outliers (shadows, clouds,
        saturation). Percentile clipping produces much better visual
        quality than simple min-max scaling.
        """
        low = self.config.rgb_clip_percentile_low
        high = self.config.rgb_clip_percentile_high

        # Compute percentiles per channel
        for c in range(rgb.shape[-1]):
            channel = rgb[:, :, c]
            p_low = np.percentile(channel, low)
            p_high = np.percentile(channel, high)

            if p_high - p_low < 1e-6:
                # Degenerate case: constant band
                rgb[:, :, c] = 0
            else:
                channel = np.clip(channel, p_low, p_high)
                channel = (channel - p_low) / (p_high - p_low) * 255.0
                rgb[:, :, c] = channel

        return rgb.astype(np.uint8)

    def _normalize_spectral(
        self,
        spectral: np.ndarray,
        provenance: PreprocessingProvenance,
    ) -> np.ndarray:
        """Apply per-band normalization to harmonized spectral stack.

        Args:
            spectral: (C, H, W) float32 array of harmonized bands.

        Returns:
            Normalized (C, H, W) float32 array.
        """
        normalized = spectral.copy()
        band_names = provenance.bands_used

        for i, name in enumerate(band_names):
            if self.config.normalize_method == "standardize":
                if name in S2_NORM_STATS:
                    mean, std = S2_NORM_STATS[name]
                    normalized[i] = (normalized[i] - mean) / (std + 1e-8)
                else:
                    # Fallback: per-image z-score
                    mean = np.mean(normalized[i])
                    std = np.std(normalized[i])
                    normalized[i] = (normalized[i] - mean) / (std + 1e-8)
                    logger.warning(
                        f"No preset stats for band {name}. Using per-image z-score."
                    )
            elif self.config.normalize_method == "minmax":
                vmin = np.min(normalized[i])
                vmax = np.max(normalized[i])
                if vmax - vmin > 1e-8:
                    normalized[i] = (normalized[i] - vmin) / (vmax - vmin)
                else:
                    normalized[i] = 0.0

        return normalized

    def _compute_spectral_indices(
        self,
        bands_dict: dict[str, np.ndarray],
        target_h: int,
        target_w: int,
    ) -> dict[str, np.ndarray]:
        """Compute spectral vegetation/water/built-up indices.

        These provide additional features for change detection and
        fusion heads without increasing VLM input complexity.
        """
        indices = {}

        def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute (a - b) / (a + b) safely, avoiding division by zero."""
            num = a.astype(np.float32) - b.astype(np.float32)
            den = a.astype(np.float32) + b.astype(np.float32)
            return np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-8)

        def _get_band(name: str) -> Optional[np.ndarray]:
            if name not in bands_dict:
                return None
            arr = bands_dict[name].astype(np.float32)
            if arr.shape != (target_h, target_w):
                arr = self._resample_band(arr, target_h, target_w, "bilinear")
            return arr

        if self.config.compute_ndvi:
            nir = _get_band("B08")
            red = _get_band("B04")
            if nir is not None and red is not None:
                indices["NDVI"] = _safe_index(nir, red)
                logger.debug("Computed NDVI")

        if self.config.compute_ndwi:
            green = _get_band("B03")
            nir = _get_band("B08")
            if green is not None and nir is not None:
                indices["NDWI"] = _safe_index(green, nir)
                logger.debug("Computed NDWI")

        if self.config.compute_ndbi:
            swir = _get_band("B11")
            nir = _get_band("B08")
            if swir is not None and nir is not None:
                indices["NDBI"] = _safe_index(swir, nir)
                logger.debug("Computed NDBI")

        return indices

    def get_provenance_dict(self, provenance: PreprocessingProvenance) -> dict:
        """Convert provenance to a serializable dict for audit logging."""
        return {
            "bands_used": provenance.bands_used,
            "bands_dropped": provenance.bands_dropped,
            "drop_reasons": provenance.drop_reasons,
            "resample_method": provenance.resample_method,
            "target_resolution_m": provenance.target_resolution_m,
            "normalization": provenance.normalization,
            "norm_stats_source": provenance.norm_stats_source,
            "spectral_indices": provenance.spectral_indices,
            "original_crs": provenance.original_crs,
            "original_shape": list(provenance.original_shape),
            "output_shape": list(provenance.output_shape),
        }
