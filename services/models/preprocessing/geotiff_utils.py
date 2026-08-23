"""
GeoTIFF Validation & Metadata Extraction for SatQuery AI.

Handles:
- Format validation (GeoTIFF/TIFF primary, PNG/JPEG only for benchmarks)
- CRS and geographic compatibility checking for pairs
- Band configuration detection (how many bands? what sensor?)
- Resolution extraction and mismatch detection
- Co-registration validation for cross-modal and bi-temporal pairs

This feeds into the Ontology layer (D1) and Input Validation (F9).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ImageModality(Enum):
    """Detected image modality."""
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    RGB = "rgb"
    UNKNOWN = "unknown"


class ImageFormat(Enum):
    """Supported image formats."""
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    """Structured metadata for an uploaded/fetched image.

    This is the 'OntologyImage' object from the architecture (D1).
    Every image gets one of these before any model block touches it.
    """
    filepath: str
    format: ImageFormat = ImageFormat.UNKNOWN
    modality: ImageModality = ImageModality.UNKNOWN

    # Spatial
    crs: Optional[str] = None           # e.g., "EPSG:32633"
    bounds: Optional[dict] = None       # {"left", "bottom", "right", "top"}
    resolution: Optional[tuple] = None  # (x_res, y_res) in CRS units
    width: int = 0
    height: int = 0

    # Spectral
    band_count: int = 0
    band_names: list[str] = field(default_factory=list)
    dtype: str = ""
    nodata: Optional[float] = None

    # Sensor inference
    sensor_guess: str = ""  # e.g., "Sentinel-2", "Sentinel-1", "Cartosat-2S"
    cloud_cover: Optional[float] = None

    # Flags
    is_benchmark_format: bool = False  # True if PNG/JPEG from benchmark dataset
    has_georeference: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)


class GeoTIFFValidator:
    """Validates and extracts metadata from satellite imagery.

    Enforces the format rules from the project docs:
    - GeoTIFF/TIFF is primary format for user uploads
    - PNG/JPEG accepted ONLY for prescribed benchmark datasets
    - Band configuration must be validated before any model block
    """

    BENCHMARK_DATASETS = {"VRSBench", "RSVQA", "CDVQA", "BigEarthNet"}

    def validate_and_extract(
        self,
        filepath: Union[str, Path],
        dataset_source: Optional[str] = None,
    ) -> ImageMetadata:
        """Validate an image file and extract its metadata.

        Args:
            filepath: Path to the image file.
            dataset_source: If from a benchmark dataset, specify which one.
                            Enables PNG/JPEG acceptance for that file.

        Returns:
            ImageMetadata with validation results.
        """
        filepath = Path(filepath)
        meta = ImageMetadata(filepath=str(filepath))

        if not filepath.exists():
            meta.validation_errors.append(f"File not found: {filepath}")
            return meta

        # Detect format
        meta.format = self._detect_format(filepath)

        # Check format compliance
        if meta.format in (ImageFormat.PNG, ImageFormat.JPEG):
            if dataset_source and dataset_source in self.BENCHMARK_DATASETS:
                meta.is_benchmark_format = True
                meta.validation_warnings.append(
                    f"PNG/JPEG accepted because source is benchmark dataset: "
                    f"{dataset_source}. This bypass is logged per FR1."
                )
            else:
                meta.validation_errors.append(
                    f"Format {meta.format.value} is only accepted for prescribed "
                    f"benchmark datasets (VRSBench, RSVQA, CDVQA). "
                    f"User uploads must be GeoTIFF/TIFF."
                )

        # Extract metadata based on format
        if meta.format in (ImageFormat.GEOTIFF, ImageFormat.TIFF):
            self._extract_geotiff_metadata(filepath, meta)
        elif meta.format in (ImageFormat.PNG, ImageFormat.JPEG):
            self._extract_raster_metadata(filepath, meta)

        # Infer modality and sensor
        self._infer_modality(meta)
        self._infer_sensor(meta)

        return meta

    def validate_pair_compatibility(
        self,
        meta_a: ImageMetadata,
        meta_b: ImageMetadata,
        pair_type: str = "cross_modal",
    ) -> dict:
        """Check if two images are compatible as a pair.

        Args:
            pair_type: "cross_modal" (optical+SAR) or "bitemporal" (t1+t2)

        Returns:
            Dict with 'compatible' (bool), 'errors' (list), 'warnings' (list).
        """
        errors = []
        warnings = []

        # CRS compatibility
        if meta_a.crs and meta_b.crs:
            if meta_a.crs != meta_b.crs:
                # Not necessarily a hard error — could be reprojected
                warnings.append(
                    f"CRS mismatch: {meta_a.crs} vs {meta_b.crs}. "
                    f"Images may need reprojection before processing."
                )

        # Geographic overlap (co-registration check)
        if meta_a.bounds and meta_b.bounds:
            overlap = self._compute_overlap(meta_a.bounds, meta_b.bounds)
            if overlap < 0.5:
                errors.append(
                    f"Insufficient geographic overlap ({overlap:.1%}). "
                    f"Images may not be co-registered."
                )
            elif overlap < 0.9:
                warnings.append(
                    f"Partial geographic overlap ({overlap:.1%}). "
                    f"Co-registration quality may be limited."
                )

        # Resolution compatibility
        if meta_a.resolution and meta_b.resolution:
            ratio = max(meta_a.resolution[0], meta_b.resolution[0]) / \
                    min(meta_a.resolution[0], meta_b.resolution[0])
            if ratio > 10:
                errors.append(
                    f"Resolution mismatch too large ({ratio:.1f}x). "
                    f"Harmonization may produce meaningless results."
                )
            elif ratio > 2:
                warnings.append(
                    f"Resolution mismatch ({ratio:.1f}x). "
                    f"Resampling will be applied during harmonization."
                )

        # Modality checks for cross-modal pairs
        if pair_type == "cross_modal":
            modalities = {meta_a.modality, meta_b.modality}
            if ImageModality.SAR not in modalities:
                errors.append(
                    "Cross-modal pair requires one SAR image. "
                    "Neither image appears to be SAR."
                )
            if modalities == {ImageModality.SAR}:
                errors.append(
                    "Cross-modal pair requires one optical/multispectral image. "
                    "Both images appear to be SAR."
                )

        return {
            "compatible": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _detect_format(self, filepath: Path) -> ImageFormat:
        """Detect image format from file extension and magic bytes."""
        suffix = filepath.suffix.lower()
        if suffix in (".tif", ".tiff", ".geotiff"):
            # Check if it's a GeoTIFF (has CRS) or plain TIFF
            try:
                import rasterio
                with rasterio.open(filepath) as src:
                    if src.crs is not None:
                        return ImageFormat.GEOTIFF
                    return ImageFormat.TIFF
            except Exception:
                return ImageFormat.TIFF
        elif suffix in (".png",):
            return ImageFormat.PNG
        elif suffix in (".jpg", ".jpeg"):
            return ImageFormat.JPEG
        else:
            return ImageFormat.UNKNOWN

    def _extract_geotiff_metadata(self, filepath: Path, meta: ImageMetadata):
        """Extract metadata from a GeoTIFF using rasterio."""
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                meta.crs = str(src.crs) if src.crs else None
                meta.has_georeference = src.crs is not None
                meta.bounds = dict(
                    zip(["left", "bottom", "right", "top"], src.bounds)
                ) if src.bounds else None
                meta.resolution = src.res if src.res else None
                meta.width = src.width
                meta.height = src.height
                meta.band_count = src.count
                meta.dtype = str(src.dtypes[0]) if src.dtypes else ""
                meta.nodata = src.nodata

                # Try to extract band names
                if src.descriptions:
                    meta.band_names = [
                        d if d else f"Band_{i+1}"
                        for i, d in enumerate(src.descriptions)
                    ]
                else:
                    meta.band_names = [f"Band_{i+1}" for i in range(src.count)]

                # Look for cloud cover in tags
                tags = src.tags()
                if "CLOUD_COVER" in tags:
                    try:
                        meta.cloud_cover = float(tags["CLOUD_COVER"])
                    except (ValueError, TypeError):
                        pass

        except ImportError:
            meta.validation_errors.append(
                "rasterio not installed. Cannot read GeoTIFF metadata."
            )
        except Exception as e:
            meta.validation_errors.append(f"Failed to read GeoTIFF: {e}")

    def _extract_raster_metadata(self, filepath: Path, meta: ImageMetadata):
        """Extract basic metadata from PNG/JPEG (no geo info)."""
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                meta.width = img.width
                meta.height = img.height
                mode_to_bands = {"L": 1, "RGB": 3, "RGBA": 4}
                meta.band_count = mode_to_bands.get(img.mode, len(img.getbands()))
                meta.dtype = "uint8"
                meta.has_georeference = False
        except Exception as e:
            meta.validation_errors.append(f"Failed to read image: {e}")

    def _infer_modality(self, meta: ImageMetadata):
        """Infer image modality from band count and metadata."""
        if meta.band_count == 2:
            # Likely SAR (VV + VH)
            meta.modality = ImageModality.SAR
        elif meta.band_count == 3:
            meta.modality = ImageModality.RGB
        elif meta.band_count >= 4:
            meta.modality = ImageModality.MULTISPECTRAL
        elif meta.band_count == 1:
            # Could be single-pol SAR or panchromatic
            meta.modality = ImageModality.UNKNOWN
        else:
            meta.modality = ImageModality.UNKNOWN

    def _infer_sensor(self, meta: ImageMetadata):
        """Make a best-guess at the sensor from metadata clues."""
        if meta.band_count == 2 and meta.modality == ImageModality.SAR:
            if meta.resolution and meta.resolution[0] == 10.0:
                meta.sensor_guess = "Sentinel-1"
            else:
                meta.sensor_guess = "SAR (unknown sensor)"
        elif meta.band_count == 12:
            meta.sensor_guess = "Sentinel-2 (BigEarthNet v2, no B10)"
        elif meta.band_count == 13:
            meta.sensor_guess = "Sentinel-2 (full L2A)"
        elif meta.band_count == 4:
            if meta.resolution and meta.resolution[0] <= 1.0:
                meta.sensor_guess = "Cartosat-2S (possible)"
            else:
                meta.sensor_guess = "Optical 4-band (RGBNIR)"
        elif meta.band_count == 3:
            meta.sensor_guess = "RGB (sensor unknown)"

    def _compute_overlap(
        self, bounds_a: dict, bounds_b: dict
    ) -> float:
        """Compute the geographic overlap ratio between two bounding boxes.

        Returns fraction of the smaller box that overlaps with the larger.
        """
        # Intersection
        x_left = max(bounds_a["left"], bounds_b["left"])
        y_bottom = max(bounds_a["bottom"], bounds_b["bottom"])
        x_right = min(bounds_a["right"], bounds_b["right"])
        y_top = min(bounds_a["top"], bounds_b["top"])

        if x_right <= x_left or y_top <= y_bottom:
            return 0.0

        intersection = (x_right - x_left) * (y_top - y_bottom)

        area_a = (bounds_a["right"] - bounds_a["left"]) * \
                 (bounds_a["top"] - bounds_a["bottom"])
        area_b = (bounds_b["right"] - bounds_b["left"]) * \
                 (bounds_b["top"] - bounds_b["bottom"])

        # Overlap as fraction of smaller area
        smaller = min(area_a, area_b)
        if smaller <= 0:
            return 0.0

        return intersection / smaller
