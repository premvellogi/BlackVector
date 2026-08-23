"""
Sentinel-1 SAR Preprocessing for SatQuery AI.

SAR imagery has fundamentally different value distributions than optical:
- SAR backscatter values are typically in dB (decibel) scale
- Sentinel-1 provides VV and VH polarization channels (dual-pol)
- RISAT (ISRO evaluation) may provide different polarization configurations
- No "natural color" — visualization requires specific processing

BigEarthNet v2 Sentinel-1 Format:
- 2 bands: VV and VH polarization
- Patch size: 120 x 120 pixels at 10m resolution (matching S2 10m bands)
- Values: backscatter coefficient in dB (typically -25 to +5 dB range)

For the VLM:
- SAR is converted to a pseudo-RGB visualization (VV, VH, VV/VH ratio)
  so InternVL2-2B can process it as a standard image.
- The raw dB values are also preserved for the fusion head.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Sentinel-1 Band Specification
# =============================================================================

@dataclass(frozen=True)
class S1BandSpec:
    """Specification for a Sentinel-1 polarization channel."""
    name: str            # e.g., "VV"
    description: str
    typical_range_db: tuple[float, float]  # typical min/max in dB


S1_BANDS = [
    S1BandSpec("VV", "Co-polarization (Vertical-Vertical)", (-25.0, 5.0)),
    S1BandSpec("VH", "Cross-polarization (Vertical-Horizontal)", (-35.0, -5.0)),
]

# Per-channel normalization stats for Sentinel-1 (BigEarthNet v2 training set)
# Values in dB scale
# Source: Official BigEarthNet v2 training split (from ben_txt_datamodule.py)
S1_NORM_STATS = {
    "VV": {"mean": -12.643863677978516, "std": 5.133493900299072, "min": -25.0, "max": 5.0},
    "VH": {"mean": -19.352558135986328, "std": 5.590505599975586, "min": -35.0, "max": -5.0},
}


@dataclass
class S1PreprocessConfig:
    """Configuration for Sentinel-1 SAR preprocessing."""
    # Polarization channels to use
    use_vv: bool = True
    use_vh: bool = True

    # Normalization
    normalize_method: str = "standardize"  # "standardize" or "minmax"
    clip_range_db: Optional[tuple[float, float]] = (-30.0, 5.0)

    # VLM visualization
    # SAR → pseudo-RGB for VLM: channel 1 = VV, channel 2 = VH, channel 3 = VV/VH
    vlm_pseudo_rgb: bool = True
    vlm_tile_size: int = 448  # InternVL2 tile size


@dataclass
class S1PreprocessingProvenance:
    """Audit trail for SAR preprocessing decisions."""
    channels_used: list[str] = field(default_factory=list)
    normalize_method: str = ""
    clip_range_db: Optional[tuple[float, float]] = None
    original_shape: tuple = ()
    output_shape: tuple = ()
    input_value_range: dict = field(default_factory=dict)


class Sentinel1Preprocessor:
    """Sentinel-1 SAR preprocessing pipeline.

    Handles:
    - dB-scale normalization (SAR values are NOT reflectance)
    - Pseudo-RGB generation for VLM consumption
    - Raw channel extraction for fusion head
    - Provenance tracking
    """

    def __init__(self, config: Optional[S1PreprocessConfig] = None):
        self.config = config or S1PreprocessConfig()

    def process_geotiff(
        self,
        filepath: Union[str, Path],
        channel_names: Optional[list[str]] = None,
    ) -> dict:
        """Process a Sentinel-1 SAR GeoTIFF.

        Returns:
            Dict with keys:
                - 'vlm_rgb': pseudo-RGB np.ndarray (H, W, 3) uint8
                - 'sar_normalized': np.ndarray (C, H, W) float32
                - 'provenance': S1PreprocessingProvenance
                - 'metadata': dict
        """
        import rasterio

        filepath = Path(filepath)
        provenance = S1PreprocessingProvenance()

        with rasterio.open(filepath) as src:
            provenance.original_shape = (src.count, src.height, src.width)
            metadata = {
                "crs": str(src.crs),
                "bounds": dict(zip(["left", "bottom", "right", "top"], src.bounds)),
                "resolution": src.res,
                "band_count": src.count,
                "dtype": str(src.dtypes[0]),
            }
            all_data = src.read().astype(np.float32)
            num_bands = all_data.shape[0]

            if channel_names is None:
                if num_bands == 2:
                    channel_names = ["VV", "VH"]
                elif num_bands == 1:
                    channel_names = ["VV"]
                else:
                    channel_names = [f"SAR_{i}" for i in range(num_bands)]

        channels_dict = {}
        for i, name in enumerate(channel_names):
            if i < num_bands:
                channels_dict[name] = all_data[i]

        return self.process_channels(channels_dict, provenance, metadata)

    def process_channels(
        self,
        channels_dict: dict[str, np.ndarray],
        provenance: Optional[S1PreprocessingProvenance] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Process raw SAR channel arrays."""
        if provenance is None:
            provenance = S1PreprocessingProvenance()
        if metadata is None:
            metadata = {}

        # Record input value ranges
        for name, arr in channels_dict.items():
            provenance.input_value_range[name] = {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }

        # Select channels
        selected = {}
        if self.config.use_vv and "VV" in channels_dict:
            selected["VV"] = channels_dict["VV"]
        if self.config.use_vh and "VH" in channels_dict:
            selected["VH"] = channels_dict["VH"]

        # Fallback: use whatever channels are available
        if not selected:
            selected = channels_dict

        provenance.channels_used = list(selected.keys())
        provenance.normalize_method = self.config.normalize_method

        # Clip extreme values
        if self.config.clip_range_db:
            provenance.clip_range_db = self.config.clip_range_db
            for name in selected:
                lo, hi = self.config.clip_range_db
                selected[name] = np.clip(selected[name], lo, hi)

        # Normalize
        normalized = self._normalize(selected)

        # Stack to (C, H, W)
        channel_list = [normalized[n] for n in sorted(normalized.keys())]
        sar_normalized = np.stack(channel_list, axis=0)
        provenance.output_shape = sar_normalized.shape

        # VLM pseudo-RGB
        vlm_rgb = None
        if self.config.vlm_pseudo_rgb:
            vlm_rgb = self._create_pseudo_rgb(selected)

        return {
            "vlm_rgb": vlm_rgb,
            "sar_normalized": sar_normalized,
            "provenance": provenance,
            "metadata": metadata,
        }

    def _normalize(self, channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Normalize SAR channels."""
        result = {}
        for name, arr in channels.items():
            arr = arr.astype(np.float32)
            if self.config.normalize_method == "standardize":
                stats = S1_NORM_STATS.get(name)
                if stats:
                    result[name] = (arr - stats["mean"]) / (stats["std"] + 1e-8)
                else:
                    result[name] = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
            elif self.config.normalize_method == "minmax":
                stats = S1_NORM_STATS.get(name)
                if stats:
                    vmin, vmax = stats["min"], stats["max"]
                else:
                    vmin, vmax = np.min(arr), np.max(arr)
                if vmax - vmin > 1e-8:
                    result[name] = (arr - vmin) / (vmax - vmin)
                else:
                    result[name] = np.zeros_like(arr)
            else:
                result[name] = arr
        return result

    def _create_pseudo_rgb(self, channels: dict[str, np.ndarray]) -> np.ndarray:
        """Create a pseudo-RGB visualization of SAR data for VLM input.

        Standard false-color composite:
            R = VV
            G = VH
            B = VV / VH (ratio, highlights scattering differences)

        This gives the VLM visual structure to reason about, even though
        SAR has no natural color. The ratio channel is particularly useful
        for distinguishing water bodies, vegetation, and built-up areas.
        """
        vv = channels.get("VV")
        vh = channels.get("VH")

        if vv is None and vh is None:
            raise ValueError("Need at least VV or VH channel for pseudo-RGB.")

        h, w = (vv if vv is not None else vh).shape

        if vv is not None and vh is not None:
            # Full composite: VV, VH, VV/VH ratio
            ratio = np.divide(
                vv, vh,
                out=np.ones_like(vv),
                where=np.abs(vh) > 1e-8
            )
            rgb_raw = np.stack([vv, vh, ratio], axis=-1)
        elif vv is not None:
            # Single-pol: replicate to 3 channels
            rgb_raw = np.stack([vv, vv, vv], axis=-1)
        else:
            rgb_raw = np.stack([vh, vh, vh], axis=-1)

        # Scale to uint8 via per-channel percentile clipping
        rgb_uint8 = np.zeros_like(rgb_raw, dtype=np.uint8)
        for c in range(3):
            ch = rgb_raw[:, :, c]
            p2, p98 = np.percentile(ch, [2, 98])
            if p98 - p2 > 1e-8:
                ch = np.clip(ch, p2, p98)
                ch = (ch - p2) / (p98 - p2) * 255.0
            else:
                ch = np.zeros_like(ch)
            rgb_uint8[:, :, c] = ch.astype(np.uint8)

        return rgb_uint8
