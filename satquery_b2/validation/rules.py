"""
SatQuery AI - B2 Validation Rules & Checks
=========================================
Modular, deterministic validation rules for single images and cross-modal pairs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from satquery_b2.ontology.schema import Modality, OntologyImage, PairType, SpatialBounds
from satquery_b2.validation.errors import ValidationErrorCode, ValidationErrorDetail

# Default supported coordinate reference systems
DEFAULT_SUPPORTED_EPSG: Set[int] = {
    4326,   # WGS 84 (lat/lon)
    3857,   # Web Mercator
    32601, 32602, 32603, 32604, 32605, 32606, 32607, 32608, 32609, 32610,
    32611, 32612, 32613, 32614, 32615, 32616, 32617, 32618, 32619, 32620,
    32621, 32622, 32623, 32624, 32625, 32626, 32627, 32628, 32629, 32630,
    32631, 32632, 32633, 32634, 32635, 32636, 32637, 32638, 32639, 32640,
    32641, 32642, 32643, 32644, 32645, 32646, 32647, 32648, 32649, 32650,
    32651, 32652, 32653, 32654, 32655, 32656, 32657, 32658, 32659, 32660,
    32701, 32702, 32703, 32704, 32705, 32706, 32707, 32708, 32709, 32710,
    32732, 32733, 32734, 32735, 32736, 32737, 32738, 32739, 32740, 32743,
}


@dataclass
class ValidationConfig:
    """Configurable constraints for image and pair validation."""
    allow_benchmark_formats: bool = False
    allowed_formats: Set[str] = field(default_factory=lambda: {"GEOTIFF", "TIFF"})
    require_crs: bool = True
    supported_epsg_codes: Set[int] = field(default_factory=lambda: set(DEFAULT_SUPPORTED_EPSG))
    min_resolution_m: Optional[float] = 0.01
    max_resolution_m: Optional[float] = 1000.0
    min_width: int = 16
    min_height: int = 16
    max_width: int = 65536
    max_height: int = 65536
    required_optical_bands: Optional[List[str]] = None
    required_sar_bands: Optional[List[str]] = None
    required_band_count: Optional[int] = None
    min_spatial_overlap_iou: float = 0.50
    max_resolution_ratio_difference: float = 0.25  # Max 25% resolution difference between pair
    grid_tolerance_ratio: float = 0.05             # Max 5% pixel offset for co-registration


def validate_format(image: OntologyImage, config: ValidationConfig) -> List[ValidationErrorDetail]:
    """Validates image format against allowed formats and benchmark contracts."""
    fmt = image.file_format.upper()
    if fmt in ("PNG", "JPEG", "JPG"):
        if not config.allow_benchmark_formats:
            return [
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.UNSUPPORTED_BENCHMARK_FORMAT,
                    field="file_format",
                    message=f"Format '{fmt}' is only permitted when allow_benchmark_formats=True",
                    actual_value=fmt,
                    expected_value="GEOTIFF / TIFF",
                    remediation_hint="Enable benchmark format flag or provide native GeoTIFF/TIFF imagery",
                )
            ]
        return []

    if fmt not in config.allowed_formats:
        return [
            ValidationErrorDetail(
                error_code=ValidationErrorCode.INVALID_FORMAT,
                field="file_format",
                message=f"Unsupported format '{fmt}'. Allowed: {sorted(list(config.allowed_formats))}",
                actual_value=fmt,
                expected_value=sorted(list(config.allowed_formats)),
                remediation_hint="Convert input raster to standard GeoTIFF/TIFF",
            )
        ]
    return []


def validate_crs(image: OntologyImage, config: ValidationConfig) -> List[ValidationErrorDetail]:
    """Validates presence and support of CRS and EPSG identifiers."""
    errors: List[ValidationErrorDetail] = []
    if config.require_crs and not image.crs and not image.epsg:
        errors.append(
            ValidationErrorDetail(
                error_code=ValidationErrorCode.MISSING_CRS,
                field="crs",
                message="Image lacks spatial reference system (CRS / EPSG)",
                actual_value=None,
                expected_value="Valid CRS (e.g. EPSG:4326 or EPSG:32643)",
                remediation_hint="Georeference the image or supply EPSG code in metadata",
            )
        )
        return errors

    if image.epsg and config.supported_epsg_codes:
        if image.epsg not in config.supported_epsg_codes:
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.UNSUPPORTED_CRS,
                    field="epsg",
                    message=f"EPSG code {image.epsg} is not in the supported projection catalog",
                    actual_value=image.epsg,
                    expected_value=f"One of {len(config.supported_epsg_codes)} supported EPSG codes",
                    remediation_hint="Reproject to standard WGS84 (EPSG:4326) or UTM zone",
                )
            )
    return errors


def validate_dimensions(image: OntologyImage, config: ValidationConfig) -> List[ValidationErrorDetail]:
    """Validates pixel width and height constraints."""
    errors: List[ValidationErrorDetail] = []
    if image.width < config.min_width or image.height < config.min_height:
        errors.append(
            ValidationErrorDetail(
                error_code=ValidationErrorCode.DIMENSION_MISMATCH,
                field="dimensions",
                message=f"Image dimensions ({image.width}x{image.height}) below minimum ({config.min_width}x{config.min_height})",
                actual_value=(image.width, image.height),
                expected_value=(config.min_width, config.min_height),
                remediation_hint="Provide imagery with sufficient spatial extent",
            )
        )
    elif image.width > config.max_width or image.height > config.max_height:
        errors.append(
            ValidationErrorDetail(
                error_code=ValidationErrorCode.DIMENSION_MISMATCH,
                field="dimensions",
                message=f"Image dimensions ({image.width}x{image.height}) exceed maximum ({config.max_width}x{config.max_height})",
                actual_value=(image.width, image.height),
                expected_value=(config.max_width, config.max_height),
                remediation_hint="Tile or chip the image before inference",
            )
        )
    return errors


def validate_bands(image: OntologyImage, config: ValidationConfig) -> List[ValidationErrorDetail]:
    """Validates band counts and required band configurations."""
    errors: List[ValidationErrorDetail] = []
    if config.required_band_count is not None:
        if image.band_count != config.required_band_count:
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.INVALID_BAND_CONFIGURATION,
                    field="band_count",
                    message=f"Image has {image.band_count} bands, expected {config.required_band_count}",
                    actual_value=image.band_count,
                    expected_value=config.required_band_count,
                    remediation_hint=f"Supply a raster with exactly {config.required_band_count} bands",
                )
            )

    # Check named required bands
    required_bands = config.required_optical_bands if image.is_optical() else (
        config.required_sar_bands if image.is_sar() else None
    )
    if required_bands and image.bands:
        present_band_ids = {b.band_id.upper() for b in image.bands if b.band_id}
        missing_bands = [b for b in required_bands if b.upper() not in present_band_ids]
        if missing_bands:
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.MISSING_BAND,
                    field="bands",
                    message=f"Missing required bands: {missing_bands}",
                    actual_value=sorted(list(present_band_ids)),
                    expected_value=required_bands,
                    remediation_hint=f"Ensure raster includes channels: {missing_bands}",
                )
            )
    return errors


def validate_resolution(image: OntologyImage, config: ValidationConfig) -> List[ValidationErrorDetail]:
    """Validates spatial resolution bounds."""
    if image.spatial_resolution_m is None:
        return []

    errors: List[ValidationErrorDetail] = []
    if config.min_resolution_m is not None and image.spatial_resolution_m < config.min_resolution_m:
        errors.append(
            ValidationErrorDetail(
                error_code=ValidationErrorCode.RESOLUTION_MISMATCH,
                field="spatial_resolution_m",
                message=f"Spatial resolution {image.spatial_resolution_m}m is below minimum {config.min_resolution_m}m",
                actual_value=image.spatial_resolution_m,
                expected_value=f">={config.min_resolution_m}m",
                remediation_hint="Verify pixel scale in GeoTIFF metadata",
            )
        )
    elif config.max_resolution_m is not None and image.spatial_resolution_m > config.max_resolution_m:
        errors.append(
            ValidationErrorDetail(
                error_code=ValidationErrorCode.RESOLUTION_MISMATCH,
                field="spatial_resolution_m",
                message=f"Spatial resolution {image.spatial_resolution_m}m exceeds maximum {config.max_resolution_m}m",
                actual_value=image.spatial_resolution_m,
                expected_value=f"<={config.max_resolution_m}m",
                remediation_hint="Provide higher resolution imagery suited for target model",
            )
        )
    return errors


def normalize_crs_representation(crs: Optional[str], epsg: Optional[int] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Normalizes a CRS string and EPSG code to canonical form.
    Returns (canonical_epsg, canonical_crs_name).
    """
    if epsg is not None and epsg > 0:
        return epsg, f"EPSG:{epsg}"

    if not crs or not crs.strip():
        return None, None

    s = crs.strip().upper()

    # Extract EPSG prefix e.g. "EPSG:32643"
    if "EPSG:" in s:
        code_part = s.split("EPSG:")[-1].split()[0].strip()
        if code_part.isdigit():
            code = int(code_part)
            return code, f"EPSG:{code}"

    # Known standard aliases
    if s in ("WGS 84", "WGS84", "CRS:84", "OGC:CRS84", "4326", "EPSG:4326", "WGS 84 (GEOGRAPHIC LAT/LON)"):
        return 4326, "EPSG:4326"
    if s in ("WGS 84 / PSEUDO-MERCATOR", "WEB MERCATOR", "3857", "EPSG:3857", "EPSG:900913"):
        return 3857, "EPSG:3857"

    # Match UTM zone patterns e.g. "WGS 84 / UTM ZONE 43N" -> 32643
    m = re.search(r"UTM\s*(?:ZONE\s*)?(\d{1,2})\s*([NS])", s)
    if m:
        zone = int(m.group(1))
        hemi = m.group(2)
        if 1 <= zone <= 60:
            code = (32600 + zone) if hemi == "N" else (32700 + zone)
            return code, f"EPSG:{code}"

    return None, s


def are_crs_compatible(img1: OntologyImage, img2: OntologyImage) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether two images have compatible/equivalent coordinate reference systems.
    Prioritizes canonical EPSG integer matching, falling back to normalized CRS representation.
    """
    epsg1, norm1 = normalize_crs_representation(img1.crs, img1.epsg)
    epsg2, norm2 = normalize_crs_representation(img2.crs, img2.epsg)

    # Both have canonical EPSG code
    if epsg1 is not None and epsg2 is not None:
        if epsg1 == epsg2:
            return True, None
        return False, f"EPSG mismatch: {epsg1} vs {epsg2}"

    # One or both have normalized strings
    if norm1 and norm2:
        if norm1 == norm2:
            return True, None
        return False, f"CRS mismatch: '{norm1}' vs '{norm2}'"

    return False, "Missing spatial reference system (CRS/EPSG)"


def validate_optical_sar_pair(
    optical_image: OntologyImage,
    sar_image: OntologyImage,
    config: ValidationConfig,
    pair_type: PairType = PairType.OPTICAL_SAR,
) -> List[ValidationErrorDetail]:
    """
    Pair validation supporting Optical+SAR, Optical Temporal, and SAR Temporal pairs:
    - Verifies modality constraints based on pair_type
    - Verifies CRS equivalence (via EPSG and normalized CRS strings)
    - Calculates geographical overlap IoU
    - Compares pixel resolutions
    - Checks grid co-registration
    """
    errors: List[ValidationErrorDetail] = []

    # 1. Modality check based on pair_type
    if pair_type == PairType.OPTICAL_SAR:
        if not optical_image.is_optical():
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.INCOMPATIBLE_MODALITY,
                    field="optical_image.modality",
                    message=f"Expected optical modality, got '{optical_image.modality.value}'",
                    actual_value=optical_image.modality.value,
                    expected_value="optical",
                    remediation_hint="Provide an optical/multispectral raster as optical input",
                )
            )
        if not sar_image.is_sar():
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.INCOMPATIBLE_MODALITY,
                    field="sar_image.modality",
                    message=f"Expected SAR modality, got '{sar_image.modality.value}'",
                    actual_value=sar_image.modality.value,
                    expected_value="sar",
                    remediation_hint="Provide a SAR raster as SAR input",
                )
            )
    elif pair_type == PairType.OPTICAL_TEMPORAL:
        if not (optical_image.is_optical() and sar_image.is_optical()):
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.INCOMPATIBLE_MODALITY,
                    field="pair.modality",
                    message=f"OPTICAL_TEMPORAL requires both images to be optical, got '{optical_image.modality.value}' and '{sar_image.modality.value}'",
                    actual_value=(optical_image.modality.value, sar_image.modality.value),
                    expected_value="optical + optical",
                    remediation_hint="Provide two optical scenes for bi-temporal optical pair",
                )
            )
    elif pair_type == PairType.SAR_TEMPORAL:
        if not (optical_image.is_sar() and sar_image.is_sar()):
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.INCOMPATIBLE_MODALITY,
                    field="pair.modality",
                    message=f"SAR_TEMPORAL requires both images to be SAR, got '{optical_image.modality.value}' and '{sar_image.modality.value}'",
                    actual_value=(optical_image.modality.value, sar_image.modality.value),
                    expected_value="sar + sar",
                    remediation_hint="Provide two SAR scenes for bi-temporal SAR pair",
                )
            )

    # 2. CRS Compatibility using canonical EPSG and normalized representation
    is_compat, reason = are_crs_compatible(optical_image, sar_image)
    if not is_compat:
        if config.require_crs and (not optical_image.crs and not optical_image.epsg or not sar_image.crs and not sar_image.epsg):
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.MISSING_CRS,
                    field="crs",
                    message="Both images in pair must have valid spatial reference systems (CRS/EPSG)",
                    actual_value={"img1_crs": optical_image.crs, "img2_crs": sar_image.crs},
                    expected_value="Valid CRS in both images",
                    remediation_hint="Supply georeferenced rasters",
                )
            )
        else:
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.UNSUPPORTED_CRS,
                    field="crs",
                    message=f"CRS mismatch between pair: {reason}",
                    actual_value=(optical_image.crs or optical_image.epsg, sar_image.crs or sar_image.epsg),
                    expected_value="Equivalent CRS / EPSG",
                    remediation_hint="Reproject images to matching CRS before inference",
                )
            )

    # 3. Spatial Footprint Overlap
    if optical_image.spatial_bounds and sar_image.spatial_bounds:
        iou = optical_image.spatial_bounds.calculate_intersection_iou(sar_image.spatial_bounds)
        if iou < config.min_spatial_overlap_iou:
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.SPATIAL_MISMATCH,
                    field="spatial_bounds",
                    message=f"Spatial overlap IoU ({iou:.3f}) below required threshold ({config.min_spatial_overlap_iou:.3f})",
                    actual_value=round(iou, 4),
                    expected_value=f">={config.min_spatial_overlap_iou}",
                    remediation_hint="Ensure both images cover the same geographical target area",
                )
            )

    # 4. Spatial Resolution Compatibility
    if optical_image.spatial_resolution_m and sar_image.spatial_resolution_m:
        res_opt = optical_image.spatial_resolution_m
        res_sar = sar_image.spatial_resolution_m
        diff_ratio = abs(res_opt - res_sar) / max(res_opt, res_sar)
        if diff_ratio > config.max_resolution_ratio_difference:
            errors.append(
                ValidationErrorDetail(
                    error_code=ValidationErrorCode.RESOLUTION_MISMATCH,
                    field="spatial_resolution_m",
                    message=f"Resolution mismatch: Image 1 ({res_opt}m) vs Image 2 ({res_sar}m) diff ratio {diff_ratio:.2%}",
                    actual_value=(res_opt, res_sar),
                    expected_value=f"Diff ratio <= {config.max_resolution_ratio_difference:.0%}",
                    remediation_hint="Resample imagery to matching pixel scale",
                )
            )

    # 5. Grid Co-Registration Check
    if optical_image.affine_transform and sar_image.affine_transform:
        aff_opt = optical_image.affine_transform
        aff_sar = sar_image.affine_transform
        pixel_size = abs(aff_opt.a)
        offset_x = abs(aff_opt.c - aff_sar.c)
        offset_y = abs(aff_opt.f - aff_sar.f)
        if pixel_size > 0:
            offset_ratio_x = (offset_x % pixel_size) / pixel_size
            offset_ratio_y = (offset_y % pixel_size) / pixel_size
            if (offset_ratio_x > config.grid_tolerance_ratio and offset_ratio_x < (1.0 - config.grid_tolerance_ratio)) or \
               (offset_ratio_y > config.grid_tolerance_ratio and offset_ratio_y < (1.0 - config.grid_tolerance_ratio)):
                errors.append(
                    ValidationErrorDetail(
                        error_code=ValidationErrorCode.COREGISTRATION_FAILURE,
                        field="affine_transform",
                        message=f"Pixel grids are misaligned (subpixel offset x={offset_ratio_x:.3f}, y={offset_ratio_y:.3f})",
                        actual_value=(round(offset_ratio_x, 3), round(offset_ratio_y, 3)),
                        expected_value="Pixel grid aligned (tolerance <= 5%)",
                        remediation_hint="Co-register images to identical pixel grid alignment",
                    )
                )

    return errors
