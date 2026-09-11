"""
SatQuery AI - B2 Centralized Input Validation Pipeline
======================================================
Coordinates modular validators and provides a unified entry point for F9 / FR1 / FR2
input verification.
"""

from __future__ import annotations

from typing import List, Optional, Union

from satquery_b2.ontology.schema import OntologyImage, PairType
from satquery_b2.validation.errors import (
    InputValidationError,
    ValidationErrorCode,
    ValidationErrorDetail,
    ValidationResult,
)
from satquery_b2.validation.rules import (
    ValidationConfig,
    validate_bands,
    validate_crs,
    validate_dimensions,
    validate_format,
    validate_optical_sar_pair,
    validate_resolution,
)

# Deterministic Error Precedence Hierarchy
ERROR_PRECEDENCE = [
    ValidationErrorCode.INVALID_FORMAT,
    ValidationErrorCode.UNSUPPORTED_BENCHMARK_FORMAT,
    ValidationErrorCode.UNREADABLE_RASTER,
    ValidationErrorCode.INCOMPATIBLE_MODALITY,
    ValidationErrorCode.MISSING_CRS,
    ValidationErrorCode.UNSUPPORTED_CRS,
    ValidationErrorCode.DIMENSION_MISMATCH,
    ValidationErrorCode.INVALID_BAND_CONFIGURATION,
    ValidationErrorCode.MISSING_BAND,
    ValidationErrorCode.RESOLUTION_MISMATCH,
    ValidationErrorCode.SPATIAL_MISMATCH,
    ValidationErrorCode.COREGISTRATION_FAILURE,
    ValidationErrorCode.INVALID_CLOUD_COVER,
    ValidationErrorCode.MISSING_REQUIRED_METADATA,
]


def _sort_errors_by_precedence(errors: List[ValidationErrorDetail]) -> List[ValidationErrorDetail]:
    """Sorts errors deterministically according to standardized precedence."""
    def get_rank(err: ValidationErrorDetail) -> int:
        try:
            return ERROR_PRECEDENCE.index(err.error_code)
        except ValueError:
            return len(ERROR_PRECEDENCE) + 1

    return sorted(errors, key=get_rank)


def validate_single_image(
    image: OntologyImage,
    config: Optional[ValidationConfig] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Validates a single `OntologyImage` against scientific, format, CRS, and dimension rules.

    Parameters
    ----------
    image : OntologyImage
        The canonical ontology image to validate.
    config : ValidationConfig, optional
        Custom validation configuration.
    strict : bool, default False
        If True, raises `InputValidationError` on validation failure.

    Returns
    -------
    ValidationResult
        Detailed validation report with errors, warnings, and status.
    """
    cfg = config or ValidationConfig()
    errors: List[ValidationErrorDetail] = []
    warnings: List[str] = []

    # 1. Format validation
    errors.extend(validate_format(image, cfg))

    # 2. CRS validation
    errors.extend(validate_crs(image, cfg))

    # 3. Dimensions validation
    errors.extend(validate_dimensions(image, cfg))

    # 4. Band configuration validation
    errors.extend(validate_bands(image, cfg))

    # 5. Resolution validation
    errors.extend(validate_resolution(image, cfg))

    # Warnings for non-fatal metadata issues
    if image.cloud_cover_percentage is not None and image.cloud_cover_percentage > 50.0:
        warnings.append(f"High cloud cover detected ({image.cloud_cover_percentage:.1f}%), which may reduce optical model accuracy")

    if not image.sensor:
        warnings.append("Sensor metadata not specified; default processing assumptions apply")

    # Sort errors by precedence
    sorted_errors = _sort_errors_by_precedence(errors)
    is_valid = len(sorted_errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        errors=sorted_errors,
        warnings=warnings,
        context={"image_id": image.image_id, "modality": image.modality.value, "format": image.file_format},
    )

    if strict and not is_valid:
        raise InputValidationError(result)

    return result


def validate_image_pair(
    optical_image: OntologyImage,
    sar_image: OntologyImage,
    pair_type: Union[PairType, str] = PairType.OPTICAL_SAR,
    config: Optional[ValidationConfig] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Validates a pair of images (Optical + SAR or Bi-temporal) for cross-modal or change inference.

    Parameters
    ----------
    optical_image : OntologyImage
        First image instance (optical image in Optical+SAR pair, or T1 in temporal pair).
    sar_image : OntologyImage
        Second image instance (SAR image in Optical+SAR pair, or T2 in temporal pair).
    pair_type : PairType | str, default PairType.OPTICAL_SAR
        Pair configuration type (OPTICAL_SAR, OPTICAL_TEMPORAL, SAR_TEMPORAL).
    config : ValidationConfig, optional
        Custom validation configuration.
    strict : bool, default False
        If True, raises `InputValidationError` on validation failure.

    Returns
    -------
    ValidationResult
        Detailed pair validation report.
    """
    cfg = config or ValidationConfig()
    p_type = PairType.from_str(pair_type) if isinstance(pair_type, str) else pair_type
    errors: List[ValidationErrorDetail] = []
    warnings: List[str] = []

    # Validate individual images first
    res_opt = validate_single_image(optical_image, cfg, strict=False)
    res_sar = validate_single_image(sar_image, cfg, strict=False)

    errors.extend(res_opt.errors)
    errors.extend(res_sar.errors)
    warnings.extend([f"[Image 1] {w}" for w in res_opt.warnings])
    warnings.extend([f"[Image 2] {w}" for w in res_sar.warnings])

    # Validate pair compatibility
    pair_errors = validate_optical_sar_pair(optical_image, sar_image, cfg, pair_type=p_type)
    errors.extend(pair_errors)

    # Sort errors by precedence
    sorted_errors = _sort_errors_by_precedence(errors)
    is_valid = len(sorted_errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        errors=sorted_errors,
        warnings=warnings,
        context={
            "image1_id": optical_image.image_id,
            "image2_id": sar_image.image_id,
            "pair_type": p_type.value,
        },
    )

    if strict and not is_valid:
        raise InputValidationError(result)

    return result
