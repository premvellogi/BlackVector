"""
SatQuery AI - B2 Validation Errors and Codes
===========================================
Machine-readable validation error models and standardized error codes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationErrorCode(str, Enum):
    """Standard machine-readable error codes for SatQuery AI B2 validation."""
    INVALID_FORMAT = "INVALID_FORMAT"
    UNREADABLE_RASTER = "UNREADABLE_RASTER"
    MISSING_CRS = "MISSING_CRS"
    UNSUPPORTED_CRS = "UNSUPPORTED_CRS"
    MISSING_BAND = "MISSING_BAND"
    INVALID_BAND_CONFIGURATION = "INVALID_BAND_CONFIGURATION"
    RESOLUTION_MISMATCH = "RESOLUTION_MISMATCH"
    SPATIAL_MISMATCH = "SPATIAL_MISMATCH"
    COREGISTRATION_FAILURE = "COREGISTRATION_FAILURE"
    UNSUPPORTED_BENCHMARK_FORMAT = "UNSUPPORTED_BENCHMARK_FORMAT"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    INVALID_CLOUD_COVER = "INVALID_CLOUD_COVER"
    MISSING_REQUIRED_METADATA = "MISSING_REQUIRED_METADATA"
    INCOMPATIBLE_MODALITY = "INCOMPATIBLE_MODALITY"
    FILE_SIZE_LIMIT_EXCEEDED = "FILE_SIZE_LIMIT_EXCEEDED"


@dataclass
class ValidationErrorDetail:
    """Detailed machine-readable validation error record."""
    error_code: ValidationErrorCode
    field: str
    message: str
    actual_value: Any = None
    expected_value: Any = None
    remediation_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "field": self.field,
            "message": self.message,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "remediation_hint": self.remediation_hint,
        }


@dataclass
class ValidationResult:
    """Result of validating an OntologyImage or image pair."""
    is_valid: bool
    errors: List[ValidationErrorDetail]
    warnings: List[str]
    context: Dict[str, Any]

    @property
    def primary_error(self) -> Optional[ValidationErrorDetail]:
        """Returns the highest precedence error if any."""
        return self.errors[0] if self.errors else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "context": self.context,
        }


class InputValidationError(Exception):
    """Exception raised when an image or pair fails validation in strict mode."""
    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        primary = validation_result.primary_error
        msg = primary.message if primary else "Input validation failed"
        super().__init__(f"[{primary.error_code.value if primary else 'VALIDATION_ERROR'}] {msg}")
