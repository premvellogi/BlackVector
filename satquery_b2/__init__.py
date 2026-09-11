"""
SatQuery AI - B2 Subsystem (Ontology + Reliability + Confidence)
================================================================
B2 ML infrastructure, validation, reliability harness, confidence aggregation,
disagreement surfacing, and provenance layer.
"""

from satquery_b2.api import (
    build_ontology,
    compute_confidence,
    create_provenance,
    execute_ml_service,
    surface_disagreement,
    validate_input,
    validate_pair,
)
from satquery_b2.confidence.aggregator import (
    ConfidenceReport,
    aggregate_confidence,
)
from satquery_b2.confidence.config import ConfidenceConfig
from satquery_b2.disagreement.models import (
    BoundingBox,
    DisagreementReport,
    DisagreementStatus,
    ModelOutput,
)
from satquery_b2.disagreement.surfacing import detect_disagreement
from satquery_b2.extraction.raster_metadata import extract_raster_metadata
from satquery_b2.ontology.schema import (
    AffineTransform,
    BandInfo,
    FileFormat,
    Modality,
    OntologyImage,
    PairedOntologyImage,
    PairType,
    SpatialBounds,
)
from satquery_b2.provenance.hasher import calculate_file_hash, verify_file_hash
from satquery_b2.provenance.models import (
    PairedProvenanceRecord,
    ProvenanceRecord,
)
from satquery_b2.provenance.tracker import ProvenanceTracker
from satquery_b2.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from satquery_b2.reliability.exceptions import (
    MaxRetriesExceededError,
    MLServiceCircuitOpenError,
    ReliabilityError,
)
from satquery_b2.reliability.retry_wrapper import (
    RetryConfig,
    execute_with_reliability,
    execute_with_retry,
)
from satquery_b2.validation.errors import (
    InputValidationError,
    ValidationErrorCode,
    ValidationErrorDetail,
    ValidationResult,
)
from satquery_b2.validation.rules import ValidationConfig
from satquery_b2.validation.validator import (
    validate_image_pair,
    validate_single_image,
)

__version__ = "1.0.0"

__all__ = [
    # Facade API
    "build_ontology",
    "validate_input",
    "validate_pair",
    "execute_ml_service",
    "surface_disagreement",
    "compute_confidence",
    "create_provenance",
    # Ontology
    "OntologyImage",
    "PairedOntologyImage",
    "PairType",
    "Modality",
    "FileFormat",
    "SpatialBounds",
    "AffineTransform",
    "BandInfo",
    # Extraction
    "extract_raster_metadata",
    # Validation
    "validate_single_image",
    "validate_image_pair",
    "ValidationConfig",
    "ValidationResult",
    "ValidationErrorDetail",
    "ValidationErrorCode",
    "InputValidationError",
    # Reliability
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "RetryConfig",
    "execute_with_retry",
    "execute_with_reliability",
    "ReliabilityError",
    "MLServiceCircuitOpenError",
    "MaxRetriesExceededError",
    # Confidence
    "ConfidenceReport",
    "ConfidenceConfig",
    "aggregate_confidence",
    # Disagreement
    "ModelOutput",
    "BoundingBox",
    "DisagreementReport",
    "DisagreementStatus",
    "detect_disagreement",
    # Provenance
    "ProvenanceRecord",
    "PairedProvenanceRecord",
    "ProvenanceTracker",
    "calculate_file_hash",
    "verify_file_hash",
]
