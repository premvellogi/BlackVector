"""
SatQuery AI - B2 Public API Facade
==================================
Unified, clean entry points for downstream blocks (B1 Orchestrator, VQA,
Grounding, Change Detection, Optical-SAR).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from satquery_b2.confidence.aggregator import (
    ConfidenceReport,
    aggregate_confidence,
)
from satquery_b2.confidence.config import ConfidenceConfig
from satquery_b2.disagreement.models import (
    DisagreementReport,
    ModelOutput,
)
from satquery_b2.disagreement.surfacing import detect_disagreement
from satquery_b2.extraction.raster_metadata import extract_raster_metadata
from satquery_b2.ontology.schema import (
    Modality,
    OntologyImage,
    PairedOntologyImage,
    PairType,
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
)
from satquery_b2.reliability.retry_wrapper import (
    RetryConfig,
    execute_with_reliability,
    execute_with_retry,
)
from satquery_b2.validation.errors import ValidationResult
from satquery_b2.validation.rules import ValidationConfig
from satquery_b2.validation.validator import (
    validate_image_pair,
    validate_single_image,
)

T = TypeVar("T")


def build_ontology(
    file_path: str,
    source_id: Optional[str] = None,
    image_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    default_modality: Optional[Modality] = None,
) -> OntologyImage:
    """
    Extracts metadata from a raster file and returns a validated canonical OntologyImage.
    """
    return extract_raster_metadata(
        file_path=file_path,
        source_id=source_id,
        image_id=image_id,
        dataset_name=dataset_name,
        default_modality=default_modality,
    )


def validate_input(
    image: OntologyImage,
    config: Optional[ValidationConfig] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Validates a single OntologyImage against F9 / FR1 / FR2 constraints.
    """
    return validate_single_image(image=image, config=config, strict=strict)


def validate_pair(
    image_a: OntologyImage,
    image_b: OntologyImage,
    pair_type: Union[PairType, str] = PairType.OPTICAL_SAR,
    config: Optional[ValidationConfig] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Validates a pair of images (Optical + SAR, Optical Temporal, or SAR Temporal).
    """
    return validate_image_pair(
        optical_image=image_a,
        sar_image=image_b,
        pair_type=pair_type,
        config=config,
        strict=strict,
    )


def execute_ml_service(
    circuit_breaker: CircuitBreaker,
    func: Callable[..., T],
    *args: Any,
    retry_config: Optional[RetryConfig] = None,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> T:
    """
    Executes an inference call against a laptop-served ML service with integrated circuit breaking and retries.
    """
    return execute_with_reliability(
        circuit_breaker,
        func,
        *args,
        retry_config=retry_config,
        correlation_id=correlation_id,
        **kwargs,
    )


def surface_disagreement(
    model_outputs: List[ModelOutput],
    spatial_iou_threshold: float = 0.50,
) -> DisagreementReport:
    """
    Detects and reports agreement, partial agreement, or material disagreement across model outputs.
    """
    return detect_disagreement(model_outputs=model_outputs, spatial_iou_threshold=spatial_iou_threshold)


def compute_confidence(
    model_confidence: Optional[float] = None,
    optical_model_confidence: Optional[float] = None,
    sar_model_confidence: Optional[float] = None,
    agreement_score: Optional[float] = None,
    cloud_cover_percentage: Optional[float] = None,
    metadata_quality: Optional[Dict[str, Any]] = None,
    spatial_alignment_iou: Optional[float] = None,
    disagreement_detected: bool = False,
    disagreement_severity: str = "NONE",
    config: Optional[ConfidenceConfig] = None,
) -> ConfidenceReport:
    """
    Aggregates composite confidence score according to §6.3 specification.
    """
    return aggregate_confidence(
        model_confidence=model_confidence,
        optical_model_confidence=optical_model_confidence,
        sar_model_confidence=sar_model_confidence,
        agreement_score=agreement_score,
        cloud_cover_percentage=cloud_cover_percentage,
        metadata_quality=metadata_quality,
        spatial_alignment_iou=spatial_alignment_iou,
        disagreement_detected=disagreement_detected,
        disagreement_severity=disagreement_severity,
        config=config,
    )


def create_provenance(
    image_or_path: Union[OntologyImage, str],
    model_name: str,
    task: str,
    model_version: Optional[str] = None,
    checkpoint: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ProvenanceRecord:
    """
    Creates an immutable provenance record linking input file hash, model version, checkpoint, and timestamp.
    """
    tracker = ProvenanceTracker(run_id=run_id)
    return tracker.create_provenance_record(
        image_or_path=image_or_path,
        model_name=model_name,
        task=task,
        model_version=model_version,
        checkpoint=checkpoint,
    )
