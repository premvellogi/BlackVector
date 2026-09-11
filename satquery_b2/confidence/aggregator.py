"""
SatQuery AI - B2 Confidence Aggregator (§6.3)
==============================================
Produces a deterministic, audit-traceable final confidence score combining model
confidence, cross-modal agreement, and geospatial input quality factors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from satquery_b2.confidence.config import ConfidenceConfig


@dataclass
class ConfidenceReport:
    """Standardized §6.3 Confidence Evaluation Output."""
    model_confidence: float
    agreement_score: Optional[float]
    input_quality_score: float
    final_confidence: float
    confidence_components: Dict[str, Any]
    disagreement_detected: bool
    disagreement_penalty_applied: float
    confidence_method_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_confidence": round(self.model_confidence, 4),
            "agreement_score": round(self.agreement_score, 4) if self.agreement_score is not None else None,
            "input_quality_score": round(self.input_quality_score, 4),
            "final_confidence": round(self.final_confidence, 4),
            "confidence_components": self.confidence_components,
            "disagreement_detected": self.disagreement_detected,
            "disagreement_penalty_applied": round(self.disagreement_penalty_applied, 4),
            "confidence_method_version": self.confidence_method_version,
        }


def _clamp_01(val: float) -> float:
    """Clamps a floating point value to the [0.0, 1.0] interval."""
    if math.isnan(val):
        return 0.0
    return max(0.0, min(1.0, float(val)))


def calculate_input_quality_score(
    cloud_cover_percentage: Optional[float] = None,
    metadata_quality: Optional[Dict[str, Any]] = None,
    spatial_alignment_iou: Optional[float] = None,
    is_optical: bool = True,
    config: Optional[ConfidenceConfig] = None,
) -> tuple[float, Dict[str, float]]:
    """
    Computes normalized input quality factor from geospatial quality attributes.
    SAR images are invariant to cloud cover.
    """
    cfg = config or ConfidenceConfig()
    components: Dict[str, float] = {}

    # 1. Cloud cover factor
    if is_optical and cloud_cover_percentage is not None:
        clamped_cloud = _clamp_01(cloud_cover_percentage / 100.0)
        cloud_factor = 1.0 - clamped_cloud
    else:
        # SAR is unaffected by clouds; optical with no cloud metadata assumes standard 1.0
        cloud_factor = 1.0
    components["cloud_quality_factor"] = round(cloud_factor, 4)

    # 2. Metadata completeness factor
    meta_factor = 1.0
    if metadata_quality:
        if not metadata_quality.get("is_complete", True):
            meta_factor = 0.70
        if not metadata_quality.get("has_georeference", True) and "has_georeference" in metadata_quality:
            meta_factor = min(meta_factor, 0.50)
    components["metadata_quality_factor"] = round(meta_factor, 4)

    # 3. Spatial alignment factor
    if spatial_alignment_iou is not None:
        alignment_factor = _clamp_01(spatial_alignment_iou)
    else:
        alignment_factor = 1.0
    components["spatial_alignment_factor"] = round(alignment_factor, 4)

    # Weighted input quality calculation
    q_score = (
        (cloud_factor * cfg.cloud_cover_weight) +
        (meta_factor * cfg.metadata_completeness_weight) +
        (alignment_factor * cfg.spatial_alignment_weight)
    )
    return _clamp_01(q_score), components


def aggregate_confidence(
    model_confidence: Optional[float] = None,
    optical_model_confidence: Optional[float] = None,
    sar_model_confidence: Optional[float] = None,
    agreement_score: Optional[float] = None,
    cloud_cover_percentage: Optional[float] = None,
    metadata_quality: Optional[Dict[str, Any]] = None,
    spatial_alignment_iou: Optional[float] = None,
    disagreement_detected: bool = False,
    disagreement_severity: str = "NONE",  # NONE | PARTIAL | MATERIAL
    config: Optional[ConfidenceConfig] = None,
) -> ConfidenceReport:
    """
    Aggregates specialist model probabilities, cross-modal agreement, and input quality
    into a single auditable confidence score according to §6.3 specification.
    """
    cfg = config or ConfidenceConfig()
    components: Dict[str, Any] = {}

    # 1. Determine model confidence
    if optical_model_confidence is not None and sar_model_confidence is not None:
        # Paired mode: average reported confidences
        opt_c = _clamp_01(optical_model_confidence)
        sar_c = _clamp_01(sar_model_confidence)
        eff_model_conf = (opt_c + sar_c) / 2.0
        components["optical_model_confidence"] = opt_c
        components["sar_model_confidence"] = sar_c
        is_paired = True
    elif optical_model_confidence is not None:
        eff_model_conf = _clamp_01(optical_model_confidence)
        components["optical_model_confidence"] = eff_model_conf
        is_paired = False
    elif sar_model_confidence is not None:
        eff_model_conf = _clamp_01(sar_model_confidence)
        components["sar_model_confidence"] = eff_model_conf
        is_paired = False
    elif model_confidence is not None:
        eff_model_conf = _clamp_01(model_confidence)
        components["raw_model_confidence"] = eff_model_conf
        is_paired = agreement_score is not None
    else:
        # Missing model confidence: default baseline
        eff_model_conf = 0.50
        components["model_confidence_assumed_default"] = True
        is_paired = False

    # 2. Compute Input Quality
    is_optical = optical_model_confidence is not None or (sar_model_confidence is None)
    input_q_score, q_components = calculate_input_quality_score(
        cloud_cover_percentage=cloud_cover_percentage,
        metadata_quality=metadata_quality,
        spatial_alignment_iou=spatial_alignment_iou,
        is_optical=is_optical,
        config=cfg,
    )
    components.update(q_components)

    # 3. Disagreement Penalty
    penalty = 0.0
    if disagreement_detected:
        if disagreement_severity.upper() == "PARTIAL":
            penalty = cfg.partial_disagreement_penalty
        else:
            penalty = cfg.disagreement_penalty
    components["disagreement_penalty"] = penalty

    # 4. Final Aggregation Formula
    if is_paired and agreement_score is not None:
        norm_agreement = _clamp_01(agreement_score)
        weights = cfg.paired_weights
        raw_final = (
            (weights.weight_model * eff_model_conf) +
            (weights.weight_agreement * norm_agreement) +
            (weights.weight_input_quality * input_q_score) -
            penalty
        )
    else:
        norm_agreement = None
        weights_single = cfg.single_modality_weights
        raw_final = (
            (weights_single.weight_model * eff_model_conf) +
            (weights_single.weight_input_quality * input_q_score) -
            penalty
        )

    final_conf = _clamp_01(raw_final)

    return ConfidenceReport(
        model_confidence=eff_model_conf,
        agreement_score=norm_agreement,
        input_quality_score=input_q_score,
        final_confidence=final_conf,
        confidence_components=components,
        disagreement_detected=disagreement_detected,
        disagreement_penalty_applied=penalty,
        confidence_method_version=cfg.version,
    )
