"""
SatQuery AI - B2 Disagreement Surfacing Pipeline (D3, FR9)
===========================================================
Detects, represents, and surfaces disagreement across multimodal specialist models.
Preserves all conflicting outputs without silently discarding any prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from satquery_b2.disagreement.models import (
    DisagreementReport,
    DisagreementStatus,
    ModelOutput,
)
from satquery_b2.disagreement.spatial import compare_spatial_bounding_boxes


def _normalize_pred(pred: Any) -> str:
    """Normalizes prediction values to a comparable string format."""
    if isinstance(pred, bool):
        return "true" if pred else "false"
    if pred is None:
        return ""
    return str(pred).strip().lower()


def detect_disagreement(
    model_outputs: List[ModelOutput],
    spatial_iou_threshold: float = 0.50,
) -> DisagreementReport:
    """
    Analyzes multiple specialist model outputs and generates a comprehensive disagreement report.

    Parameters
    ----------
    model_outputs : List[ModelOutput]
        Outputs collected from participating specialist models.
    spatial_iou_threshold : float, default 0.50
        IoU threshold for spatial bounding box matches.

    Returns
    -------
    DisagreementReport
        Structured report classifying agreement, detailing individual predictions,
        and providing human-readable explanations.
    """
    # 1. Handle Edge Cases
    if not model_outputs:
        return DisagreementReport(
            status=DisagreementStatus.INSUFFICIENT_INFORMATION,
            participating_models=[],
            individual_predictions={},
            individual_confidences={},
            comparison_summary="No model outputs provided for disagreement analysis.",
            human_readable_explanation="Disagreement analysis could not run because no model outputs were received.",
            confidence_impact="NO_IMPACT",
        )

    participating_models = [m.model_name for m in model_outputs]
    individual_preds = {m.model_name: m.prediction for m in model_outputs}
    individual_confs = {m.model_name: m.confidence for m in model_outputs}
    provenance_refs = [m.provenance_id for m in model_outputs if m.provenance_id]

    if len(model_outputs) == 1:
        single = model_outputs[0]
        return DisagreementReport(
            status=DisagreementStatus.AGREEMENT,
            participating_models=participating_models,
            individual_predictions=individual_preds,
            individual_confidences=individual_confs,
            comparison_summary=f"Single model output received from '{single.model_name}'.",
            human_readable_explanation=f"Single model evaluation ({single.model_name}) produced prediction '{single.prediction}'. Cross-modal comparison is not applicable.",
            confidence_impact="NO_IMPACT",
            provenance_refs=provenance_refs,
        )

    # 2. Check for Spatial Detections
    spatial_details: Optional[Dict[str, Any]] = None
    has_spatial = all(m.spatial_output is not None for m in model_outputs)
    if has_spatial and len(model_outputs) == 2:
        spatial_details = compare_spatial_bounding_boxes(
            model_outputs[0].spatial_output or [],
            model_outputs[1].spatial_output or [],
            iou_threshold=spatial_iou_threshold,
        )

    # 3. Analyze Prediction Agreement
    norm_preds = [_normalize_pred(m.prediction) for m in model_outputs]
    all_identical = len(set(norm_preds)) == 1

    # Check if predictions are partially overlapping (e.g. substring or shared words)
    is_partial = False
    if not all_identical and len(norm_preds) == 2:
        p1, p2 = norm_preds[0], norm_preds[1]
        words1 = set(p1.replace(",", " ").replace(";", " ").split())
        words2 = set(p2.replace(",", " ").replace(";", " ").split())
        if words1.intersection(words2):
            is_partial = True

    # 4. Status and Severity Classification
    if all_identical:
        if spatial_details and spatial_details["status"] == "DISAGREEMENT":
            status = DisagreementStatus.PARTIAL_AGREEMENT
            reason = "Predictions agree in label but spatial bounding boxes are misaligned."
            impact = "MINOR_PENALTY"
        else:
            status = DisagreementStatus.AGREEMENT
            reason = "All participating models produced matching predictions."
            impact = "NO_IMPACT"
    elif is_partial:
        status = DisagreementStatus.PARTIAL_AGREEMENT
        reason = "Participating models produced overlapping or partially aligned predictions."
        impact = "MINOR_PENALTY"
    else:
        status = DisagreementStatus.DISAGREEMENT
        reason = "Participating models produced contradictory predictions."
        impact = "MAJOR_PENALTY"

    # 5. Generate Human-Readable Explanation for UI
    m1 = model_outputs[0]
    m2 = model_outputs[1]
    c1_str = f" (confidence: {m1.confidence:.2f})" if m1.confidence is not None else ""
    c2_str = f" (confidence: {m2.confidence:.2f})" if m2.confidence is not None else ""

    if status == DisagreementStatus.AGREEMENT:
        explanation = (
            f"Both {m1.model_name}{c1_str} and {m2.model_name}{c2_str} agreed on "
            f"the prediction: '{m1.prediction}'."
        )
    elif status == DisagreementStatus.PARTIAL_AGREEMENT:
        explanation = (
            f"{m1.model_name} predicted '{m1.prediction}'{c1_str}, while {m2.model_name} "
            f"predicted '{m2.prediction}'{c2_str}. The predictions partially overlap but exhibit minor discrepancy."
        )
    else:
        explanation = (
            f"{m1.model_name} predicted '{m1.prediction}'{c1_str}, whereas {m2.model_name} "
            f"predicted '{m2.prediction}'{c2_str}. Material disagreement was detected between the models."
        )

    return DisagreementReport(
        status=status,
        participating_models=participating_models,
        individual_predictions=individual_preds,
        individual_confidences=individual_confs,
        comparison_summary=f"Evaluated {len(model_outputs)} model outputs with status: {status.value}.",
        human_readable_explanation=explanation,
        disagreement_reason=reason,
        confidence_impact=impact,
        spatial_details=spatial_details,
        provenance_refs=provenance_refs,
    )
