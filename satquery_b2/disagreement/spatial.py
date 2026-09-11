"""
SatQuery AI - B2 Spatial Disagreement Utilities
===============================================
Calculates bounding box overlaps, multi-box matching IoUs, and spatial consensus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from satquery_b2.disagreement.models import BoundingBox


def compare_spatial_bounding_boxes(
    boxes_a: List[BoundingBox],
    boxes_b: List[BoundingBox],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Compares two lists of detected bounding boxes and calculates match statistics.

    Returns
    -------
    Dict with matched pairs, unmatched boxes, mean IoU, and spatial overlap ratio.
    """
    if not boxes_a and not boxes_b:
        return {
            "status": "EMPTY_MATCH",
            "mean_iou": 1.0,
            "matched_count": 0,
            "unmatched_a_count": 0,
            "unmatched_b_count": 0,
            "spatial_agreement_ratio": 1.0,
        }

    if not boxes_a or not boxes_b:
        return {
            "status": "DISJOINT",
            "mean_iou": 0.0,
            "matched_count": 0,
            "unmatched_a_count": len(boxes_a),
            "unmatched_b_count": len(boxes_b),
            "spatial_agreement_ratio": 0.0,
        }

    matched_pairs: List[Dict[str, Any]] = []
    unmatched_a = list(boxes_a)
    unmatched_b = list(boxes_b)
    ious: List[float] = []

    # Greedy matching based on maximum IoU
    for box_a in boxes_a:
        best_iou = 0.0
        best_match_b: Optional[BoundingBox] = None
        for box_b in unmatched_b:
            iou = box_a.calculate_iou(box_b)
            if iou > best_iou:
                best_iou = iou
                best_match_b = box_b

        if best_match_b is not None and best_iou >= iou_threshold:
            matched_pairs.append({
                "box_a": box_a.to_dict(),
                "box_b": best_match_b.to_dict(),
                "iou": round(best_iou, 4),
                "labels_match": (box_a.label or "").lower() == (best_match_b.label or "").lower(),
            })
            ious.append(best_iou)
            if box_a in unmatched_a:
                unmatched_a.remove(box_a)
            if best_match_b in unmatched_b:
                unmatched_b.remove(best_match_b)

    total_unique = len(matched_pairs) + len(unmatched_a) + len(unmatched_b)
    spatial_ratio = len(matched_pairs) / total_unique if total_unique > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return {
        "status": "MATCHED" if spatial_ratio >= 0.8 else ("PARTIAL" if spatial_ratio > 0.3 else "DISAGREEMENT"),
        "mean_iou": round(mean_iou, 4),
        "matched_count": len(matched_pairs),
        "unmatched_a_count": len(unmatched_a),
        "unmatched_b_count": len(unmatched_b),
        "spatial_agreement_ratio": round(spatial_ratio, 4),
        "matched_pairs": matched_pairs,
    }
