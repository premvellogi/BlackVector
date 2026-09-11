"""
SatQuery AI - B2 Disagreement Models & Statuses (D3, FR9)
=========================================================
Structured data representations for multi-model outputs, spatial regions, and
disagreement reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from satquery_b2.ontology.schema import Modality


class DisagreementStatus(str, Enum):
    """Classification of cross-modal or multi-model agreement."""
    AGREEMENT = "AGREEMENT"
    PARTIAL_AGREEMENT = "PARTIAL_AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


@dataclass
class BoundingBox:
    """2D Spatial bounding box for detections or grounding."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    label: Optional[str] = None
    confidence: Optional[float] = None

    def calculate_iou(self, other: BoundingBox) -> float:
        """Computes IoU between two bounding boxes."""
        inter_min_x = max(self.min_x, other.min_x)
        inter_min_y = max(self.min_y, other.min_y)
        inter_max_x = min(self.max_x, other.max_x)
        inter_max_y = min(self.max_y, other.max_y)

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return 0.0

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area_self = (self.max_x - self.min_x) * (self.max_y - self.min_y)
        area_other = (other.max_x - other.min_x) * (other.max_y - other.min_y)
        union_area = area_self + area_other - inter_area

        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        return cls(
            min_x=float(data["min_x"]),
            min_y=float(data["min_y"]),
            max_x=float(data["max_x"]),
            max_y=float(data["max_y"]),
            label=data.get("label"),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        )


@dataclass
class ModelOutput:
    """Structured output record from a specialist model."""
    model_name: str
    task: str
    prediction: Any
    confidence: Optional[float] = None
    modality: Optional[Union[Modality, str]] = None
    model_version: Optional[str] = "UNKNOWN"
    checkpoint: Optional[str] = "UNKNOWN"
    spatial_output: Optional[List[BoundingBox]] = None
    timestamp: Optional[str] = None
    provenance_id: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "checkpoint": self.checkpoint,
            "task": self.task,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "modality": self.modality.value if isinstance(self.modality, Modality) else self.modality,
            "spatial_output": [b.to_dict() for b in self.spatial_output] if self.spatial_output else None,
            "timestamp": self.timestamp,
            "provenance_id": self.provenance_id,
        }


@dataclass
class DisagreementReport:
    """Comprehensive disagreement report preserving raw outputs and explaining discrepancies."""
    status: DisagreementStatus
    participating_models: List[str]
    individual_predictions: Dict[str, Any]
    individual_confidences: Dict[str, Optional[float]]
    comparison_summary: str
    human_readable_explanation: str
    disagreement_reason: Optional[str] = None
    confidence_impact: str = "NO_IMPACT"  # NO_IMPACT | MINOR_PENALTY | MAJOR_PENALTY
    spatial_details: Optional[Dict[str, Any]] = None
    provenance_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "participating_models": self.participating_models,
            "individual_predictions": self.individual_predictions,
            "individual_confidences": self.individual_confidences,
            "comparison_summary": self.comparison_summary,
            "human_readable_explanation": self.human_readable_explanation,
            "disagreement_reason": self.disagreement_reason,
            "confidence_impact": self.confidence_impact,
            "spatial_details": self.spatial_details,
            "provenance_refs": self.provenance_refs,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
