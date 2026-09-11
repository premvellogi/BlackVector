"""
SatQuery AI - B2 Disagreement Package
"""

from satquery_b2.disagreement.models import (
    BoundingBox,
    DisagreementReport,
    DisagreementStatus,
    ModelOutput,
)
from satquery_b2.disagreement.spatial import compare_spatial_bounding_boxes
from satquery_b2.disagreement.surfacing import detect_disagreement

__all__ = [
    "ModelOutput",
    "BoundingBox",
    "DisagreementStatus",
    "DisagreementReport",
    "detect_disagreement",
    "compare_spatial_bounding_boxes",
]
