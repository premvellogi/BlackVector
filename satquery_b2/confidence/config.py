"""
SatQuery AI - B2 Confidence Aggregation Configuration (§6.3)
=============================================================
Defines configurable weights, quality factor scaling, and disagreement penalties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ConfidenceWeights:
    """Configurable weights for confidence components. Must sum to 1.0."""
    weight_model: float = 0.50       # Weight given to model's reported probability
    weight_agreement: float = 0.30   # Weight given to cross-modal / multi-model agreement
    weight_input_quality: float = 0.20  # Weight given to input geospatial & sensor quality

    def __post_init__(self) -> None:
        total = self.weight_model + self.weight_agreement + self.weight_input_quality
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Confidence weights must sum to 1.0, got {total:.3f}")


@dataclass
class SingleModalityConfidenceWeights:
    """Weights applied when only a single modality (e.g. Optical-only or SAR-only) is present."""
    weight_model: float = 0.70       # No cross-modal agreement available
    weight_input_quality: float = 0.30

    def __post_init__(self) -> None:
        total = self.weight_model + self.weight_input_quality
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Single modality confidence weights must sum to 1.0, got {total:.3f}")


@dataclass
class ConfidenceConfig:
    """Configuration for SatQuery AI §6.3 Confidence Aggregator."""
    version: str = "1.0.0"
    paired_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    single_modality_weights: SingleModalityConfidenceWeights = field(default_factory=SingleModalityConfidenceWeights)
    disagreement_penalty: float = 0.20       # Direct score reduction if material disagreement is detected
    partial_disagreement_penalty: float = 0.08  # Penalty for partial disagreement
    cloud_cover_weight: float = 0.40         # Weight of cloud cover within input quality score
    metadata_completeness_weight: float = 0.30  # Weight of metadata completeness
    spatial_alignment_weight: float = 0.30   # Weight of spatial alignment IoU
