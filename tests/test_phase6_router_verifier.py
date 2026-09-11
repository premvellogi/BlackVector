import sys
from pathlib import Path

# Add project root to sys.path
NETRA_ROOT = str(Path(__file__).resolve().parents[1])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

import pytest
import asyncio
from services.models.serving.schemas import MLRequest, MLResponse, TaskType, InputScope
from services.models.core.qwen3_router import Qwen3Router, RouterDecision
from services.models.core.evidence_verifier import EvidenceVerifier
from services.models.serving.model_registry import ModelRegistry


def test_qwen3_router_classification():
    router = Qwen3Router()

    # 1. Grounding query
    d1 = router.route("Locate and segment all water bodies and buildings")
    assert d1.task_type == TaskType.GROUNDING
    assert "water body" in d1.extracted_entities or "water" in d1.extracted_entities
    assert "building" in d1.extracted_entities or "buildings" in d1.extracted_entities

    # 2. Change detection query
    d2 = router.route("Show me the building change between these bi-temporal images", input_scope=InputScope.BI_TEMPORAL)
    assert d2.task_type == TaskType.CHANGE

    # 3. Fusion query
    d3 = router.route("Analyze this Sentinel-1 SAR and Sentinel-2 optical image", input_scope=InputScope.CROSS_MODAL)
    assert d3.task_type == TaskType.FUSION

    # 4. Caption query
    d4 = router.route("Provide a detailed caption and summary of the scene")
    assert d4.task_type == TaskType.CAPTION

    # 5. VQA query
    d5 = router.route("What percentage of this area is agricultural land?")
    assert d5.task_type == TaskType.VQA


def test_evidence_verifier_bbox_validation():
    verifier = EvidenceVerifier()

    # Valid boxes
    valid_boxes = [[100, 100, 500, 500], [200, 300, 400, 600]]
    is_valid, quality, issues = verifier.verify_bounding_boxes(valid_boxes)
    assert is_valid is True
    assert quality == 1.0
    assert len(issues) == 0

    # Invalid boxes
    invalid_boxes = [[500, 500, 100, 100], [-10, 20, 30, 40], [0, 0, 1200, 500]]
    is_valid, quality, issues = verifier.verify_bounding_boxes(invalid_boxes)
    assert is_valid is False
    assert quality == 0.0
    assert len(issues) == 3


def test_evidence_verifier_response_verification():
    verifier = EvidenceVerifier(strict_hallucination=True)

    resp = MLResponse(
        status="success",
        task_type=TaskType.VQA,
        facts={
            "answer": "This area is located in Serbia during summer 2023.",
            "land_cover": "forest",
        },
        confidence=0.9,
    )

    verified = verifier.verify_response(resp)
    assert verified.status == "success"
    # Stripped hallucinated claims ("Serbia", "summer", "2023")
    assert "Serbia" not in verified.facts["answer"]
    assert verified.facts["verification_report"]["verified"] is False  # Violations found
    assert verified.confidence < 0.9  # Confidence penalized for unsupported claims


def test_model_registry_dispatch():
    registry = ModelRegistry.get_instance()

    req = MLRequest(
        query="Where are the buildings in this satellite image?",
        input_scope=InputScope.SINGLE,
    )

    routed = registry.route_request(req)
    assert routed.task_type == TaskType.GROUNDING
    assert "router_decision" in routed.metadata
