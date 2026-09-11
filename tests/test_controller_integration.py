"""
End-to-End Integration Test for SatQuery AI.

Flow:
Client Query -> B1 Controller -> B2 Validation -> Real ML Service -> Response & ExecutionTrace
"""

import sys
import uuid
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add D:\Netra-main\Controller and d:\Netra to sys.path
NETRA_MAIN_CONTROLLER = "D:/Netra-main/Controller"
if NETRA_MAIN_CONTROLLER not in sys.path:
    sys.path.insert(0, NETRA_MAIN_CONTROLLER)

NETRA_ROOT = str(Path(__file__).resolve().parents[1])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

from services.models.serving.server import app as ml_app
from mock_b2 import app as b2_app

def test_full_pipeline_vqa():
    """Test /vqa flow through Real ML Service."""
    with TestClient(ml_app) as ml_client, TestClient(b2_app) as b2_client:
        req = {
            "request_id": str(uuid.uuid4()),
            "task_type": "vqa",
            "query": "Is there a river in this satellite patch?",
            "input_scope": "single",
            "image_ids": ["patch_0"],
            "metadata": {},
        }

        # 1. B2 Validation
        val_res = b2_client.post("/validate", json={
            "request_id": req["request_id"],
            "input_scope": req["input_scope"],
            "image_ids": req["image_ids"],
            "task_type": req["task_type"],
        })
        assert val_res.status_code == 200
        assert val_res.json()["valid"] is True

        # 2. Real ML Service Execution
        ml_res = ml_client.post("/vqa", json=req)
        assert ml_res.status_code == 200
        ml_data = ml_res.json()
        assert ml_data["status"] == "success"
        assert "answer" in ml_data["facts"]

        # 3. B2 Confidence Aggregation
        conf_res = b2_client.post("/confidence", json={
            "request_id": req["request_id"],
            "task_type": req["task_type"],
            "model_confidence": ml_data["confidence"],
            "cross_modal_agreement": 1.0,
            "input_quality": 0.95,
            "raw_block_output": ml_data,
        })
        assert conf_res.status_code == 200
        assert conf_res.json()["score"] > 0.70


def test_full_pipeline_change():
    """Test /change flow through Real ML Service."""
    with TestClient(ml_app) as ml_client, TestClient(b2_app) as b2_client:
        req = {
            "request_id": str(uuid.uuid4()),
            "task_type": "change",
            "query": "Detect changes between T1 and T2",
            "input_scope": "bi_temporal",
            "image_ids": ["patch_t1_0", "patch_t2_0"],
            "metadata": {},
        }

        # 1. B2 Validation
        val_res = b2_client.post("/validate", json={
            "request_id": req["request_id"],
            "input_scope": req["input_scope"],
            "image_ids": req["image_ids"],
            "task_type": req["task_type"],
        })
        assert val_res.status_code == 200
        assert val_res.json()["valid"] is True

        # 2. Real ML Service Execution
        ml_res = ml_client.post("/change", json=req)
        assert ml_res.status_code == 200
        ml_data = ml_res.json()
        assert ml_data["status"] == "success"
        assert "change_detected" in ml_data["facts"]


def test_full_pipeline_fusion():
    """Test /fusion flow through Real ML Service."""
    with TestClient(ml_app) as ml_client, TestClient(b2_app) as b2_client:
        req = {
            "request_id": str(uuid.uuid4()),
            "task_type": "fusion",
            "query": "Combine optical and SAR imagery",
            "input_scope": "cross_modal",
            "image_ids": ["patch_opt_0", "patch_sar_0"],
            "metadata": {},
        }

        # 1. B2 Validation
        val_res = b2_client.post("/validate", json={
            "request_id": req["request_id"],
            "input_scope": req["input_scope"],
            "image_ids": req["image_ids"],
            "task_type": req["task_type"],
        })
        assert val_res.status_code == 200
        assert val_res.json()["valid"] is True

        # 2. Real ML Service Execution
        ml_res = ml_client.post("/fusion", json=req)
        assert ml_res.status_code == 200
        ml_data = ml_res.json()
        assert ml_data["status"] == "success"
        assert "fused_answer" in ml_data["facts"]
