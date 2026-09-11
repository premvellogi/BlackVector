"""
Unit test for /fusion endpoint in real ML serving service.
"""

import uuid
import sys
from pathlib import Path
from fastapi.testclient import TestClient

NETRA_ROOT = str(Path(__file__).resolve().parents[1])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

from services.models.serving.server import app

def test_fusion_endpoint():
    with TestClient(app) as client:
        req_payload = {
            "request_id": str(uuid.uuid4()),
            "task_type": "fusion",
            "query": "Combine optical and SAR imagery",
            "input_scope": "cross_modal",
            "image_ids": ["patch_opt_0", "patch_sar_0"],
            "metadata": {},
        }
        response = client.post("/fusion", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "fusion"
        assert data["status"] == "success"
        assert "answer" in data["facts"]
        assert "modality_attribution" in data["facts"]
        assert "optical_weight" in data["facts"]["modality_attribution"]
