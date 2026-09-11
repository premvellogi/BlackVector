"""
Unit test for /change endpoint in real ML serving service.
"""

import uuid
import sys
from pathlib import Path
from fastapi.testclient import TestClient

NETRA_ROOT = str(Path(__file__).resolve().parents[1])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

from services.models.serving.server import app

def test_change_endpoint():
    with TestClient(app) as client:
        req_payload = {
            "request_id": str(uuid.uuid4()),
            "task_type": "change",
            "query": "Detect changes between T1 and T2",
            "input_scope": "bi_temporal",
            "image_ids": ["patch_t1_0", "patch_t2_0"],
            "metadata": {},
        }
        response = client.post("/change", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "change"
        assert data["status"] == "success"
        assert "answer" in data["facts"]
        assert "change_score" in data["facts"]
