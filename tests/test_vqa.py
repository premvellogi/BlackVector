"""
Unit test for /vqa endpoint in real ML serving service.
"""

import uuid
import sys
from pathlib import Path
from fastapi.testclient import TestClient

NETRA_ROOT = str(Path(__file__).resolve().parents[1])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

from services.models.serving.server import app

def test_vqa_endpoint():
    with TestClient(app) as client:
        req_payload = {
            "request_id": str(uuid.uuid4()),
            "task_type": "vqa",
            "query": "Is there a river in this satellite patch?",
            "input_scope": "single",
            "image_ids": ["patch_0"],
            "metadata": {},
        }
        response = client.post("/vqa", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "vqa"
        assert data["status"] == "success"
        assert "answer" in data["facts"]
        assert data["confidence"] > 0.0
