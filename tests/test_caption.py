"""
Unit test for /caption endpoint in real ML serving service.
"""

import uuid
import sys
from pathlib import Path
from fastapi.testclient import TestClient

NETRA_ROOT = str(Path(__file__).resolve().parents[1])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

from services.models.serving.server import app

def test_caption_endpoint():
    with TestClient(app) as client:
        req_payload = {
            "request_id": str(uuid.uuid4()),
            "task_type": "caption",
            "query": "Describe the image",
            "input_scope": "single",
            "image_ids": ["patch_0"],
            "metadata": {},
        }
        response = client.post("/caption", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "caption"
        assert data["status"] == "success"
        assert "caption" in data["facts"]
        assert "tags" in data["facts"]
