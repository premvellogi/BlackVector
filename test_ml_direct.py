"""
Direct ML Service Tester
========================
Sends a request straight to the ML service (port 8200), bypassing the
frontend and the controller, so you can see:
  1. The exact prompt sent to InternVL2
  2. The raw JSON response from the model

Usage:
  python test_ml_direct.py                         # uses default test image
  python test_ml_direct.py --image path/to/img.jpg --question "What is this?"
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import requests

ML_URL = "http://localhost:8200"


def upload_image(image_path: str) -> str:
    """Upload an image to the ML service and return the image_id (server path)."""
    print(f"\n[UPLOAD] Sending image: {image_path}")
    with open(image_path, "rb") as f:
        r = requests.post(f"{ML_URL}/upload", files={"file": (Path(image_path).name, f)})
    r.raise_for_status()
    resp = r.json()
    print(f"[UPLOAD] OK -> image_id = {resp['image_id']}")
    return resp["image_id"]


def call_vqa(image_id: str, question: str) -> dict:
    """Call /vqa directly and return the full raw JSON response."""
    payload = {
        "request_id": str(uuid.uuid4()),
        "task_type": "vqa",
        "query": question,
        "input_scope": "single",
        "image_ids": [image_id],
        "metadata": {},
    }

    print("\n" + "=" * 60)
    print("REQUEST PAYLOAD SENT TO ML SERVICE /vqa:")
    print("=" * 60)
    print(json.dumps(payload, indent=2))
    print("=" * 60)

    r = requests.post(f"{ML_URL}/vqa", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="Direct ML Service Tester")
    parser.add_argument("--image", default=None, help="Path to image file (JPG/PNG/TIF)")
    parser.add_argument(
        "--question",
        default=(
            "Describe the visible water feature in detail. "
            "Is it more likely to be a river, coastline, estuary, lake, or ocean? "
            "Explain your answer using only visible features."
        ),
        help="Question to ask the model",
    )
    parser.add_argument(
        "--task",
        default="vqa",
        choices=["vqa", "caption"],
        help="Which endpoint to test (default: vqa)",
    )
    args = parser.parse_args()

    # --- Health check ---
    print(f"[HEALTH] Checking ML service at {ML_URL}/health ...")
    try:
        h = requests.get(f"{ML_URL}/health", timeout=5)
        print(f"[HEALTH] {h.json()}")
    except Exception as e:
        print(f"[ERROR] ML service not reachable: {e}")
        sys.exit(1)

    # --- Image ---
    image_path = args.image
    if not image_path:
        # Try to find any uploaded image in the temp dir
        import tempfile, glob
        temp_uploads = Path(tempfile.gettempdir()) / "satquery_uploads"
        if temp_uploads.exists():
            found = sorted(temp_uploads.glob("*.jpg")) + sorted(temp_uploads.glob("*.png"))
            if found:
                image_path = str(found[-1])  # Use most recent
                print(f"[AUTO] Using most recent upload: {image_path}")

    if not image_path or not Path(image_path).exists():
        # Fallback: send a synthetic request with a dummy id so the model uses
        # its synthetic fallback — still shows you prompt + raw output
        print("[WARN] No image found — using 'default' image_id (synthetic fallback)")
        image_id = "default"
    else:
        image_id = upload_image(image_path)

    # --- Ask the model ---
    print(f"\n[QUESTION] {args.question}")
    print("\n[CALLING] POST /vqa ...")

    raw = call_vqa(image_id, args.question)

    # --- Print raw response ---
    print("\n" + "=" * 60)
    print("RAW JSON RESPONSE FROM ML MODEL:")
    print("=" * 60)
    print(json.dumps(raw, indent=2))
    print("=" * 60)

    # --- Highlight the answer ---
    facts = raw.get("facts", {})
    answer = facts.get("answer", facts.get("caption_facts", "-- no answer field --"))
    print(f"\n[ANSWER]\n{answer}")
    print(f"\n[Confidence]: {raw.get('confidence', 'n/a')}")
    print(f"[Task type returned]: {raw.get('task_type', 'n/a')}")
    print(f"[Model version]: {raw.get('model_version', 'n/a')}")


if __name__ == "__main__":
    main()
