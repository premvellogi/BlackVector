"""Quick end-to-end test of EOV2B VQA and Caption endpoints."""
import requests
import json
import time
import uuid

IMG_PATH = r"d:\Netra\outputs\test_images\agriculture_pasture_optical.png"
BASE = "http://localhost:8200"

# Upload
print("Uploading image...")
with open(IMG_PATH, "rb") as f:
    resp = requests.post(f"{BASE}/upload", files={"file": ("test.png", f, "image/png")})
upload = resp.json()
image_id = upload["image_id"]
print(f"Uploaded -> {image_id}")

# VQA test
print("\n" + "=" * 60)
print("TEST: VQA")
print("=" * 60)
t0 = time.time()
payload = {
    "request_id": str(uuid.uuid4()),
    "query": "What land cover types are visible in this satellite image?",
    "image_ids": [image_id],
    "task_type": "vqa",
    "input_scope": "single",
}
resp = requests.post(f"{BASE}/vqa", json=payload)
dt = time.time() - t0
print(f"HTTP Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error: {resp.text[:500]}")
else:
    result = resp.json()
    print(f"Status: {result.get('status')}")
    print(f"Time: {dt:.1f}s")
    answer = result.get("facts", {}).get("answer", "N/A")
    print(f"Answer: {answer[:500]}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Model: {result.get('model_version')}")

# Caption test
print("\n" + "=" * 60)
print("TEST: CAPTION")
print("=" * 60)
t0 = time.time()
payload = {
    "request_id": str(uuid.uuid4()),
    "query": "",
    "image_ids": [image_id],
    "task_type": "caption",
    "input_scope": "single",
}
resp = requests.post(f"{BASE}/caption", json=payload)
dt = time.time() - t0
print(f"HTTP Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error: {resp.text[:500]}")
else:
    result = resp.json()
    print(f"Status: {result.get('status')}")
    print(f"Time: {dt:.1f}s")
    caption = result.get("facts", {}).get("caption", "N/A")
    print(f"Caption: {caption[:500]}")
    tags = result.get("facts", {}).get("tags", [])
    print(f"Tags: {tags}")
    scene = result.get("facts", {}).get("scene_type", "N/A")
    print(f"Scene type: {scene}")
