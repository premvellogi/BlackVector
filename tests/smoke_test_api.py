"""
SatQuery AI — API Smoke Test.

Tests all 5 endpoints with both .npz multiband and RGB fallback paths.
Runs directly against the endpoint functions (no uvicorn needed) to
avoid dual-GPU-process issues.

Usage:
    python tests/smoke_test_api.py
"""

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

# ─── Test utilities ──────────────────────────────────────────────────────────

class FakeUploadFile:
    """Mock UploadFile for testing endpoints without uvicorn."""

    def __init__(self, content: bytes, filename: str, content_type: str = "application/octet-stream"):
        self._content = content
        self._pos = 0
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        self._pos = 0
        return self._content

    async def seek(self, pos):
        self._pos = pos


def make_rgb_upload(width=120, height=120, filename="test.png") -> FakeUploadFile:
    """Create a synthetic RGB PNG upload."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return FakeUploadFile(buf.getvalue(), filename, "image/png")


def make_npz_upload(patch_id: str = None, cache_dir: str = "data/imagery_cache") -> FakeUploadFile:
    """Create an .npz upload from a real or synthetic patch."""
    from services.models.training.multiband_dataset import S2_BANDS, S2_STATS, S1_STATS

    if patch_id:
        npz_path = os.path.join(cache_dir, f"{patch_id}.npz")
        if os.path.exists(npz_path):
            return FakeUploadFile(
                open(npz_path, "rb").read(),
                f"{patch_id}.npz",
                "application/octet-stream",
            )

    # Synthetic fallback
    rng = np.random.RandomState(42)
    s2 = np.stack([
        np.clip(rng.normal(S2_STATS[b]["mean"], S2_STATS[b]["std"] * 0.3, (120, 120)), 0, 10000)
        for b in S2_BANDS
    ]).astype(np.float32)
    s1 = np.stack([
        rng.normal(S1_STATS[p]["mean"], S1_STATS[p]["std"] * 0.5, (120, 120))
        for p in ["VV", "VH"]
    ]).astype(np.float32)

    buf = io.BytesIO()
    np.savez(buf, s2=s2, s1=s1)
    return FakeUploadFile(buf.getvalue(), "synthetic.npz", "application/octet-stream")


# ─── Find an unseen patch for testing ────────────────────────────────────────

def find_unseen_patch(cache_dir="data/imagery_cache",
                      train_jsonl="data/training_exp1/train.jsonl",
                      val_jsonl="data/training_exp1/val.jsonl") -> str:
    """Return the ID of an unseen cached patch."""
    used = set()
    for path in [train_jsonl, val_jsonl]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    used.add(json.loads(line)["image"])
    cached = [f.stem for f in Path(cache_dir).glob("*.npz")]
    unseen = [p for p in cached if p not in used]
    return unseen[0] if unseen else None


# ─── Test functions ──────────────────────────────────────────────────────────

results = []

def record(endpoint: str, input_type: str, passed: bool, latency: float,
           details: str = "", response: dict = None):
    """Record a test result."""
    status = "[PASS]" if passed else "[FAIL]"
    results.append({
        "endpoint": endpoint,
        "input": input_type,
        "status": "PASS" if passed else "FAIL",
        "latency_s": round(latency, 2),
        "details": details,
    })
    print(f"  {status} {endpoint:12s} [{input_type:5s}] {latency:.1f}s -- {details}")


async def test_health(app_state):
    """Test GET /health."""
    from services.models.api.app import health
    t0 = time.time()
    try:
        resp = await health()
        elapsed = time.time() - t0
        ok = resp["status"] == "ready" and resp["model_loaded"]
        adapter_ok = "spectral_adapter" in resp.get("vram", {})
        record("/health", "--", ok and adapter_ok, elapsed,
               f"status={resp['status']}, adapter={'loaded' if adapter_ok else 'MISSING'}")
    except Exception as e:
        record("/health", "--", False, time.time() - t0, f"ERROR: {e}")


async def test_vqa_npz(unseen_patch_id):
    """Test POST /vqa with .npz multiband input."""
    from services.models.api.app import vqa
    upload = make_npz_upload(unseen_patch_id)
    t0 = time.time()
    try:
        resp = await vqa(image=upload, question="What type of land cover is present?")
        elapsed = time.time() - t0
        ok = (len(resp.answer) > 0
              and resp.provenance.checkpoint_version == "satquery-lora-exp1"
              and resp.execution_time_ms > 0)
        record("/vqa", ".npz", ok, elapsed,
               f"answer='{resp.answer[:60]}', tokens={resp.confidence.factors.get('tokens_generated', '?')}")
    except Exception as e:
        record("/vqa", ".npz", False, time.time() - t0, f"ERROR: {e}")


async def test_vqa_rgb():
    """Test POST /vqa with RGB PNG input."""
    from services.models.api.app import vqa
    upload = make_rgb_upload(filename="test.png")
    t0 = time.time()
    try:
        resp = await vqa(image=upload, question="Describe what you see in this image.")
        elapsed = time.time() - t0
        ok = len(resp.answer) > 0
        record("/vqa", ".png", ok, elapsed, f"answer='{resp.answer[:60]}'")
    except Exception as e:
        record("/vqa", ".png", False, time.time() - t0, f"ERROR: {e}")


async def test_caption_npz(unseen_patch_id):
    """Test POST /caption with .npz input."""
    from services.models.api.app import caption
    upload = make_npz_upload(unseen_patch_id)
    t0 = time.time()
    try:
        resp = await caption(image=upload)
        elapsed = time.time() - t0
        ok = len(resp.caption) > 50
        record("/caption", ".npz", ok, elapsed, f"len={len(resp.caption)} chars")
    except Exception as e:
        record("/caption", ".npz", False, time.time() - t0, f"ERROR: {e}")


async def test_caption_rgb():
    """Test POST /caption with RGB PNG input."""
    from services.models.api.app import caption
    upload = make_rgb_upload(filename="test.png")
    t0 = time.time()
    try:
        resp = await caption(image=upload)
        elapsed = time.time() - t0
        ok = len(resp.caption) > 0
        record("/caption", ".png", ok, elapsed, f"len={len(resp.caption)} chars")
    except Exception as e:
        record("/caption", ".png", False, time.time() - t0, f"ERROR: {e}")


async def test_change_npz(unseen_patch_id1, unseen_patch_id2):
    """Test POST /change with .npz inputs."""
    from services.models.api.app import change_detection
    upload1 = make_npz_upload(unseen_patch_id1)
    upload2 = make_npz_upload(unseen_patch_id2)
    t0 = time.time()
    try:
        resp = await change_detection(image_t1=upload1, image_t2=upload2)
        elapsed = time.time() - t0
        ok = len(resp.description) > 0 and resp.execution_time_ms > 0
        record("/change", ".npz", ok, elapsed,
               f"detected={resp.change_detected}, score={resp.confidence.score:.2f}")
    except Exception as e:
        record("/change", ".npz", False, time.time() - t0, f"ERROR: {e}")


async def test_change_rgb():
    """Test POST /change with RGB inputs."""
    from services.models.api.app import change_detection
    upload1 = make_rgb_upload(filename="t1.png")
    upload2 = make_rgb_upload(filename="t2.png")
    t0 = time.time()
    try:
        resp = await change_detection(image_t1=upload1, image_t2=upload2)
        elapsed = time.time() - t0
        ok = len(resp.description) > 0
        record("/change", ".png", ok, elapsed, f"detected={resp.change_detected}")
    except Exception as e:
        record("/change", ".png", False, time.time() - t0, f"ERROR: {e}")


async def test_fusion_npz(unseen_patch_id):
    """Test POST /fusion with .npz input (single file contains both S1+S2)."""
    from services.models.api.app import fusion
    # Send same .npz as both optical and sar (it contains both S1+S2)
    upload_opt = make_npz_upload(unseen_patch_id)
    upload_sar = make_npz_upload(unseen_patch_id)
    t0 = time.time()
    try:
        resp = await fusion(
            optical_image=upload_opt,
            sar_image=upload_sar,
            question="What land cover is in this area?",
        )
        elapsed = time.time() - t0
        attr_ok = (len(resp.modality_attribution.optical_evidence) > 0
                   and len(resp.modality_attribution.sar_evidence) > 0
                   and len(resp.modality_attribution.fused_reasoning) > 0)
        ok = len(resp.fused_answer) > 0 and attr_ok
        record("/fusion", ".npz", ok, elapsed,
               f"attr={'populated' if attr_ok else 'EMPTY'}, "
               f"disagree={resp.disagreement_flag}")
    except Exception as e:
        record("/fusion", ".npz", False, time.time() - t0, f"ERROR: {e}")


async def test_fusion_rgb():
    """Test POST /fusion with RGB inputs."""
    from services.models.api.app import fusion
    upload_opt = make_rgb_upload(filename="optical.png")
    upload_sar = make_rgb_upload(filename="sar.png")
    t0 = time.time()
    try:
        resp = await fusion(optical_image=upload_opt, sar_image=upload_sar)
        elapsed = time.time() - t0
        ok = len(resp.fused_answer) > 0
        record("/fusion", ".png", ok, elapsed, f"fused_len={len(resp.fused_answer)}")
    except Exception as e:
        record("/fusion", ".png", False, time.time() - t0, f"ERROR: {e}")


# ─── Main ────────────────────────────────────────────────────────────────────

async def run_all_tests():
    """Run all smoke tests."""
    from services.models.api.app import app_state

    print("=" * 65)
    print("SatQuery AI — API Smoke Test")
    print("=" * 65)

    # --- Startup: load model ---
    print("\n[1/3] Loading model...")
    t0 = time.time()
    await app_state.startup()
    print(f"  Model loaded in {time.time()-t0:.1f}s\n")

    # --- Find unseen patches ---
    print("[2/3] Preparing test data...")
    unseen1 = find_unseen_patch()
    # Get a second different unseen patch for change detection
    used = set()
    for path in ["data/training_exp1/train.jsonl", "data/training_exp1/val.jsonl"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    used.add(json.loads(line)["image"])
    cached = [f.stem for f in Path("data/imagery_cache").glob("*.npz") if f.stem not in used]
    unseen2 = cached[1] if len(cached) > 1 else unseen1
    print(f"  Unseen patch 1: {unseen1}")
    print(f"  Unseen patch 2: {unseen2}")

    # --- Run tests ---
    print(f"\n[3/3] Running smoke tests...\n")

    await test_health(app_state)
    await test_vqa_npz(unseen1)
    await test_vqa_rgb()
    await test_caption_npz(unseen1)
    await test_caption_rgb()
    await test_change_npz(unseen1, unseen2)
    await test_change_rgb()
    await test_fusion_npz(unseen1)
    await test_fusion_rgb()

    # --- Summary ---
    print("\n" + "=" * 65)
    print("SMOKE TEST SUMMARY")
    print("=" * 65)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    for r in results:
        icon = "[OK]" if r["status"] == "PASS" else "[!!]"
        print(f"  {icon} {r['endpoint']:12s} [{r['input']:5s}] {r['latency_s']:5.1f}s -- {r['details'][:70]}")

    print(f"\n  TOTAL: {passed}/{total} passed, {failed} failed")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Save results
    report_path = "tests/smoke_test_results.json"
    with open(report_path, "w") as f:
        json.dump({"results": results, "passed": passed, "failed": failed, "total": total}, f, indent=2)
    print(f"  Report: {report_path}")

    # Shutdown
    await app_state.shutdown()

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
