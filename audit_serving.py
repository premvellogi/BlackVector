# SatQuery AI - Independent Audit Script.
# Run from d:/Netra as working directory.
import sys, os, json, time
from pathlib import Path

# Ensure d:\Netra is the root
NETRA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NETRA_ROOT))

import numpy as np
import torch

print("=" * 70)
print("SatQuery AI — Independent Audit")
print(f"Working dir: {os.getcwd()}")
print(f"NETRA_ROOT: {NETRA_ROOT}")
print("=" * 70)

# =========================================================================
# AUDIT 1: Check available real satellite imagery
# =========================================================================
print("\n[AUDIT 1] Checking available real satellite imagery...")
cache_dir = NETRA_ROOT / "data" / "imagery_cache"
npz_files = sorted(cache_dir.glob("*.npz")) if cache_dir.exists() else []
print(f"  Total .npz files in cache: {len(npz_files)}")

real_patch_id = None
if npz_files:
    real_patch_id = npz_files[0].stem
    print(f"  Using real patch: {real_patch_id}")
    data = np.load(npz_files[0])
    print(f"  Keys: {list(data.keys())}")
    for k in data.keys():
        print(f"    {k}: shape={data[k].shape}, dtype={data[k].dtype}")
else:
    print("  WARNING: No cached imagery found.")

# =========================================================================
# AUDIT 2: Check checkpoint files exist
# =========================================================================
print("\n[AUDIT 2] Checking Exp 3 checkpoint...")
adapter_path = NETRA_ROOT / "checkpoints" / "satquery-lora-exp3" / "best"
required_files = ["adapter_config.json", "adapter_model.safetensors", "spectral_adapter.pt"]
for f in required_files:
    p = adapter_path / f
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    status = f"EXISTS ({size:,} bytes)" if exists else "MISSING"
    print(f"  {f}: {status}")

change_head_path = NETRA_ROOT / "checkpoints" / "change_head" / "best.pt"
print(f"  change_head/best.pt: {'EXISTS' if change_head_path.exists() else 'MISSING (will use random init)'}")

# =========================================================================
# AUDIT 3: ModelManager singleton test
# =========================================================================
print("\n[AUDIT 3] Testing ModelManager singleton loading...")
from services.models.serving.model_manager import ModelManager

ModelManager._instance = None

manager = ModelManager.get_instance()
assert not manager.is_loaded, "Manager should not be loaded yet"

t0 = time.time()
manager.load()
load_time = time.time() - t0
print(f"  Load time: {load_time:.1f}s")
assert manager.is_loaded, "Manager should be loaded now"

manager2 = ModelManager.get_instance()
assert manager is manager2, "Singleton broken"
print("  Singleton: VERIFIED (same object identity)")

print(f"  VLM loaded: {manager.vlm is not None and manager.vlm.is_loaded}")
print(f"  SpectralAdapter loaded: {manager.vlm.spectral_adapter is not None}")
print(f"  ChangeDetector loaded: {manager.change_detector is not None}")

if torch.cuda.is_available():
    vram_mb = torch.cuda.memory_allocated() / 1e6
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"  VRAM used: {vram_mb:.1f} MB / {vram_total:.0f} MB")
    print(f"  VRAM safe for RTX 3060: {'YES' if vram_mb < 5500 else 'NO - DANGER'}")

# =========================================================================
# AUDIT 4: VQA - Real model inference
# =========================================================================
print("\n[AUDIT 4] /vqa — Real inference check...")
from services.models.serving.inference.vqa import process_vqa_request
from services.models.serving.schemas import MLRequest, TaskType, InputScope
import uuid

patch_id = real_patch_id if real_patch_id else "synthetic_test"
vqa_req = MLRequest(
    request_id=uuid.uuid4(),
    task_type=TaskType.VQA,
    query="Is there urban or built-up land cover in this satellite image?",
    input_scope=InputScope.SINGLE,
    image_ids=[patch_id],
)

t0 = time.time()
vqa_res = process_vqa_request(vqa_req)
vqa_time = time.time() - t0

print(f"  Status: {vqa_res.status}")
answer = vqa_res.facts.get("answer", "")
print(f"  Answer: {answer[:200]}")
print(f"  Confidence field: {vqa_res.confidence}")
print(f"  Inference time: {vqa_time:.2f}s")

MOCK_PHRASES = ["Yes, the requested feature is present", "No answer generated", "Scene combines structural"]
is_mock = any(p in answer for p in MOCK_PHRASES)
print(f"  Appears mock/hardcoded: {'YES - CRITICAL' if is_mock else 'NO - Real inference confirmed'}")
print(f"  Confidence hardcoded to 0.88: {'YES - ISSUE' if vqa_res.confidence == 0.88 else 'NO'}")

# =========================================================================
# AUDIT 5: Caption - Real model inference
# =========================================================================
print("\n[AUDIT 5] /caption — Real inference check...")
from services.models.serving.inference.caption import process_caption_request

cap_req = MLRequest(
    request_id=uuid.uuid4(),
    task_type=TaskType.CAPTION,
    query="Describe this satellite image",
    input_scope=InputScope.SINGLE,
    image_ids=[patch_id],
)

t0 = time.time()
cap_res = process_caption_request(cap_req)
cap_time = time.time() - t0

caption = cap_res.facts.get("caption_facts", "")
print(f"  Status: {cap_res.status}")
print(f"  Caption: {caption[:300]}")
print(f"  Tags: {cap_res.facts.get('tags', [])}")
print(f"  Inference time: {cap_time:.2f}s")
print(f"  Confidence hardcoded to 0.90: {'YES - ISSUE' if cap_res.confidence == 0.90 else 'NO'}")

# =========================================================================
# AUDIT 6: Fusion - DualModalityAdapter verification
# =========================================================================
print("\n[AUDIT 6] /fusion — DualModalityAdapter verification...")
from services.models.serving.inference.fusion import process_fusion_request

fusion_req = MLRequest(
    request_id=uuid.uuid4(),
    task_type=TaskType.FUSION,
    query="Use the optical and SAR images together to identify built-up regions",
    input_scope=InputScope.CROSS_MODAL,
    image_ids=[patch_id, patch_id],
)

t0 = time.time()
fus_res = process_fusion_request(fusion_req)
fus_time = time.time() - t0

print(f"  Status: {fus_res.status}")
print(f"  Fused answer: {fus_res.facts.get('fused_answer', 'NONE')[:200]}")
opt_answer = fus_res.facts.get("optical_answer", "")
sar_answer = fus_res.facts.get("sar_answer", "")
print(f"  Optical answer: {opt_answer[:100]}")
print(f"  SAR answer: {sar_answer[:100]}")
print(f"  Cross-modal agreement: {fus_res.facts.get('cross_modal_agreement')}")
print(f"  Inference time: {fus_time:.2f}s")
print(f"  Confidence hardcoded to 0.86: {'YES - ISSUE' if fus_res.confidence == 0.86 else 'NO'}")

opt_is_template = "Optical imagery confirms surface features for query:" in opt_answer
sar_is_template = "SAR radar response provides structural geometry for query:" in sar_answer
print(f"  Optical answer is TEMPLATE (not model): {'YES - CRITICAL' if opt_is_template else 'NO'}")
print(f"  SAR answer is TEMPLATE (not model): {'YES - CRITICAL' if sar_is_template else 'NO'}")

cma = fus_res.facts.get("cross_modal_agreement")
print(f"  cross_modal_agreement hardcoded to 0.85: {'YES - ISSUE' if cma == 0.85 else 'NO'}")

# Check DualModalityAdapter.forward() signature
print("\n  Checking DualModalityAdapter.forward() signature...")
from services.models.core.spectral_adapter import DualModalityAdapter
import inspect
sig = inspect.signature(DualModalityAdapter.forward)
print(f"  forward() params: {list(sig.parameters.keys())}")
has_return_weights = "return_weights" in sig.parameters
print(f"  Has return_weights param: {has_return_weights}")
if not has_return_weights:
    print("  CRITICAL: fusion.py calls spectral_adapter(s2, s1, return_weights=True)")
    print("            but DualModalityAdapter.forward() does NOT accept return_weights!")
    print("            This silently falls back to hardcoded 0.70/0.30 weights")

# =========================================================================
# AUDIT 7: Change Detection
# =========================================================================
print("\n[AUDIT 7] /change — ChangeDetector verification...")
from services.models.serving.inference.change import process_change_request

change_req = MLRequest(
    request_id=uuid.uuid4(),
    task_type=TaskType.CHANGE,
    query="What changed between these two dates?",
    input_scope=InputScope.BI_TEMPORAL,
    image_ids=[patch_id, patch_id],
)

t0 = time.time()
chg_res = process_change_request(change_req)
chg_time = time.time() - t0

print(f"  Status: {chg_res.status}")
print(f"  Change detected: {chg_res.facts.get('change_detected')}")
print(f"  Change score: {chg_res.facts.get('change_score')}")
print(f"  Embedding similarity: {chg_res.facts.get('embedding_similarity')}")
print(f"  Inference time: {chg_time:.2f}s")

score = chg_res.facts.get("change_score", 0.5)
if score < 0.2:
    print(f"  Same-patch sanity check: PASSED (low score = {score})")
elif score == 0.50:
    print(f"  Same-patch sanity check: FAILED (stuck at default 0.50)")
else:
    print(f"  Same-patch sanity check: UNCERTAIN (score = {score})")

# =========================================================================
# AUDIT 8: GeoTIFF support
# =========================================================================
print("\n[AUDIT 8] GeoTIFF support check...")
print("  image_loader.py direct file handler: Uses PIL.Image.open()")
print("  Problem: PIL cannot open uint16 GeoTIFFs (Sentinel-2)")
print("  Problem: For direct file path, s2/s1 tensors are ZERO-filled (line 69-70)")
print("  Status: GeoTIFF support is PARTIAL — works via .npz cache only")

# =========================================================================
# AUDIT 9: Schema contract matching
# =========================================================================
print("\n[AUDIT 9] Schema contract matching vs B1 Controller...")
controller_fields = {"task_type", "status", "facts", "evidence", "confidence",
                     "modality_attribution", "disagreement", "error", "model_version"}
our_fields = set(vqa_res.model_dump().keys())
missing = controller_fields - our_fields
extra = our_fields - controller_fields
print(f"  Missing from Controller contract: {missing if missing else 'None'}")
print(f"  Extra fields: {extra if extra else 'None'}")

# =========================================================================
# AUDIT 10: Grounding status
# =========================================================================
print("\n[AUDIT 10] Grounding endpoint status...")
from services.models.serving.server import app
from fastapi.testclient import TestClient

with TestClient(app) as client:
    gres = client.post("/grounding", json={
        "request_id": str(uuid.uuid4()),
        "task_type": "grounding",
        "query": "Where are the buildings?",
        "input_scope": "single",
        "image_ids": ["test"],
    })
    gdata = gres.json()
    correctly_disabled = gdata.get("status") == "failure" and "not yet integration ready" in gdata.get("error", "")
    print(f"  Correctly marked NOT READY: {'YES' if correctly_disabled else 'NO'}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
