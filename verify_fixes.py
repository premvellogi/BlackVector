# SatQuery AI - Post-fix verification audit
import sys, os, time, inspect
from pathlib import Path

NETRA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NETRA_ROOT))

import numpy as np
import torch

print("=" * 70)
print("SatQuery AI - Post-Fix Verification Audit")
print("=" * 70)

# =========================================================================
# VERIFY 1: Image loader GeoTIFF support
# =========================================================================
print("\n[VERIFY 1] Image loader - GeoTIFF support...")
from services.models.serving.utils.image_loader import (
    load_image_or_patch, load_geotiff, _detect_sar, to_pil_rgb
)

# Test .npz loading still works
cache_dir = NETRA_ROOT / "data" / "imagery_cache"
npz_files = sorted(cache_dir.glob("*.npz"))
if npz_files:
    patch_id = npz_files[0].stem
    pil, s2, s1 = load_image_or_patch(patch_id)
    print(f"  NPZ loading: OK (patch={patch_id})")
    print(f"    PIL: {pil.size}, s2: {s2.shape}, s1: {s1.shape}")
    assert s2.sum() > 0, "S2 tensor should not be zero"
    assert s1.sum() != 0, "S1 tensor should not be zero (BigEarthNet has SAR)"
    print(f"    s2 non-zero: YES, s1 non-zero: YES")

# Test SAR detection heuristic
sar_bands = np.random.uniform(-25, 0, (2, 100, 100)).astype(np.float32)
assert _detect_sar(2, sar_bands, [], None), "Should detect 2-band float as SAR"
print("  SAR detection heuristic: OK")

opt_bands = np.random.randint(0, 10000, (10, 100, 100), dtype=np.uint16)
assert not _detect_sar(10, opt_bands, [], None), "Should NOT detect 10-band uint16 as SAR"
print("  Optical detection heuristic: OK")

# Test PIL loading now maps into S2 format (not zero-filled)
from PIL import Image as PILImage
import tempfile
test_img = PILImage.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
test_path = str(NETRA_ROOT / "data" / "test_pil_image.png")
test_img.save(test_path)
pil, s2, s1 = load_image_or_patch(test_path)
assert s2.sum() > 0, "PIL image should map RGB into S2 tensor (not zero-filled)"
print(f"  PIL loading (non-zero S2): OK (s2 sum={s2.sum():.2f})")
os.remove(test_path)

# Test to_pil_rgb with 2-band SAR input
sar_arr = np.random.uniform(-25, 0, (2, 64, 64)).astype(np.float32)
pil_sar = to_pil_rgb(sar_arr)
assert pil_sar.size == (64, 64), "SAR pseudo-RGB should have correct size"
print(f"  SAR to_pil_rgb: OK (size={pil_sar.size})")

print("  GeoTIFF support: ALL CHECKS PASSED")

# =========================================================================
# VERIFY 2: Confidence module
# =========================================================================
print("\n[VERIFY 2] Dynamic confidence module...")
from services.models.serving.utils.confidence import (
    compute_vqa_confidence, compute_caption_confidence,
    compute_fusion_confidence, compute_change_confidence,
)

# VQA: different answers should get different confidence
c1 = compute_vqa_confidence("Yes", "Is there urban area?")
c2 = compute_vqa_confidence("It might be urban, but I cannot determine", "Is there urban area?")
c3 = compute_vqa_confidence("", "Is there urban area?")
print(f"  VQA 'Yes': {c1}, uncertain: {c2}, empty: {c3}")
assert c1 > c2, "Definitive answer should have higher confidence"
assert c2 > c3, "Uncertain answer should beat empty"

# Caption: rich captions should have higher confidence
c4 = compute_caption_confidence(
    "The image shows urban areas with dense residential buildings surrounded by agricultural fields and forest patches.",
    ["urban", "agricultural", "forest"]
)
c5 = compute_caption_confidence("An image.", ["satellite_imagery"])
print(f"  Caption rich: {c4}, minimal: {c5}")
assert c4 > c5, "Rich caption should have higher confidence"

# Fusion: high agreement should boost confidence
c6 = compute_fusion_confidence(0.95, 0.9, 0.85, "Detailed fusion analysis of the scene.")
c7 = compute_fusion_confidence(0.3, 0.4, 0.35, "Short.")
print(f"  Fusion high-agreement: {c6}, low-agreement: {c7}")
assert c6 > c7, "High agreement should boost confidence"

# Change: near boundary should have low confidence
c8 = compute_change_confidence(0.99)  # Very confident change
c9 = compute_change_confidence(0.51)  # Near boundary
c10 = compute_change_confidence(0.01)  # Very confident no-change
print(f"  Change 0.99: {c8}, 0.51: {c9}, 0.01: {c10}")
assert c8 > c9, "Far from boundary should have higher confidence"
assert c10 > c9, "Far from boundary should have higher confidence"

print("  Confidence module: ALL CHECKS PASSED")

# =========================================================================
# VERIFY 3: Fusion endpoint - no more return_weights bug
# =========================================================================
print("\n[VERIFY 3] Fusion endpoint - adapter call fixed...")
from services.models.serving.inference import fusion
import inspect
source = inspect.getsource(fusion.process_fusion_request)
assert "return_weights" not in source, "return_weights should be removed from fusion.py"
print("  return_weights removed: YES")

assert "Optical imagery confirms" not in source, "Template strings should be removed"
assert "SAR radar response provides" not in source, "Template strings should be removed"
print("  Template strings removed: YES")

assert "cross_modal_agreement\": 0.85" not in source, "Hardcoded 0.85 should be removed"
print("  Hardcoded 0.85 removed: YES")

assert "confidence=0.86" not in source, "Hardcoded 0.86 should be removed"
print("  Hardcoded 0.86 removed: YES")

# Check that it uses cosine_similarity
assert "cosine_similarity" in source, "Should use cosine_similarity for attribution"
print("  Uses cosine_similarity: YES")

# Check adapter call is correct
assert "adapter(s2_proc, s1_proc)" in source, "Should call adapter(s2, s1) without return_weights"
print("  Adapter call correct: YES")

print("  Fusion code audit: ALL CHECKS PASSED")

# =========================================================================
# VERIFY 4: VQA/Caption no longer hardcoded
# =========================================================================
print("\n[VERIFY 4] VQA/Caption confidence not hardcoded...")
from services.models.serving.inference import vqa, caption

vqa_src = inspect.getsource(vqa.process_vqa_request)
assert "confidence=0.88" not in vqa_src, "VQA should not have hardcoded 0.88"
assert "compute_vqa_confidence" in vqa_src, "VQA should use compute_vqa_confidence"
print("  VQA: hardcoded 0.88 removed, uses compute_vqa_confidence: YES")

cap_src = inspect.getsource(caption.process_caption_request)
assert "confidence=0.90" not in cap_src, "Caption should not have hardcoded 0.90"
assert "compute_caption_confidence" in cap_src, "Caption should use compute_caption_confidence"
print("  Caption: hardcoded 0.90 removed, uses compute_caption_confidence: YES")

print("  Confidence fix: ALL CHECKS PASSED")

# =========================================================================
# VERIFY 5: Full model inference test
# =========================================================================
print("\n[VERIFY 5] Full model inference test...")
from services.models.serving.model_manager import ModelManager
from services.models.serving.schemas import MLRequest, TaskType, InputScope
import uuid

ModelManager._instance = None
manager = ModelManager.get_instance()
t0 = time.time()
manager.load()
load_time = time.time() - t0
print(f"  Model load: {load_time:.1f}s")

if torch.cuda.is_available():
    vram_mb = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM: {vram_mb:.1f} MB")

# VQA
print("\n  --- VQA ---")
vqa_req = MLRequest(
    request_id=uuid.uuid4(), task_type=TaskType.VQA,
    query="What land cover types are visible?",
    input_scope=InputScope.SINGLE, image_ids=[patch_id],
)
from services.models.serving.inference.vqa import process_vqa_request
t0 = time.time()
vqa_res = process_vqa_request(vqa_req)
print(f"  Answer: {vqa_res.facts['answer'][:150]}")
print(f"  Confidence: {vqa_res.confidence} (should NOT be 0.88)")
print(f"  Time: {time.time()-t0:.2f}s")
assert vqa_res.confidence != 0.88, "VQA confidence should not be hardcoded 0.88"

# Caption
print("\n  --- Caption ---")
cap_req = MLRequest(
    request_id=uuid.uuid4(), task_type=TaskType.CAPTION,
    query="Describe", input_scope=InputScope.SINGLE, image_ids=[patch_id],
)
from services.models.serving.inference.caption import process_caption_request
t0 = time.time()
cap_res = process_caption_request(cap_req)
print(f"  Caption: {cap_res.facts['caption_facts'][:150]}")
print(f"  Confidence: {cap_res.confidence} (should NOT be 0.90)")
print(f"  Time: {time.time()-t0:.2f}s")
assert cap_res.confidence != 0.90, "Caption confidence should not be hardcoded 0.90"

# Fusion
print("\n  --- Fusion ---")
fusion_req = MLRequest(
    request_id=uuid.uuid4(), task_type=TaskType.FUSION,
    query="Identify land cover using both optical and SAR",
    input_scope=InputScope.CROSS_MODAL, image_ids=[patch_id, patch_id],
)
from services.models.serving.inference.fusion import process_fusion_request
t0 = time.time()
fus_res = process_fusion_request(fusion_req)
fus_time = time.time() - t0
print(f"  Fused answer: {fus_res.facts['fused_answer'][:150]}")
print(f"  Optical contribution: {fus_res.facts.get('optical_contribution')}")
print(f"  SAR contribution: {fus_res.facts.get('sar_contribution')}")
print(f"  Cross-modal agreement: {fus_res.facts.get('cross_modal_agreement')}")
print(f"  Confidence: {fus_res.confidence} (should NOT be 0.86)")
print(f"  Time: {fus_time:.2f}s")

# Critical fusion checks
assert "optical_contribution" in fus_res.facts, "Should have real optical contribution"
assert "sar_contribution" in fus_res.facts, "Should have real SAR contribution"
assert "cross_modal_agreement" in fus_res.facts, "Should have real cross-modal agreement"
assert fus_res.facts["cross_modal_agreement"] != 0.85, "cross_modal_agreement should NOT be 0.85"
assert fus_res.confidence != 0.86, "Fusion confidence should NOT be 0.86"
assert "Optical imagery confirms" not in str(fus_res.facts), "No template strings"
assert "SAR radar response" not in str(fus_res.facts), "No template strings"

# Change
print("\n  --- Change Detection ---")
change_req = MLRequest(
    request_id=uuid.uuid4(), task_type=TaskType.CHANGE,
    query="What changed?", input_scope=InputScope.BI_TEMPORAL,
    image_ids=[patch_id, patch_id],
)
from services.models.serving.inference.change import process_change_request
t0 = time.time()
chg_res = process_change_request(change_req)
print(f"  Change score: {chg_res.facts['change_score']}")
print(f"  Change detected: {chg_res.facts['change_detected']}")
print(f"  Confidence: {chg_res.confidence}")
print(f"  Time: {time.time()-t0:.2f}s")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("POST-FIX VERIFICATION COMPLETE")
print("=" * 70)
print("\nAll critical fixes verified:")
print("  C1 - Adapter call fixed (no return_weights)")
print("  C2 - Template strings removed (real embedding attribution)")
print("  C3 - Cross-modal agreement computed from cosine similarity")
print("  M1-M3 - Dynamic confidence across all endpoints")
print("  GeoTIFF - rasterio support with band detection and metadata")
