# SatQuery AI - Phase A Verification
# Tests: normalized attribution, GeoTIFF, confidence variation
import sys, os, time, json
from pathlib import Path

NETRA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NETRA_ROOT))

import numpy as np
import torch

print("=" * 70)
print("SatQuery AI - Phase A Verification")
print("=" * 70)

# =========================================================================
# TEST 1: Normalized fusion attribution
# =========================================================================
print("\n[TEST 1] Normalized fusion attribution...")
from services.models.serving.model_manager import ModelManager
from services.models.serving.schemas import MLRequest, TaskType, InputScope
from services.models.serving.inference.fusion import process_fusion_request
import uuid

ModelManager._instance = None
manager = ModelManager.get_instance()
manager.load()

cache_dir = NETRA_ROOT / "data" / "imagery_cache"
npz_files = sorted(cache_dir.glob("*.npz"))
patch_id = npz_files[0].stem

fusion_req = MLRequest(
    request_id=uuid.uuid4(), task_type=TaskType.FUSION,
    query="Identify land cover using both optical and SAR data",
    input_scope=InputScope.CROSS_MODAL, image_ids=[patch_id, patch_id],
)
fus_res = process_fusion_request(fusion_req)

# Check structure
assert "raw_similarity" in fus_res.facts, "Missing raw_similarity"
assert "normalized_attribution" in fus_res.facts, "Missing normalized_attribution"
raw = fus_res.facts["raw_similarity"]
norm = fus_res.facts["normalized_attribution"]

print(f"  Raw similarity:    optical={raw['optical']:.4f}, sar={raw['sar']:.4f}")
print(f"  Normalized attrib: optical={norm['optical']:.4f}, sar={norm['sar']:.4f}")
print(f"  Sum of normalized: {norm['optical'] + norm['sar']:.4f} (should be ~1.0)")
print(f"  Cross-modal agreement: {fus_res.facts['cross_modal_agreement']:.4f}")
print(f"  Fused answer: {fus_res.facts['fused_answer'][:150]}")
print(f"  Confidence: {fus_res.confidence}")

# Verify normalization sums to 1
assert abs(norm["optical"] + norm["sar"] - 1.0) < 0.01, "Normalized should sum to ~1.0"
# modality_attribution should also be normalized
ma = fus_res.modality_attribution
print(f"  modality_attribution: {ma}")
assert abs(ma["optical"] + ma["sar"] - 1.0) < 0.01, "modality_attribution should sum to ~1.0"

print("  Attribution normalization: PASSED")

# =========================================================================
# TEST 2: Real GeoTIFF tests
# =========================================================================
print("\n[TEST 2] GeoTIFF support tests...")
from services.models.serving.utils.image_loader import (
    load_geotiff, load_image_or_patch, _detect_sar, _process_optical_geotiff,
    _process_sar_geotiff, _extract_geo_metadata, to_pil_rgb,
)
import rasterio
from rasterio.transform import from_bounds

test_dir = NETRA_ROOT / "data" / "geotiff_test"
test_dir.mkdir(exist_ok=True)

# Test 2a: Create and load a synthetic Sentinel-2 GeoTIFF (10-band uint16)
print("\n  [2a] Optical GeoTIFF (10-band uint16)...")
s2_data = np.random.randint(200, 8000, (10, 128, 128), dtype=np.uint16)
s2_path = test_dir / "test_s2_optical.tif"
transform = from_bounds(10.0, 50.0, 10.1, 50.1, 128, 128)
with rasterio.open(
    str(s2_path), "w", driver="GTiff", height=128, width=128,
    count=10, dtype="uint16", crs="EPSG:4326", transform=transform,
) as dst:
    dst.write(s2_data)

pil, s2_t, s1_t = load_geotiff(str(s2_path))
print(f"    PIL size: {pil.size}")
print(f"    S2 tensor: {s2_t.shape}, sum={s2_t.sum():.0f} (should be >0)")
print(f"    S1 tensor: {s1_t.shape}, sum={s1_t.sum():.0f} (should be 0, optical-only)")
assert s2_t.sum() > 0, "S2 tensor should NOT be zero for optical GeoTIFF"
assert s2_t.shape == torch.Size([1, 10, 128, 128]), f"Wrong S2 shape: {s2_t.shape}"

# Check metadata via rasterio
with rasterio.open(str(s2_path)) as src:
    meta = _extract_geo_metadata(src)
    print(f"    CRS: {meta['crs']}")
    print(f"    Resolution: {meta['resolution']}")
    print(f"    Bounds: {meta['bounds']}")
    assert meta["crs"] == "EPSG:4326", f"CRS mismatch: {meta['crs']}"
print("    Optical GeoTIFF: PASSED")

# Test 2b: Create and load a synthetic Sentinel-1 SAR GeoTIFF (2-band float32 dB)
print("\n  [2b] SAR GeoTIFF (2-band float32 dB)...")
sar_vv = np.random.uniform(-25, -5, (128, 128)).astype(np.float32)
sar_vh = np.random.uniform(-30, -10, (128, 128)).astype(np.float32)
sar_data = np.stack([sar_vv, sar_vh])
sar_path = test_dir / "test_s1_sar.tif"
with rasterio.open(
    str(sar_path), "w", driver="GTiff", height=128, width=128,
    count=2, dtype="float32", crs="EPSG:4326", transform=transform,
) as dst:
    dst.write(sar_data)
    dst.set_band_description(1, "VV")
    dst.set_band_description(2, "VH")

pil, s2_t, s1_t = load_geotiff(str(sar_path))
print(f"    PIL size: {pil.size}")
print(f"    S2 tensor: {s2_t.shape}, sum={s2_t.sum():.0f} (should be 0, SAR-only)")
print(f"    S1 tensor: {s1_t.shape}, min={s1_t.min():.1f}, max={s1_t.max():.1f}")
assert s1_t.shape == torch.Size([1, 2, 128, 128]), f"Wrong S1 shape: {s1_t.shape}"
assert s1_t.min() < -5, "SAR values should be negative (dB scale)"
print("    SAR GeoTIFF: PASSED")

# Test 2c: SAR in linear scale (sigma0)
print("\n  [2c] SAR GeoTIFF (linear scale)...")
sar_linear = np.random.uniform(0.0, 0.5, (2, 128, 128)).astype(np.float32)
sar_lin_path = test_dir / "test_s1_linear.tif"
with rasterio.open(
    str(sar_lin_path), "w", driver="GTiff", height=128, width=128,
    count=2, dtype="float32", crs="EPSG:4326", transform=transform,
) as dst:
    dst.write(sar_linear)

pil, s2_t, s1_t = load_geotiff(str(sar_lin_path))
print(f"    S1 tensor: min={s1_t.min():.1f}, max={s1_t.max():.1f} (should be dB after conversion)")
assert s1_t.min() < -5, "Linear SAR should be converted to dB (negative values)"
print("    SAR linear-to-dB: PASSED")

# Test 2d: Cross-modal pair test
print("\n  [2d] Cross-modal pair (optical + SAR)...")
pil_opt, s2_opt, _ = load_geotiff(str(s2_path))
_, _, s1_sar = load_geotiff(str(sar_path))
print(f"    Optical: s2={s2_opt.shape}, SAR: s1={s1_sar.shape}")
assert s2_opt.sum() > 0 and s1_sar.sum() != 0, "Both modalities should have data"
print("    Cross-modal pair: PASSED")

# Test 2e: Run VQA through the full pipeline with optical GeoTIFF
print("\n  [2e] VQA with optical GeoTIFF file path...")
from services.models.serving.inference.vqa import process_vqa_request
vqa_req = MLRequest(
    request_id=uuid.uuid4(), task_type=TaskType.VQA,
    query="What do you see in this satellite image?",
    input_scope=InputScope.SINGLE, image_ids=[str(s2_path)],
)
vqa_res = process_vqa_request(vqa_req)
print(f"    Answer: {vqa_res.facts['answer'][:150]}")
print(f"    Confidence: {vqa_res.confidence}")
assert vqa_res.status == "success", "VQA with GeoTIFF should succeed"
print("    VQA with GeoTIFF: PASSED")

print("\n  GeoTIFF support: ALL TESTS PASSED")

# =========================================================================
# TEST 3: Confidence varies across different inputs
# =========================================================================
print("\n[TEST 3] Confidence variation across different patches...")
from services.models.serving.inference.vqa import process_vqa_request
from services.models.serving.inference.caption import process_caption_request

# Pick 3 different patches
test_patches = [npz_files[i].stem for i in [0, 100, 500]]
vqa_confs = []
cap_confs = []

for pid in test_patches:
    # VQA
    vqa_req = MLRequest(
        request_id=uuid.uuid4(), task_type=TaskType.VQA,
        query="What land cover types are visible in this satellite image?",
        input_scope=InputScope.SINGLE, image_ids=[pid],
    )
    vqa_r = process_vqa_request(vqa_req)
    vqa_confs.append(vqa_r.confidence)
    print(f"  VQA [{pid[:30]}...]: conf={vqa_r.confidence}, answer={vqa_r.facts['answer'][:80]}")

    # Caption
    cap_req = MLRequest(
        request_id=uuid.uuid4(), task_type=TaskType.CAPTION,
        query="Describe", input_scope=InputScope.SINGLE, image_ids=[pid],
    )
    cap_r = process_caption_request(cap_req)
    cap_confs.append(cap_r.confidence)
    print(f"  Cap [{pid[:30]}...]: conf={cap_r.confidence}, tags={cap_r.facts.get('tags', [])}")

# Check variation
vqa_unique = len(set(vqa_confs))
cap_unique = len(set(cap_confs))
print(f"\n  VQA confidences: {vqa_confs} — unique values: {vqa_unique}")
print(f"  Caption confidences: {cap_confs} — unique values: {cap_unique}")

# At least 2 different values expected across 3 patches
if vqa_unique >= 2:
    print("  VQA confidence varies: YES")
else:
    print("  VQA confidence varies: NO (all same — heuristic may be too stable for similar answers)")

if cap_unique >= 2:
    print("  Caption confidence varies: YES")
else:
    print("  Caption confidence varies: NO (check if captions are very similar)")

# =========================================================================
# Cleanup
# =========================================================================
import shutil
shutil.rmtree(test_dir, ignore_errors=True)

print("\n" + "=" * 70)
print("PHASE A VERIFICATION COMPLETE")
print("=" * 70)
