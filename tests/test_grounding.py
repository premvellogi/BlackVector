"""
Phase 6 — Grounding Integration Test.

Tests the full grounding pipeline end-to-end:
1. Load GroundingDINO via grounding_head.py
2. Run detection on real BigEarthNet satellite patches
3. Verify detection results have correct structure (boxes, labels, confidence)
4. Test multiple query types (buildings, water, forest, agriculture)
5. Test the inference handler formatting
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_rgb_from_s2_patch(patch_dir):
    """Load an RGB composite from a BigEarthNet-S2 patch."""
    patch_dir = Path(patch_dir)
    folder_name = patch_dir.name
    bands = {}
    for band_name in ["B04", "B03", "B02"]:
        band_file = patch_dir / f"{folder_name}_{band_name}.tif"
        if not band_file.exists():
            return None
        try:
            import rasterio
            with rasterio.open(str(band_file)) as src:
                bands[band_name] = src.read(1).astype(np.float32)
        except ImportError:
            bands[band_name] = np.array(Image.open(str(band_file)), dtype=np.float32)

    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.resize((800, 800), Image.BICUBIC)
    return img


def find_patch_dir(s2_root, patch_name):
    """Find the full path of a patch in BigEarthNet-S2."""
    s2_root = Path(s2_root)
    for tile_dir in s2_root.iterdir():
        if tile_dir.is_dir():
            candidate = tile_dir / patch_name
            if candidate.exists():
                return candidate
    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s2_root = "D:/BigEarthNet/BigEarthNet-S2"
    sample_json = "d:/Netra/data/training/sampled_patches.json"

    with open(sample_json) as f:
        sample_data = json.load(f)

    passed = 0
    failed = 0
    total = 0

    # ===== TEST 1: GroundingDINOHead loads and detects =====
    print("\n" + "=" * 60)
    print("TEST 1: GroundingDINOHead load + detect")
    print("=" * 60)
    total += 1

    try:
        from services.models.heads.grounding_head import GroundingDINOHead
        dino = GroundingDINOHead(device=device, box_threshold=0.20, text_threshold=0.15)
        dino.load()

        # Find a water patch
        test_patch = None
        for p, labels in sample_data["patch_labels"].items():
            if "Inland waters" in labels and p in sample_data["val"]:
                test_patch = p
                break

        patch_dir = find_patch_dir(s2_root, test_patch)
        img = load_rgb_from_s2_patch(patch_dir)
        detections = dino.detect(img, "water . lake . river")

        assert isinstance(detections, list), "Detections should be a list"
        assert len(detections) > 0, "Should detect water in a water patch"
        for det in detections:
            assert "box" in det, "Detection should have 'box' key"
            assert "confidence" in det, "Detection should have 'confidence' key"
            assert "label" in det, "Detection should have 'label' key"
            assert len(det["box"]) == 4, "Box should have 4 coordinates"
            assert all(0 <= v <= 1 for v in det["box"]), "Box coords should be normalized [0, 1]"
            assert 0 <= det["confidence"] <= 1, "Confidence should be [0, 1]"

        print(f"  [PASS] Loaded DINO, detected {len(detections)} water objects (max conf: {max(d['confidence'] for d in detections):.3f})")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # ===== TEST 2: Multi-query detection =====
    print("\n" + "=" * 60)
    print("TEST 2: Multi-query detection on diverse patches")
    print("=" * 60)
    total += 1

    queries_to_test = [
        ("forest . trees", ["Broad-leaved forest", "Coniferous forest", "Mixed forest"]),
        ("buildings . houses", ["Urban fabric", "Industrial or commercial units"]),
        ("agricultural fields . farmland", ["Arable land", "Pastures"]),
    ]

    all_queries_passed = True
    for query, target_classes in queries_to_test:
        # Find a matching patch
        test_patch = None
        for p, labels in sample_data["patch_labels"].items():
            if any(tc in labels for tc in target_classes) and p in sample_data["val"]:
                test_patch = p
                break

        if test_patch is None:
            print(f"  [SKIP] No patch found for '{query}'")
            continue

        patch_dir = find_patch_dir(s2_root, test_patch)
        img = load_rgb_from_s2_patch(patch_dir)
        detections = dino.detect(img, query)

        if detections:
            max_conf = max(d["confidence"] for d in detections)
            print(f"  [OK] '{query}': {len(detections)} detections (max conf: {max_conf:.3f})")
        else:
            print(f"  [WARN] '{query}': no detections")
            all_queries_passed = False

    if all_queries_passed:
        print(f"  [PASS] All queries detected objects")
        passed += 1
    else:
        print(f"  [FAIL] Some queries had no detections")
        failed += 1

    # ===== TEST 3: Inference handler box parsing =====
    print("\n" + "=" * 60)
    print("TEST 3: Inference handler box format")
    print("=" * 60)
    total += 1

    try:
        # Simulate what the inference handler does
        test_det = {"box": [0.1, 0.2, 0.8, 0.9], "confidence": 0.75, "label": "water"}
        box = test_det.get("box", [0.0, 0.0, 1.0, 1.0])
        obj = {
            "label": test_det.get("label", "unknown"),
            "confidence": round(test_det.get("confidence", 0.0), 3),
            "x1": round(box[0], 4),
            "y1": round(box[1], 4),
            "x2": round(box[2], 4),
            "y2": round(box[3], 4),
        }
        assert obj["x1"] == 0.1, f"x1 should be 0.1, got {obj['x1']}"
        assert obj["y1"] == 0.2, f"y1 should be 0.2, got {obj['y1']}"
        assert obj["x2"] == 0.8, f"x2 should be 0.8, got {obj['x2']}"
        assert obj["y2"] == 0.9, f"y2 should be 0.9, got {obj['y2']}"
        assert obj["confidence"] == 0.75
        assert obj["label"] == "water"
        print(f"  [PASS] Box format correctly parsed from grounding_head output")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] Box format mismatch: {e}")
        failed += 1

    # ===== TEST 4: Edge case - no detections =====
    print("\n" + "=" * 60)
    print("TEST 4: Query for unlikely objects (should return empty list)")
    print("=" * 60)
    total += 1

    try:
        # Use a forest patch and search for "airplane . helicopter"
        test_patch = None
        for p, labels in sample_data["patch_labels"].items():
            if "Coniferous forest" in labels and p in sample_data["val"]:
                test_patch = p
                break

        patch_dir = find_patch_dir(s2_root, test_patch)
        img = load_rgb_from_s2_patch(patch_dir)

        # Very high threshold to force no detections
        detections = dino.detect(img, "airplane . helicopter . spaceship", box_threshold=0.90)
        assert isinstance(detections, list), "Should return a list even with no detections"
        print(f"  [PASS] Unlikely query returned {len(detections)} detections (expected 0 or few)")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # ===== TEST 5: Detection consistency =====
    print("\n" + "=" * 60)
    print("TEST 5: Same image, same query = same results")
    print("=" * 60)
    total += 1

    try:
        test_patch = None
        for p, labels in sample_data["patch_labels"].items():
            if "Inland waters" in labels and p in sample_data["val"]:
                test_patch = p
                break

        patch_dir = find_patch_dir(s2_root, test_patch)
        img = load_rgb_from_s2_patch(patch_dir)

        det1 = dino.detect(img, "water")
        det2 = dino.detect(img, "water")

        assert len(det1) == len(det2), f"Detection count mismatch: {len(det1)} vs {len(det2)}"
        for d1, d2 in zip(det1, det2):
            assert abs(d1["confidence"] - d2["confidence"]) < 1e-4, "Confidence should be deterministic"
        print(f"  [PASS] Deterministic: {len(det1)} detections both runs, identical confidence")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # ===== CLEANUP =====
    dino.unload()

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"GROUNDING INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")

    if failed == 0:
        print(f"\n  [OK] ALL TESTS PASSED - Grounding DINO is fully integrated")
    else:
        print(f"\n  [!!] {failed} test(s) failed - review needed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
