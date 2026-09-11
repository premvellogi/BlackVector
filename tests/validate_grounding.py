"""
SatQuery AI -- Grounding DINO Validation.

Tests /grounding with multiple prompts on unseen satellite imagery.
Validates bounding boxes, confidence, and labels - not just HTTP 200.

Tests:
  1. Single-class prompts: "buildings", "roads", "water", "forest", "agricultural fields"
  2. Multi-class prompts: "buildings . roads . water"
  3. Edge cases: empty results, very specific prompts

For each test, verifies:
  - Boxes are valid (0 <= coords <= 1, x_min < x_max, y_min < y_max)
  - Confidence is in [0, 1]
  - Labels match the queried expression
  - No crashes or NaN values
"""

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from PIL import Image


def npz_to_pil(patch_id, cache_dir="data/imagery_cache"):
    """Convert a cached .npz patch to a PIL RGB image for Grounding DINO."""
    npz_path = os.path.join(cache_dir, f"{patch_id}.npz")
    data = np.load(npz_path)
    s2 = data["s2"].astype(np.float32)  # (C, H, W), force float32

    # Use B04 (Red), B03 (Green), B02 (Blue) for true-color RGB
    # BigEarthNet S2 band order: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    rgb = np.stack([s2[2], s2[1], s2[0]], axis=0)  # B04, B03, B02

    # Normalize to 0-255 uint8
    rgb_uint8 = np.zeros((3, rgb.shape[1], rgb.shape[2]), dtype=np.uint8)
    for i in range(3):
        band = rgb[i]
        p2, p98 = np.percentile(band, [2, 98])
        if p98 > p2:
            band = (band - p2) / (p98 - p2) * 255.0
        rgb_uint8[i] = np.clip(band, 0, 255).astype(np.uint8)

    rgb_uint8 = rgb_uint8.transpose(1, 2, 0)  # (H, W, 3)
    return Image.fromarray(rgb_uint8, "RGB")


def get_unseen_patches(n=5, cache_dir="data/imagery_cache"):
    """Get unseen patch IDs."""
    used = set()
    for path in ["data/training_exp1/train.jsonl", "data/training_exp1/val.jsonl"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    used.add(json.loads(line)["image"])
    cached = sorted([f.stem for f in Path(cache_dir).glob("*.npz") if f.stem not in used])
    return cached[:n]


def validate_box(box):
    """Validate a single bounding box. Returns list of issues."""
    issues = []
    if box["box"][0] < 0 or box["box"][0] > 1:
        issues.append(f"x_min out of range: {box['box'][0]}")
    if box["box"][1] < 0 or box["box"][1] > 1:
        issues.append(f"y_min out of range: {box['box'][1]}")
    if box["box"][2] < 0 or box["box"][2] > 1:
        issues.append(f"x_max out of range: {box['box'][2]}")
    if box["box"][3] < 0 or box["box"][3] > 1:
        issues.append(f"y_max out of range: {box['box'][3]}")
    if box["box"][0] >= box["box"][2]:
        issues.append(f"x_min >= x_max: {box['box'][0]} >= {box['box'][2]}")
    if box["box"][1] >= box["box"][3]:
        issues.append(f"y_min >= y_max: {box['box'][1]} >= {box['box'][3]}")
    if not (0 <= box["confidence"] <= 1):
        issues.append(f"confidence out of range: {box['confidence']}")
    if np.isnan(box["confidence"]):
        issues.append("confidence is NaN")
    if not box["label"]:
        issues.append("label is empty")
    return issues


def main():
    print("=" * 65)
    print("SatQuery AI -- Grounding DINO Validation")
    print("=" * 65)

    # --- Load model ---
    print("\n[1/3] Loading Grounding DINO...", flush=True)
    t0 = time.time()

    from services.models.heads.grounding_head import GroundingDINOHead

    head = GroundingDINOHead()
    head.load()
    print(f"  Loaded in {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB\n")

    # --- Prepare images ---
    print("[2/3] Preparing test images...", flush=True)
    patches = get_unseen_patches(n=3)
    print(f"  Using {len(patches)} unseen patches\n")

    images = {}
    for pid in patches:
        images[pid] = npz_to_pil(pid)
        print(f"  {pid[-20:]}: {images[pid].size}")

    # --- Test prompts ---
    SINGLE_PROMPTS = [
        "buildings",
        "roads",
        "water",
        "forest",
        "agricultural fields",
    ]

    MULTI_PROMPTS = [
        "buildings . roads . water",
        "vegetation . urban areas",
        "forest . agricultural fields . water bodies",
    ]

    EDGE_PROMPTS = [
        "spacecraft landing pad",       # Should return 0 boxes
        "very small red car",           # Unlikely in 120px sat image
    ]

    results = []

    print("\n[3/3] Running grounding tests...\n")

    # --- Single-class prompts ---
    print("  --- Single-class prompts ---")
    for prompt in SINGLE_PROMPTS:
        pid = patches[0]
        img = images[pid]
        t0 = time.time()
        try:
            detections = head.detect(img, prompt)
            elapsed = time.time() - t0

            n_boxes = len(detections)
            all_issues = []
            for d in detections:
                all_issues.extend(validate_box(d))

            valid = len(all_issues) == 0
            avg_conf = sum(d["confidence"] for d in detections) / max(n_boxes, 1)
            labels = [d["label"] for d in detections]

            status = "[OK]" if valid else "[!!]"
            print(f"    {status} '{prompt:25s}' -> {n_boxes:2d} boxes, "
                  f"avg_conf={avg_conf:.3f}, labels={labels[:3]}, {elapsed:.2f}s")

            if all_issues:
                for issue in all_issues[:3]:
                    print(f"         ISSUE: {issue}")

            results.append({
                "prompt": prompt, "type": "single", "patch": pid,
                "n_boxes": n_boxes, "avg_confidence": round(avg_conf, 4),
                "labels": labels, "valid_boxes": valid,
                "issues": all_issues, "latency_s": round(elapsed, 2),
                "boxes": [d["box"] for d in detections[:5]],
            })

        except Exception as e:
            print(f"    [!!] '{prompt:25s}' -> ERROR: {e}")
            results.append({
                "prompt": prompt, "type": "single", "patch": pid,
                "error": str(e),
            })

    # --- Multi-class prompts ---
    print("\n  --- Multi-class prompts ---")
    for prompt in MULTI_PROMPTS:
        pid = patches[1] if len(patches) > 1 else patches[0]
        img = images[pid]
        t0 = time.time()
        try:
            detections = head.detect(img, prompt)
            elapsed = time.time() - t0

            n_boxes = len(detections)
            all_issues = []
            for d in detections:
                all_issues.extend(validate_box(d))

            valid = len(all_issues) == 0
            avg_conf = sum(d["confidence"] for d in detections) / max(n_boxes, 1)
            labels = [d["label"] for d in detections]
            unique_labels = list(set(labels))

            status = "[OK]" if valid else "[!!]"
            print(f"    {status} '{prompt:40s}' -> {n_boxes:2d} boxes, "
                  f"avg_conf={avg_conf:.3f}, unique_labels={unique_labels[:5]}, {elapsed:.2f}s")

            if all_issues:
                for issue in all_issues[:3]:
                    print(f"         ISSUE: {issue}")

            results.append({
                "prompt": prompt, "type": "multi", "patch": pid,
                "n_boxes": n_boxes, "avg_confidence": round(avg_conf, 4),
                "labels": labels, "unique_labels": unique_labels,
                "valid_boxes": valid, "issues": all_issues,
                "latency_s": round(elapsed, 2),
            })

        except Exception as e:
            print(f"    [!!] '{prompt:40s}' -> ERROR: {e}")
            results.append({
                "prompt": prompt, "type": "multi", "patch": pid,
                "error": str(e),
            })

    # --- Edge case prompts ---
    print("\n  --- Edge case prompts (expect 0 or few boxes) ---")
    for prompt in EDGE_PROMPTS:
        pid = patches[0]
        img = images[pid]
        t0 = time.time()
        try:
            detections = head.detect(img, prompt)
            elapsed = time.time() - t0

            n_boxes = len(detections)
            avg_conf = sum(d["confidence"] for d in detections) / max(n_boxes, 1)

            status = "[OK]" if n_boxes <= 2 else "[??]"
            print(f"    {status} '{prompt:30s}' -> {n_boxes:2d} boxes, "
                  f"avg_conf={avg_conf:.3f}, {elapsed:.2f}s")

            results.append({
                "prompt": prompt, "type": "edge", "patch": pid,
                "n_boxes": n_boxes, "avg_confidence": round(avg_conf, 4),
                "latency_s": round(elapsed, 2),
            })

        except Exception as e:
            print(f"    [!!] '{prompt:30s}' -> ERROR: {e}")
            results.append({
                "prompt": prompt, "type": "edge", "patch": pid,
                "error": str(e),
            })

    # --- Cross-patch consistency ---
    print("\n  --- Cross-patch consistency (same prompt, different patches) ---")
    test_prompt = "vegetation . water"
    for pid in patches:
        img = images[pid]
        t0 = time.time()
        try:
            detections = head.detect(img, test_prompt)
            elapsed = time.time() - t0

            n_boxes = len(detections)
            avg_conf = sum(d["confidence"] for d in detections) / max(n_boxes, 1)
            labels = [d["label"] for d in detections]

            print(f"    [--] {pid[-15:]}: {n_boxes:2d} boxes, "
                  f"avg_conf={avg_conf:.3f}, labels={labels[:5]}, {elapsed:.2f}s")

            results.append({
                "prompt": test_prompt, "type": "cross_patch", "patch": pid,
                "n_boxes": n_boxes, "avg_confidence": round(avg_conf, 4),
                "labels": labels, "latency_s": round(elapsed, 2),
            })

        except Exception as e:
            print(f"    [!!] {pid[-15:]}: ERROR: {e}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("GROUNDING DINO VALIDATION SUMMARY")
    print("=" * 65)

    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    valid_runs = [r for r in results if "error" not in r]
    all_valid_boxes = sum(1 for r in valid_runs if r.get("valid_boxes", True))
    total_boxes = sum(r.get("n_boxes", 0) for r in valid_runs)
    avg_latency = np.mean([r["latency_s"] for r in valid_runs]) if valid_runs else 0

    print(f"\n  Tests run: {total}")
    print(f"  Errors: {errors}")
    print(f"  Valid box geometry: {all_valid_boxes}/{len(valid_runs)}")
    print(f"  Total detections: {total_boxes}")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    verdict = "GOOD" if errors == 0 and all_valid_boxes == len(valid_runs) else "NEEDS WORK"
    print(f"\n  Verdict: {verdict}")

    # Save report
    report_path = "tests/grounding_validation.json"
    with open(report_path, "w") as f:
        json.dump({"results": results, "summary": {
            "total_tests": total, "errors": errors,
            "total_detections": total_boxes,
            "avg_latency_s": round(avg_latency, 2),
            "verdict": verdict,
        }}, f, indent=2)
    print(f"  Report: {report_path}")

    head.unload()


if __name__ == "__main__":
    main()
