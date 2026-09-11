"""
Full End-to-End Visual Test — Grounding + BBox Drawing + Mistral + Validator.

Tests the complete Phase 6 pipeline on real satellite imagery:
1. Loads a satellite patch from BigEarthNet
2. Runs Grounding DINO for object detection
3. Draws bounding boxes with labels on the image
4. Generates natural language description via Mistral (or template fallback)
5. Validates the answer through the grounding validator
6. Saves annotated images as visual proof
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set Mistral API key
os.environ["MISTRAL_API_KEY"] = "jWjpFRorwfpIbG058855GFPC1WPlTLHX"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(r"C:\Users\AJINKYA SINGH\.gemini\antigravity-ide\brain\54410963-3611-4c2c-9f25-5a3ae3fb6406")


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
    output_dir = Path("d:/Netra/outputs/grounding_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(sample_json) as f:
        sample_data = json.load(f)

    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    from services.models.heads.grounding_head import GroundingDINOHead
    from services.models.serving.utils.bbox_drawer import (
        draw_detections,
        create_detection_summary_image,
    )
    from services.models.core.mistral_client import MistralClient
    from services.models.core.answer_validator import validate_answer

    dino = GroundingDINOHead(device=device, box_threshold=0.20, text_threshold=0.15)
    dino.load()
    mistral = MistralClient()
    print(f"  Mistral API available: {mistral.is_available}")

    # Define test scenarios
    test_scenarios = [
        {
            "name": "water_detection",
            "query": "water . lake . river",
            "ben_classes": ["Inland waters", "Marine waters"],
            "description": "Detecting water bodies",
        },
        {
            "name": "urban_detection",
            "query": "buildings . houses . urban area",
            "ben_classes": ["Urban fabric", "Industrial or commercial units"],
            "description": "Detecting urban structures",
        },
        {
            "name": "forest_detection",
            "query": "forest . trees . woodland",
            "ben_classes": ["Coniferous forest", "Mixed forest", "Broad-leaved forest"],
            "description": "Detecting forested areas",
        },
        {
            "name": "agriculture_detection",
            "query": "agricultural fields . farmland . crops",
            "ben_classes": ["Arable land", "Pastures"],
            "description": "Detecting agricultural land",
        },
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['description']}")
        print(f"  Query: '{scenario['query']}'")
        print(f"{'='*60}")

        # Find a suitable patch
        test_patch = None
        patch_labels = []
        for p, labels in sample_data["patch_labels"].items():
            if p in sample_data["val"]:
                for cls in scenario["ben_classes"]:
                    if cls in labels:
                        test_patch = p
                        patch_labels = labels
                        break
            if test_patch:
                break

        if not test_patch:
            print(f"  [SKIP] No patch found")
            continue

        print(f"  Patch: {test_patch}")
        print(f"  Ground truth: {patch_labels}")

        # Load image
        patch_dir = find_patch_dir(s2_root, test_patch)
        img = load_rgb_from_s2_patch(patch_dir)
        if img is None:
            print(f"  [SKIP] Could not load image")
            continue

        # Step 1: Run Grounding DINO
        print(f"\n  [1/4] Running Grounding DINO...")
        detections = dino.detect(img, scenario["query"])
        print(f"        Detections: {len(detections)}")
        for i, det in enumerate(detections):
            box = det["box"]
            print(f"        #{i+1}: '{det['label']}' conf={det['confidence']:.3f} "
                  f"box=[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]")

        # Step 2: Draw bounding boxes
        print(f"\n  [2/4] Drawing bounding boxes...")
        annotated = create_detection_summary_image(
            img, detections,
            title=f"{scenario['description']} ({len(detections)} detections)",
        )
        out_path = output_dir / f"{scenario['name']}.png"
        annotated.save(str(out_path), quality=95)
        print(f"        Saved: {out_path}")

        # Also save to artifacts for display
        artifact_path = ARTIFACTS_DIR / f"grounding_{scenario['name']}.png"
        annotated.save(str(artifact_path), quality=95)

        # Step 3: Generate description via Mistral
        print(f"\n  [3/4] Generating description via Mistral...")
        grounding_facts = {
            "query": scenario["query"],
            "num_detections": len(detections),
            "detections": [
                {
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "location": f"[{d['box'][0]:.2f}, {d['box'][1]:.2f}, {d['box'][2]:.2f}, {d['box'][3]:.2f}]",
                }
                for d in detections
            ],
            "ground_truth_classes": patch_labels,
            "land_cover_tags": [
                {"class": cls, "confidence": 0.8}
                for cls in patch_labels
            ],
            "dominant_class": patch_labels[0] if patch_labels else "unknown",
            "scene_complexity": len(patch_labels),
        }

        answer = mistral.generate(
            facts=grounding_facts,
            question=f"Describe what was detected when searching for '{scenario['query']}' in this satellite image.",
            task_type="land_cover",
        )
        print(f"        Answer ({len(answer)} chars): {answer[:200]}...")

        # Step 4: Validate answer
        print(f"\n  [4/4] Validating answer...")
        cleaned, violations = validate_answer(answer, grounding_facts, strict=True)
        if violations:
            print(f"        Violations found: {len(violations)}")
            for v in violations:
                print(f"          - [{v['category']}] '{v['match']}' -> {v['action']}")
            print(f"        Cleaned: {cleaned[:200]}...")
        else:
            print(f"        [PASS] No violations - answer is grounded")

        all_results.append({
            "scenario": scenario["name"],
            "patch": test_patch,
            "ground_truth": patch_labels,
            "num_detections": len(detections),
            "detections": [
                {"label": d["label"], "confidence": d["confidence"]}
                for d in detections
            ],
            "answer": cleaned if violations else answer,
            "violations": len(violations),
            "annotated_image": str(out_path),
        })

    # Cleanup
    dino.unload()

    # Summary
    print(f"\n{'='*60}")
    print("FULL E2E TEST SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {'Scenario':<25} {'Detections':>12} {'Violations':>12} {'Status':>10}")
    print(f"  {'-'*60}")

    for r in all_results:
        status = "PASS" if r["num_detections"] > 0 else "WARN"
        print(f"  {r['scenario']:<25} {r['num_detections']:>12} {r['violations']:>12} {status:>10}")

    # Save results
    results_path = output_dir / "e2e_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    total_detections = sum(r["num_detections"] for r in all_results)
    total_violations = sum(r["violations"] for r in all_results)
    print(f"\n  Total detections: {total_detections}")
    print(f"  Total violations: {total_violations}")
    print(f"  Annotated images: {output_dir}")

    if all(r["num_detections"] > 0 for r in all_results):
        print(f"\n  [OK] ALL SCENARIOS PASSED")
    else:
        print(f"\n  [!!] Some scenarios had no detections")

    return True


if __name__ == "__main__":
    main()
