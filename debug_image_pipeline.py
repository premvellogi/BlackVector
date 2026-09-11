"""
Diagnostic: What does InternVL2 actually see?

This script:
1. Loads the same TIFF file the user uploads
2. Runs it through the EXACT same preprocessing pipeline
3. Saves the resulting PIL image to disk so we can inspect it visually
4. Runs inference and logs the raw output

IMPORTANT: Run with the virtual environment Python:
    d:\\Netra\\satquery_env\\Scripts\\python.exe debug_image_pipeline.py <path_to_your_tiff>

NOT: python debug_image_pipeline.py (this uses system Python which lacks rasterio/transformers)
"""
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add Netra root
NETRA_ROOT = str(Path(__file__).resolve().parent)
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)


def check_dependencies():
    """Check that critical packages are available."""
    missing = []
    for pkg in ["rasterio", "torch", "PIL", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error("=" * 60)
        logger.error("MISSING PACKAGES: %s", ", ".join(missing))
        logger.error("")
        logger.error("You are running with: %s", sys.executable)
        logger.error("")
        logger.error("You MUST use the virtual environment Python:")
        logger.error('  d:\\Netra\\satquery_env\\Scripts\\python.exe debug_image_pipeline.py "path\\to\\image.tiff"')
        logger.error("=" * 60)
        sys.exit(1)

    logger.info("Python: %s", sys.executable)
    import rasterio
    logger.info("rasterio: %s ✅", rasterio.__version__)
    import torch
    logger.info("torch: %s (CUDA: %s) ✅", torch.__version__, torch.cuda.is_available())


def run_diagnostic(image_path: str):
    check_dependencies()

    from services.models.serving.utils.image_loader import load_image_or_patch
    import numpy as np

    logger.info("=" * 60)
    logger.info("STEP 1: Loading image through image_loader pipeline")
    logger.info("File: %s", image_path)
    logger.info("File size: %d bytes", os.path.getsize(image_path))
    logger.info("=" * 60)

    try:
        pil_img, s2_tensor, s1_tensor = load_image_or_patch(image_path)
    except Exception as exc:
        logger.error("Image loading FAILED: %s", exc)
        logger.error("This means InternVL2 cannot process this file at all.")
        return

    logger.info("PIL image size: %s", pil_img.size)
    logger.info("PIL image mode: %s", pil_img.mode)
    logger.info("S2 tensor shape: %s (non-zero: %s)", s2_tensor.shape, (s2_tensor != 0).any().item())
    logger.info("S1 tensor shape: %s (non-zero: %s)", s1_tensor.shape, (s1_tensor != 0).any().item())

    # Save what the model sees
    out_path = Path("debug_what_model_sees.png")
    pil_img.save(str(out_path))
    logger.info("✅ Saved PIL image to: %s", out_path.absolute())
    logger.info("   >>> OPEN THIS FILE to see EXACTLY what InternVL2 receives <<<")

    # Analyze pixel statistics
    arr = np.array(pil_img)
    logger.info("Pixel stats: min=%d, max=%d, mean=%.1f, std=%.1f",
                arr.min(), arr.max(), arr.mean(), arr.std())

    if arr.mean() < 10:
        logger.warning("⚠️  IMAGE IS NEARLY BLACK! Model will hallucinate.")
    elif arr.std() < 5:
        logger.warning("⚠️  IMAGE HAS NO CONTRAST! Model may hallucinate.")
    else:
        logger.info("✅ Image has reasonable pixel values")

    # Per-channel stats
    for i, ch in enumerate(["R", "G", "B"]):
        logger.info("  %s channel: min=%d, max=%d, mean=%.1f",
                    ch, arr[:, :, i].min(), arr[:, :, i].max(), arr[:, :, i].mean())

    all_same = (arr[:, :, 0] == arr[:, :, 1]).all() and (arr[:, :, 1] == arr[:, :, 2]).all()
    if all_same:
        logger.warning("⚠️  ALL 3 CHANNELS IDENTICAL — single-band grayscale as RGB")
        logger.warning("   Model gets NO color information. This limits visual analysis.")

    logger.info("=" * 60)
    logger.info("STEP 2: Running inference (if ML server is running)")
    logger.info("=" * 60)

    try:
        import requests
        health = requests.get("http://localhost:8200/health", timeout=3)
        if health.ok and health.json().get("model_loaded"):
            logger.info("ML server is running and model is loaded ✅")

            # Upload the file
            with open(image_path, "rb") as f:
                upload_resp = requests.post(
                    "http://localhost:8200/upload",
                    files={"file": (os.path.basename(image_path), f)},
                    timeout=30,
                )
            if not upload_resp.ok:
                logger.error("Upload failed: %s", upload_resp.text)
                return

            uploaded_id = upload_resp.json()["image_id"]
            logger.info("Uploaded → image_id: %s", uploaded_id)

            # Send VQA request directly to ML server
            import uuid
            vqa_payload = {
                "request_id": str(uuid.uuid4()),
                "task_type": "vqa",
                "query": "What do you see in this image? Describe only what is visually observable.",
                "input_scope": "single",
                "image_ids": [uploaded_id],
                "metadata": {},
            }
            logger.info("Sending VQA request...")
            vqa_resp = requests.post("http://localhost:8200/vqa", json=vqa_payload, timeout=120)
            if vqa_resp.ok:
                result = vqa_resp.json()
                answer = result.get("facts", {}).get("answer", "N/A")
                img_stats = result.get("facts", {}).get("image_stats", {})
                logger.info("=" * 60)
                logger.info("MODEL ANSWER: %s", answer)
                logger.info("Image stats from server: %s", img_stats)
                logger.info("Confidence: %s", result.get("confidence"))
                logger.info("=" * 60)

                # Check for hallucination markers
                hallu_markers = ["Finland", "Ireland", "spring", "climate zone",
                                 "square meters", "square kilometres", "hectares"]
                found = [m for m in hallu_markers if m.lower() in answer.lower()]
                if found:
                    logger.warning("⚠️  HALLUCINATION DETECTED: answer contains %s", found)
                else:
                    logger.info("✅ No obvious hallucination markers detected")
            else:
                logger.error("VQA request failed: %s", vqa_resp.text)
        else:
            logger.warning("ML server not running or model not loaded. Skipping inference test.")
            logger.info("Start the server first:")
            logger.info('  d:\\Netra\\satquery_env\\Scripts\\python.exe -m uvicorn services.models.serving.server:app --host 0.0.0.0 --port 8200')
    except requests.ConnectionError:
        logger.warning("ML server not reachable at localhost:8200. Skipping inference test.")
    except Exception as e:
        logger.error("Inference test failed: %s", e)
        import traceback
        traceback.print_exc()

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("Key file: %s", out_path.absolute())
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  d:\\Netra\\satquery_env\\Scripts\\python.exe debug_image_pipeline.py "C:\\path\\to\\image.tiff"')
        print("")
        print("⚠️  You MUST use the virtual environment Python, not system python!")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    run_diagnostic(image_path)
