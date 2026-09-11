"""Quick check: GroundingDINO availability and GPU status."""
import sys

# GPU info
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"VRAM Used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    print(f"VRAM Free: {(props.total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")

# GroundingDINO
print()
try:
    import groundingdino
    print(f"groundingdino: INSTALLED ({groundingdino.__file__})")
except ImportError as e:
    print(f"groundingdino: NOT INSTALLED ({e})")

# Check if weights exist
from pathlib import Path
weights = Path("models/grounding_dino/groundingdino_swint_ogc.pth")
print(f"DINO weights cached: {weights.exists()}")
if weights.exists():
    print(f"  Size: {weights.stat().st_size / 1e6:.1f} MB")

# Alternative: try transformers-based GroundingDINO
print()
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print("transformers GroundingDINO: AVAILABLE")
except ImportError as e:
    print(f"transformers GroundingDINO: NOT AVAILABLE ({e})")

# Check for autodistill or other alternatives
print()
try:
    from autodistill_grounding_dino import GroundingDINO
    print("autodistill GroundingDINO: AVAILABLE")
except ImportError:
    print("autodistill GroundingDINO: NOT AVAILABLE")
