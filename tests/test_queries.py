"""Test multiple query types against the trained model."""
import sys
sys.path.insert(0, r"d:\Netra")

import torch
from services.models.inference.run_inference import load_model, load_patch, ask

model, adapter, tokenizer = load_model()
patch_id = "S2B_MSIL2A_20170924T093019_N9999_R136_T35VNH_02_39"
s2, s1, source = load_patch(patch_id)

queries = [
    "What land cover types are visible in this satellite image?",
    "Is there any urban area in this image?",
    "Are there water bodies adjacent to agricultural land?",
    "Describe the vegetation patterns visible.",
    "What season does this image appear to be from?",
]

print(f"Patch: {patch_id} ({source})\n")

for q in queries:
    with torch.no_grad():
        response = ask(model, adapter, tokenizer, q, s2, s1, max_new_tokens=100)
    print(f"Q: {q}")
    print(f"A: {response}\n")
