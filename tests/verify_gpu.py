"""
GPU Verification: Load InternVL2-2B in 4-bit and run a forward pass.
Confirms the model fits in 6GB VRAM before training.
"""
import sys
import gc
import time
import torch

print("=" * 60)
print("SatQuery AI — GPU Verification Test")
print("=" * 60)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
free_vram = torch.cuda.mem_get_info()[0] / 1e9
print(f"GPU: {gpu_name}")
print(f"Total VRAM: {total_vram:.1f} GB")
print(f"Free VRAM: {free_vram:.1f} GB")
print()

# Step 1: Load model in 4-bit
print("[1/4] Loading InternVL2-2B in 4-bit quantization...")
start = time.time()

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_path = "models/InternVL2-2B"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False
)

model = AutoModel.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

load_time = time.time() - start
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
free_after = torch.cuda.mem_get_info()[0] / 1e9

print(f"  Load time: {load_time:.1f}s")
print(f"  VRAM allocated: {allocated:.2f} GB")
print(f"  VRAM reserved: {reserved:.2f} GB")
print(f"  VRAM free: {free_after:.2f} GB")
print()

# Step 2: Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"[2/4] Total parameters: {total_params:,}")
print()

# Step 3: Text-only forward pass (no image needed)
print("[3/4] Running text-only inference via model.chat()...")
start = time.time()

try:
    generation_config = dict(max_new_tokens=50, do_sample=False)
    
    # Text-only test (pixel_values=None is explicitly supported)
    response = model.chat(
        tokenizer,
        pixel_values=None,
        question="What land cover types would you expect in a European agricultural region?",
        generation_config=generation_config,
    )
    
    infer_time = time.time() - start
    print(f"  Response: {response[:300]}")
    print(f"  Inference time: {infer_time:.1f}s")
    print(f"  Status: OK")
except Exception as e:
    infer_time = time.time() - start
    print(f"  Chat error: {type(e).__name__}: {e}")
    
    # Fallback: raw language model forward pass
    print("  Trying raw LM forward pass...")
    inputs = tokenizer("What are the land cover types in Europe?", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.language_model.generate(
            **inputs, max_new_tokens=30, do_sample=False
        )
    
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  LM response: {decoded[:200]}")
    print(f"  Status: OK (via fallback)")

print()

# Step 4: VRAM summary
allocated_final = torch.cuda.memory_allocated() / 1e9
peak = torch.cuda.max_memory_allocated() / 1e9
free_final = torch.cuda.mem_get_info()[0] / 1e9

print(f"[4/4] Final VRAM Report:")
print(f"  Allocated: {allocated_final:.2f} GB")
print(f"  Peak:      {peak:.2f} GB")
print(f"  Free:      {free_final:.2f} GB")
print(f"  Headroom:  {free_final:.2f} GB")
print()

if free_final > 0.5:
    print("PASS: Model fits in VRAM with headroom for LoRA training")
else:
    print("WARNING: Very tight VRAM. May need to reduce max_dynamic_patch.")

# Cleanup
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("\nCleanup done.")
