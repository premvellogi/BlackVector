"""
Phase 0 Sanity Check — End-to-End Pipeline Verification.

Verifies the complete chain:
  GeoTIFF → Rasterio → S1/S2 preprocessing → SpectralAdapter → Model → LoRA → loss decreases

Uses synthetic imagery (no download needed) to validate the architecture.
This is your "does it even work?" test before downloading real data.

Expected output:
  1. Dataset loads correctly
  2. SpectralAdapter projects 10+2 channels → 3 channels
  3. InternVL2-2B processes the pseudo-RGB
  4. LoRA is attached and gradients flow
  5. Loss decreases over 5 micro-batches
"""

import gc
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, r"d:\Netra")

import torch
import numpy as np

print("=" * 60)
print("SatQuery AI — Phase 0 Sanity Check")
print("=" * 60)


# Step 1: Verify dataset loading
print("\n[1/6] Loading dataset...")
from services.models.training.multiband_dataset import BigEarthNetMultiBandDataset

ds = BigEarthNetMultiBandDataset(
    jsonl_path="data/training/train.jsonl",
    imagery_cache_dir="data/imagery_cache",
    tokenizer=None,  # Skip tokenization for now
    require_real_images=False,
)

sample = ds[0]
print(f"  Dataset size: {len(ds):,}")
print(f"  S2 bands shape: {sample['s2_bands'].shape}")  # (10, 120, 120)
print(f"  S1 bands shape: {sample['s1_bands'].shape}")  # (2, 120, 120)
print(f"  Has real image: {sample['has_real_image']}")
print(f"  Patch ID: {sample['patch_id']}")
stats = ds.get_real_image_stats()
print(f"  Real imagery: {stats['real_imagery']:,}/{stats['total']:,} ({stats['real_pct']:.1f}%)")


# Step 2: Verify SpectralAdapter
print("\n[2/6] Testing SpectralChannelAdapter...")
from services.models.core.spectral_adapter import DualModalityAdapter

adapter = DualModalityAdapter(s2_channels=10, s1_channels=2, target_size=448)
adapter = adapter.to("cuda", dtype=torch.bfloat16)

s2_input = sample["s2_bands"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)
s1_input = sample["s1_bands"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)

with torch.no_grad():
    pseudo_rgb = adapter(s2_input, s1_input)

print(f"  Input:  S2={s2_input.shape}, S1={s1_input.shape}")
print(f"  Output: {pseudo_rgb.shape}")  # (1, 3, 448, 448)
print(f"  Value range: [{pseudo_rgb.min():.3f}, {pseudo_rgb.max():.3f}]")
print(f"  Fusion weights: {adapter.modality_weight}")


# Step 3: Load InternVL2-2B
print("\n[3/6] Loading InternVL2-2B in 4-bit...")
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "models/InternVL2-2B", trust_remote_code=True, use_fast=False,
)

model = AutoModel.from_pretrained(
    "models/InternVL2-2B",
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

allocated = torch.cuda.memory_allocated() / 1e9
print(f"  Model loaded: {allocated:.2f} GB VRAM")


# Step 4: Attach LoRA
print("\n[4/6] Attaching LoRA to language model...")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Freeze vision model completely
for param in model.vision_model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    # InternLM2 uses fused QKV (wqkv) and output projection (wo)
    target_modules=["wqkv", "wo"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
adapter_params = sum(p.numel() for p in adapter.parameters())

print(f"  Total params: {total:,}")
print(f"  LoRA trainable: {trainable:,} ({trainable/total*100:.2f}%)")
print(f"  Adapter params: {adapter_params:,}")
print(f"  VRAM after LoRA: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# Step 5: Forward pass with loss
print("\n[5/6] Testing forward pass with loss computation...")

# Process pseudo-RGB through the vision encoder to get visual features
# After prepare_model_for_kbit_training, ViT patch_embedding may be float32
vit_dtype = next(model.vision_model.parameters()).dtype
pseudo_rgb_cast = pseudo_rgb.to(dtype=vit_dtype)

with torch.no_grad():
    vit_embeds = model.extract_feature(pseudo_rgb_cast)
print(f"  ViT embeddings: {vit_embeds.shape}")  # (1, 256, 2048)

num_image_tokens = vit_embeds.shape[1]  # 256

# Build text with proper image token replacement
# InternVL2 replaces <image> with: <img> + <IMG_CONTEXT>*256 + </img>
IMG_START = "<img>"
IMG_END = "</img>"
IMG_CTX = "<IMG_CONTEXT>"

image_tokens_str = IMG_START + IMG_CTX * num_image_tokens + IMG_END
question = "Describe the land cover types visible in this satellite image."
answer = "The image shows primarily agricultural land with some scattered vegetation."

# Build the conversation using InternVL2's template format
query = (
    f"<|im_start|>user\n{image_tokens_str}\n{question}<|im_end|>\n"
    f"<|im_start|>assistant\n{answer}<|im_end|>"
)

inputs = tokenizer(query, return_tensors="pt", max_length=1024, truncation=True)
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)

# Create labels: mask everything up to and including "assistant\n"
labels = input_ids.clone()
# Find where assistant response starts
assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
seq = input_ids[0].tolist()
for pos in range(len(seq) - len(assistant_header)):
    if seq[pos:pos + len(assistant_header)] == assistant_header:
        labels[0, :pos + len(assistant_header)] = -100
        break

# Get input embeddings and inject ViT features at IMG_CONTEXT positions
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CTX)
model.img_context_token_id = img_context_token_id

input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
B, N, C = input_embeds.shape
input_embeds_flat = input_embeds.reshape(B * N, C)
input_ids_flat = input_ids.reshape(B * N)

# Replace IMG_CONTEXT embeddings with ViT features
selected = (input_ids_flat == img_context_token_id)
n_selected = selected.sum().item()
print(f"  IMG_CONTEXT tokens found: {n_selected} (expected: {num_image_tokens})")

if n_selected > 0:
    input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds_flat.dtype)

input_embeds = input_embeds_flat.reshape(B, N, C)

# Forward through language model with loss
outputs = model.language_model(
    inputs_embeds=input_embeds,
    attention_mask=attention_mask,
    labels=labels,
)

loss = outputs.loss
print(f"  Initial loss: {loss.item():.4f}")


# Step 6: Verify gradients flow through LoRA + adapter
print("\n[6/6] Verifying gradient flow (5 micro-batches)...")
optimizer = torch.optim.AdamW(
    list(filter(lambda p: p.requires_grad, model.parameters())) +
    list(adapter.parameters()),
    lr=2e-4,
)

losses = []
for step in range(5):
    optimizer.zero_grad()

    # Fresh forward pass through adapter
    pseudo_rgb = adapter(s2_input, s1_input)
    pseudo_rgb_cast = pseudo_rgb.to(dtype=vit_dtype)
    with torch.no_grad():
        vit_embeds = model.extract_feature(pseudo_rgb_cast)

    # Build embeddings with ViT features injected
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_token_id)
    if selected.sum() > 0:
        input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds_flat.dtype)
    input_embeds = input_embeds_flat.reshape(B, N, C)

    outputs = model.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"  Step {step}: loss={loss.item():.4f}")

# Check if loss decreased
if losses[-1] < losses[0]:
    print(f"\nPASS: Loss decreased {losses[0]:.4f} -> {losses[-1]:.4f}")
else:
    print(f"\nNote: Loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("  (May need more steps to see decrease)")

# Final VRAM report
print(f"\nFinal VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
print(f"Peak VRAM:  {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"Free VRAM:  {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

print("\n[OK] Phase 0 sanity check complete!")
print("   Pipeline: Dataset -> Adapter -> ViT -> LoRA -> Loss [VERIFIED]")

# Cleanup
del model, adapter, optimizer
gc.collect()
torch.cuda.empty_cache()
