"""
SatQuery AI -- Inference Script.

Load the LoRA-adapted InternVL2-2B and run inference on satellite imagery.

Usage:
    # Interactive mode (type questions)
    python services/models/inference/run_inference.py

    # Single query
    python services/models/inference/run_inference.py --query "Describe the land cover"

    # With a specific patch
    python services/models/inference/run_inference.py --patch-id S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np


def load_model(
    base_model_path: str = "models/InternVL2-2B",
    lora_checkpoint: str = "checkpoints/satquery-lora/best",
    s2_channels: int = 10,
    s1_channels: int = 2,
):
    """Load InternVL2-2B + LoRA + SpectralAdapter for inference."""
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from services.models.core.spectral_adapter import DualModalityAdapter

    print("Loading InternVL2-2B + LoRA adapter...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, use_fast=False,
    )

    model = AutoModel.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Load LoRA weights
    if os.path.exists(lora_checkpoint):
        model = PeftModel.from_pretrained(model, lora_checkpoint)
        print(f"  LoRA loaded from: {lora_checkpoint}")
    else:
        print(f"  WARNING: No LoRA checkpoint at {lora_checkpoint}, using base model")

    # Load spectral adapter
    adapter = DualModalityAdapter(
        s2_channels=s2_channels,
        s1_channels=s1_channels,
        target_size=448,
    ).to("cuda", dtype=torch.bfloat16)

    adapter_path = os.path.join(lora_checkpoint, "spectral_adapter.pt")
    if os.path.exists(adapter_path):
        adapter.load_state_dict(torch.load(adapter_path, map_location="cuda", weights_only=True))
        print(f"  SpectralAdapter loaded from: {adapter_path}")
    else:
        print(f"  WARNING: No adapter checkpoint, using random init")

    model.eval()
    adapter.eval()

    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Fusion weights: {adapter.modality_weight}")

    return model, adapter, tokenizer


def load_patch(patch_id: str, cache_dir: str = "data/imagery_cache"):
    """Load a patch from cache and normalize."""
    from services.models.training.multiband_dataset import S2_BANDS, S2_STATS, S1_STATS

    npz_path = os.path.join(cache_dir, f"{patch_id}.npz")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        s2 = data["s2"].astype(np.float32)
        s1 = data.get("s1", np.zeros((2, 120, 120), dtype=np.float32)).astype(np.float32)
        source = "real"
    else:
        # Generate synthetic
        import hashlib
        seed = int(hashlib.md5(patch_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        s2 = np.stack([
            np.clip(rng.normal(S2_STATS[b]["mean"], S2_STATS[b]["std"] * 0.4, (120, 120)), 0, 10000)
            for b in S2_BANDS
        ]).astype(np.float32)
        s1 = np.stack([
            rng.normal(S1_STATS[p]["mean"], S1_STATS[p]["std"] * 0.5, (120, 120))
            for p in ["VV", "VH"]
        ]).astype(np.float32)
        source = "synthetic"

    # Normalize
    s2_mean = np.array([S2_STATS[b]["mean"] for b in S2_BANDS]).reshape(-1, 1, 1)
    s2_std = np.array([S2_STATS[b]["std"] for b in S2_BANDS]).reshape(-1, 1, 1)
    s1_mean = np.array([S1_STATS["VV"]["mean"], S1_STATS["VH"]["mean"]]).reshape(-1, 1, 1)
    s1_std = np.array([S1_STATS["VV"]["std"], S1_STATS["VH"]["std"]]).reshape(-1, 1, 1)

    s2_norm = (s2 - s2_mean) / (s2_std + 1e-8)
    s1_norm = (s1 - s1_mean) / (s1_std + 1e-8)

    s2_tensor = torch.from_numpy(s2_norm).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    s1_tensor = torch.from_numpy(s1_norm).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

    return s2_tensor, s1_tensor, source


@torch.no_grad()
def ask(model, adapter, tokenizer, question: str, s2_bands, s1_bands, max_new_tokens=256):
    """Ask a question about a satellite image."""

    # Get visual features
    vit_dtype = next(model.vision_model.parameters()).dtype
    pseudo_rgb = adapter(s2_bands, s1_bands).to(dtype=vit_dtype)
    vit_embeds = model.extract_feature(pseudo_rgb)

    num_image_tokens = vit_embeds.shape[1]
    image_tokens_str = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"

    # Build prompt
    prompt = (
        f"<|im_start|>user\n{image_tokens_str}\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Inject ViT features
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    B, N, C = input_embeds.shape
    flat = input_embeds.reshape(B * N, C)
    ids_flat = input_ids.reshape(B * N)
    sel = (ids_flat == img_ctx_id)

    if sel.sum() > 0:
        vit_flat = vit_embeds.reshape(-1, C).to(flat.dtype)
        n_use = min(sel.sum().item(), vit_flat.shape[0])
        indices = sel.nonzero(as_tuple=True)[0][:n_use]
        flat[indices] = vit_flat[:n_use]

    input_embeds = flat.reshape(B, N, C)

    # Manual autoregressive generation
    # (HF generate() doesn't work reliably with inputs_embeds)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generated_ids = []
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            # First step: use inputs_embeds (with ViT features injected)
            outputs = model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
        else:
            # Subsequent steps: use last generated token + KV cache
            new_token_embed = model.language_model.get_input_embeddings()(
                torch.tensor([[generated_ids[-1]]], device=model.device)
            )
            new_attn = torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_attn], dim=1)

            outputs = model.language_model(
                inputs_embeds=new_token_embed,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        # Greedy decoding
        next_token = logits.argmax(dim=-1).item()

        if next_token == eos_id or next_token == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token)

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Remove trailing <|im_end|> if present
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()

    return response


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery Inference")
    parser.add_argument("--model-path", default="models/InternVL2-2B")
    parser.add_argument("--lora-path", default="checkpoints/satquery-lora/best")
    parser.add_argument("--patch-id", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--cache-dir", default="data/imagery_cache")
    args = parser.parse_args()

    model, adapter, tokenizer = load_model(args.model_path, args.lora_path)

    # Default patch if none specified
    patch_id = args.patch_id or "S2B_MSIL2A_20170924T093019_N9999_R136_T35VNH_02_39"
    s2, s1, source = load_patch(patch_id, args.cache_dir)
    print(f"\nPatch: {patch_id} ({source} imagery)")

    if args.query:
        # Single query mode
        response = ask(model, adapter, tokenizer, args.query, s2, s1)
        print(f"\nQ: {args.query}")
        print(f"A: {response}")
    else:
        # Interactive mode
        print("\n--- SatQuery AI Interactive Mode ---")
        print("Type a question about the satellite image (or 'quit' to exit)")
        print("Type 'patch <id>' to switch to a different patch\n")

        while True:
            try:
                user_input = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() == "quit":
                break

            if user_input.lower().startswith("patch "):
                patch_id = user_input[6:].strip()
                s2, s1, source = load_patch(patch_id, args.cache_dir)
                print(f"  Switched to: {patch_id} ({source})")
                continue

            response = ask(model, adapter, tokenizer, user_input, s2, s1)
            print(f"A: {response}\n")


if __name__ == "__main__":
    main()
