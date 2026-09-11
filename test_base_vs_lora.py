"""
Base vs LoRA Model Comparison Test (Step 7)
============================================
Tests the SAME image + prompt through:
  A) Base InternVL2-2B (no LoRA)
  B) InternVL2-2B + satquery-lora-exp3 (your fine-tuned model)

This tells you definitively whether the short-answer behavior
is from the LoRA training or the generation config.

Run:
  python -X utf8 test_base_vs_lora.py
"""

import sys
import json
import gc
import logging
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "OpenGVLab/InternVL2-2B"
LORA_PATH  = "checkpoints/satquery-lora-exp3/best"

QUESTION = (
    "Describe this satellite image in detail. Write a structured analysis in complete sentences. "
    "Discuss visible land cover, water features, vegetation, terrain, and human-made structures. "
    "Do not guess the location."
)

SYSTEM_INSTRUCTION = (
    "You are a satellite image analysis assistant. "
    "Analyse ONLY what is visually observable in the provided image. "
    "Give a complete, detailed answer in full sentences.\n"
)

PROMPT = (
    f"{SYSTEM_INSTRUCTION}\n"
    f"<image>\n"
    f"Question: {QUESTION}\n"
    f"Answer:"
)

# Generation configs to compare
GEN_CONFIGS = {
    "short (current: temp=0.1, max=512, no sample)": {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.1,
    },
    "long (experimental: temp=0.7, max=512, sample)": {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}


def preprocess_image(pil_img):
    transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def load_base_model():
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    logger.info("Loading BASE InternVL2-2B (no LoRA)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb,
        device_map="auto",
    ).eval()
    logger.info("Base model loaded.")
    return model, tokenizer


def apply_lora(model, lora_path):
    from peft import PeftModel
    logger.info(f"Applying LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    logger.info("LoRA applied.")
    return model


def run_inference(model, tokenizer, pil_img, gen_config: dict, label: str) -> str:
    pixel_values = preprocess_image(pil_img).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, PROMPT, gen_config)
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    print(f"ANSWER: {response}")
    print(f"TOKENS: {len(tokenizer.encode(response))}")
    return response


def find_test_image():
    temp_uploads = Path(tempfile.gettempdir()) / "satquery_uploads"
    if temp_uploads.exists():
        found = sorted(temp_uploads.glob("*.jpg")) + sorted(temp_uploads.glob("*.png"))
        if found:
            return str(found[-1])
    return None


def main():
    print("\n" + "=" * 60)
    print("STEP 7: Base InternVL2-2B vs LoRA Comparison")
    print("=" * 60)
    print(f"PROMPT:\n{PROMPT}\n")

    img_path = find_test_image()
    if img_path:
        print(f"Using image: {img_path}")
        pil_img = Image.open(img_path).convert("RGB")
    else:
        print("No uploaded image found. Creating synthetic test image...")
        import numpy as np
        arr = (np.random.rand(343, 343, 3) * 255).astype("uint8")
        pil_img = Image.fromarray(arr)

    results = {}

    # ── Test A: Base model ─────────────────────────────────────
    print("\n\n>>> TEST A: BASE InternVL2-2B (no LoRA)")
    base_model, tokenizer = load_base_model()
    for cfg_name, cfg in GEN_CONFIGS.items():
        key = f"BASE | {cfg_name}"
        results[key] = run_inference(base_model, tokenizer, pil_img, cfg, key)

    # Free VRAM
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Base model freed from VRAM.")

    # ── Test B: Base + LoRA ─────────────────────────────────────
    lora_path = Path(LORA_PATH)
    if lora_path.exists():
        print("\n\n>>> TEST B: InternVL2-2B + satquery-lora-exp3")
        lora_model, tokenizer = load_base_model()
        lora_model = apply_lora(lora_model, str(lora_path))
        for cfg_name, cfg in GEN_CONFIGS.items():
            key = f"LORA | {cfg_name}"
            results[key] = run_inference(lora_model, tokenizer, pil_img, cfg, key)
        del lora_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[SKIP] LoRA adapter not found at {lora_path}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, answer in results.items():
        tokens = len(answer.split())
        print(f"\n[{label}]")
        print(f"  Words: {tokens}  |  Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")

    print("\n\nINTERPRETATION:")
    print("  Base detailed, LoRA short   -> LoRA training caused the short-answer behavior")
    print("  Both short                  -> Generation config / prompt problem")
    print("  Both detailed               -> Routing or display bug (model is fine)")


if __name__ == "__main__":
    main()
