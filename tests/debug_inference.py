"""Debug inference to see what tokens are being generated."""
import sys
sys.path.insert(0, r"d:\Netra")

import torch
from services.models.inference.run_inference import load_model, load_patch

model, adapter, tokenizer = load_model()
patch_id = "S2B_MSIL2A_20170924T093019_N9999_R136_T35VNH_02_39"
s2, s1, source = load_patch(patch_id)

# Get visual features
vit_dtype = next(model.vision_model.parameters()).dtype
with torch.no_grad():
    pseudo_rgb = adapter(s2, s1).to(dtype=vit_dtype)
    vit_embeds = model.extract_feature(pseudo_rgb)

num_image_tokens = vit_embeds.shape[1]
image_tokens_str = "<img>" + "<IMG_CONTEXT>" * num_image_tokens + "</img>"

question = "What land cover types are visible?"
prompt = (
    f"<|im_start|>user\n{image_tokens_str}\n{question}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)

print(f"Input length: {input_ids.shape[1]}")
print(f"Last 5 input tokens: {input_ids[0, -5:].tolist()}")
print(f"Decoded last 5: {tokenizer.decode(input_ids[0, -5:])}")

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

print(f"IMG_CONTEXT injected: {sel.sum().item()}")

# Try generate with different settings
with torch.no_grad():
    # Method 1: greedy
    out1 = model.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=False,
    )
    gen1 = out1[0][input_ids.shape[1]:]
    print(f"\nGreedy output ({len(gen1)} tokens): {gen1.tolist()[:20]}")
    print(f"Decoded: '{tokenizer.decode(gen1)}'")

    # Method 2: try with input_ids instead of embeds (no ViT injection)
    out2 = model.language_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=False,
    )
    gen2 = out2[0][input_ids.shape[1]:]
    print(f"\nInput_ids output ({len(gen2)} tokens): {gen2.tolist()[:20]}")
    print(f"Decoded: '{tokenizer.decode(gen2)}'")

    # Method 3: use model.chat() directly (bypasses our adapter)
    print("\n--- Testing model.chat() directly ---")
    try:
        response, history = model.chat(tokenizer, pixel_values=None, question=question, history=[], generation_config=dict(max_new_tokens=50))
        print(f"model.chat() response: '{response}'")
    except Exception as e:
        print(f"model.chat() failed: {e}")
