"""Inspect InternLM2 tokenizer special tokens and chat template."""
import sys
sys.path.insert(0, r"d:\Netra")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/InternVL2-2B", trust_remote_code=True, use_fast=False)

# Check special tokens
print("EOS:", tok.eos_token, tok.eos_token_id)
print("BOS:", tok.bos_token, tok.bos_token_id)
print("PAD:", tok.pad_token, tok.pad_token_id)

# Check image-related tokens
for name in ["<img>", "</img>", "<IMG_CONTEXT>", "<image>", "<|im_start|>", "<|im_end|>"]:
    tid = tok.convert_tokens_to_ids(name)
    print(f"  {name}: id={tid}")

# Check the conversation template
sys.path.insert(0, r"d:\Netra\models\InternVL2-2B")
from conversation import get_conv_template
template = get_conv_template("internlm2-chat")
print(f"\nTemplate sep: repr={repr(template.sep)}")
print(f"Template roles: {template.roles}")
sys_msg = template.system_message or "None"
print(f"System: {sys_msg[:80]}")

# Build a sample conversation to see the exact format
template.append_message(template.roles[0], "Describe this image.")
template.append_message(template.roles[1], None)
prompt = template.get_prompt()
print(f"\nSample prompt:\n{repr(prompt[:300])}")

# Tokenize it and check
ids = tok.encode(prompt)
print(f"\nTokenized length: {len(ids)}")
print(f"First 20 token ids: {ids[:20]}")
print(f"Decoded first 20: {tok.decode(ids[:20])}")
