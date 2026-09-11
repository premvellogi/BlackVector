"""Test Mistral API with encoding fix."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import httpx
import time
import json

API_KEY = "jWjpFRorwfpIbG058855GFPC1WPlTLHX"
URL = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Test with open-mistral-nemo (free tier model)
models_to_try = ["open-mistral-nemo", "mistral-small-latest"]

for model in models_to_try:
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"{'='*50}")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a satellite image analysis expert. Never use emojis."},
            {"role": "user", "content": (
                "A classification system detected these land cover types:\n"
                "- Coniferous forest (92%)\n"
                "- Mixed forest (78%)\n"  
                "- Inland waters (65%)\n\n"
                "Describe this satellite image in 2-3 sentences. "
                "Do NOT mention countries, seasons, or measurements."
            )},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    
    try:
        resp = httpx.post(URL, headers=headers, json=payload, timeout=30.0)
        rl = resp.headers.get("x-ratelimit-remaining-req-minute", "?")
        rl_max = resp.headers.get("x-ratelimit-limit-req-minute", "?")
        print(f"  Status: {resp.status_code}")
        print(f"  Rate limit: {rl}/{rl_max}")
        
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            # Replace problematic chars for console
            safe_text = text.encode('ascii', errors='replace').decode('ascii')
            print(f"  [PASS] Response ({len(text)} chars):")
            print(f"  {safe_text}")
            print(f"  Tokens: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")
        elif resp.status_code == 429:
            print(f"  [RATE LIMITED] Quota exhausted")
        elif resp.status_code == 401:
            print(f"  [INVALID KEY]")
        else:
            print(f"  [ERROR] {resp.text[:200]}")
    except Exception as e:
        print(f"  [EXCEPTION] {type(e).__name__}: {e}")
    
    time.sleep(2)
