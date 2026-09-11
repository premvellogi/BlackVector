"""Test Mistral API key and full pipeline."""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set the API key
os.environ["MISTRAL_API_KEY"] = "1fXdnGCqtfqOkEJxahZffPmyiVCLv51f"

from services.models.core.mistral_client import MistralClient

print("=" * 60)
print("TEST 1: Mistral API Key Verification")
print("=" * 60)

client = MistralClient()
print(f"  API key set: {bool(client.api_key)}")
print(f"  API available: {client.is_available}")

# Simple test call
test_facts = {
    "land_cover_tags": [
        {"class": "Coniferous forest", "confidence": 0.92},
        {"class": "Mixed forest", "confidence": 0.78},
        {"class": "Inland waters", "confidence": 0.65},
    ],
    "dominant_class": "Coniferous forest",
    "scene_complexity": 3,
}

print("\n  Sending test request to Mistral API...")
try:
    response = client.generate(
        facts=test_facts,
        question="What land cover types are visible in this satellite image?",
        task_type="land_cover",
    )
    print(f"\n  [PASS] API responded ({len(response)} chars)")
    print(f"\n  Response:\n  {response[:500]}")
    print(f"\n  Usage: {client.usage_stats}")
except Exception as e:
    print(f"\n  [FAIL] API error: {e}")

# Test caption generation
print("\n" + "=" * 60)
print("TEST 2: Caption Generation via Mistral")
print("=" * 60)

try:
    caption = client.generate(
        facts=test_facts,
        question="Describe this satellite image.",
        task_type="caption",
    )
    print(f"\n  [PASS] Caption generated ({len(caption)} chars)")
    print(f"\n  Caption:\n  {caption[:500]}")
except Exception as e:
    print(f"\n  [FAIL] Caption error: {e}")

# Test change description
print("\n" + "=" * 60)
print("TEST 3: Change Detection Description via Mistral")
print("=" * 60)

change_facts = {
    "change_score": 0.87,
    "change_detected": True,
    "tags_t1": {"Coniferous forest": 0.91, "Arable land": 0.72},
    "tags_t2": {"Urban fabric": 0.85, "Industrial or commercial units": 0.78},
    "tag_diff": {
        "Coniferous forest": -0.91,
        "Urban fabric": 0.85,
        "Arable land": -0.72,
        "Industrial or commercial units": 0.78,
    },
}

try:
    change_desc = client.generate(
        facts=change_facts,
        question="What changes occurred between the two satellite images?",
        task_type="change",
    )
    print(f"\n  [PASS] Change description generated ({len(change_desc)} chars)")
    print(f"\n  Description:\n  {change_desc[:500]}")
except Exception as e:
    print(f"\n  [FAIL] Change error: {e}")

# Test answer validator on the response
print("\n" + "=" * 60)
print("TEST 4: Answer Validator on Mistral Output")
print("=" * 60)

from services.models.core.answer_validator import validate_answer

# Simulate a hallucinated answer
fake_answer = (
    "This satellite image captures a forested region in Finland during summer. "
    "The area spans approximately 1,440,000 square meters of boreal landscape. "
    "The Mediterranean climate supports diverse vegetation."
)

cleaned, violations = validate_answer(fake_answer, test_facts, strict=True)
print(f"\n  Original:  {fake_answer}")
print(f"\n  Cleaned:   {cleaned}")
print(f"\n  Violations found: {len(violations)}")
for v in violations:
    print(f"    - [{v['category']}] '{v['match']}' -> {v['action']}")

# Now validate the actual Mistral response
print("\n  Validating actual Mistral response...")
cleaned_real, violations_real = validate_answer(response, test_facts, strict=True)
print(f"  Violations in real response: {len(violations_real)}")
if violations_real:
    for v in violations_real:
        print(f"    - [{v['category']}] '{v['match']}' -> {v['action']}")
else:
    print("  [PASS] Mistral response passed validation with 0 violations")

print(f"\n  Total API calls: {client.usage_stats}")
