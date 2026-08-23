# BigEarthNet.txt Adaptation Log (§5.1a Compliance)

## Component Adapted
- **Model:** InternVL2-2B (OpenGVLab/InternVL2-2B)
- **Adaptation method:** QLoRA (4-bit NF4 quantization, LoRA rank=8, alpha=16)
- **Target modules:** wqkv, wo (InternLM2 fused QKV + output projection)
- **Vision encoder:** Frozen (not fine-tuned -- critical for 6GB VRAM constraint)

## Dataset: BigEarthNet.txt
- **Source:** https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt
- **Total available:** 9,553,962 annotations across 464,044 co-registered S1+S2 image pairs
- **License:** CDLA Permissive 1.0

## Portion Used
- **Split used:** Official `train` split only (4,674,281 annotations)
- **Patches sampled:** 4,994 of 229,114
- **Annotations after sampling:** 102,283
- **After task capping:** 102,283 instruction entries

## Sampling Strategy
- **Method:** Stratified by acquisition country for geographic diversity
- **Random seed:** 42
- **Task type capping:** Applied to prevent binary VQA from dominating
  - captioning: uncapped (all available)
  - binary: capped to 100,000
  - mcq: capped to 80,000
  - bounding box: capped to 60,000

## Adaptation Objective
Multi-task instruction tuning:
1. **Captioning** — Scene description from satellite imagery
2. **Binary VQA** — Yes/no questions about land cover (presence, area, adjacency, etc.)
3. **MCQ VQA** — Multiple-choice questions (presence, season, climate zone, country)
4. **Bounding Box** — Referring expression detection / LULC localization

## Training Configuration
- Batch size: 1 (gradient accumulation: 8, effective batch: 8)
- Learning rate: 2e-4 (cosine schedule with warmup)
- Epochs: 3
- Max sequence length: 1024 tokens
- Optimizer: paged_adamw_8bit
- Gradient checkpointing: enabled

## Train/Validation Split
- **Train:** 92,251 entries (90%)
- **Val:** 10,032 entries (10%)
- **Split method:** Patch-level split (all annotations for one image stay together) with seed 42

## Task Distribution After Formatting
```
{
  "binary": 38926,
  "bounding box": 23411,
  "captioning": 4994,
  "mcq": 34952
}
```

## Checkpoint
- **Version:** [TO BE FILLED AFTER TRAINING]
- **Path:** checkpoints/satquery-lora/final/
- **Referenced in provenance (D4):** Yes — every inference response includes checkpoint version
