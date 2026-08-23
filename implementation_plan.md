# SatQuery AI — Revised Solo ML Lead Roadmap (Post Dataset Research)

> [!IMPORTANT]
> **Major revision from v1:** After reading the actual BigEarthNet.txt paper (arXiv:2603.29630), the architecture changes significantly. BigEarthNet.txt is **not** just a multi-label classification dataset — it's a **full vision-language dataset** with 9.6M text annotations. This unlocks a much better training strategy.

---

## What is BigEarthNet.txt (The Real Dataset)

**Paper:** [BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for EO](https://arxiv.org/abs/2603.29630) (March 2026)

| Fact | Value |
|---|---|
| **Image pairs** | 464,044 co-registered Sentinel-1 (SAR) + Sentinel-2 (multispectral) pairs |
| **Text annotations** | 9.6 million total |
| **Annotation types** | Geographically anchored captions (LULC classes, spatial relations, environmental context), VQA pairs (15 task types), referring expression detection instructions for bounding box prediction |
| **Task coverage** | Presence, Area, Counting, Adjacency, Relative Position, Country, Season, Climate Zone, and more |
| **Benchmark split** | 1,082 manually-verified image pairs with 15,029 text annotations |
| **Access** | [HuggingFace: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) (annotations) + [bigearth.net](https://bigearth.net/) (imagery) |

### Why This Changes Everything

The old plan assumed BigEarthNet was only good for multi-label classification (train a tag head, pass tags to LLM). **Now we know it has:**
- ✅ **Captions** → directly train/fine-tune captioning
- ✅ **VQA pairs** → directly train/fine-tune VQA  
- ✅ **Referring expressions + bounding boxes** → directly train/fine-tune grounding
- ✅ **Co-registered SAR + optical pairs** → directly train fusion
- ✅ **The dataset IS multi-sensor** → perfect for cross-modal adaptation

**This means we can fine-tune a small VLM (or adapter) on BigEarthNet.txt's own text annotations, instead of building a hacky tag-head → LLM-prompt pipeline.** This is stronger for judging AND more elegant.

---

## Complete Dataset Reference Card

| Dataset | Purpose in SatQuery | Size | Access |
|---|---|---|---|
| **BigEarthNet.txt** | Primary RS adaptation (F7). Fine-tune VLM on its captions, VQA, and referring expressions. Also use its SAR+optical pairs for fusion training. | 464K image pairs, 9.6M annotations | [HuggingFace](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) + [bigearth.net](https://bigearth.net/) |
| **VRSBench** | Evaluation for captioning, grounding, VQA | 29,614 images, 29,614 captions, 52,472 referring expressions, 123,221 VQA pairs | [HuggingFace: xiang709/VRSBench](https://huggingface.co/datasets/xiang709/VRSBench), [GitHub](https://github.com/lx709/VRSBench) |
| **RSVQA** | Evaluation for single-image VQA | LR (Sentinel-2) + HR (USGS), 1M+ QA pairs | [rsvqa.sylvainlobry.com](https://rsvqa.sylvainlobry.com/), [Zenodo](https://doi.org/10.5281/zenodo.6344366), [HuggingFace: dmarsili/RSVQA-HR-2k](https://huggingface.co/datasets/dmarsili/RSVQA-HR-2k) |
| **CDVQA** | Evaluation for change detection VQA | Multi-temporal image-question-answer triplets | [GitHub: YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA) |

---

## Revised Architecture (6GB VRAM)

The previous plan used: `Backbone → tag head → structured facts → LLM API prompt`.

**Revised approach:** Fine-tune a **small VLM** (or RS-adapted VLM) with LoRA/QLoRA on BigEarthNet.txt's rich text annotations. This is now viable because BigEarthNet.txt provides exactly the kind of instruction-tuning data needed.

```
┌─────────────────────────────────────────────────────────┐
│               One FastAPI Process (services/models/)     │
│                                                          │
│  Shared Component:                                       │
│  ┌──────────────────────────────────────┐                │
│  │ Small VLM (e.g., InternVL2-2B or    │                │
│  │ Florence-2 or Qwen2-VL-2B)          │                │
│  │ + LoRA adapter fine-tuned on         │                │
│  │ BigEarthNet.txt subset               │                │
│  │ (~2B params, 4-bit ≈ 1.5-2GB VRAM)  │                │
│  └──────────────────────────────────────┘                │
│        │          │           │           │        │     │
│  ┌─────▼──┐ ┌─────▼───┐ ┌────▼────┐ ┌───▼───┐ ┌──▼──┐ │
│  │ /vqa   │ │/caption │ │/change  │ │/fusion│ │/gnd │ │
│  │        │ │         │ │(siamese)│ │(dual) │ │DINO │ │
│  └────────┘ └─────────┘ └─────────┘ └───────┘ └─────┘ │
└─────────────────────────────────────────────────────────┘
```

### Two Architecture Options (You Must Pick One)

#### Option A: Small VLM + LoRA (Recommended)
- Use a 2B-class VLM (InternVL2-2B, Qwen2-VL-2B, or Florence-2)
- Fine-tune with QLoRA (4-bit) on BigEarthNet.txt subset
- VQA, captioning, and basic grounding all handled by one model
- Add small adapter heads for change detection (siamese) and fusion (cross-modal)
- **Fits 6GB**: 2B model in 4-bit ≈ 1.5GB + LoRA weights + activations
- **Pros**: End-to-end vision-language reasoning, strongest for judging, BigEarthNet.txt was designed for this
- **Cons**: Longer fine-tuning time, more complex setup

#### Option B: RS Backbone + Heads + LLM API (Original Plan, Still Valid)
- Use Prithvi-100M/Clay-small as vision backbone
- LoRA adapt on BigEarthNet.txt's multi-label LULC classification
- Small task heads on top, LLM API for text generation
- **Fits 6GB easily**: 100M backbone is tiny
- **Pros**: Simpler, faster training, guaranteed to fit in VRAM
- **Cons**: Weaker vision-language reasoning, LLM phrasing adds latency and cost

> [!IMPORTANT]
> **My recommendation: Option A (Small VLM + LoRA)**. BigEarthNet.txt was literally designed for instruction-driven VLM fine-tuning. Using it to train only a classification head wastes its 9.6M text annotations. A 2B VLM in 4-bit fits 6GB and gives you proper VQA/captioning without the tag-head-to-LLM hack. The judges will value this significantly more.

---

## What I (Claude) Will Build For You

### ✅ Code Deliverables

| # | Deliverable | Details |
|---|---|---|
| 1 | **Environment setup** | `environment.yml`, `requirements.txt` with exact versions for RTX 3060 6GB, CUDA, PyTorch, transformers, PEFT, rasterio, GDAL |
| 2 | **Data pipeline** | Scripts to download BigEarthNet.txt annotations from HuggingFace, sample stratified subset, download/link imagery, create train/val splits |
| 3 | **VLM LoRA training script** | Full training loop using HuggingFace PEFT + transformers, QLoRA config, BigEarthNet.txt instruction format, multi-task (VQA + captioning + referring expressions) |
| 4 | **FastAPI service** | One process, 5 endpoints (`/vqa`, `/caption`, `/grounding`, `/change`, `/fusion`), model loaded once, contract schemas enforced |
| 5 | **VQA endpoint** | Image + question → VLM inference → answer + confidence |
| 6 | **Captioning endpoint** | Image → VLM inference with captioning prompt → scene description |
| 7 | **Change detection endpoint** | Siamese processing of bi-temporal pair → change description/VQA via VLM + small change-aware adapter |
| 8 | **Fusion endpoint** | Optical + SAR pair → dual-embedding extraction → cross-attention adapter → fused answer with modality attribution (§6.4 contract) |
| 9 | **Grounding endpoint** | Pretrained Grounding DINO wrapper in fp16, inference only |
| 10 | **GeoTIFF preprocessing** | Band reading, normalization tables per sensor, resolution harmonization, CRS validation |
| 11 | **Contract schemas** | All 5 JSON schema files for `contracts/schemas/` |
| 12 | **Benchmark eval scripts** | Evaluation harnesses against VRSBench, RSVQA, CDVQA held-out splits |
| 13 | **Unit tests** | Per-endpoint schema conformance tests |
| 14 | **Documentation** | `adaptation_log.md`, `confidence_spec.md` |
| 15 | **Docker & compose** | `Dockerfile` for ML service, service entry for team `docker-compose.yml` |

---

## What YOU Need to Study & Know

### 🧠 Must-Study Topics

| # | Topic | Why | Key Resource |
|---|---|---|---|
| 1 | **QLoRA / PEFT / LoRA** | Your one real training job. Understand rank, alpha, target modules, 4-bit quantization. | [PEFT docs](https://huggingface.co/docs/peft), [QLoRA paper](https://arxiv.org/abs/2305.14314) |
| 2 | **BigEarthNet.txt** | Read the paper thoroughly. Understand the 15 task types, annotation format, how captions/VQA/referring expressions are structured. | [Paper](https://arxiv.org/abs/2603.29630), [HuggingFace page](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) |
| 3 | **Small VLMs (2B-class)** | Understand architecture of InternVL2-2B or Qwen2-VL-2B or Florence-2. Know how they process image+text, how to prompt them. | [InternVL2 GitHub](https://github.com/OpenGVLab/InternVL), [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), [Florence-2](https://huggingface.co/microsoft/Florence-2-base) |
| 4 | **GeoTIFF / Rasterio** | Reading satellite imagery, understanding CRS, bands, resolution. | [Rasterio docs](https://rasterio.readthedocs.io) |
| 5 | **Sentinel-1 & Sentinel-2** | Know the band structure: S2 has 13 bands (multispectral), S1 has VV/VH polarization (SAR). BigEarthNet.txt uses both. | [ESA Sentinel docs](https://sentinels.copernicus.eu/) |
| 6 | **Siamese Networks** | Change detection uses same model on two images, then compares embeddings. | Search: "siamese change detection remote sensing" |
| 7 | **Cross-Attention Fusion** | How to merge optical + SAR embeddings and preserve modality attribution. | Search: "DeCUR cross-modal remote sensing" |
| 8 | **Grounding DINO** | Pretrained open-vocabulary detector. Know the text prompt → bounding box flow. | [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| 9 | **FastAPI** | Your service API framework. | [FastAPI docs](https://fastapi.tiangolo.com/) |
| 10 | **VRSBench evaluation protocol** | Understand the metrics: CHIAR for captions, IoU for grounding, accuracy for VQA. | [VRSBench paper](https://github.com/lx709/VRSBench) |

### 🔑 Decisions You Must Make

| # | Decision | Options | Recommendation | Deadline |
|---|---|---|---|---|
| 1 | **Architecture choice** | **Option A** (Small VLM + LoRA) vs **Option B** (RS Backbone + Heads + LLM API) | **Option A** — the dataset was designed for this | **Day 0** |
| 2 | **VLM choice** (if Option A) | InternVL2-2B, Qwen2-VL-2B, Florence-2 | **Qwen2-VL-2B** — best documented for LoRA, strong multilingual, fits 6GB | **Day 0** |
| 3 | **Backbone choice** (if Option B) | Prithvi-100M, Clay-small, SatMAE | Prithvi-100M | **Day 0** |
| 4 | **BigEarthNet.txt subset size** | 10K, 20K, 50K, 100K pairs | **30K–50K pairs** (more text annotations per pair now!) | **Day 0–1** |
| 5 | **LLM API access** (if Option B) | Claude API, GPT-4, local tiny LLM | Depends on your API keys/budget | **Day 0** |
| 6 | **Change head architecture** | VLM with temporal tokens vs. Siamese + small head | VLM with siamese embeddings + adapter | **Day 7** |
| 7 | **Fusion strategy** | VLM with dual-image input vs. cross-attention adapter | Cross-attention adapter on VLM embeddings | **Day 9** |

---

## Revised Day-by-Day Schedule (15 Days)

### Phase 0: Setup (Day 0–1) 🔧

| Task | Details |
|---|---|
| Set up environment | conda/venv, CUDA, PyTorch, transformers, PEFT, rasterio, GDAL — I'll generate exact specs |
| Download VLM weights | Qwen2-VL-2B (or chosen model), ~4GB download |
| Download BigEarthNet.txt annotations | Clone HuggingFace repo, get the parquet annotation file |
| Sample imagery subset | Stratified sampling script — pull 30K–50K image pairs from bigearth.net |
| Agree contracts with team | I'll draft all 5 schemas, team freezes them |
| Verify GPU | Load model in 4-bit, run dummy forward pass, confirm VRAM usage |

### Phase 1: VLM LoRA Fine-Tuning on BigEarthNet.txt (Day 2–5) 🧠
**This is your one critical training job — F7 compliance depends on it.**

| Task | Details |
|---|---|
| Format BigEarthNet.txt annotations | Convert captions, VQA pairs, referring expressions into instruction-tuning format for your chosen VLM |
| Configure QLoRA | 4-bit quantization, LoRA rank 8–16, target attention layers |
| Multi-task training | Train on mixed data: captioning prompts + VQA prompts + (optionally) referring expression prompts |
| Validate | Check per-task metrics on held-out BigEarthNet.txt benchmark split (1,082 pairs) |
| Stand up embedding service | FastAPI `/embed` endpoint for downstream use |
| **Document everything** | `adaptation_log.md` — component, dataset portion, objective, split, checkpoint |

### Phase 2: VQA + Caption Endpoints (Day 5–7) 💬

| Task | Details |
|---|---|
| Wire `/vqa` endpoint | Image + question → VLM inference → answer + confidence score |
| Wire `/caption` endpoint | Image → VLM with captioning prompt → scene description |
| Test against VRSBench/RSVQA samples | Basic quality check on held-out data |
| Confidence extraction | Extract token probabilities / logits as confidence signal |

### Phase 3: Change Detection (Day 7–9) 🔄

| Task | Details |
|---|---|
| Implement bi-temporal processing | Load VLM embeddings for t1 and t2, compute difference features |
| Build change adapter head | Small MLP/transformer on concatenated embeddings |
| Train on CDVQA + BigEarthNet.txt temporal pairs | Use whatever bi-temporal pairs are available |
| Wire `/change` endpoint | Following contract schema, return change description + region |

### Phase 4: Optical–SAR Fusion (Day 9–11) 🛰️ **← Highest risk**

| Task | Details |
|---|---|
| Implement dual-modality processing | VLM processes optical and SAR images separately (or with special tokens) |
| Build cross-attention fusion adapter | Small cross-attention layer merging modality embeddings |
| Train on BigEarthNet.txt paired data | The dataset has co-registered SAR+optical — perfect for this |
| Implement modality attribution | **Required by §6.4** — which answer parts come from optical vs. SAR |
| Wire `/fusion` endpoint | Full schema with disagreement flag (FR9) |

### Phase 5: Grounding (Day 11–12) 📍

| Task | Details |
|---|---|
| Download Grounding DINO | ~350MB at fp16 |
| Build inference wrapper | Text query → bounding boxes |
| Wire `/grounding` endpoint | Following contract schema |

### Phase 6: Integration & Polish (Day 12–15) 🔗

| Task | Details |
|---|---|
| Swap real endpoints behind controller | One at a time, coordinate with B1 |
| Run benchmark evaluations | VRSBench (caption + grounding + VQA), RSVQA (VQA), CDVQA (change VQA) |
| Write documentation | `adaptation_log.md`, `confidence_spec.md` |
| Bug fixes, edge cases | GeoTIFF band mismatches, normalization |
| Freeze checkpoints | Tag final model versions |

---

## Fallback Priority (If Running Behind)

```
F1 (VQA) → F4 (Change) → F5 (Fusion) → F7 (Adaptation, done by Day 5) → F2 (Caption) → F3 (Grounding)
```

Drop grounding first (F3). You can ship captioning-only for the "F2 or F3" requirement. VQA is the highest priority and the most straightforward with a fine-tuned VLM.

---

## Critical Compliance Rules

> [!CAUTION]
> ### Rule 1: If Option B — The LLM Never Sees Raw Images
> (Only applies if you go with Option B.) The LLM prompt template must ONLY contain structured facts — never image bytes. If using Option A (VLM), this rule doesn't apply since the VLM is RS-adapted.

> [!WARNING]
> ### Rule 2: Document BigEarthNet.txt Adaptation Thoroughly
> In `/docs/adaptation_log.md`: component adapted, **portion of BigEarthNet.txt used** (not just BigEarthNet), sampling strategy, adaptation objective (instruction tuning on captions/VQA/referring expressions), train/val split, checkpoint version.

> [!IMPORTANT]
> ### Rule 3: One Process, Five Endpoints
> Ship one FastAPI process. Load VLM once in memory. Five endpoints share it.

> [!IMPORTANT]
> ### Rule 4: Modality Attribution is NOT Optional in Fusion
> The `/fusion` endpoint MUST return which parts of the answer come from optical vs. SAR. This is required by §6.4.

> [!IMPORTANT]
> ### Rule 5: The VLM Must Be RS-Adapted, Not Generic
> The whole point of F7 is that you fine-tuned/adapted a model on BigEarthNet.txt. If you ship a generic VLM without adaptation, you fail the compliance bar. The LoRA fine-tuning on BigEarthNet.txt IS the adaptation.

---

## Storage Budget (176GB free)

| Item | Estimated Size |
|---|---|
| VLM weights (4-bit quantized) | ~2–3 GB |
| BigEarthNet.txt annotations (parquet) | ~1–2 GB |
| Imagery subset (30K–50K pairs, S1+S2) | ~40–80 GB |
| LoRA adapter checkpoints | ~50–200 MB per checkpoint |
| Grounding DINO weights | ~350 MB |
| Evaluation datasets (VRSBench, RSVQA, CDVQA) | ~5–10 GB |
| Working space (embeddings cache, logs) | ~10–20 GB |
| **Total estimate** | **~60–120 GB** ✅ Fits in 176GB |

---

## Open Questions For You

1. **Option A or Option B?** (VLM fine-tuning vs. backbone + heads + LLM API) — I strongly recommend Option A.

2. **If Option A: Which VLM?** Qwen2-VL-2B (recommended), InternVL2-2B, or Florence-2?

3. **BigEarthNet.txt subset size?** I recommend 30K–50K pairs. The full 464K is too large for your disk/time.

4. **Are your 5 teammates assigned?** This affects when we freeze schemas.

5. **Shall I start coding the environment setup and data pipeline immediately after you approve?**
