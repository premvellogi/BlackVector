# SatQuery AI — Solo ML Lead Plan (RTX 3060 6GB) + 5-Person Backend/Frontend Split

**Supersedes the role split in `SatQuery_AI_Master_Plan_6person.md`** — the contracts, repo shape, and phase structure from that doc still apply. This doc replaces §2 (roles) with a 1 ML-lead + 5 backend/frontend split, and adds a GPU-budgeted model architecture that actually fits 6GB VRAM.

**Your hardware:** Ryzen 7 5800H, 16GB RAM, RTX 3060 Laptop 6GB VRAM, ~176GB free disk.

---

## 1. Why "all model work solo on 6GB" needs a different architecture, not just less time

6GB VRAM rules out fine-tuning a 7B-class VQA decoder in 4-bit QLoRA the way v1.1 §5.1 proposed as the default — that's a 24GB-GPU assumption. Doing five mandatory ML components (backbone adaptation, VQA, one of caption/grounding, change detection, fusion) solo means two constraints stack: **your time** (one person, 15 days, sequential) and **your VRAM** (6GB, one job at a time). The way through both is the same move: **keep one frozen, adapted backbone, and make every task-specific component a small head sitting on top of its embeddings** — not separate heavy models per task.

This is also the right read of the brief: F7 only requires **at least one** component to be RS-adapted via BigEarthNet — it doesn't require every block to be independently fine-tuned. Adapt the backbone once, reuse it everywhere.

### Your actual stack (all fits in 6GB, one component loaded at a time):

| Component | What it is | Why it fits 6GB |
|---|---|---|
| **Backbone** | Prithvi-100M or Clay-small, frozen, + LoRA adapter fine-tuned on a BigEarthNet subset (multi-label land-cover classification — BigEarthNet's native label task) | ~100M params, LoRA adapter adds a few MB of trainable weights. This is your one real training job (F7). |
| **VQA + Captioning (F1, F2)** | Backbone embeddings → small tag/attribute classification head (trained on BigEarthNet labels) → structured facts (land-cover tags, confidence per tag) → passed as context to an LLM API call (Claude, per v1.1 §5.2's pluggable controller LLM) which generates the actual natural-language answer/caption | The heavy generative language work happens via API, not on your GPU. The only thing you train locally is a small classification head. |
| **Grounding (F3)** | Pretrained Grounding DINO, run in fp16 inference, **not fine-tuned** | ~172M params at fp16 ≈ 350MB — inference-only, no training needed. v1.1 §5.1 already says fine-tuning here is optional. |
| **Change detection (F4, mandatory)** | Siamese pair of frozen-backbone embeddings (t1, t2) → concat/diff → small MLP or shallow transformer head, trained on CDVQA / bitemporal pairs sampled from BigEarthNet | Small head over frozen embeddings — cheap to train, cheap to run. |
| **Fusion — optical/SAR (F5, mandatory)** | Same pattern: frozen-backbone embeddings from both modalities → small cross-attention adapter or concat+MLP head, trained on whatever paired optical/SAR samples you can assemble | Same reason as above — the expensive part (the backbone) is already trained once and reused. |

Net effect: **one real training run** (backbone LoRA) plus **three lightweight heads** (VQA/caption tag-head, change-head, fusion-head) plus **one pretrained model used as-is** (grounding). That's a realistic solo scope for 15 days on 6GB.

---

## 2. Storage plan — don't touch the full BigEarthNet corpus

`BIFOLD-BigEarthNetv2-0/BigEarthNet` on Hugging Face is large. With 176GB free:
- Pull a **stratified subset** — sample a few tens of thousands of patches across land-cover classes (not the full archive). Document the sampling strategy per v1.1 §5.1a ("portion used, sampling strategy, and why").
- Delete/don't retain intermediate download artifacts once patches are extracted to your working format.
- Keep raw imagery + embeddings cache separate from checkpoints so you can prune the raw data once embeddings are extracted, if space gets tight.

---

## 3. Your personal build sequence (Day 0–15, sequential — you're one person)

This assumes the team-wide contracts (schemas, mock servers) from the master plan are locked on Day 0–1 by the whole team together, so the other 5 aren't blocked waiting on you.

| Days | You (ML) | The other 5 (in parallel, against contracts/mocks) |
|---|---|---|
| 0–1 | Env setup (conda/venv, CUDA, GDAL/rasterio), download backbone weights, sample BigEarthNet subset, agree contracts with the team | Lock schemas; build mock servers + docker-compose skeleton; start controller/GUI shells against mocks |
| 2–5 | Backbone LoRA adaptation training (F7) + stand up the embedding service behind its contract | Controller state machine (A1/A2) against mocks; GUI upload/query flow |
| 5–7 | VQA + caption pipeline: tag head + LLM-answer integration (F1, F2) | Ontology layer (D1), input validation (F9), GUI results panel |
| 7–9 | Change detection head (F4, mandatory) | Confidence aggregation (§6.3) wiring, provenance tagging (D4) |
| 9–11 | Fusion head, optical–SAR (F5, mandatory) — highest-uncertainty item, budget slack here | Disagreement surfacing (FR9, D3), reliability harness (D6) |
| 11–12 | Grounding integration (pretrained Grounding DINO, inference-only) | GUI overlays for evidence per task type (§6.5), PDF report (§3.6) |
| 12–13 | Swap your real endpoints in behind the controller (one at a time), re-test each swap | Same — integration owner runs `tests/integration/` after each swap |
| 13–14 | Bug fixes, benchmark eval scripts against VRSBench/RSVQA/CDVQA held-out splits, adaptation log + confidence spec docs | Deliverables checklist (v1.1 §9), demo video |
| 15 | Freeze, buffer | Freeze, buffer, demo rehearsal |

If you fall behind, the fallback order to protect is: **F1 (VQA) → F4 (change) → F5 (fusion) → F7 (adaptation, already done Day 5) → F2/F3 (whichever is further along)** — F2/F3 only needs one of the two, so if grounding (Day 11–12) is at risk, drop it and ship captioning-only; it's the one component you can legitimately cut.

---

## 4. Your contract boundary with the other 5

You don't need 5 separate microservices just because there are 5 model tasks. Ship **one FastAPI process** (`services/models/`) exposing five endpoints — `/vqa`, `/caption`, `/grounding`, `/change`, `/fusion` — each honoring its own locked schema from `contracts/schemas/`. Internally it's one process sharing the loaded backbone in memory (saves you from reloading a 100M-param model five times); externally it looks like five independent contract-honoring blocks to the controller team, so the split doesn't leak into their side.

Hand the backend team the same "scoped spec" pattern as the main master plan: they build the controller against your contracts and the mock servers, and only start pointing at your real `/vqa`, `/change`, etc. as each one comes online — never blocked waiting for all five at once.

---

## 5. Splitting the other 5 people (backend + frontend)

Suggested split, assuming they're not doing ML:

| # | Role | Owns |
|---|---|---|
| B1 | Controller Lead | Agentic state machine, tool registry (F6, A1, A2) |
| B2 | Ontology + Reliability | Image ontology/metadata layer (D1), input validation (F9), reliability harness (D6), confidence aggregation (§6.3) |
| F1 | GUI Lead | Upload, query box, results panel, execution trace + confidence display (F8) |
| F2 | GUI — Evidence & Reports | Overlays per task type (§6.5), disagreement flag UI (FR9), PDF report (mandatory, §3.6), GeoJSON export if time allows (S3) |
| I1 | Integration/DevOps | `docker-compose`, `tests/integration/`, environment consistency across 6 laptops, demo/deployment, deliverables checklist assembly |

This keeps the same "everyone builds against a contract, not against your in-progress model code" discipline — critical since you're the single slowest-moving, most sequential part of the pipeline; nobody else should be blocked on you finishing anything before they can build their piece.
