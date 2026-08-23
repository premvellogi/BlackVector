# SatQuery AI — Project Requirements & Technical Documentation

**Version:** 1.1
**Status:** Approved scope, revised for compliance gaps
**Compute target:** 1x consumer GPU (24GB-class), 4+ person team, 15-day build cycle

**Changelog from v1.0:** Added explicit Evaluation/Judging Criteria table; strengthened BigEarthNet.txt as primary adaptation dataset; added multispectral/SAR band-handling spec; defined cross-modal analysis output contract; defined confidence methodology; defined visual-evidence semantics per task; added ISRO/SAC → capability mapping; clarified benchmark evaluation as an external protocol; corrected PDF/GeoJSON mandatory-vs-stretch split; made controller LLM pluggable (Claude as reference implementation, not a hard dependency); separated "Requirement" from "Proposed Implementation" throughout; corrected single-image baseline to strictly match the source requirement; expanded test/demonstration deliverables.

---

## 1. Project Overview

SatQuery AI is an agentic, query-driven vision-language assistant for remote-sensing imagery. Instead of a single generic model handling every request, a controller agent interprets each natural-language query, validates the supplied imagery, selects and executes the appropriate specialist model(s) from a registry, and returns an evidence-grounded, auditable answer. The system supports single images, co-registered optical–SAR pairs, and bi-temporal pairs, and is built as an interactive GUI/web application.

Throughout this document, **Requirement** denotes what the system must satisfy per the governing brief; **Proposed Implementation** denotes the team's current technical choice for meeting that requirement. Implementation choices may change without altering compliance, as long as the underlying requirement is still met.

The core design principle: **specialist models + an agentic controller, not one monolithic VLM.** A generic LLM/VLM without remote-sensing adaptation does not satisfy the requirements.

---

## 2. Source Requirements (from the original problem statement)

These are binding constraints inherited from the governing brief.

### 2.1 Defined Input Scope
- **Single image:** one optical/multispectral or SAR image — captioning, VQA, text-guided grounding.
- **Cross-modal pair:** co-registered optical/multispectral + SAR of the same area — joint/fusion analysis.
- **Bi-temporal pair:** two spatially corresponding images of the same area at different times — change detection, change description, change-VQA.
- **Supported formats:** GeoTIFF/TIFF for geospatial imagery. PNG/JPEG accepted **only** for the prescribed public benchmark datasets (not for arbitrary user uploads).

### 2.2 Mandatory Functional Scope
- **Remote-sensing adaptation (Requirement):** at least one visual/vision-language component fine-tuned or otherwise adapted using **BigEarthNet.txt**, or any open-source training data where the evaluation brief permits an alternative.
  - **Proposed Implementation:** BigEarthNet.txt is the primary adaptation dataset. A pretrained RS foundation-model backbone (Prithvi/Clay/SatMAE) may be used to reduce compute burden, but the adaptation step performed on top of it must be explicitly documented (see §5.1a).
- **Single-image baseline (Requirement, exact wording preserved):** VQA is mandatory. Each solution must **additionally implement either** captioning/scene description **or** text-guided region grounding (not necessarily both).
  - **Target (team ambition, not a compliance requirement):** implement both captioning and grounding, since it strengthens the demo — but VQA + one of the two is the actual pass bar, and schedule risk should be managed against that bar, not against "both."
- **Multi-image change analysis:** change description or change-VQA from a bi-temporal pair is mandatory. Spatial change maps are optional, contingent on reference masks being available.
- **Cross-modal pair analysis (Requirement, exact wording preserved):** the system must **extract complementary information** from a co-registered optical/multispectral and SAR image pair. See §6.4 for the defined output contract that operationalizes "complementary information."
- **Agentic orchestration:** automatic selection, sequencing, and execution of specialist models/tools per query and input configuration.

### 2.3 Required Datasets
| Dataset | Purpose |
|---|---|
| BigEarthNet.txt | **Primary** dataset for adapting image–text representations to multisensor RS data |
| VRSBench | Single-image captioning, grounding, VQA evaluation |
| RSVQA | Single-image VQA evaluation |
| CDVQA | Multitemporal change-based VQA evaluation |

### 2.4 Evaluation / Judging Criteria

Final evaluation uses prescribed public benchmark test subsets **and** an ISRO/SAC evaluation dataset. The ISRO/SAC set contains pre-georeferenced, co-registered **Cartosat-2S** (optical) and **RISAT** (SAR) pairs with task-specific reference answers/labels/bounding boxes/masks. Scores are normalized before combination across metrics. **Evaluation annotations are not disclosed to teams beforehand** — the system must generalize, not memorize a leaderboard, and this documentation does not invent exact metrics the official brief did not specify.

**Evaluation/Judging Criteria table:**

| Evaluation Area | Dataset | Evaluation Basis | Metric Source |
|---|---|---|---|
| Single-image VQA | RSVQA | Answer correctness against reference answers | Prescribed by official benchmark protocol |
| Captioning | VRSBench | Caption quality against reference captions | Prescribed by official benchmark protocol |
| Region grounding | VRSBench | Bounding-box overlap against reference boxes | Prescribed by official benchmark protocol |
| Change VQA / description | CDVQA | Change-question answer correctness | Prescribed by official benchmark protocol |
| Cross-modal optical–SAR analysis | ISRO/SAC (Cartosat-2S + RISAT) | Reference answers/labels for fused-modality questions | ISRO/SAC evaluation protocol (not disclosed pre-hoc) |
| Change localization | ISRO/SAC | Mask overlap against reference masks, where provided | ISRO/SAC evaluation protocol |
| Generalization | ISRO/SAC (held-out) | Combined normalized score across task areas | ISRO/SAC evaluation protocol |
| Agentic orchestration | ISRO/SAC / live demo | Correctness of task/model/tool selection + completeness of execution trace | Observable execution trace, per §2.6 |

**Explicit note:** exact metric formulas, normalization weights, and test-split identifiers are the evaluator's protocol and are intentionally **not** invented in this document. The system's job is to (a) produce outputs in the correct format/contract for each evaluation area above, and (b) generalize across unseen benchmark and ISRO/SAC samples, not to optimize against a specific disclosed metric.

**ISRO/SAC → SatQuery capability mapping** (documents which system capability is exercised by which evaluation category; does not assume knowledge of undisclosed exact tasks):

| ISRO/SAC evaluation sample type | SatQuery capability exercised |
|---|---|
| Single optical or SAR image + question | VQA (F1) |
| Optical + SAR pair + question | Cross-modal reasoning (F5) |
| Bi-temporal pair + question | Change understanding (F4) |
| Sample with reference bounding box | Grounding (F3) |
| Sample with reference mask | Change localization (S2) |

### 2.5 Deliverables (as specified, expanded in §9)
1. Interactive GUI or web application with an agentic remote-sensing AI backend.
2. Code and models, including test and demonstration materials.

### 2.6 Agentic Controller Behavior (as specified)
The controller must:
- Interpret the query and classify the requested task.
- Check number, modality, format, metadata, and compatibility of input images.
- Select one or more models/tools from a predefined registry.
- Configure only permitted task parameters and execute the workflow.
- Combine textual and spatial outputs, estimate confidence, return visual evidence.
- Provide an auditable execution summary (selected task, model/tool names, key parameters).

Internal chain-of-thought/planning text is **not** required or evaluated — only the observable execution trace is.

---

## 3. Confirmed Feature Scope

### 3.1 Core Mandatory (must ship)
| # | Feature | Compliance level | Notes |
|---|---|---|---|
| F1 | Single-image VQA | Requirement | Mandatory baseline |
| F2 | Single-image captioning/scene description | Requirement satisfied by F2 **or** F3 | Team targets both; pass bar is one |
| F3 | Text-guided region grounding | Requirement satisfied by F2 **or** F3 | Team targets both; pass bar is one |
| F4 | Multi-temporal change description / change-VQA | Requirement | Mandatory |
| F5 | Cross-modal optical–SAR joint analysis | Requirement | See §6.4 for output contract |
| F6 | Agentic orchestration (task routing, tool selection, execution) | Requirement | Mandatory |
| F7 | RS domain adaptation via BigEarthNet.txt (primary) | Requirement | See §5.1a for documentation fields |
| F8 | Interactive GUI/web app | Requirement | Upload, query, results, execution trace, downloadable report |
| F9 | Input validation | Requirement | Format, modality, CRS/geographic compatibility |

### 3.2 Differentiator Features (AIP-inspired, in core scope — team ambition beyond the source brief)
| # | Feature |
|---|---|
| D1 | Ontology/metadata layer — structured objects for images with sensor, date, CRS, and cross-image relationships |
| D2 | Proposal-based actions — agent states its plan and asks confirmation on ambiguous/low-confidence decisions |
| D3 | Disagreement resolution — surfaces conflicts between modalities/models instead of silently resolving them |
| D4 | Provenance tracking — every output tagged with model used, confidence, input hash, timestamp |
| D5 | Session memory — caches processed embeddings/results within a session for follow-ups |
| D6 | Reliability harness — retries, malformed-output handling, circuit breakers around tool calls |

### 3.3 Architecture-Level Novelty (in core scope — team ambition beyond the source brief)
| # | Feature |
|---|---|
| A1 | Block-based tool registry — every specialist model is a swappable block with a declared input/output schema |
| A2 | State-machine agent execution — deterministic graph of nodes/edges rather than a freeform LLM loop, producing the auditable execution trace |

### 3.4 Stretch Features (confirmed in documented scope)
| # | Feature | Notes |
|---|---|---|
| S1 | Live satellite data fetch | Query a location, pull fresh Sentinel-1/2 tiles instead of requiring upload |
| S2 | Spatial change mask generation | Pixel-level output, contingent on reference masks being available |
| S3 | GeoJSON export | Stretch, in addition to the mandatory PDF report (see §3.6 correction) |
| S4 | Click-to-follow-up interaction | Click a returned box/change region, ask a follow-up query about it |

### 3.5 Appendix Feature (optional, build only if ahead of schedule)
| # | Feature |
|---|---|
| AP1 | Live "Model Health" dashboard — runs pipeline against held-out benchmark slices, reports per-task accuracy in-app |

### 3.6 Mandatory vs. Stretch Correction — Downloadable Reports
The source brief's "Expected Solution" lists **downloadable reports** as part of the expected solution — this is treated as **mandatory** and satisfied by:
- **Mandatory:** downloadable **PDF report** (contains answer, visual evidence, confidence, execution trace).
- **Stretch (S3):** **GeoJSON export**, additive to the PDF, for GIS-tool interoperability (e.g., QGIS).

### 3.7 Explicitly Out of Scope
- Training any vision encoder fully from scratch on large-scale data.
- A single end-to-end multimodal model replacing the modular agent+specialist architecture (this would violate the mandatory architecture, not just be a simplification).
- Enterprise-grade security/access-control layer (AIP-style governance is borrowed conceptually for D2–D4, not built as a full permissions system).
- Inventing exact ISRO/SAC metrics, weights, or test-split identifiers not disclosed in the governing brief.

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────┐
│                  GUI / Web Application                │
│  Upload · Query box · Results · Overlays ·             │
│  Execution trace · Confidence · PDF report (+GeoJSON)  │
└───────────────────────┬────────────────────────────────┘
                         │
┌───────────────────────▼────────────────────────────────┐
│              Agentic Controller (state machine)          │
│  1. Query interpretation → task classification            │
│  2. Input validation against Ontology layer                │
│  3. Block selection from Tool Registry                     │
│  4. Execution (with reliability harness: retry/circuit-brk)│
│  5. Output fusion + confidence + disagreement resolution   │
│  6. Provenance tagging + execution trace generation         │
└───┬─────────┬─────────┬─────────┬─────────┬──────────────┘
    │         │         │         │         │
┌───▼──┐ ┌───▼───┐ ┌───▼────┐ ┌──▼──────┐ ┌▼─────────────┐
│ VQA  │ │Caption│ │Grounding│ │ Change  │ │ Optical–SAR   │
│ block│ │ block │ │  block  │ │  block  │ │ Fusion block  │
└──┬───┘ └───┬───┘ └────┬────┘ └────┬────┘ └───────┬───────┘
   └─────────┴───────────┴────────────┴─────────────┘
             Shared RS-adapted vision-language backbone
      (Proposed Implementation: Prithvi/Clay/SatMAE + BigEarthNet.txt adaptation)
```

**Ontology layer** sits alongside the controller: every uploaded/fetched image becomes a structured object (sensor, date, CRS, resolution, cloud cover, band configuration, links to paired images). The controller queries this layer before dispatching to blocks — this is what allows compatibility checks (F9) to be principled rather than ad hoc.

**Tool Registry** is a config-driven list mapping task type → block endpoint → expected input/output schema → permitted parameters (A1). Adding a new specialist model means registering a new block, not rewriting controller logic.

**Session Memory (D5)** is a lightweight cache (image hash → embeddings/results) held for the duration of a session, checked before re-running a block on already-seen input.

---

## 5. Tech Stack

### 5.1 Vision / ML

| Component | Requirement | Proposed Implementation |
|---|---|---|
| RS-adapted vision-language component | At least one component fine-tuned/adapted on BigEarthNet.txt | Pretrained RS foundation backbone (Prithvi, Clay, **or** SatMAE — one selected at kickoff) + adaptation step on BigEarthNet.txt (see §5.1a) |
| Optical–SAR fusion | Extract complementary information from co-registered pairs | DeCUR-style cross-attention adapter, or custom concat + shared head, over backbone embeddings |
| VQA / captioning head | VQA mandatory; captioning as one option for the second single-image task | Small (≈7B-class) LLM decoder + QLoRA adapter on backbone embeddings (4-bit, fits 24GB VRAM) |
| Grounding | One option for the second single-image task | Pretrained Grounding DINO (open-vocabulary detector); fine-tuning optional if time allows |
| Change detection/VQA | Mandatory | Siamese head sharing the RS backbone + small transformer/LLM head |
| Change mask (S2) | Stretch, contingent on reference masks | U-Net-style decoder on backbone features |
| Fine-tuning method | Not specified by brief | QLoRA/LoRA (4-bit) — only viable approach on a single consumer GPU |

#### 5.1a BigEarthNet.txt Adaptation — Required Documentation Fields
To keep F7 defensible under judging, the following must be recorded (in `/docs/adaptation_log.md` or equivalent) once training begins:
- **Component adapted:** which module (e.g., backbone projection layer, VQA head, or both).
- **Portion of BigEarthNet.txt used:** subset size, sampling strategy, and why (full dataset vs. a compute-bounded subset).
- **Adaptation objective:** e.g., image–text contrastive alignment, captioning loss, or VQA loss — stated explicitly.
- **Training/validation split:** how the split was constructed and its size.
- **Resulting checkpoint:** version-tagged and referenced in the execution trace's provenance field (D4), so any answer can be traced to the exact adapted checkpoint that produced it.

#### 5.1b Multispectral and SAR Band Handling — Technical Specification
This closes a previously undefined gap and must be implemented, not left implicit:
- **Multispectral band count:** the system must read the actual band count from GeoTIFF metadata (not assume RGB-3-band) and document which bands the backbone consumes (e.g., RGB-only subset, or full multispectral stack if the chosen backbone supports it).
- **Band selection/mapping:** an explicit mapping table from sensor band names/indices (e.g., Cartosat-2S bands) to the channels the backbone expects, stored in the ontology layer per image.
- **Normalization:** per-band normalization statistics (mean/std or min-max) applied consistently at inference and training time; documented per sensor type, since optical and SAR have very different value distributions.
- **Non-RGB band handling:** bands beyond RGB either passed through if the backbone supports multi-channel input, or explicitly dropped with that decision logged in provenance (D4) — never silently discarded.
- **SAR channel handling:** SAR is documented as single- or dual-polarization (e.g., VV/VH) based on RISAT's actual product; the fusion block's expected SAR channel count is declared in its block schema (A1).
- **Spatial resolution harmonization:** when optical and SAR (or bi-temporal) images differ in resolution, a defined resampling step (e.g., bilinear/nearest to a common grid) is applied before fusion or change comparison, and this step is logged in the execution trace.

### 5.2 Agent / Orchestration

| Component | Requirement | Proposed Implementation |
|---|---|---|
| Controller LLM | An agentic controller capable of function/tool-calling; not tied to a specific vendor by the brief | **Pluggable function-calling LLM interface.** Claude is the development/reference implementation; the controller is not architecturally hard-dependent on any single LLM provider |
| Execution model | Auditable execution trace | Explicit state machine (LangGraph or equivalent graph library) — deterministic, trace derived directly from the graph path taken |
| Tool/block interface | Predefined registry of models/tools | JSON schema contract per block (A1) |
| Reliability layer | Not specified by brief (team addition) | Retry + circuit breaker wrapper around each block call |

### 5.3 Backend / Serving
| Component | Proposed Implementation |
|---|---|
| Model serving | FastAPI microservice per model block (decouples GPU serving from agent logic) |
| Data validation | GDAL/rasterio for GeoTIFF metadata (CRS, bounding box, resolution, band count) |
| Live data fetch (S1) | Sentinel Hub API or equivalent Sentinel-1/2 access |
| Storage | Local filesystem for images/embeddings during dev; object storage (S3-compatible) if deploying beyond local |

### 5.4 Frontend / GUI
| Component | Proposed Implementation |
|---|---|
| Framework | React (or Streamlit/Gradio if prioritizing speed over polish — decided by team D) |
| Map/GeoTIFF rendering | Leaflet or MapLibre GL with a GeoTIFF plugin, for overlays and click-to-follow-up (S4) |
| Report export | PDF (mandatory, §3.6) via a lightweight report generator; GeoJSON export (S3, stretch) via standard geojson serialization |

### 5.5 Dev tooling
| Component | Proposed Implementation |
|---|---|
| IDE / agent tooling | Claude + Google Antigravity (multi-agent Manager View, shared skills) |
| Version control | Git, monorepo structure (see companion master plan for branch strategy) |
| Package management | Python (pip/conda) for ML; npm for frontend |
| Eval harness | Scripts against VRSBench/RSVQA/CDVQA test splits; surfaced via AP1 if built. Treated as internal validation only — does not substitute for or predict the official ISRO/SAC evaluation protocol |

---

## 6. Functional Requirements (detailed)

- **FR1** — System accepts GeoTIFF/TIFF uploads for arbitrary imagery; PNG/JPEG accepted only when sourced from the prescribed benchmark datasets, and this bypass is explicitly flagged in the ontology record for that image (not silently treated the same as a GeoTIFF upload).
- **FR2** — System validates image count, modality, format, band configuration, and geographic compatibility before dispatching to any model block, and rejects/flags incompatible inputs with a clear reason.
- **FR3** — Given a single image and a natural-language question, system returns a VQA answer grounded in the image (F1).
- **FR4** — Given a single image, system can produce a caption/scene description (F2) and/or a grounded region for a referring expression (F3), per query intent; at minimum one of the two is functional (pass bar), both is the target.
- **FR5** — Given a bi-temporal pair, system produces a natural-language change description and/or answers change-specific questions (F4); optionally a spatial change mask (S2) when reference masks exist.
- **FR6** — Given a co-registered optical–SAR pair, system produces a joint analysis satisfying the output contract in §6.4 (F5).
- **FR7** — The controller classifies query intent and selects the correct block(s) automatically — no manual task selection by the user (F6).
- **FR8** — Every response includes: the answer, visual evidence per the semantics in §6.5, a confidence estimate computed per §6.3, and an execution summary listing selected task, model(s)/block(s) used, and key parameters.
- **FR9** — When model outputs disagree (e.g., optical vs. SAR), the system surfaces the disagreement explicitly rather than silently resolving it (D3).
- **FR10** — For low-confidence or ambiguous task selection, the system proposes its plan and requests confirmation before executing (D2).
- **FR11** — Users can issue follow-up queries referencing a previously returned image/region without re-uploading, within a session (D5, S4).
- **FR12** — Users can download a PDF report (mandatory) and optionally export GeoJSON (S3, stretch).
- **FR13** — Users can query a named location to fetch live Sentinel-1/2 imagery in place of uploading files (S1).

### 6.3 Confidence Methodology (previously undefined — now specified)
Confidence is not a single global number; it is computed per response using a **rule-based aggregation** of the following signals, combined via a documented weighting (finalized during implementation, logged in `/docs/confidence_spec.md`):
- **Model-reported confidence:** where the underlying block exposes a probability/logit-derived score (e.g., VQA answer probability, grounding IoU-proxy score).
- **Cross-modal agreement (F5 specifically):** degree to which optical- and SAR-derived sub-answers agree; disagreement lowers confidence and triggers FR9.
- **Input-quality factors:** e.g., cloud cover percentage, resolution mismatch after harmonization (§5.1b), co-registration quality.
- **Ensemble agreement**, only where multiple blocks independently address the same sub-question.
This methodology is deliberately simple and rule-based (not a learned calibration model) given the 15-day timeline — documented as such rather than implied to be a calibrated probability.

### 6.4 Cross-Modal Output Contract (previously vague — now specified)
To operationalize "extract complementary information" (§2.2), a successful cross-modal (optical–SAR) response must include:
1. A fused textual answer combining both modalities.
2. **Modality attribution:** which parts of the answer are supported by optical evidence vs. SAR evidence (e.g., "built-up area identified from optical texture; extent confirmed by SAR backscatter").
3. **Modality-specific evidence overlays** (see §6.5), shown separately, not merged into one indistinguishable overlay.
4. Explicit resolution of any ambiguity between modalities where one modality alone would have been insufficient or misleading (e.g., cloud-obscured optical, resolved via SAR).
5. A disagreement flag (FR9) if the two modalities produce conflicting sub-answers that cannot be confidently resolved.

### 6.5 Visual Evidence Semantics (previously underspecified — now specified)
Per task type, "visual evidence" means:
| Task | Evidence returned |
|---|---|
| VQA | Highlighted region(s) the answer is grounded in, where applicable |
| Captioning | No mandatory overlay (caption is evidence itself); optional salient-region highlight |
| Grounding | Bounding box(es) for the referring expression |
| Change VQA/description | Change region overlay on both t1 and t2 images |
| Change mask (S2) | Pixel-level mask overlay |
| Cross-modal analysis | Separate optical evidence overlay **and** SAR evidence overlay, plus modality attribution per §6.4 |

---

## 7. Non-Functional Requirements

- **Extensibility:** New specialist models must be addable by registering a new block against the schema (A1), without controller code changes.
- **Auditability:** Every response must be traceable to the specific block(s), model version/checkpoint (§5.1a), and input(s) that produced it (D4), independent of whether the eval dashboard (AP1) is built.
- **Robustness:** A single failed/malformed block call must not crash the session — the reliability harness (D6) must catch and report it gracefully.
- **Reasonable latency:** Given a single shared GPU, concurrent multi-user load is not a target; single-query turnaround should be acceptable for a live demo (seconds, not minutes, for cached/small models — foundation-model backbone calls may take longer and should show progress in the GUI).
- **Reproducibility:** Fine-tuning runs, dataset splits, and evaluation results must be logged (git + `/docs/adaptation_log.md` + `/docs/confidence_spec.md`) so results can be reproduced or explained during judging.
- **Provider independence:** The agentic controller must not be architecturally hard-coded to a single LLM vendor (§5.2), preserving defensibility of the "agentic orchestration architecture" claim independent of any one provider's availability.

---

## 8. Data Handling Rules

- Primary formats: GeoTIFF/TIFF for all user-supplied or live-fetched imagery.
- PNG/JPEG accepted strictly for prescribed benchmark datasets (VRSBench, RSVQA, CDVQA) during evaluation — not a general upload path, and flagged as such in the ontology record (FR1).
- Cross-modal pairs must be checked for co-registration (same geographic area) before being routed to the fusion block; mismatched pairs are rejected with a clear message (FR2).
- Bi-temporal pairs must be checked for spatial correspondence before being routed to the change block.
- Band configuration (§5.1b) is validated and normalized before any model block receives the image.
- Evaluation annotations for the ISRO/SAC set are not available pre-hoc — no component assumes access to them; design and validate purely against BigEarthNet/VRSBench/RSVQA/CDVQA and held-out splits thereof. The official ISRO/SAC and public-benchmark evaluation procedures (exact metrics, splits, normalization) are treated as **external protocols owned by the evaluator**, not something this documentation defines or predicts.

---

## 9. Deliverables Checklist (expanded per source requirement: "codes and models including test and demonstration")

**Core deliverables:**
- [ ] Interactive GUI/web application with agentic RS backend (F8)
- [ ] Working single-image VQA (F1)
- [ ] Working captioning (F2) **and/or** grounding (F3) — at least one is the pass bar, both is the target
- [ ] Working change description/VQA (F4), with optional change mask (S2)
- [ ] Working optical–SAR fusion analysis satisfying the §6.4 output contract (F5)
- [ ] Functioning agentic controller with block registry and state-machine execution (F6, A1, A2)
- [ ] At least one RS-adapted vision-language component, documented per §5.1a (F7)
- [ ] Input validation + ontology-backed compatibility checking, including band handling (F9, D1, §5.1b)
- [ ] Confidence methodology implemented per §6.3
- [ ] Visual evidence per task-type semantics in §6.5
- [ ] Disagreement resolution, provenance tagging, session memory, reliability harness (D2–D6)
- [ ] Downloadable PDF report (mandatory, §3.6)
- [ ] GeoJSON export (S3, stretch)
- [ ] Live satellite fetch (S1, stretch)
- [ ] Click-to-follow-up (S4, stretch)
- [ ] (If time permits) Model Health dashboard (AP1)

**Test artifacts (explicit, per "test and demonstration" requirement):**
- [ ] Unit tests for each model block (input/output schema conformance)
- [ ] Integration tests for the controller's task-routing logic (correct block selected per query type)
- [ ] Benchmark evaluation scripts against VRSBench/RSVQA/CDVQA held-out splits
- [ ] Sample input images (one per input scope: single, cross-modal pair, bi-temporal pair)
- [ ] Sample queries covering every representative query type from the source brief
- [ ] Expected outputs for each sample query (for reviewer/judge spot-checking)
- [ ] Model checkpoints/adapters (versioned, referenced in provenance per §5.1a)
- [ ] Setup/run instructions (README covering environment, dependencies, how to launch the GUI and backend)
- [ ] Demonstration video (fallback for live-demo risk, per master plan Day 14)

---

## 10. Team, Roles, and Timeline

Role distribution (4-person team), day-by-day 15-day build schedule, GPU scheduling discipline, and the Claude+Antigravity collaboration workflow (repo structure, branching, shared skills, background task usage) are specified in the companion document: **`SatQuery_AI_Master_Plan.md`**. This documentation defines *what* is being built and *why*; the master plan defines *how and when* it gets built and by whom.

---

## 11. Risk Notes

- Single-GPU + 4-person contention is the primary schedule risk — mitigated by the GPU calendar in the master plan and by decoupling model serving (FastAPI microservices) from agent/GUI work so most of the team isn't GPU-blocked most of the time.
- Optical–SAR fusion (F5) and the RS foundation backbone choice (§5.1) are the highest-uncertainty technical items — resolve backbone choice at kickoff (Day 0) so downstream work isn't built against a moving target.
- Targeting **both** captioning and grounding (§2.2) is a scope expansion beyond the strict pass bar (one of the two) — schedule risk should be managed so that if time runs short, the team falls back to shipping one fully working, rather than two partially working.
- Stretch features (S1–S4) should not begin until F1–F9 and D1–D6/A1–A2 are functioning end-to-end; the master plan places them at Day 11+ deliberately.
- The confidence methodology (§6.3) and cross-modal output contract (§6.4) are team-defined operationalizations of brief language that did not specify exact mechanisms — if the official evaluator later clarifies expectations for either, this documentation should be updated to match before final submission.
