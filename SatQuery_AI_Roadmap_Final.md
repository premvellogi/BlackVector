# SatQuery AI — Compliance Check + Final Phase-Wise Roadmap (5-Person Backend/Frontend Build)

**Reads together with:** `SatQuery_AI_Master_Plan_6person.md` (contracts, repo shape) and `SatQuery_AI_Solo_ML_Plan.md` (your ML architecture). This doc: (1) checks the constrained architecture against the governing brief, (2) gives the final system shape, (3) gives each of the 5 backend/frontend people an exact, collision-free day-by-day roadmap.

---

## 1. Compliance check against the governing brief

Going through each mandatory item from `project_brief.pdf` / v1.1 §2.2 against the 6GB-constrained plan:

| Requirement | Constrained design | Compliant? |
|---|---|---|
| RS adaptation via BigEarthNet (at least one component) | Backbone LoRA-adapted on a BigEarthNet subset | **Yes** — brief says "at least one," not "every component." Document sampling strategy per §5.1a so it's defensible, not silently a toy subset. |
| VQA mandatory | Backbone embeddings → tag/attribute head (RS-adapted) → LLM phrases the answer | **Yes, with one condition** — see §1.1 below. |
| Caption **or** grounding (only one required) | Both implemented (caption via tag-head+LLM, grounding via pretrained Grounding DINO) | **Yes** — exceeds the pass bar. |
| Change description/VQA mandatory | Siamese head → LLM phrases the description/answer | **Yes, same condition as VQA** — see §1.1. |
| Cross-modal optical–SAR, extract complementary info | Fusion head → per-modality sub-answers + fused answer, modality attribution | **Yes, if the §6.4 output contract fields are actually populated** — see §1.2. |
| Agentic orchestration | Controller (backend team, not you) | Not affected by GPU constraints — untouched. |
| Format support (GeoTIFF/TIFF primary) | Ontology + validation layer (backend team) | Not affected by GPU constraints — untouched. |
| Visual evidence per task | Each block returns evidence per §6.5 (boxes/overlays) | **Yes, if schemas require it** — flagged in §2 below so nobody's block ships text-only. |

### 1.1 The one real compliance risk: don't let the LLM do the seeing

The brief is explicit: *"A generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements."* Your design routes final text through an LLM API call — that's fine **only if the LLM never receives the raw image and never does the actual visual reasoning.** The LLM's job must be strictly: take already-extracted, RS-adapted-model-produced facts (land-cover tags + confidence, change region coordinates + change type, grounding boxes) and phrase them as fluent, grounded natural language. If the LLM call includes the image itself and is asked to "describe what you see" or "has the built-up area changed," you've silently swapped in a generic VLM for the actual perception task, which is exactly what the brief disallows.

**Concrete rule for whoever builds these endpoints (you, on the ML side):** the LLM prompt template must only ever contain structured facts (tag lists, coordinates, class labels, confidence numbers) — never image bytes, never "look at this image." Log this explicitly in `/docs/adaptation_log.md` so it's auditable and defensible if a judge asks how VQA/change/caption text is actually generated.

### 1.2 Make sure §6.4's cross-modal contract fields are non-optional in the fusion schema

§6.4 requires: fused answer, modality attribution, separate optical/SAR evidence overlays, ambiguity resolution, and a disagreement flag (FR9) when modalities conflict. It's easy for a lightweight fusion head to quietly produce only a merged answer and skip modality attribution because it's the hardest field to fill in correctly. Lock this in the schema (§2) as a **required**, not optional, field — an empty-but-present field forces the implementer to address it rather than silently omitting it.

**Net result: nothing in the GPU-constrained design violates the brief, provided (a) the backbone adaptation is real and documented, and (b) the LLM-phrasing step is fact-in/text-out only, never image-in.** Everything else (agentic orchestration, format validation, GUI, reports) is backend/frontend work, entirely unaffected by your GPU constraint.

---

## 2. Final system shape

```
                         ┌─────────────────────────────┐
                         │      GUI (F1 + F2)            │
                         │ Upload · Query · Results ·    │
                         │ Overlays · Trace · PDF/GeoJSON │
                         └───────────────┬────────────────┘
                                         │
                         ┌───────────────▼────────────────┐
                         │   Controller (B1) — state machine │
                         │  query→task · block select ·      │
                         │  execute · fuse · trace            │
                         └───┬───────────────────────────┬───┘
                             │                           │
                ┌────────────▼──────────────┐  ┌─────────▼─────────┐
                │  Ontology + Reliability     │  │  Confidence (§6.3) │
                │  (B2) — image validation,   │  │  + Disagreement     │
                │  band handling, retries      │  │  (B2) — FR9, D3     │
                └────────────┬──────────────┘  └─────────┬─────────┘
                             │                           │
                ┌────────────▼───────────────────────────▼─────────┐
                │        ML Model Service (you) — one process        │
                │  /vqa  /caption  /grounding  /change  /fusion      │
                │  (frozen+LoRA backbone shared internally)           │
                └──────────────────────────────────────────────────┘

                Cutting across all of the above:
                Integration/DevOps (I1) — docker-compose, contract
                tests, cross-machine env consistency, deployment
```

Five contract surfaces exist, and only these five — everything else is internal to each person's folder:
1. **GUI ↔ Controller** (query submission, results retrieval)
2. **Controller ↔ Ontology/Reliability** (validated image objects, retry-wrapped calls)
3. **Controller ↔ Confidence/Disagreement** (raw block outputs in, aggregated confidence + disagreement flags out)
4. **Controller ↔ ML Model Service** (the 5 schemas from the master plan: vqa/caption/grounding/change/fusion)
5. **Everyone ↔ Integration/DevOps** (docker-compose service definitions, contract test fixtures)

---

## 3. Roles & Responsibilities — the 5 backend/frontend people

v1.1 §10 defers role definitions to "the companion master plan" but never actually states them — this section is that missing piece, stated plainly, once, before the day-by-day breakdown.

| Person | Title | Mission (one line) | Exactly what they build | Primary v1.1 refs |
|---|---|---|---|---|
| **B1** | Controller Lead | Turn a natural-language query + validated images into the right sequence of model calls, and produce an auditable trace of what happened. | Query interpreter (task classification), tool registry (config-driven task→block mapping), state-machine execution graph, execution trace/audit output. | F6, A1, A2 |
| **B2** | Ontology + Reliability + Confidence Lead | Make sure only valid, compatible inputs ever reach a model, calls don't crash the session, and every answer carries an honest confidence score. | Image ontology/metadata parsing (GeoTIFF via GDAL/rasterio), input validation (format/band/co-registration checks), retry + circuit-breaker wrapper around every block call, confidence aggregation (§6.3), disagreement detection between modalities. | D1, F9, D6, §6.3, D3, FR9 |
| **F1** | GUI Lead — Core Flow | The interactive front door: upload, ask, see the answer. | Upload flow for all 3 input scopes (single/cross-modal/bi-temporal), query box, task-type indicator, results panel (answer + confidence + execution trace display), map/GeoTIFF rendering. | F8 |
| **F2** | GUI Lead — Evidence & Reports | Make every answer visually provable and exportable. | Bounding-box/overlay rendering per task type (§6.5), modality attribution display (§6.4), disagreement flag UI, downloadable PDF report (mandatory, §3.6), GeoJSON export (stretch, S3). | §6.5, §6.4, FR9, §3.6, S3 |
| **I1** | Integration/DevOps Lead | Make sure the 6 people's work actually runs together, on 6 different laptops, without surprises. | Mock servers for all ML endpoints, `docker-compose` for the whole stack, contract-validation tests, locked per-service environments, nightly integration runs, deliverables checklist assembly, demo fallback. | v1.1 §7, §9 |

**Dependency shape, so nobody is confused about who blocks whom:**
- F1 and F2 depend on B1's trace schema and B2's confidence/validation schema — but only the *schema*, frozen Day 0–1, not on B1/B2's finished implementation. Build against mocks first.
- B1 depends on your (ML) 5 endpoints — again, only the schema. B1 builds the full state machine against mock servers before any real ML endpoint exists.
- I1 depends on nobody's implementation — only needs the frozen schemas to build mocks and tests, and is the one role that can start productive work literally on Day 0.
- Nobody depends on another *person* finishing — everyone depends on the *contract* being frozen, which happens once, on Day 0–1.

---

## 4. Exact phase-wise roadmap — 5 people, solo-buildable, collision-free

**Ground rule for all 5:** each person works only inside their own folder (§2 of the master plan) and reads/writes `contracts/` only via a reviewed PR. Day 0–1 is a joint meeting — everyone leaves it with schemas frozen and mock servers running. After that, nobody needs to wait on anyone else's *implementation*, only on the *contracts*, which don't change.

### Day 0–1 — Joint kickoff (all 5, together)
- Freeze all schemas in `contracts/schemas/`: `vqa`, `caption`, `grounding`, `change`, `fusion`, `confidence`, `provenance`, `ontology_image`, plus new ones for this phase: `controller_query.schema.json` (GUI→Controller) and `confidence_input.schema.json` (Controller→Confidence module, if you split it from Ontology as below).
- I1 builds the 5 mock servers (stubs matching the ML schemas) + `docker-compose.yml` skeleton with a placeholder container per person's folder.
- Everyone confirms they can `docker-compose up` and hit each other's stub endpoint with a dummy request by end of Day 1.

---

### B1 — Controller Lead
*Owns: `controller/` — query interpretation, task classification, block selection, execution, trace generation (F6, A1, A2)*

| Days | Task |
|---|---|
| 2–3 | Query interpreter: classify incoming NL query → task type (VQA/caption/grounding/change/fusion) using rules or a small LLM-call classifier. Test against mock servers. |
| 4–5 | Tool registry (A1): config-driven task→block-endpoint mapping. Adding a new block = new registry entry, no controller code change — verify this by registering a dummy 6th block. |
| 6–7 | State-machine execution graph (A2) — LangGraph or equivalent. Deterministic path per task type; trace = the path taken. |
| 8–9 | Wire in Ontology/Reliability (B2) calls before dispatch — controller checks compatibility before calling any block. |
| 10–11 | Wire in real ML endpoints one at a time as you (ML) ship each of /vqa, /change, /fusion, /caption, /grounding — re-run integration test after each swap. |
| 12–13 | Execution summary / audit trace output (D4 fields: task, block(s), key params) — hand this shape to F1 for the GUI trace display. |
| 14–15 | Bug fixes, buffer, help I1 with integration test failures. |

---

### B2 — Ontology + Reliability + Confidence Lead
*Owns: image ontology/metadata layer (D1), input validation (F9), reliability harness (D6), confidence aggregation (§6.3), disagreement surfacing (FR9, D3)*

| Days | Task |
|---|---|
| 2–3 | `ontology_image` schema implementation: parse GeoTIFF metadata via GDAL/rasterio (sensor, date, CRS, resolution, cloud cover, band count) into the shared `OntologyImage` object every block consumes. |
| 4–5 | Input validation (F9/FR1/FR2): format check (GeoTIFF/TIFF primary, PNG/JPEG flagged if from benchmark sets only), co-registration check for pairs, band-config validation per §5.1b. Reject with clear reasons. |
| 6–7 | Reliability harness (D6): retry + circuit breaker wrapper around every block call — this matters more than usual since ML endpoints are laptop-served, not a stable cluster. Test by having I1 deliberately kill your ML service mid-call. |
| 8–9 | Confidence aggregation (§6.3): combine model-reported confidence + cross-modal agreement + input-quality factors into one documented weighted score. Log the weighting in `/docs/confidence_spec.md`. |
| 10–11 | Disagreement resolution (D3, FR9): compare optical vs. SAR sub-answers from the fusion block; surface conflicts rather than silently picking one. |
| 12–13 | Provenance tagging (D4): every response gets model/checkpoint version, input hash, timestamp — coordinate the exact field names with B1's trace and your own ML checkpoint versioning docs. |
| 14–15 | Bug fixes, buffer. |

---

### F1 — GUI Lead (core flow)
*Owns: `frontend/` upload, query box, results panel, execution trace + confidence display (F8)*

| Days | Task |
|---|---|
| 2–3 | Upload flow: single image / cross-modal pair / bi-temporal pair modes, hitting B2's validation endpoint (or a stub of it) for compatibility feedback. |
| 4–5 | Query box + task-type indicator (what the controller inferred, once B1's classifier exists — use a stub result until then). |
| 6–7 | Results panel: answer text, confidence display (against B2's schema), execution trace display (against B1's schema). |
| 8–9 | Map/GeoTIFF rendering (Leaflet/MapLibre) for image display — foundation for F2's overlay work. |
| 10–11 | Wire against real Controller (B1) once it's routing to real ML endpoints — replace stubs incrementally. |
| 12–13 | Polish, loading states for slower ML calls (per v1.1 §7 non-functional: foundation-model calls may be slow — show progress). |
| 14–15 | Bug fixes, demo rehearsal support. |

---

### F2 — GUI Lead (evidence, reports, disagreement)
*Owns: visual evidence overlays per §6.5, disagreement flag UI (FR9), PDF report (mandatory, §3.6), GeoJSON export (S3, stretch)*

| Days | Task |
|---|---|
| 2–3 | Overlay component shell: bounding boxes (VQA/grounding), change region overlay (both t1/t2), separate optical + SAR overlays (fusion) — build against mock schema data first. |
| 4–5 | Modality attribution display for fusion results (§6.4) — which parts of the answer come from optical vs. SAR. |
| 6–7 | Disagreement flag UI (FR9) — surfaced clearly, not buried, when B2 reports conflicting modality sub-answers. |
| 8–9 | PDF report generator (mandatory, §3.6): answer + evidence + confidence + execution trace, downloadable. Start this early — it's an easy thing to leave to the last two days and regret. |
| 10–11 | Wire overlays against real ML outputs (grounding boxes, change masks) as they land. |
| 12–13 | GeoJSON export (S3, stretch) — only after everything above is solid. |
| 14–15 | Bug fixes, demo rehearsal support. |

---

### I1 — Integration/DevOps Lead
*Owns: `docker-compose`, `tests/integration/`, cross-machine environment consistency, deployment, deliverables checklist*

| Days | Task |
|---|---|
| 0–1 | Mock servers for all 5 ML endpoints + `docker-compose.yml` (all 6 people's services as containers, even as empty shells Day 1). |
| 2–4 | Contract validation tests: for each schema, a test that fixture input → any implementation (mock or real) → output validates against schema. This is what actually catches breakage, not just "it ran." |
| 5–7 | `requirements.txt`/`environment.yml` per service, locked — prevent the "works on my laptop" problem across 6 different machines (v1.1 §7 + solo-ML-plan §5 risk notes). |
| 8–9 | Nightly integration run automation: `docker-compose up` + full `tests/integration/` suite, run daily, report failures back to whoever's contract broke. |
| 10–12 | Cross-service smoke tests as real implementations swap in for mocks — verify each swap didn't silently change response shape. |
| 13–14 | Assemble the full deliverables checklist (v1.1 §9): unit tests, sample inputs (one per input scope), sample queries, expected outputs, checkpoints referenced, README, adaptation log, confidence spec. |
| 15 | Record fallback demo video; final freeze. |

---

## 5. Cross-team checkpoints (don't skip these)

- **End of Day 1:** everyone can hit everyone else's mock endpoint. If not, don't proceed to Phase 1 — fix the contract/mock gap first.
- **Day 5:** your backbone + embedding service should be live (per Solo ML Plan §3) — this is when B1/B2 can start swapping their first real ML call in instead of a mock.
- **Day 9, 11, 12:** each time you ship a new real endpoint (change, fusion, grounding respectively per your sequence), ping B1 immediately — don't let a shipped-but-unannounced endpoint sit unintegrated for days.
- **Day 13:** hard checkpoint — every mandatory item (F1, F4, F5, F6, F7, F8, F9, plus one of F2/F3) must be working end-to-end through the real stack, not mocks. If something mandatory is still mocked here, that's the last day to make the call to simplify rather than ship broken.
