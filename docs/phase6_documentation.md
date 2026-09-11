# SatQuery AI — Phase 6 Technical Documentation

**Module**: Qwen 3 Router, Evidence Verifier & Dynamic Model Registry  
**Version**: v3.0.0 (EOV2B Agentic Architecture)  
**Date**: September 2026  

---

## 1. Executive Summary

Phase 6 implements the **Agentic Orchestration Layer** for SatQuery AI. This layer bridges incoming natural language user queries with the underlying multi-modal vision-language models (EOV2B, ChangeFormer, GroundingDINO, MobileSAM, and SAREncoder).

Key capabilities delivered in Phase 6:
1. **Query Intent Routing (`Qwen3Router`)**: Automatically classifies natural language user queries and input scopes (`SINGLE`, `BI_TEMPORAL`, `CROSS_MODAL`) into target task handlers (`VQA`, `CAPTION`, `GROUNDING`, `CHANGE`, `FUSION`) while extracting target land-cover entities.
2. **Evidence Verification (`EvidenceVerifier`)**: Validates spatial bounding box sanity, change mask coverage, detects cross-modal disagreement, and filters LLM hallucinated facts (dates, location claims, measurements) using `answer_validator`.
3. **Dynamic Task Dispatcher (`ModelRegistry`)**: Provides a dynamic, config-driven registry for task handlers matching B1 Controller specs without requiring controller code changes.
4. **Agentic Serving Endpoints**: Exposes `/analyze` for unified intent-routed execution and `/route` for standalone intent classification.

---

## 2. System Architecture

```
                                  +-------------------------------------------------------+
                                  |              SatQuery AI API (server.py)             |
                                  |                                                       |
                                  |  POST /analyze   POST /route   POST /vqa  POST /caption...|
                                  +---------------------------+---------------------------+
                                                              |
                                             +----------------v----------------+
                                             |           Qwen3Router           |
                                             | (Intent Classify & Entity Extr) |
                                             +----------------+----------------+
                                                              |
                                             +----------------v----------------+
                                             |          ModelRegistry          |
                                             |  (Dynamic Handler Dispatcher)   |
                                             +----------------+----------------+
                                                              |
                 +-----------------------+--------------------+-----------------------+-----------------------+
                 |                       |                    |                       |                       |
        +--------v-------+      +--------v-------+   +--------v-------+      +--------v-------+      +--------v-------+
        |   VQAHandler   |      | CaptionHandler |   |GroundingHandler|      | ChangeHandler  |      | FusionHandler  |
        | (EOV2B 4-bit)  |      | (EOV2B 4-bit)  |   | (GDINO + SAM)  |      | (ChangeFormer) |      | (SAR Encoder)  |
        +--------+-------+      +--------+-------+   +--------+-------+      +--------+-------+      +--------+-------+
                 |                       |                    |                       |                       |
                 +-----------------------+--------------------+-----------------------+-----------------------+
                                                              |
                                             +----------------v----------------+
                                             |        EvidenceVerifier         |
                                             | (Bbox, Mask & Hallucination Filter)|
                                             +----------------+----------------+
                                                              |
                                                     Verified MLResponse
```

---

## 3. VRAM Budget Optimization

A critical design constraint was keeping total active VRAM within system bounds (~6.4 GB available), avoiding OOM errors when multiple models operate simultaneously.

| Component | VRAM Footprint | Loading Strategy |
|---|---|---|
| **EOV2B (Qwen2-VL-2B)** | ~1,525 MB | Resident (4-bit NF4 quantized) |
| **Qwen3Router** | 0 MB (Rule/Pattern + Zero-shot) | Resident in host RAM |
| **EvidenceVerifier** | 0 MB (CPU validation algorithms) | Resident in host RAM |
| **ChangeFormer-Lite** | ~28 MB | Resident on GPU |
| **SAREncoder + FusionHead** | ~80 MB | Resident on GPU |
| **MobileSAM (vit_t)** | ~48 MB | Resident on GPU |
| **GroundingDINO-Tiny** | ~800 MB | VRAM-swapped dynamically on demand |
| **Total Active VRAM** | **~1,680 MB** | Fits comfortably with ~4.7 GB free buffer |

---

## 4. Component Technical Breakdown

### 4.1 Qwen 3 Router (`services/models/core/qwen3_router.py`)

The `Qwen3Router` processes the raw user query, input scope, and metadata to generate a `RouterDecision`.

#### Intent Classification Logic
- **`TaskType.GROUNDING`**: Triggered by spatial detection keywords (`find`, `locate`, `detect`, `highlight`, `segment`, `where is`, `bounding box`, `coordinates`).
- **`TaskType.CHANGE`**: Triggered by `InputScope.BI_TEMPORAL`, multi-image temporal inputs, or change keywords (`change`, `difference`, `altered`, `built`, `deforestation`, `urban growth`).
- **`TaskType.FUSION`**: Triggered by `InputScope.CROSS_MODAL` or radar/optical keywords (`sar`, `radar`, `sentinel-1`, `sentinel-2`, `optical+sar`, `penetrate clouds`).
- **`TaskType.CAPTION`**: Triggered by description requests (`describe`, `caption`, `summary`, `overview`) or empty queries with an image attached.
- **`TaskType.VQA`**: Default fallback for general remote sensing questions.

#### Target Entity Extraction
Regex-based extraction parses remote sensing target categories (`water body`, `river`, `lake`, `building`, `urban area`, `forest`, `farmland`, `road`, `solar panel`, etc.) from user text to guide downstream segmentation heads.

---

### 4.2 Evidence Verifier (`services/models/core/evidence_verifier.py`)

The `EvidenceVerifier` acts as a quality gate for all model outputs prior to returning responses to the user or Controller.

#### Verification Pipeline Steps
1. **Bounding Box Validation**:
   - Verifies coordinate range (normalized `[0, 1000]` scale).
   - Rejects inverted coordinates ($y_{max} \le y_{min}$ or $x_{max} \le x_{min}$) and zero-area boxes.
2. **Change Mask Verification**:
   - Validates pixel change percentage $0\% \le \Delta \le 100\%$.
3. **Hallucination & Unsupported Claim Filter**:
   - Walks answer text using `answer_validator`.
   - Strips unsupported geographic names (e.g. "Serbia", "Europe"), dates/years (e.g. "summer 2023"), or exact area measurements unless backed by image metadata facts.
4. **Cross-Modal Disagreement Detection**:
   - Evaluates modality attribution weights ($\text{optical\_weight}$ vs $\text{sar\_weight}$). Flags `disagreement = True` if one modality is severely suppressed ($< 5\%$).
5. **Composite Confidence Calculation**:
   $$\text{Confidence}_{\text{final}} = \text{Confidence}_{\text{base}} \times \overline{\text{QualityFactors}}$$

---

### 4.3 Dynamic Model Registry (`services/models/serving/model_registry.py`)

The `ModelRegistry` decouples handler implementation from server routing:
- **`register_handler(task_type, handler)`**: Allows dynamic handler registration without modifying core server logic.
- **`route_request(request)`**: Automates intent classification when `task_type` is unassigned or `auto_route` is set.
- **`dispatch(request, manager)`**: Orchestrates execution flow: `Route -> Execute Handler -> Verify Evidence -> Return MLResponse`.

---

### 4.4 Serving API Integration (`server.py`)

New endpoints added to FastAPI:
- **`POST /analyze`**: Accepts `MLRequest`, auto-routes through `Qwen3Router`, invokes corresponding handler, runs `EvidenceVerifier`, and returns verified `MLResponse`.
- **`POST /route`**: Accepts `MLRequest` and returns `RouterDecision` JSON (`task_type`, `confidence`, `reasoning`, `extracted_entities`, `execution_plan`).

---

## 5. Verification & Test Results

Unit and integration tests were created in `tests/test_phase6_router_verifier.py` and run alongside existing serving tests.

### Test Execution Summary

| Test Suite | Target Component | Status | Passed Tests |
|---|---|---|---|
| `test_phase6_router_verifier.py` | `Qwen3Router`, `EvidenceVerifier`, `ModelRegistry` | **PASSED** | 4 / 4 |
| `test_vqa.py` | `/vqa` endpoint & EOV2B VQA pipeline | **PASSED** | 1 / 1 |
| `test_caption.py` | `/caption` endpoint & land cover classification | **PASSED** | 1 / 1 |
| `test_change.py` | `/change` endpoint & ChangeFormer masks | **PASSED** | 1 / 1 |
| `test_fusion.py` | `/fusion` endpoint & SAR Cross-Attention | **PASSED** | 1 / 1 |
| **Total Test Suite** | **All Serving & Intent Verification Pipeline** | **PASSED** | **8 / 8 (100%)** |

---

## 6. How to Run & Verify

1. **Launch Full Stack**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File d:\Netra\run_satquery.ps1
   ```
2. **Check Health**:
   ```bash
   curl http://localhost:8200/health
   ```
3. **Test Agentic Router Endpoint**:
   ```bash
   curl -X POST http://localhost:8200/route \
     -H "Content-Type: application/json" \
     -d '{"query": "Find and locate all water bodies and buildings", "input_scope": "single"}'
   ```
4. **Test Agentic Analysis Endpoint**:
   ```bash
   curl -X POST http://localhost:8200/analyze \
     -H "Content-Type: application/json" \
     -d '{"query": "Describe this satellite image", "image_ids": ["patch_0"]}'
   ```
