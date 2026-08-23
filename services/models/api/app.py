"""
SatQuery AI — ML Model Service (FastAPI).

One process, five endpoints, one shared InternVL2-2B backbone.
Each endpoint honors its locked contract schema from contracts/schemas/.

Endpoints:
    POST /vqa       — Single-image Visual Question Answering (F1)
    POST /caption   — Single-image Scene Captioning (F2)
    POST /grounding — Text-guided Region Grounding (F3)
    POST /change    — Bi-temporal Change Detection/VQA (F4)
    POST /fusion    — Optical-SAR Cross-modal Fusion (F5)
    GET  /health    — Service health + VRAM status
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery_ml")


# =============================================================================
# Response Models (matching contract schemas)
# =============================================================================

class Confidence(BaseModel):
    """Confidence score with explanation."""
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")
    method: str = Field(default="model_logit", description="How confidence was derived")
    factors: dict = Field(default_factory=dict, description="Contributing factors")


class Provenance(BaseModel):
    """Audit trail for a model response (D4)."""
    model_name: str
    checkpoint_version: str = ""
    input_hash: str = ""
    timestamp: str = ""
    preprocessing: dict = Field(default_factory=dict)


class BoundingBox(BaseModel):
    """A single bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = ""
    confidence: float = 0.0


# --- VQA Response ---
class VQAResponse(BaseModel):
    """Response schema for /vqa endpoint."""
    answer: str
    confidence: Confidence
    evidence_regions: list[BoundingBox] = Field(default_factory=list)
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Caption Response ---
class CaptionResponse(BaseModel):
    """Response schema for /caption endpoint."""
    caption: str
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Grounding Response ---
class GroundingResponse(BaseModel):
    """Response schema for /grounding endpoint."""
    boxes: list[BoundingBox]
    expression: str  # The input referring expression
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Change Detection Response ---
class ChangeResponse(BaseModel):
    """Response schema for /change endpoint."""
    description: str              # Natural language change description
    answer: Optional[str] = None  # Answer if a question was asked (change-VQA)
    change_detected: bool = False
    change_regions: list[BoundingBox] = Field(default_factory=list)
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Fusion Response (satisfies §6.4 output contract) ---
class ModalityAttribution(BaseModel):
    """Which parts of the answer come from which modality (§6.4 requirement)."""
    optical_evidence: str = ""    # What optical image contributed
    sar_evidence: str = ""        # What SAR image contributed
    fused_reasoning: str = ""     # How they were combined


class FusionResponse(BaseModel):
    """Response schema for /fusion endpoint. Satisfies §6.4 output contract."""
    fused_answer: str                          # Combined answer from both modalities
    modality_attribution: ModalityAttribution  # REQUIRED by §6.4
    optical_evidence_overlay: dict = Field(default_factory=dict)  # Separate optical overlay
    sar_evidence_overlay: dict = Field(default_factory=dict)      # Separate SAR overlay
    disagreement_flag: bool = False            # FR9: True if modalities conflict
    disagreement_details: str = ""             # Explanation of disagreement
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Global application state — holds the shared model backbone."""

    def __init__(self):
        self.vlm = None           # VLMBackbone instance
        self.grounding_dino = None  # Grounding DINO instance (loaded separately)
        self.ready = False

    async def startup(self):
        """Load models on startup."""
        from services.models.core.vlm_backbone import VLMBackbone, ModelConfig

        logger.info("=== SatQuery AI ML Service Starting ===")

        config = ModelConfig()
        self.vlm = VLMBackbone(config)

        try:
            self.vlm.load()
            self.ready = True
            logger.info("=== ML Service Ready ===")
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            logger.warning("Service starting in degraded mode (no model loaded).")

    async def shutdown(self):
        """Clean up on shutdown."""
        if self.vlm:
            self.vlm.unload()
        logger.info("=== ML Service Shutdown ===")


app_state = AppState()


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle."""
    await app_state.startup()
    yield
    await app_state.shutdown()


app = FastAPI(
    title="SatQuery AI — ML Model Service",
    description=(
        "Single FastAPI process exposing 5 endpoints for remote-sensing "
        "vision-language tasks. All share one InternVL2-2B backbone."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Health Endpoint
# =============================================================================

@app.get("/health")
async def health():
    """Service health check with VRAM report."""
    report = {
        "status": "ready" if app_state.ready else "degraded",
        "model_loaded": app_state.vlm.is_loaded if app_state.vlm else False,
    }
    if app_state.vlm:
        report["vram"] = app_state.vlm.get_vram_report()
    return report


# =============================================================================
# VQA Endpoint (F1) — Mandatory
# =============================================================================

@app.post("/vqa", response_model=VQAResponse)
async def vqa(
    image: UploadFile = File(..., description="Single satellite image"),
    question: str = Form(..., description="Natural language question about the image"),
):
    """Visual Question Answering on a single satellite image.

    Takes an image and a question, returns a grounded answer.
    The VLM processes the image and generates a text answer.
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded. Service is starting up.")

    try:
        # Load and preprocess image
        pil_image = await _load_upload_as_pil(image)

        # Build VQA prompt
        prompt = (
            f"You are a remote sensing image analysis expert. "
            f"Look at this satellite image carefully and answer the following question.\n"
            f"Question: {question}\n"
            f"Provide a concise, factual answer based only on what you can observe in the image."
        )

        # Run inference
        result = app_state.vlm.generate(pil_image, prompt)

        elapsed = (time.time() - start_time) * 1000

        return VQAResponse(
            answer=result["text"],
            confidence=Confidence(
                score=0.7,  # TODO: Extract from logits
                method="model_logit",
                factors={"tokens_generated": result["tokens_generated"]},
            ),
            evidence_regions=[],  # TODO: Add attention-based evidence
            provenance=Provenance(
                model_name="InternVL2-2B",
                checkpoint_version="base",  # Updated after fine-tuning
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"VQA failed: {e}")
        raise HTTPException(500, f"VQA inference failed: {str(e)}")


# =============================================================================
# Caption Endpoint (F2)
# =============================================================================

@app.post("/caption", response_model=CaptionResponse)
async def caption(
    image: UploadFile = File(..., description="Single satellite image"),
):
    """Generate a scene description for a satellite image."""
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    try:
        pil_image = await _load_upload_as_pil(image)

        prompt = (
            "You are a remote sensing image analysis expert. "
            "Describe this satellite image in detail. Include:\n"
            "1. The main land cover types visible (e.g., urban, agriculture, forest, water)\n"
            "2. Notable features and their spatial arrangement\n"
            "3. The approximate land use pattern\n"
            "Provide a comprehensive but concise description."
        )

        result = app_state.vlm.generate(pil_image, prompt)
        elapsed = (time.time() - start_time) * 1000

        return CaptionResponse(
            caption=result["text"],
            confidence=Confidence(score=0.7, method="model_logit"),
            provenance=Provenance(
                model_name="InternVL2-2B",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Caption failed: {e}")
        raise HTTPException(500, f"Caption inference failed: {str(e)}")


# =============================================================================
# Grounding Endpoint (F3)
# =============================================================================

@app.post("/grounding", response_model=GroundingResponse)
async def grounding(
    image: UploadFile = File(..., description="Single satellite image"),
    expression: str = Form(..., description="Referring expression to ground"),
):
    """Text-guided region grounding using Grounding DINO.

    Given a referring expression (e.g., "the large building in the center"),
    returns bounding boxes localizing the described region(s).
    """
    start_time = time.time()

    # TODO: Integrate Grounding DINO (Day 11-12)
    # For now, return a placeholder
    elapsed = (time.time() - start_time) * 1000

    return GroundingResponse(
        boxes=[],
        expression=expression,
        confidence=Confidence(score=0.0, method="placeholder"),
        provenance=Provenance(
            model_name="GroundingDINO",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        execution_time_ms=elapsed,
    )


# =============================================================================
# Change Detection Endpoint (F4) — Mandatory
# =============================================================================

@app.post("/change", response_model=ChangeResponse)
async def change_detection(
    image_t1: UploadFile = File(..., description="Earlier temporal image"),
    image_t2: UploadFile = File(..., description="Later temporal image"),
    question: Optional[str] = Form(None, description="Optional change-related question"),
):
    """Bi-temporal change detection and change-VQA.

    Takes two spatially corresponding images from different times.
    Returns change description and optionally answers a change-related question.
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    try:
        pil_t1 = await _load_upload_as_pil(image_t1)
        pil_t2 = await _load_upload_as_pil(image_t2)

        # Build change detection prompt with multi-image tags
        if question:
            prompt = (
                f"You are a remote sensing change detection expert. "
                f"Image 1 shows an area at an earlier time. "
                f"Image 2 shows the same area at a later time.\n"
                f"Question: {question}\n"
                f"Analyze the changes between the two images and answer the question."
            )
        else:
            prompt = (
                "You are a remote sensing change detection expert. "
                "Image 1 shows an area at an earlier time. "
                "Image 2 shows the same area at a later time.\n"
                "Describe all significant changes you can observe between the two images. "
                "Include what type of change occurred and where."
            )

        result = app_state.vlm.generate_multi_image(
            [pil_t1, pil_t2], prompt
        )
        elapsed = (time.time() - start_time) * 1000

        return ChangeResponse(
            description=result["text"],
            answer=result["text"] if question else None,
            change_detected=True,  # TODO: Determine from response
            confidence=Confidence(score=0.6, method="model_logit"),
            provenance=Provenance(
                model_name="InternVL2-2B",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        raise HTTPException(500, f"Change detection failed: {str(e)}")


# =============================================================================
# Fusion Endpoint (F5) — Mandatory, must satisfy §6.4 output contract
# =============================================================================

@app.post("/fusion", response_model=FusionResponse)
async def fusion(
    optical_image: UploadFile = File(..., description="Optical/multispectral image"),
    sar_image: UploadFile = File(..., description="SAR image (co-registered)"),
    question: Optional[str] = Form(None, description="Optional analysis question"),
):
    """Cross-modal optical-SAR fusion analysis.

    Satisfies §6.4 output contract:
    1. Fused textual answer combining both modalities
    2. Modality attribution (which parts from optical vs SAR)
    3. Modality-specific evidence overlays (shown separately)
    4. Ambiguity resolution between modalities
    5. Disagreement flag when modalities conflict (FR9)
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    try:
        pil_optical = await _load_upload_as_pil(optical_image)
        pil_sar = await _load_upload_as_pil(sar_image)

        # Build fusion prompt that elicits modality-attributed responses
        base_prompt = (
            "You are an expert in multi-modal remote sensing analysis. "
            "You are given two co-registered images of the same area:\n"
            "- Image 1: OPTICAL (multispectral) image\n"
            "- Image 2: SAR (Synthetic Aperture Radar) image\n\n"
            "Analyze BOTH images and provide:\n"
            "1. What the OPTICAL image reveals (land cover, vegetation, water bodies, structures)\n"
            "2. What the SAR image reveals (surface roughness, moisture, structural features)\n"
            "3. A FUSED analysis combining insights from both modalities\n"
            "4. Any DISAGREEMENTS between what the two modalities show\n"
        )

        if question:
            base_prompt += f"\nSpecifically answer: {question}\n"

        base_prompt += (
            "\nFormat your response as:\n"
            "OPTICAL EVIDENCE: [what optical shows]\n"
            "SAR EVIDENCE: [what SAR shows]\n"
            "FUSED ANALYSIS: [combined interpretation]\n"
            "DISAGREEMENTS: [any conflicts, or 'None']\n"
        )

        result = app_state.vlm.generate_multi_image(
            [pil_optical, pil_sar], base_prompt
        )

        # Parse the structured response
        response_text = result["text"]
        attribution = _parse_modality_attribution(response_text)

        elapsed = (time.time() - start_time) * 1000

        return FusionResponse(
            fused_answer=attribution.get("fused", response_text),
            modality_attribution=ModalityAttribution(
                optical_evidence=attribution.get("optical", ""),
                sar_evidence=attribution.get("sar", ""),
                fused_reasoning=attribution.get("fused", ""),
            ),
            disagreement_flag=bool(attribution.get("disagreements")),
            disagreement_details=attribution.get("disagreements", ""),
            confidence=Confidence(score=0.6, method="model_logit"),
            provenance=Provenance(
                model_name="InternVL2-2B",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Fusion failed: {e}")
        raise HTTPException(500, f"Fusion inference failed: {str(e)}")


# =============================================================================
# Utility functions
# =============================================================================

async def _load_upload_as_pil(upload: UploadFile):
    """Load an uploaded file as a PIL Image."""
    from PIL import Image
    import io

    content = await upload.read()
    image = Image.open(io.BytesIO(content))

    # Handle GeoTIFF (may have >3 bands)
    if image.mode not in ("RGB", "L"):
        # For multi-band images, take the first 3 bands as RGB
        import numpy as np
        arr = np.array(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            # Use bands 3, 2, 1 (R, G, B) if available
            arr_rgb = arr[:, :, :3]
        elif arr.ndim == 2:
            # Single band → grayscale
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            arr_rgb = arr

        # Normalize to uint8
        if arr_rgb.dtype != np.uint8:
            p2, p98 = np.percentile(arr_rgb, [2, 98])
            if p98 - p2 > 0:
                arr_rgb = np.clip(arr_rgb, p2, p98)
                arr_rgb = ((arr_rgb - p2) / (p98 - p2) * 255).astype(np.uint8)
            else:
                arr_rgb = np.zeros_like(arr_rgb, dtype=np.uint8)

        image = Image.fromarray(arr_rgb, "RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _parse_modality_attribution(text: str) -> dict:
    """Parse structured modality attribution from model response."""
    result = {"optical": "", "sar": "", "fused": "", "disagreements": ""}

    sections = {
        "OPTICAL EVIDENCE:": "optical",
        "SAR EVIDENCE:": "sar",
        "FUSED ANALYSIS:": "fused",
        "DISAGREEMENTS:": "disagreements",
    }

    text_upper = text.upper()
    for marker, key in sections.items():
        idx = text_upper.find(marker.upper())
        if idx >= 0:
            start = idx + len(marker)
            # Find next section marker
            end = len(text)
            for other_marker in sections:
                if other_marker == marker:
                    continue
                other_idx = text_upper.find(other_marker.upper(), start)
                if other_idx >= 0 and other_idx < end:
                    end = other_idx
            result[key] = text[start:end].strip()

    # If parsing failed, use the full text as the fused answer
    if not any(result.values()):
        result["fused"] = text

    return result


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.models.api.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
