from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SatQuery Mock B2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Validation(BaseModel):
    request_id: str
    input_scope: str
    image_ids: list[str]
    task_type: str
    metadata: dict = {}


class Confidence(BaseModel):
    request_id: str
    task_type: str
    model_confidence: float | None = None
    cross_modal_agreement: float | None = None
    input_quality: float | None = None
    raw_block_output: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate")
def validate(req: Validation):
    reasons: list[str] = []
    if req.task_type in {"change"} and req.input_scope != "bi_temporal":
        reasons.append("change requires bi_temporal input scope")
    if req.task_type == "fusion" and req.input_scope != "cross_modal":
        reasons.append("fusion requires cross_modal input scope")
    if req.task_type in {"vqa", "caption", "grounding"} and req.input_scope != "single":
        reasons.append(f"{req.task_type} requires single input scope")
    if len(req.image_ids) > 2:
        reasons.append("maximum of two images is supported")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "normalized_input": {"input_quality": 0.95, "validated": not reasons},
    }


@app.post("/confidence")
def confidence(req: Confidence):
    model = req.model_confidence if req.model_confidence is not None else 0.75
    agreement = req.cross_modal_agreement if req.cross_modal_agreement is not None else 1.0
    quality = req.input_quality if req.input_quality is not None else 0.9
    score = 0.55 * model + 0.25 * agreement + 0.20 * quality
    disagreement = agreement < 0.60
    return {
        "score": round(score, 4),
        "components": {"model": model, "agreement": agreement, "input_quality": quality},
        "disagreement": disagreement,
        "explanation": "Weighted combination of model confidence, agreement, and input quality.",
    }
