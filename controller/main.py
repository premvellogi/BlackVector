from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .classifier import QueryClassifier
from .clients import ServiceClient
from .models import ControllerResponse, QueryRequest, TaskType, ToolSpec
from .registry import get_registry, register_tool
from .state_machine import ControllerEngine

app = FastAPI(title="SatQuery AI — B1 Controller", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = ControllerEngine(ServiceClient(), QueryClassifier())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "satquery-b1-controller"}


@app.get("/tools")
async def tools() -> dict:
    return {"tools": [spec.model_dump(mode="json") for spec in get_registry().values()]}


@app.post("/tools")
async def add_tool(spec: ToolSpec) -> dict:
    register_tool(spec)
    return {"registered": spec.model_dump(mode="json")}


@app.post("/query", response_model=ControllerResponse)
async def query(request: QueryRequest) -> ControllerResponse:
    if len(request.query) > 2000:
        raise HTTPException(status_code=422, detail="query exceeds 2000 characters")
    return await engine.execute(request)


@app.get("/tasks")
async def tasks() -> dict:
    return {"tasks": [task.value for task in TaskType]}
