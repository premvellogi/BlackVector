"""
SatQuery AI - B2 Provenance Data Models (D4)
============================================
Immutable lineage and origin records linking inputs, hashes, models, and tasks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def current_utc_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProvenanceRecord:
    """
    Canonical audit record tracking an ML inference or transformation.
    """
    input_hash: str
    model_name: str
    task: str
    timestamp: str = field(default_factory=current_utc_iso)
    run_id: str = "run-default"
    input_id: Optional[str] = None
    source_filename: Optional[str] = None
    hash_algorithm: str = "sha256"
    file_format: Optional[str] = None
    modality: Optional[str] = None
    sensor: Optional[str] = None
    model_version: str = "UNKNOWN"
    checkpoint: str = "UNKNOWN"
    model_provider: Optional[str] = None
    execution_parameters: Dict[str, Any] = field(default_factory=dict)
    processing_stage: str = "INFERENCE"
    output_id: Optional[str] = None
    lineage_parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "hash_algorithm": self.hash_algorithm,
            "input_id": self.input_id,
            "source_filename": self.source_filename,
            "file_format": self.file_format,
            "modality": self.modality,
            "sensor": self.sensor,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "checkpoint": self.checkpoint,
            "model_provider": self.model_provider,
            "task": self.task,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "processing_stage": self.processing_stage,
            "execution_parameters": self.execution_parameters,
            "output_id": self.output_id,
            "lineage_parent_ids": self.lineage_parent_ids,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceRecord:
        return cls(
            input_hash=str(data["input_hash"]),
            model_name=str(data["model_name"]),
            task=str(data["task"]),
            timestamp=str(data.get("timestamp", current_utc_iso())),
            run_id=str(data.get("run_id", "run-default")),
            input_id=data.get("input_id"),
            source_filename=data.get("source_filename"),
            hash_algorithm=str(data.get("hash_algorithm", "sha256")),
            file_format=data.get("file_format"),
            modality=data.get("modality"),
            sensor=data.get("sensor"),
            model_version=str(data.get("model_version", "UNKNOWN")),
            checkpoint=str(data.get("checkpoint", "UNKNOWN")),
            model_provider=data.get("model_provider"),
            execution_parameters=data.get("execution_parameters", {}),
            processing_stage=str(data.get("processing_stage", "INFERENCE")),
            output_id=data.get("output_id"),
            lineage_parent_ids=data.get("lineage_parent_ids", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ProvenanceRecord:
        return cls.from_dict(json.loads(json_str))


@dataclass
class PairedProvenanceRecord:
    """Provenance record for cross-modal (Optical + SAR) or bi-temporal operations."""
    pair_id: str
    optical_provenance: ProvenanceRecord
    sar_provenance: ProvenanceRecord
    task: str
    timestamp: str = field(default_factory=current_utc_iso)
    run_id: str = "run-default"
    fusion_method: str = "LATE_FUSION"
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "optical_provenance": self.optical_provenance.to_dict(),
            "sar_provenance": self.sar_provenance.to_dict(),
            "task": self.task,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "fusion_method": self.fusion_method,
            "extra_metadata": self.extra_metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PairedProvenanceRecord:
        return cls(
            pair_id=str(data["pair_id"]),
            optical_provenance=ProvenanceRecord.from_dict(data["optical_provenance"]),
            sar_provenance=ProvenanceRecord.from_dict(data["sar_provenance"]),
            task=str(data["task"]),
            timestamp=str(data.get("timestamp", current_utc_iso())),
            run_id=str(data.get("run_id", "run-default")),
            fusion_method=str(data.get("fusion_method", "LATE_FUSION")),
            extra_metadata=data.get("extra_metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PairedProvenanceRecord:
        return cls.from_dict(json.loads(json_str))
