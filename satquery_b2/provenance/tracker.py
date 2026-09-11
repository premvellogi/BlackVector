"""
SatQuery AI - B2 Provenance Tracker Pipeline (D4)
=================================================
Manages end-to-end lineage tracking linking inputs, models, checkpoints, and outputs.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

from satquery_b2.ontology.schema import OntologyImage, PairedOntologyImage
from satquery_b2.provenance.hasher import calculate_file_hash
from satquery_b2.provenance.models import (
    PairedProvenanceRecord,
    ProvenanceRecord,
    current_utc_iso,
)


class ProvenanceTracker:
    """
    Central provenance tracking coordinator for SatQuery AI.
    """
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self._records: List[ProvenanceRecord] = []

    def create_provenance_record(
        self,
        image_or_path: Union[OntologyImage, str],
        model_name: str,
        task: str,
        model_version: Optional[str] = None,
        checkpoint: Optional[str] = None,
        model_provider: Optional[str] = None,
        execution_parameters: Optional[Dict[str, Any]] = None,
        processing_stage: str = "INFERENCE",
        output_id: Optional[str] = None,
        lineage_parent_ids: Optional[List[str]] = None,
    ) -> ProvenanceRecord:
        """
        Constructs and records a single ProvenanceRecord for an inference run.
        """
        if isinstance(image_or_path, OntologyImage):
            source_file = image_or_path.source_id
            input_id = image_or_path.image_id
            file_format = image_or_path.file_format
            modality = image_or_path.modality.value
            sensor = image_or_path.sensor
        else:
            source_file = str(image_or_path)
            input_id = f"input-{uuid.uuid4().hex[:8]}"
            file_format = "GEOTIFF"
            modality = "unknown"
            sensor = None

        # Compute hash
        try:
            input_hash = calculate_file_hash(source_file)
        except Exception:
            input_hash = f"unhashed-ref-{source_file}"

        record = ProvenanceRecord(
            input_hash=input_hash,
            input_id=input_id,
            source_filename=source_file,
            file_format=file_format,
            modality=modality,
            sensor=sensor,
            model_name=model_name,
            model_version=model_version or "UNKNOWN",
            checkpoint=checkpoint or "UNKNOWN",
            model_provider=model_provider,
            task=task,
            timestamp=current_utc_iso(),
            run_id=self.run_id,
            processing_stage=processing_stage,
            execution_parameters=execution_parameters or {},
            output_id=output_id or f"out-{uuid.uuid4().hex[:8]}",
            lineage_parent_ids=lineage_parent_ids or [],
        )

        self._records.append(record)
        return record

    def create_paired_provenance_record(
        self,
        paired_image: PairedOntologyImage,
        optical_model_name: str,
        sar_model_name: str,
        task: str,
        optical_version: Optional[str] = None,
        sar_version: Optional[str] = None,
        optical_checkpoint: Optional[str] = None,
        sar_checkpoint: Optional[str] = None,
        fusion_method: str = "CROSS_MODAL_FUSION",
    ) -> PairedProvenanceRecord:
        """
        Constructs a PairedProvenanceRecord linking both optical and SAR lineage.
        """
        opt_record = self.create_provenance_record(
            image_or_path=paired_image.optical_image,
            model_name=optical_model_name,
            task=f"{task}_optical",
            model_version=optical_version,
            checkpoint=optical_checkpoint,
        )

        sar_record = self.create_provenance_record(
            image_or_path=paired_image.sar_image,
            model_name=sar_model_name,
            task=f"{task}_sar",
            model_version=sar_version,
            checkpoint=sar_checkpoint,
        )

        return PairedProvenanceRecord(
            pair_id=paired_image.pair_id,
            optical_provenance=opt_record,
            sar_provenance=sar_record,
            task=task,
            timestamp=current_utc_iso(),
            run_id=self.run_id,
            fusion_method=fusion_method,
        )

    def get_records(self) -> List[ProvenanceRecord]:
        """Returns all recorded provenance entries for this tracker session."""
        return list(self._records)
