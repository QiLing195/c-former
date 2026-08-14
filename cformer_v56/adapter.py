from __future__ import annotations

import hashlib
import math
import struct

from cformer_v55 import CognitiveMemory

from .retriever import CandidateRecord


def _semantic_key(object_id: int, dimensions: int = 8) -> tuple[float, ...]:
    digest = hashlib.sha256(f"cformer-object:{object_id}".encode("utf-8")).digest()
    values = [
        (struct.unpack_from("<I", digest, 4 * index)[0] / 2**31) - 1.0
        for index in range(dimensions)
    ]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


class CognitiveCandidateAdapter:
    """Maps V5.5 canonical states/dynamics to bounded neural candidate records."""

    def __init__(self, key_dimensions: int = 8) -> None:
        self.key_dimensions = key_dimensions

    def key_for_object(self, object_id: int) -> tuple[float, ...]:
        return _semantic_key(object_id, self.key_dimensions)

    def adapt(
        self, memory: CognitiveMemory, version: int | None = None
    ) -> tuple[CandidateRecord, ...]:
        records: list[CandidateRecord] = []
        for state in memory.snapshot_states(version):
            records.append(
                CandidateRecord(
                    candidate_id=state.state_id,
                    object_id=state.object_id,
                    kind=0,
                    semantic_key=self.key_for_object(state.object_id),
                    valid_from=state.valid_from,
                    valid_to=state.valid_to,
                    ingest_time=state.ingest_time,
                    confidence=state.confidence,
                    visibility_scope=state.visibility_scope,
                    equivalence_group=state.version,
                    evidence_id=state.state_id,
                )
            )
        offset = 1_000_000_000
        for transformation in memory.snapshot_transformations(version):
            for object_id in transformation.input_object_ids:
                records.append(
                    CandidateRecord(
                        candidate_id=offset + transformation.transformation_id,
                        object_id=object_id,
                        kind=1,
                        semantic_key=self.key_for_object(object_id),
                        valid_from=transformation.valid_from,
                        valid_to=transformation.valid_to,
                        ingest_time=transformation.ingest_time,
                        confidence=transformation.confidence,
                        visibility_scope=transformation.visibility_scope,
                        equivalence_group=transformation.version,
                        evidence_id=transformation.transformation_id,
                    )
                )
        return tuple(records)
