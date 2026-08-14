from __future__ import annotations

from dataclasses import dataclass

import torch

from cformer_v55.schema import CognitiveStatus


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: int
    object_id: int
    kind: int
    semantic_key: tuple[float, ...]
    perspective_scope: int = 0
    valid_from: int = 0
    valid_to: int = 2**31 - 1
    ingest_time: int = 0
    confidence: float = 1.0
    visibility_scope: frozenset[int] | None = None
    equivalence_group: int = 0
    evidence_id: int = 0


@dataclass(frozen=True)
class NeuralQuery:
    target_object_id: int
    kind: int
    semantic_key: tuple[float, ...]
    observer_scope: int
    query_time: int
    ingest_cutoff: int


@dataclass(frozen=True)
class PrefilterResult:
    status: CognitiveStatus
    candidates: tuple[CandidateRecord, ...]
    correct_candidate_ids: tuple[int, ...]
    reason: str


class GovernedRetriever:
    """Deterministic boundary layer followed by bounded semantic Top-K recall."""

    def __init__(self, candidates: tuple[CandidateRecord, ...], *, top_k: int = 64) -> None:
        self.candidates = candidates
        self.top_k = top_k

    @staticmethod
    def _active(candidate: CandidateRecord, query: NeuralQuery) -> bool:
        return (
            candidate.kind == query.kind
            and candidate.ingest_time <= query.ingest_cutoff
            and candidate.valid_from <= query.query_time < candidate.valid_to
            and candidate.confidence >= 0.5
        )

    @staticmethod
    def _visible(candidate: CandidateRecord, query: NeuralQuery) -> bool:
        return (
            candidate.visibility_scope is None
            or query.observer_scope in candidate.visibility_scope
        )

    def prefilter(self, query: NeuralQuery) -> PrefilterResult:
        target = [
            candidate
            for candidate in self.candidates
            if candidate.object_id == query.target_object_id
            and self._active(candidate, query)
            and candidate.perspective_scope in (0, query.observer_scope)
        ]
        if not target:
            return PrefilterResult(
                CognitiveStatus.UNKNOWN, (), (), "no_active_target_candidate"
            )
        visible_target = [candidate for candidate in target if self._visible(candidate, query)]
        if not visible_target:
            return PrefilterResult(
                CognitiveStatus.ACCESS_DENIED, (), (), "target_candidate_not_visible"
            )
        groups = {candidate.equivalence_group for candidate in visible_target}
        if len(groups) > 1:
            return PrefilterResult(
                CognitiveStatus.CONFLICT,
                (),
                (),
                "incompatible_target_candidates",
            )
        legal = [
            candidate
            for candidate in self.candidates
            if self._active(candidate, query) and self._visible(candidate, query)
        ]
        query_key = torch.tensor(query.semantic_key, dtype=torch.float32)
        keys = torch.tensor([candidate.semantic_key for candidate in legal])
        scores = torch.mv(keys, query_key)
        top = scores.topk(min(self.top_k, len(legal))).indices.tolist()
        shortlist = tuple(legal[index] for index in top)
        correct = tuple(candidate.candidate_id for candidate in visible_target)
        return PrefilterResult(
            CognitiveStatus.ANSWER, shortlist, correct, "governed_topk_ready"
        )
