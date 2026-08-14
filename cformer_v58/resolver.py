from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .store import LayeredAliasStore, SearchHit


class AliasResolutionStatus(str, Enum):
    VERIFIED = "verified"
    PROPOSED = "proposed"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AliasResolution:
    status: AliasResolutionStatus
    object_id: int | None
    candidate_ids: tuple[int, ...]
    score: float
    margin: float
    candidate_record_id: int | None
    reason: str


class AliasCandidateResolver:
    def __init__(
        self,
        store: LayeredAliasStore,
        *,
        candidate_limit: int = 64,
        minimum_score: float = 0.05,
    ) -> None:
        self.store = store
        self.candidate_limit = candidate_limit
        self.minimum_score = minimum_score

    def resolve(self, surface_form: str, *, source: str = "query") -> AliasResolution:
        exact = self.store.lookup_alias(surface_form)
        if len(exact) == 1:
            return AliasResolution(
                AliasResolutionStatus.VERIFIED,
                exact[0],
                tuple(exact),
                1.0,
                1.0,
                None,
                "verified_alias_index",
            )
        if len(exact) > 1:
            return AliasResolution(
                AliasResolutionStatus.AMBIGUOUS,
                None,
                tuple(exact),
                1.0,
                0.0,
                None,
                "alias_maps_to_multiple_objects",
            )
        hits = self.store.search(surface_form, self.candidate_limit)
        if not hits or hits[0].score < self.minimum_score:
            return AliasResolution(
                AliasResolutionStatus.UNKNOWN,
                None,
                tuple(hit.object_id for hit in hits),
                hits[0].score if hits else 0.0,
                0.0,
                None,
                "no_supported_object_candidate",
            )
        best: SearchHit = hits[0]
        runner_up = hits[1].score if len(hits) > 1 else -1.0
        margin = best.score - runner_up
        record_id = self.store.propose_alias(
            surface_form, best.object_id, best.score, margin, source
        )
        return AliasResolution(
            AliasResolutionStatus.PROPOSED,
            best.object_id,
            tuple(hit.object_id for hit in hits),
            best.score,
            margin,
            record_id,
            "candidate_requires_external_review",
        )
