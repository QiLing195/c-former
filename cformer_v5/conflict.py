from __future__ import annotations

from dataclasses import dataclass

import torch

from cformer_v3.data import (
    FACT_AT,
    LOCATION_BASE,
    NUM_LOCATIONS,
    NUM_PEOPLE,
    Q_PERSON_LOCATION,
)
from cformer_v4.data import STATUS_ANSWER, STATUS_CONFLICT, STATUS_UNKNOWN


@dataclass
class FactMetadata:
    valid_from: torch.Tensor
    valid_to: torch.Tensor
    source: torch.Tensor
    confidence: torch.Tensor
    version: torch.Tensor
    supersedes: torch.Tensor

    @classmethod
    def defaults(cls, num_facts: int) -> "FactMetadata":
        return cls(
            valid_from=torch.zeros(num_facts, dtype=torch.long),
            valid_to=torch.full((num_facts,), 100, dtype=torch.long),
            source=torch.zeros(num_facts, dtype=torch.long),
            confidence=torch.ones(num_facts),
            version=torch.ones(num_facts, dtype=torch.long),
            supersedes=torch.full((num_facts,), -1, dtype=torch.long),
        )

    def clone(self) -> "FactMetadata":
        return FactMetadata(
            self.valid_from.clone(),
            self.valid_to.clone(),
            self.source.clone(),
            self.confidence.clone(),
            self.version.clone(),
            self.supersedes.clone(),
        )


@dataclass(frozen=True)
class Resolution:
    status: int
    answer: int
    evidence: tuple[int, ...]
    reason: str


class ConflictResolver:
    """Trusted structured resolver for temporal, version and source boundaries."""

    def __init__(self, minimum_source_confidence: float = 0.5) -> None:
        self.minimum_source_confidence = minimum_source_confidence

    def resolve(
        self,
        memory: torch.Tensor,
        metadata: FactMetadata,
        question: torch.Tensor,
        query_time: int,
    ) -> Resolution | None:
        if question[0].item() != Q_PERSON_LOCATION:
            return None
        target = question[1].item()
        candidates = (
            memory[:, 0].eq(FACT_AT)
            & memory[:, 1].eq(target)
            & metadata.valid_from.le(query_time)
            & metadata.valid_to.ge(query_time)
            & metadata.confidence.ge(self.minimum_source_confidence)
        ).nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            return Resolution(STATUS_UNKNOWN, 0, (), "no_active_trusted_fact")

        superseded = set()
        candidate_set = set(candidates.tolist())
        for index in candidates.tolist():
            old = metadata.supersedes[index].item()
            if old in candidate_set:
                superseded.add(old)
        active = [index for index in candidates.tolist() if index not in superseded]
        if not active:
            return Resolution(STATUS_UNKNOWN, 0, (), "all_candidates_superseded")

        values: dict[int, list[int]] = {}
        for index in active:
            location_token = memory[index, 2].item()
            if not LOCATION_BASE <= location_token < LOCATION_BASE + NUM_LOCATIONS:
                continue
            values.setdefault(location_token, []).append(index)
        if not values:
            return Resolution(STATUS_UNKNOWN, 0, tuple(active), "malformed_location")
        evidence = tuple(index for indices in values.values() for index in indices)
        if len(values) > 1:
            return Resolution(STATUS_CONFLICT, 0, evidence, "overlapping_incompatible_values")
        location_token = next(iter(values))
        return Resolution(
            STATUS_ANSWER,
            NUM_PEOPLE + location_token - LOCATION_BASE,
            evidence,
            "single_active_value",
        )

