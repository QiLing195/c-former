from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


Scalar = int | float | str | bool
Value = Scalar | tuple[float, float]
Properties = tuple[tuple[str, Value], ...]


class ObjectLifecycle(str, Enum):
    ACTIVE = "active"
    DECAYED = "decayed"
    ARCHIVED = "archived"
    SPLIT = "split"
    MERGED = "merged"


class CognitiveStatus(str, Enum):
    ANSWER = "answer"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    ACCESS_DENIED = "access_denied"
    ALIGNMENT_MISSING = "alignment_missing"
    CONSTRAINT_UNSATISFIED = "constraint_unsatisfied"
    CONSTRAINT_MISSING = "constraint_missing"
    VERSION_CHANGED = "version_changed"


class ConstraintOperator(str, Enum):
    EQ = "eq"
    LE = "le"
    GE = "ge"
    IN = "in"


class EffectOperator(str, Enum):
    SET = "set"
    ADD = "add"
    ADD_VECTOR = "add_vector"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    COMMITTED = "committed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Constraint:
    field: str
    operator: ConstraintOperator
    expected: Value | tuple[Scalar, ...]


@dataclass(frozen=True)
class Effect:
    field: str
    operator: EffectOperator
    value: Value


@dataclass(frozen=True)
class SemanticObject:
    object_id: int
    canonical_name: str
    aliases: frozenset[str] = frozenset()
    object_type: str = "entity"
    valid_from: int = 0
    valid_to: int = 2**31 - 1
    lifecycle: ObjectLifecycle = ObjectLifecycle.ACTIVE
    predecessor_ids: tuple[int, ...] = ()
    successor_ids: tuple[int, ...] = ()
    evidence_ids: tuple[int, ...] = ()
    confidence: float = 1.0
    visibility_scope: frozenset[int] | None = None

    @property
    def names(self) -> frozenset[str]:
        return self.aliases | {self.canonical_name}


@dataclass(frozen=True)
class ObjectState:
    state_id: int
    object_id: int
    properties: Properties
    event_time: int = 0
    ingest_time: int = 0
    valid_from: int = 0
    valid_to: int = 2**31 - 1
    version: int = 1
    source: int = 0
    confidence: float = 1.0
    supersedes: int | None = None
    visibility_scope: frozenset[int] | None = None
    coordinate_frame: int | None = None

    def get(self, field: str) -> Value | None:
        return dict(self.properties).get(field)


@dataclass(frozen=True)
class Transformation:
    transformation_id: int
    operator: str
    input_object_ids: tuple[int, ...]
    output_object_id: int
    constraints: tuple[Constraint, ...]
    effects: tuple[Effect, ...]
    valid_from: int = 0
    valid_to: int = 2**31 - 1
    ingest_time: int = 0
    version: int = 1
    source: int = 0
    confidence: float = 1.0
    evidence_ids: tuple[int, ...] = ()
    visibility_scope: frozenset[int] | None = None


@dataclass(frozen=True)
class ObserverContext:
    scope: int
    frame: int
    query_time: int
    ingest_cutoff: int
    world_version: int | None = None


@dataclass(frozen=True)
class QueryResult:
    status: CognitiveStatus
    object_id: int | None
    value: Properties | None
    evidence_ids: tuple[int, ...]
    reason: str
    world_version: int

    def get(self, field: str) -> Value | None:
        return dict(self.value or ()).get(field)


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: int
    transformation_id: int
    source_object_id: int
    target_object_id: int
    predicted_properties: Properties
    base_world_version: int
    query_time: int
    evidence_ids: tuple[int, ...]
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    reason: str = "simulation_only_not_committed"
