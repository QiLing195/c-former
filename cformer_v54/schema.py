from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class QueryStatus(str, Enum):
    ANSWER = "answer"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    ACCESS_DENIED = "access_denied"
    CYCLE = "cycle"
    DEPTH_LIMIT = "depth_limit"
    ALIGNMENT_MISSING = "alignment_missing"
    VERSION_CHANGED = "version_changed"


@dataclass(frozen=True)
class Fact:
    fact_id: int
    subject: int
    relation: str
    object_id: int | None = None
    position: tuple[float, float] | None = None
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


@dataclass(frozen=True)
class RecursiveState:
    current_entity: int
    observer_scope: int
    observer_frame: int
    query_time: int
    ingest_cutoff: int
    world_version: int
    visited_entities: tuple[int, ...]
    evidence_ids: tuple[int, ...]
    depth: int


@dataclass(frozen=True)
class QueryResult:
    status: QueryStatus
    value: tuple[float, float] | int | None
    evidence_ids: tuple[int, ...]
    depth: int
    reason: str
    world_version: int


HopCallback = Callable[[int, RecursiveState], None]
