from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .text import HashedTextEncoder


class RouteStatus(str, Enum):
    ANSWER = "answer"
    UNKNOWN = "unknown"
    ACCESS_DENIED = "access_denied"
    CONFLICT = "conflict"
    CYCLE = "cycle"
    DEPTH_LIMIT = "depth_limit"
    VERSION_CHANGED = "version_changed"


@dataclass(frozen=True)
class TextTransition:
    transition_id: int
    source_object: int
    target_object: int
    operator: str
    description: str
    valid_from: int = 0
    valid_to: int = 2**31 - 1
    ingest_time: int = 0
    visibility_scope: frozenset[int] | None = None


@dataclass(frozen=True)
class RouteResult:
    status: RouteStatus
    object_id: int | None
    evidence_ids: tuple[int, ...]
    depth: int
    reason: str


class ControlledTransformationEngine:
    def __init__(self, transitions: tuple[TextTransition, ...]) -> None:
        self.transitions = transitions
        self.version = 1
        self.encoder = HashedTextEncoder(128)

    def touch_version(self) -> None:
        self.version += 1

    def route(
        self,
        start_object: int,
        operator_queries: tuple[str, ...],
        *,
        observer_scope: int,
        query_time: int,
        ingest_cutoff: int,
        max_depth: int = 4,
        on_hop: Callable[[int], None] | None = None,
    ) -> RouteResult:
        snapshot = self.version
        current = start_object
        visited: list[int] = []
        evidence: list[int] = []
        for depth, operator_query in enumerate(operator_queries):
            if depth >= max_depth:
                return RouteResult(RouteStatus.DEPTH_LIMIT, None, tuple(evidence), depth, "maximum_depth")
            if on_hop is not None:
                on_hop(depth)
            if self.version != snapshot:
                return RouteResult(RouteStatus.VERSION_CHANGED, None, tuple(evidence), depth, "world_changed")
            if current in visited:
                return RouteResult(RouteStatus.CYCLE, None, tuple(evidence), depth, "cycle_detected")
            visited.append(current)
            active = [
                edge
                for edge in self.transitions
                if edge.source_object == current
                and edge.ingest_time <= ingest_cutoff
                and edge.valid_from <= query_time < edge.valid_to
            ]
            if not active:
                return RouteResult(RouteStatus.UNKNOWN, None, tuple(evidence), depth, "no_transition")
            visible = [
                edge
                for edge in active
                if edge.visibility_scope is None or observer_scope in edge.visibility_scope
            ]
            if not visible:
                return RouteResult(RouteStatus.ACCESS_DENIED, None, tuple(evidence), depth, "transition_hidden")
            query = self.encoder.encode(operator_query)
            scores = [float(query @ self.encoder.encode(edge.description)) for edge in visible]
            best = max(scores)
            selected = [edge for edge, score in zip(visible, scores) if abs(score - best) < 1e-8]
            if len(selected) != 1:
                return RouteResult(RouteStatus.CONFLICT, None, tuple(evidence), depth, "ambiguous_transition")
            edge = selected[0]
            evidence.append(edge.transition_id)
            current = edge.target_object
        if current in visited:
            return RouteResult(RouteStatus.CYCLE, None, tuple(evidence), len(operator_queries), "cycle_detected")
        return RouteResult(RouteStatus.ANSWER, current, tuple(evidence), len(operator_queries), "route_resolved")
