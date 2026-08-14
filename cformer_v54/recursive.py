from __future__ import annotations

from dataclasses import dataclass

from .memory import VersionedMemory
from .schema import Fact, HopCallback, QueryResult, QueryStatus, RecursiveState
from .spatial import FrameRegistry


@dataclass(frozen=True)
class _ClaimResolution:
    status: QueryStatus
    value: tuple[float, float] | int | None
    evidence_ids: tuple[int, ...]
    reason: str


class RecursiveQueryEngine:
    def __init__(
        self,
        memory: VersionedMemory,
        frames: FrameRegistry,
        *,
        indexed: bool = True,
        minimum_confidence: float = 0.5,
    ) -> None:
        self.memory = memory
        self.frames = frames
        self.indexed = indexed
        self.minimum_confidence = minimum_confidence

    def _active_visible(
        self,
        subject: int,
        relation: str,
        state: RecursiveState,
    ) -> _ClaimResolution | list[Fact]:
        candidates = self.memory.candidates(
            subject, relation, state.world_version, indexed=self.indexed
        )
        temporally_active = [
            fact
            for fact in candidates
            if fact.ingest_time <= state.ingest_cutoff
            and fact.valid_from <= state.query_time < fact.valid_to
            and fact.confidence >= self.minimum_confidence
        ]
        if not temporally_active:
            return _ClaimResolution(QueryStatus.UNKNOWN, None, (), "no_active_fact")
        visible = [
            fact
            for fact in temporally_active
            if fact.visibility_scope is None or state.observer_scope in fact.visibility_scope
        ]
        if not visible:
            return _ClaimResolution(
                QueryStatus.ACCESS_DENIED,
                None,
                (),
                "active_fact_not_visible_to_observer",
            )
        superseded = {fact.supersedes for fact in visible if fact.supersedes is not None}
        return [fact for fact in visible if fact.fact_id not in superseded]

    def _resolve(self, subject: int, relation: str, state: RecursiveState) -> _ClaimResolution:
        active = self._active_visible(subject, relation, state)
        if isinstance(active, _ClaimResolution):
            return active
        groups: dict[tuple, list[int]] = {}
        for fact in active:
            if relation == "position":
                if fact.position is None or fact.coordinate_frame is None:
                    return _ClaimResolution(
                        QueryStatus.ALIGNMENT_MISSING,
                        None,
                        (fact.fact_id,),
                        "position_missing_coordinate_frame",
                    )
                global_position = self.frames.to_global(
                    fact.coordinate_frame, fact.position, state.query_time
                )
                if global_position is None:
                    return _ClaimResolution(
                        QueryStatus.ALIGNMENT_MISSING,
                        None,
                        (fact.fact_id,),
                        "coordinate_transform_not_active",
                    )
                key = (round(global_position[0], 6), round(global_position[1], 6))
            else:
                if fact.object_id is None:
                    return _ClaimResolution(
                        QueryStatus.UNKNOWN, None, (fact.fact_id,), "relation_missing_object"
                    )
                key = (fact.object_id,)
            groups.setdefault(key, []).append(fact.fact_id)
        evidence = tuple(fact_id for ids in groups.values() for fact_id in ids)
        if len(groups) > 1:
            return _ClaimResolution(
                QueryStatus.CONFLICT, None, evidence, "overlapping_incompatible_claims"
            )
        key = next(iter(groups))
        value = (key[0], key[1]) if relation == "position" else key[0]
        return _ClaimResolution(QueryStatus.ANSWER, value, evidence, "single_claim_group")

    def query_position(
        self,
        entity: int,
        *,
        observer_scope: int,
        observer_frame: int,
        query_time: int,
        ingest_cutoff: int | None = None,
        world_version: int | None = None,
        max_depth: int = 4,
        on_hop: HopCallback | None = None,
    ) -> QueryResult:
        use_current_version = world_version is None
        snapshot = self.memory.version if world_version is None else world_version
        cutoff = query_time if ingest_cutoff is None else ingest_cutoff
        current = entity
        visited: list[int] = []
        evidence: list[int] = []
        for depth in range(max_depth + 1):
            state = RecursiveState(
                current,
                observer_scope,
                observer_frame,
                query_time,
                cutoff,
                snapshot,
                tuple(visited),
                tuple(evidence),
                depth,
            )
            if on_hop is not None:
                on_hop(depth, state)
            if use_current_version and self.memory.version != snapshot:
                return QueryResult(
                    QueryStatus.VERSION_CHANGED,
                    None,
                    tuple(evidence),
                    depth,
                    "world_changed_during_recursive_query",
                    snapshot,
                )
            if current in visited:
                return QueryResult(
                    QueryStatus.CYCLE,
                    None,
                    tuple(evidence),
                    depth,
                    "entity_cycle_detected",
                    snapshot,
                )
            visited.append(current)
            position = self._resolve(current, "position", state)
            if position.status == QueryStatus.ANSWER:
                evidence.extend(position.evidence_ids)
                local = self.frames.from_global(
                    observer_frame, position.value, query_time  # type: ignore[arg-type]
                )
                if local is None:
                    return QueryResult(
                        QueryStatus.ALIGNMENT_MISSING,
                        None,
                        tuple(evidence),
                        depth,
                        "observer_coordinate_transform_not_active",
                        snapshot,
                    )
                return QueryResult(
                    QueryStatus.ANSWER,
                    (round(local[0], 6), round(local[1], 6)),
                    tuple(evidence),
                    depth,
                    "position_resolved",
                    snapshot,
                )
            if position.status != QueryStatus.UNKNOWN:
                return QueryResult(
                    position.status,
                    None,
                    tuple(evidence) + position.evidence_ids,
                    depth,
                    position.reason,
                    snapshot,
                )
            if depth == max_depth:
                return QueryResult(
                    QueryStatus.DEPTH_LIMIT,
                    None,
                    tuple(evidence),
                    depth,
                    "maximum_recursion_depth_reached",
                    snapshot,
                )
            parent = self._resolve(current, "parent", state)
            if parent.status != QueryStatus.ANSWER:
                return QueryResult(
                    parent.status,
                    None,
                    tuple(evidence) + parent.evidence_ids,
                    depth,
                    parent.reason,
                    snapshot,
                )
            evidence.extend(parent.evidence_ids)
            current = int(parent.value)  # type: ignore[arg-type]
        raise AssertionError("unreachable")
