from __future__ import annotations

from dataclasses import dataclass

from cformer_v54.spatial import FrameRegistry

from .memory import CognitiveMemory
from .schema import (
    CognitiveStatus,
    Constraint,
    ConstraintOperator,
    Effect,
    EffectOperator,
    ObjectState,
    ObserverContext,
    Properties,
    QueryResult,
    SemanticObject,
    Transformation,
    Value,
)


@dataclass(frozen=True)
class SimulationResult:
    status: CognitiveStatus
    transformation: Transformation | None
    source_object_id: int | None
    target_object_id: int | None
    predicted_properties: Properties | None
    evidence_ids: tuple[int, ...]
    reason: str
    world_version: int


class CognitiveEngine:
    def __init__(
        self,
        memory: CognitiveMemory,
        frames: FrameRegistry,
        *,
        minimum_confidence: float = 0.5,
        enforce_constraints: bool = True,
        project_observer: bool = True,
        enforce_ingest_cutoff: bool = True,
        detect_identity_conflict: bool = True,
    ) -> None:
        self.memory = memory
        self.frames = frames
        self.minimum_confidence = minimum_confidence
        self.enforce_constraints = enforce_constraints
        self.project_observer = project_observer
        self.enforce_ingest_cutoff = enforce_ingest_cutoff
        self.detect_identity_conflict = detect_identity_conflict

    def _snapshot(self, observer: ObserverContext) -> int:
        return self.memory.version if observer.world_version is None else observer.world_version

    def resolve_object(
        self, name: str, observer: ObserverContext
    ) -> QueryResult | SemanticObject:
        version = self._snapshot(observer)
        candidates = self.memory.object_candidates(name, observer.query_time, version)
        if not candidates:
            return QueryResult(
                CognitiveStatus.UNKNOWN, None, None, (), "object_identity_not_found", version
            )
        visible = [
            obj
            for obj in candidates
            if obj.visibility_scope is None or observer.scope in obj.visibility_scope
        ]
        if not visible:
            return QueryResult(
                CognitiveStatus.ACCESS_DENIED,
                None,
                None,
                (),
                "object_identity_not_visible",
                version,
            )
        if len(visible) > 1 and self.detect_identity_conflict:
            return QueryResult(
                CognitiveStatus.CONFLICT,
                None,
                None,
                tuple(evidence for obj in visible for evidence in obj.evidence_ids),
                "alias_maps_to_multiple_active_objects",
                version,
            )
        return visible[0]

    def observe(self, name: str, observer: ObserverContext) -> QueryResult:
        version = self._snapshot(observer)
        resolved = self.resolve_object(name, observer)
        if isinstance(resolved, QueryResult):
            return resolved
        states = [
            state
            for state in self.memory.state_candidates(resolved.object_id, version)
            if (not self.enforce_ingest_cutoff or state.ingest_time <= observer.ingest_cutoff)
            and state.valid_from <= observer.query_time < state.valid_to
            and state.confidence >= self.minimum_confidence
        ]
        if not states:
            return QueryResult(
                CognitiveStatus.UNKNOWN,
                resolved.object_id,
                None,
                resolved.evidence_ids,
                "no_active_object_state",
                version,
            )
        visible = [
            state
            for state in states
            if state.visibility_scope is None or observer.scope in state.visibility_scope
        ]
        if not visible:
            return QueryResult(
                CognitiveStatus.ACCESS_DENIED,
                resolved.object_id,
                None,
                resolved.evidence_ids,
                "active_state_not_visible_to_observer",
                version,
            )
        superseded = {state.supersedes for state in visible if state.supersedes is not None}
        visible = [state for state in visible if state.state_id not in superseded]
        latest_key = max((state.valid_from, state.version) for state in visible)
        latest = [
            state for state in visible if (state.valid_from, state.version) == latest_key
        ]
        normalized: list[tuple[Properties, int]] = []
        for state in latest:
            values = dict(state.properties)
            position = values.get("position")
            if position is not None and self.project_observer:
                if not isinstance(position, tuple) or state.coordinate_frame is None:
                    return QueryResult(
                        CognitiveStatus.ALIGNMENT_MISSING,
                        resolved.object_id,
                        None,
                        (state.state_id,),
                        "position_missing_coordinate_frame",
                        version,
                    )
                global_position = self.frames.to_global(
                    state.coordinate_frame, position, observer.query_time
                )
                local_position = (
                    self.frames.from_global(
                        observer.frame, global_position, observer.query_time
                    )
                    if global_position is not None
                    else None
                )
                if local_position is None:
                    return QueryResult(
                        CognitiveStatus.ALIGNMENT_MISSING,
                        resolved.object_id,
                        None,
                        (state.state_id,),
                        "coordinate_transform_not_active",
                        version,
                    )
                values["position"] = (
                    round(local_position[0], 6),
                    round(local_position[1], 6),
                )
            normalized.append((tuple(sorted(values.items())), state.state_id))
        groups: dict[Properties, list[int]] = {}
        for properties, state_id in normalized:
            groups.setdefault(properties, []).append(state_id)
        evidence = resolved.evidence_ids + tuple(
            state_id for ids in groups.values() for state_id in ids
        )
        if len(groups) > 1:
            return QueryResult(
                CognitiveStatus.CONFLICT,
                resolved.object_id,
                None,
                evidence,
                "incompatible_active_object_states",
                version,
            )
        return QueryResult(
            CognitiveStatus.ANSWER,
            resolved.object_id,
            next(iter(groups)),
            evidence,
            "observer_projected_object_state",
            version,
        )

    @staticmethod
    def _constraint(constraint: Constraint, values: dict[str, Value]) -> bool | None:
        if constraint.field not in values:
            return None
        actual = values[constraint.field]
        expected = constraint.expected
        if constraint.operator == ConstraintOperator.EQ:
            return actual == expected
        if constraint.operator == ConstraintOperator.LE:
            return (
                isinstance(actual, (int, float))
                and isinstance(expected, (int, float))
                and actual <= expected
            )
        if constraint.operator == ConstraintOperator.GE:
            return (
                isinstance(actual, (int, float))
                and isinstance(expected, (int, float))
                and actual >= expected
            )
        if constraint.operator == ConstraintOperator.IN:
            return isinstance(expected, tuple) and actual in expected
        raise AssertionError("unknown constraint operator")

    @staticmethod
    def _effect(effect: Effect, values: dict[str, Value]) -> bool:
        if effect.operator == EffectOperator.SET:
            values[effect.field] = effect.value
            return True
        current = values.get(effect.field)
        if effect.operator == EffectOperator.ADD:
            if not isinstance(current, (int, float)) or not isinstance(
                effect.value, (int, float)
            ):
                return False
            values[effect.field] = current + effect.value
            return True
        if effect.operator == EffectOperator.ADD_VECTOR:
            if not isinstance(current, tuple) or not isinstance(effect.value, tuple):
                return False
            values[effect.field] = (
                current[0] + effect.value[0],
                current[1] + effect.value[1],
            )
            return True
        raise AssertionError("unknown effect operator")

    def simulate(
        self,
        name: str,
        operator: str,
        observer: ObserverContext,
        context: dict[str, Value] | None = None,
    ) -> SimulationResult:
        version = self._snapshot(observer)
        observed = self.observe(name, observer)
        if observed.status != CognitiveStatus.ANSWER:
            return SimulationResult(
                observed.status,
                None,
                observed.object_id,
                None,
                None,
                observed.evidence_ids,
                observed.reason,
                version,
            )
        transformations = [
            transformation
            for transformation in self.memory.transformation_candidates(
                observed.object_id, operator, version  # type: ignore[arg-type]
            )
            if (
                not self.enforce_ingest_cutoff
                or transformation.ingest_time <= observer.ingest_cutoff
            )
            and transformation.valid_from <= observer.query_time
            < transformation.valid_to
            and transformation.confidence >= self.minimum_confidence
        ]
        visible = [
            transformation
            for transformation in transformations
            if transformation.visibility_scope is None
            or observer.scope in transformation.visibility_scope
        ]
        if transformations and not visible:
            status, reason = (
                CognitiveStatus.ACCESS_DENIED,
                "transformation_not_visible_to_observer",
            )
        elif not visible:
            status, reason = CognitiveStatus.UNKNOWN, "transformation_not_found"
        elif len(visible) > 1:
            status, reason = CognitiveStatus.CONFLICT, "multiple_active_transformations"
        else:
            status, reason = CognitiveStatus.ANSWER, "transformation_candidate_found"
        if status != CognitiveStatus.ANSWER:
            return SimulationResult(
                status,
                None,
                observed.object_id,
                None,
                None,
                observed.evidence_ids,
                reason,
                version,
            )
        transformation = visible[0]
        values = dict(observed.value or ())
        values.update(context or {})
        checks = (
            [self._constraint(constraint, values) for constraint in transformation.constraints]
            if self.enforce_constraints
            else []
        )
        if any(check is None for check in checks):
            return SimulationResult(
                CognitiveStatus.CONSTRAINT_MISSING,
                transformation,
                observed.object_id,
                transformation.output_object_id,
                None,
                observed.evidence_ids + transformation.evidence_ids,
                "required_constraint_variable_missing",
                version,
            )
        if not all(checks):
            return SimulationResult(
                CognitiveStatus.CONSTRAINT_UNSATISFIED,
                transformation,
                observed.object_id,
                transformation.output_object_id,
                None,
                observed.evidence_ids + transformation.evidence_ids,
                "transformation_constraints_not_satisfied",
                version,
            )
        for effect in transformation.effects:
            if not self._effect(effect, values):
                return SimulationResult(
                    CognitiveStatus.UNKNOWN,
                    transformation,
                    observed.object_id,
                    transformation.output_object_id,
                    None,
                    observed.evidence_ids + transformation.evidence_ids,
                    "effect_input_state_missing_or_invalid",
                    version,
                )
        predicted = tuple(
            sorted(
                (field, value)
                for field, value in values.items()
                if field not in (context or {}) or field in {effect.field for effect in transformation.effects}
            )
        )
        return SimulationResult(
            CognitiveStatus.ANSWER,
            transformation,
            observed.object_id,
            transformation.output_object_id,
            predicted,
            observed.evidence_ids + transformation.evidence_ids,
            "simulation_candidate_requires_verification",
            version,
        )
