from __future__ import annotations

from dataclasses import dataclass

from cformer_v54.spatial import FrameRegistry, FrameTransform

from .memory import CognitiveMemory
from .schema import (
    CognitiveStatus,
    Constraint,
    ConstraintOperator,
    Effect,
    EffectOperator,
    ObjectLifecycle,
    ObjectState,
    ObserverContext,
    Properties,
    SemanticObject,
    Transformation,
    Value,
)


@dataclass(frozen=True)
class CognitiveCase:
    category: str
    action: str
    name: str
    observer: ObserverContext
    expected_status: CognitiveStatus
    expected_properties: Properties = ()
    expected_object_id: int | None = None
    operator: str | None = None
    context: tuple[tuple[str, Value], ...] = ()


@dataclass(frozen=True)
class CognitiveWorld:
    name: str
    memory: CognitiveMemory
    frames: FrameRegistry
    cases: tuple[CognitiveCase, ...]


def _motion_world() -> CognitiveWorld:
    memory = CognitiveMemory()
    frames = FrameRegistry()
    frames.add(FrameTransform(1, 0, 100, translate_x=10.0))
    frames.add(FrameTransform(2, 0, 100, translate_y=5.0))
    memory.add_objects(
        [SemanticObject(1, "Rover", frozenset({"探测车"}), "vehicle", evidence_ids=(9001,))]
    )
    memory.add_states(
        [
            ObjectState(
                1,
                1,
                (("mode", "survey"), ("position", (2.0, 3.0))),
                valid_from=0,
                valid_to=50,
                coordinate_frame=1,
            ),
            ObjectState(
                2,
                1,
                (("mode", "survey"), ("position", (4.0, 4.0))),
                event_time=50,
                ingest_time=50,
                valid_from=50,
                valid_to=100,
                version=2,
                supersedes=1,
                coordinate_frame=1,
            ),
        ]
    )
    memory.add_transformations(
        [
            Transformation(
                1,
                "move",
                (1,),
                1,
                (Constraint("authorized", ConstraintOperator.EQ, True),),
                (Effect("position", EffectOperator.ADD_VECTOR, (3.0, -1.0)),),
                evidence_ids=(9101,),
            )
        ]
    )
    return CognitiveWorld(
        "motion_and_coordinates",
        memory,
        frames,
        (
            CognitiveCase(
                "historical_state",
                "observe",
                "探测车",
                ObserverContext(1, 2, 25, 25),
                CognitiveStatus.ANSWER,
                (("position", (12.0, -2.0)),),
                1,
            ),
            CognitiveCase(
                "observer_projection",
                "observe",
                "Rover",
                ObserverContext(1, 2, 75, 75),
                CognitiveStatus.ANSWER,
                (("position", (14.0, -1.0)),),
                1,
            ),
            CognitiveCase(
                "transformation_valid",
                "simulate",
                "Rover",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.ANSWER,
                (("position", (17.0, 3.0)),),
                1,
                "move",
                (("authorized", True),),
            ),
            CognitiveCase(
                "constraint_unsatisfied",
                "simulate",
                "Rover",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.CONSTRAINT_UNSATISFIED,
                expected_object_id=1,
                operator="move",
                context=(("authorized", False),),
            ),
            CognitiveCase(
                "constraint_missing",
                "simulate",
                "Rover",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.CONSTRAINT_MISSING,
                expected_object_id=1,
                operator="move",
            ),
        ),
    )


def _phase_world() -> CognitiveWorld:
    memory = CognitiveMemory()
    frames = FrameRegistry()
    memory.add_objects(
        [
            SemanticObject(10, "Water", frozenset({"水", "H2O"}), "substance"),
            SemanticObject(11, "Ice", frozenset({"冰"}), "substance"),
        ]
    )
    memory.add_states(
        [ObjectState(10, 10, (("mass", 1.0), ("phase", "liquid")), valid_to=100)]
    )
    memory.add_transformations(
        [
            Transformation(
                10,
                "freeze",
                (10,),
                11,
                (
                    Constraint("temperature", ConstraintOperator.LE, 0.0),
                    Constraint("pressure", ConstraintOperator.EQ, "normal"),
                ),
                (
                    Effect("phase", EffectOperator.SET, "solid"),
                    Effect("temperature", EffectOperator.SET, 0.0),
                ),
                evidence_ids=(9201, 9202),
            )
        ]
    )
    observer = ObserverContext(1, 0, 20, 20)
    return CognitiveWorld(
        "phase_change",
        memory,
        frames,
        (
            CognitiveCase(
                "cross_alias_identity",
                "observe",
                "H2O",
                observer,
                CognitiveStatus.ANSWER,
                (("phase", "liquid"),),
                10,
            ),
            CognitiveCase(
                "transformation_valid",
                "simulate",
                "水",
                observer,
                CognitiveStatus.ANSWER,
                (("phase", "solid"), ("temperature", 0.0)),
                11,
                "freeze",
                (("temperature", -5.0), ("pressure", "normal")),
            ),
            CognitiveCase(
                "constraint_unsatisfied",
                "simulate",
                "Water",
                observer,
                CognitiveStatus.CONSTRAINT_UNSATISFIED,
                expected_object_id=11,
                operator="freeze",
                context=(("temperature", 5.0), ("pressure", "normal")),
            ),
            CognitiveCase(
                "constraint_missing",
                "simulate",
                "Water",
                observer,
                CognitiveStatus.CONSTRAINT_MISSING,
                expected_object_id=11,
                operator="freeze",
                context=(("temperature", -5.0),),
            ),
        ),
    )


def _organization_world() -> CognitiveWorld:
    memory = CognitiveMemory()
    frames = FrameRegistry()
    memory.add_objects(
        [SemanticObject(20, "Alex", frozenset({"员工A"}), "person")]
    )
    memory.add_states(
        [
            ObjectState(
                20,
                20,
                (("role", "engineer"),),
                valid_from=0,
                valid_to=50,
                visibility_scope=frozenset({1, 2}),
            ),
            ObjectState(
                21,
                20,
                (("role", "manager"),),
                event_time=50,
                ingest_time=80,
                valid_from=50,
                valid_to=100,
                version=2,
                supersedes=20,
                visibility_scope=frozenset({1}),
            ),
        ]
    )
    return CognitiveWorld(
        "organization_and_permissions",
        memory,
        frames,
        (
            CognitiveCase(
                "historical_state",
                "observe",
                "员工A",
                ObserverContext(2, 0, 25, 25),
                CognitiveStatus.ANSWER,
                (("role", "engineer"),),
                20,
            ),
            CognitiveCase(
                "delayed_ingest_hidden",
                "observe",
                "Alex",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.UNKNOWN,
                expected_object_id=20,
            ),
            CognitiveCase(
                "delayed_ingest_visible",
                "observe",
                "Alex",
                ObserverContext(1, 0, 75, 90),
                CognitiveStatus.ANSWER,
                (("role", "manager"),),
                20,
            ),
            CognitiveCase(
                "observer_access_denied",
                "observe",
                "Alex",
                ObserverContext(2, 0, 75, 90),
                CognitiveStatus.ACCESS_DENIED,
                expected_object_id=20,
            ),
        ),
    )


def _concept_world() -> CognitiveWorld:
    memory = CognitiveMemory()
    frames = FrameRegistry()
    memory.add_objects(
        [
            SemanticObject(
                30,
                "Artificial Intelligence",
                frozenset({"AI"}),
                "concept",
                valid_from=0,
                valid_to=50,
                lifecycle=ObjectLifecycle.SPLIT,
                successor_ids=(31, 32),
            ),
            SemanticObject(
                31,
                "Symbolic AI",
                frozenset({"AI"}),
                "concept",
                valid_from=50,
                predecessor_ids=(30,),
            ),
            SemanticObject(
                32,
                "Generative AI",
                frozenset({"AI", "生成式人工智能"}),
                "concept",
                valid_from=50,
                predecessor_ids=(30,),
            ),
        ]
    )
    memory.add_states(
        [
            ObjectState(30, 30, (("scope", "general"),), valid_from=0, valid_to=50),
            ObjectState(31, 31, (("paradigm", "symbolic"),), valid_from=50),
            ObjectState(32, 32, (("paradigm", "generative"),), valid_from=50),
        ]
    )
    return CognitiveWorld(
        "concept_lifecycle",
        memory,
        frames,
        (
            CognitiveCase(
                "identity_before_split",
                "observe",
                "AI",
                ObserverContext(1, 0, 25, 25),
                CognitiveStatus.ANSWER,
                (("scope", "general"),),
                30,
            ),
            CognitiveCase(
                "identity_split_conflict",
                "observe",
                "AI",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.CONFLICT,
            ),
            CognitiveCase(
                "canonical_disambiguation",
                "observe",
                "Generative AI",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.ANSWER,
                (("paradigm", "generative"),),
                32,
            ),
            CognitiveCase(
                "unknown_object",
                "observe",
                "Nonexistent Concept",
                ObserverContext(1, 0, 75, 75),
                CognitiveStatus.UNKNOWN,
            ),
        ),
    )


def build_cognitive_worlds() -> tuple[CognitiveWorld, ...]:
    return (_motion_world(), _phase_world(), _organization_world(), _concept_world())
