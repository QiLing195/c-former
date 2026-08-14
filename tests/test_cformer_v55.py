import pytest

from cformer_v55 import (
    CognitiveEngine,
    CognitiveStatus,
    HypothesisSandbox,
    HypothesisStatus,
    ObjectState,
    ObserverContext,
    build_cognitive_worlds,
)


def _world(name: str):
    return next(world for world in build_cognitive_worlds() if world.name == name)


def test_all_cognitive_world_cases_match_expected_boundaries() -> None:
    for world in build_cognitive_worlds():
        engine = CognitiveEngine(world.memory, world.frames)
        for case in world.cases:
            if case.action == "observe":
                result = engine.observe(case.name, case.observer)
                properties = result.value
                object_id = result.object_id
            else:
                result = engine.simulate(
                    case.name, case.operator or "", case.observer, dict(case.context)
                )
                properties = result.predicted_properties
                object_id = result.target_object_id
            assert result.status == case.expected_status, (world.name, case.category)
            if case.expected_object_id is not None:
                assert object_id == case.expected_object_id
            actual = dict(properties or ())
            for field, value in case.expected_properties:
                assert actual.get(field) == value


def test_observer_projection_preserves_canonical_world_state() -> None:
    world = _world("motion_and_coordinates")
    engine = CognitiveEngine(world.memory, world.frames)
    frame_zero = engine.observe("Rover", ObserverContext(1, 0, 75, 75))
    frame_two = engine.observe("Rover", ObserverContext(1, 2, 75, 75))
    assert frame_zero.get("position") == (14.0, 4.0)
    assert frame_two.get("position") == (14.0, -1.0)
    assert frame_zero.object_id == frame_two.object_id == 1


def test_alias_split_is_conflict_not_arbitrary_identity_merge() -> None:
    world = _world("concept_lifecycle")
    engine = CognitiveEngine(world.memory, world.frames)
    before = engine.observe("AI", ObserverContext(1, 0, 25, 25))
    after = engine.observe("AI", ObserverContext(1, 0, 75, 75))
    explicit = engine.observe("Generative AI", ObserverContext(1, 0, 75, 75))
    assert before.object_id == 30
    assert after.status == CognitiveStatus.CONFLICT
    assert explicit.object_id == 32


def test_constraint_engine_uses_three_valued_boundary() -> None:
    world = _world("phase_change")
    engine = CognitiveEngine(world.memory, world.frames)
    observer = ObserverContext(1, 0, 20, 20)
    valid = engine.simulate(
        "Water", "freeze", observer, {"temperature": -5.0, "pressure": "normal"}
    )
    invalid = engine.simulate(
        "Water", "freeze", observer, {"temperature": 5.0, "pressure": "normal"}
    )
    missing = engine.simulate("Water", "freeze", observer, {"temperature": -5.0})
    assert valid.status == CognitiveStatus.ANSWER
    assert invalid.status == CognitiveStatus.CONSTRAINT_UNSATISFIED
    assert missing.status == CognitiveStatus.CONSTRAINT_MISSING


def test_unverified_simulation_cannot_write_memory() -> None:
    world = _world("phase_change")
    engine = CognitiveEngine(world.memory, world.frames)
    sandbox = HypothesisSandbox(engine)
    before = (world.memory.version, world.memory.state_count)
    hypothesis = sandbox.propose(
        "Water",
        "freeze",
        ObserverContext(1, 0, 20, 20),
        {"temperature": -5.0, "pressure": "normal"},
    )
    assert hypothesis.status == HypothesisStatus.PROPOSED
    assert (world.memory.version, world.memory.state_count) == before
    with pytest.raises(PermissionError):
        sandbox.commit(hypothesis.hypothesis_id, event_time=30)
    observed = ObjectState(
        999,
        11,
        hypothesis.predicted_properties,
        event_time=30,
        ingest_time=30,
        valid_from=30,
    )
    verified = sandbox.verify(hypothesis.hypothesis_id, observed)
    assert verified.status == HypothesisStatus.VERIFIED
    committed = sandbox.commit(hypothesis.hypothesis_id, event_time=30)
    assert committed.object_id == 11
    assert world.memory.state_count == before[1] + 1
    assert sandbox.get(hypothesis.hypothesis_id).status == HypothesisStatus.COMMITTED


def test_world_change_blocks_stale_hypothesis_commit() -> None:
    world = _world("motion_and_coordinates")
    engine = CognitiveEngine(world.memory, world.frames)
    sandbox = HypothesisSandbox(engine)
    hypothesis = sandbox.propose(
        "Rover",
        "move",
        ObserverContext(1, 0, 75, 75),
        {"authorized": True},
    )
    observed = ObjectState(
        998,
        1,
        hypothesis.predicted_properties,
        event_time=80,
        ingest_time=80,
        valid_from=80,
        coordinate_frame=0,
    )
    sandbox.verify(hypothesis.hypothesis_id, observed)
    world.memory.add_states(
        [ObjectState(997, 1, (("mode", "safe"),), valid_from=90)]
    )
    with pytest.raises(RuntimeError):
        sandbox.commit(hypothesis.hypothesis_id, event_time=80)
    assert sandbox.get(hypothesis.hypothesis_id).status == HypothesisStatus.BLOCKED
