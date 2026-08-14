from cformer_v54 import (
    Fact,
    FrameRegistry,
    FrameTransform,
    QueryStatus,
    RecursiveQueryEngine,
    VersionedMemory,
    build_world,
)


def test_v54_world_contains_requested_independent_facts() -> None:
    world = build_world(2048, 0)
    assert world.memory.fact_count == 2048
    assert world.memory.index_entries == 2048
    assert len(world.cases) == 56


def test_indexed_and_linear_engines_match_all_categories() -> None:
    world = build_world(2048, 1)
    indexed = RecursiveQueryEngine(world.memory, world.frames, indexed=True)
    linear = RecursiveQueryEngine(world.memory, world.frames, indexed=False)
    for case in world.cases:
        kwargs = dict(
            observer_scope=case.observer_scope,
            observer_frame=case.observer_frame,
            query_time=case.query_time,
            ingest_cutoff=case.ingest_cutoff,
            max_depth=case.max_depth,
        )
        indexed_result = indexed.query_position(case.entity, **kwargs)
        assert indexed_result == linear.query_position(case.entity, **kwargs)
        assert indexed_result.status == case.expected_status
        if case.expected_value is not None:
            assert indexed_result.value == case.expected_value


def test_historical_snapshot_survives_retraction() -> None:
    memory = VersionedMemory()
    frames = FrameRegistry()
    fact = Fact(1, 10, "position", position=(2.0, 3.0), coordinate_frame=0)
    version_one = memory.add_many([fact])
    memory.retract(1)
    engine = RecursiveQueryEngine(memory, frames)
    historical = engine.query_position(
        10,
        observer_scope=1,
        observer_frame=0,
        query_time=1,
        world_version=version_one,
    )
    current = engine.query_position(10, observer_scope=1, observer_frame=0, query_time=1)
    assert historical.status == QueryStatus.ANSWER
    assert current.status == QueryStatus.UNKNOWN


def test_world_change_interrupts_recursive_query() -> None:
    world = build_world(2048, 2)
    engine = RecursiveQueryEngine(world.memory, world.frames)
    case = next(case for case in world.cases if case.category == "recursive_current")
    inserted = False

    def mutate_on_second_hop(depth, _state):
        nonlocal inserted
        if depth == 1 and not inserted:
            inserted = True
            world.memory.add_many([Fact(999_999_999, 999, "attribute", object_id=1)])

    result = engine.query_position(
        case.entity,
        observer_scope=case.observer_scope,
        observer_frame=case.observer_frame,
        query_time=case.query_time,
        ingest_cutoff=case.ingest_cutoff,
        on_hop=mutate_on_second_hop,
    )
    assert result.status == QueryStatus.VERSION_CHANGED


def test_time_dependent_frame_alignment() -> None:
    frames = FrameRegistry()
    frames.add(FrameTransform(7, 0, 50, quarter_turns=0, translate_x=1.0))
    frames.add(FrameTransform(7, 50, 100, quarter_turns=1, translate_x=1.0))
    assert frames.to_global(7, (2.0, 0.0), 25) == (3.0, 0.0)
    assert frames.to_global(7, (2.0, 0.0), 75) == (1.0, 2.0)
