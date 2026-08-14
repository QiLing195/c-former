from __future__ import annotations

from dataclasses import dataclass

from .memory import VersionedMemory
from .schema import Fact, QueryStatus
from .spatial import FrameRegistry, FrameTransform


V54_SCALES = (2048, 8192, 32768)


@dataclass(frozen=True)
class QueryCase:
    category: str
    entity: int
    observer_scope: int
    observer_frame: int
    query_time: int
    ingest_cutoff: int
    max_depth: int
    expected_status: QueryStatus
    expected_value: tuple[float, float] | None


@dataclass(frozen=True)
class BuiltWorld:
    name: str
    memory: VersionedMemory
    frames: FrameRegistry
    cases: tuple[QueryCase, ...]
    independent_facts: int


def build_world(scale: int, world_index: int) -> BuiltWorld:
    if scale not in V54_SCALES:
        raise ValueError(f"unsupported V5.4 scale {scale}")
    frames = FrameRegistry()
    frames.add(FrameTransform(1, 0, 50, world_index % 4, 10.0 + world_index, -3.0))
    frames.add(
        FrameTransform(1, 50, 100, (world_index + 1) % 4, 12.0 + world_index, -2.0)
    )
    frames.add(
        FrameTransform(2, 0, 50, (world_index + 1) % 4, -4.0, 5.0 + world_index)
    )
    frames.add(
        FrameTransform(2, 50, 100, (world_index + 2) % 4, -2.0, 7.0 + world_index)
    )
    facts: list[Fact] = []
    cases: list[QueryCase] = []
    next_fact_id = world_index * 100_000 + 1
    entity_base = world_index * 1_000_000

    def add(**kwargs) -> int:
        nonlocal next_fact_id
        fact_id = next_fact_id
        next_fact_id += 1
        facts.append(Fact(fact_id=fact_id, **kwargs))
        return fact_id

    def expected(point: tuple[float, float], source_frame: int, time: int) -> tuple[float, float]:
        global_point = frames.to_global(source_frame, point, time)
        assert global_point is not None
        local = frames.from_global(2, global_point, time)
        assert local is not None
        return round(local[0], 6), round(local[1], 6)

    # Recursive chains with historical and current versions.
    for index in range(8):
        chain_length = 1 + index % 3
        leaf = entity_base + 100 + index * 10
        current = leaf
        for hop in range(chain_length):
            parent = leaf + hop + 1
            add(subject=current, relation="parent", object_id=parent, valid_from=0, valid_to=100)
            current = parent
        old_position = (float(index + 1), float(2 * index + 1))
        new_position = (old_position[0] + 1.0, old_position[1] + 1.0)
        old_id = add(
            subject=current,
            relation="position",
            position=old_position,
            coordinate_frame=1,
            event_time=0,
            ingest_time=0,
            valid_from=0,
            valid_to=50,
            version=1,
        )
        add(
            subject=current,
            relation="position",
            position=new_position,
            coordinate_frame=1,
            event_time=50,
            ingest_time=50,
            valid_from=50,
            valid_to=100,
            version=2,
            supersedes=old_id,
        )
        cases.append(
            QueryCase(
                "recursive_current",
                leaf,
                1,
                2,
                75,
                75,
                4,
                QueryStatus.ANSWER,
                expected(new_position, 1, 75),
            )
        )
        cases.append(
            QueryCase(
                "historical",
                leaf,
                1,
                2,
                25,
                25,
                4,
                QueryStatus.ANSWER,
                expected(old_position, 1, 25),
            )
        )

    for index in range(4):
        entity = entity_base + 1_000 + index
        point = (float(index), float(index + 2))
        add(
            subject=entity,
            relation="position",
            position=point,
            coordinate_frame=1,
            visibility_scope=frozenset({1}),
            valid_from=0,
            valid_to=100,
        )
        cases.append(QueryCase("access_denied", entity, 2, 2, 75, 75, 4, QueryStatus.ACCESS_DENIED, None))

    for index in range(4):
        entity = entity_base + 2_000 + index
        add(subject=entity, relation="position", position=(1.0, float(index)), coordinate_frame=1, valid_from=0, valid_to=100)
        add(subject=entity, relation="position", position=(2.0, float(index)), coordinate_frame=1, valid_from=0, valid_to=100)
        cases.append(QueryCase("conflict", entity, 1, 2, 75, 75, 4, QueryStatus.CONFLICT, None))

    for index in range(4):
        first = entity_base + 3_000 + index * 2
        second = first + 1
        add(subject=first, relation="parent", object_id=second, valid_from=0, valid_to=100)
        add(subject=second, relation="parent", object_id=first, valid_from=0, valid_to=100)
        cases.append(QueryCase("cycle", first, 1, 2, 75, 75, 4, QueryStatus.CYCLE, None))

    for index in range(4):
        leaf = entity_base + 4_000 + index * 10
        current = leaf
        for hop in range(6):
            parent = leaf + hop + 1
            add(subject=current, relation="parent", object_id=parent, valid_from=0, valid_to=100)
            current = parent
        add(subject=current, relation="position", position=(3.0, float(index)), coordinate_frame=1, valid_from=0, valid_to=100)
        cases.append(QueryCase("depth_limit", leaf, 1, 2, 75, 75, 4, QueryStatus.DEPTH_LIMIT, None))

    for index in range(4):
        entity = entity_base + 5_000 + index
        add(subject=entity, relation="position", position=(1.0, 1.0), coordinate_frame=99, valid_from=0, valid_to=100)
        cases.append(QueryCase("alignment_missing", entity, 1, 2, 75, 75, 4, QueryStatus.ALIGNMENT_MISSING, None))

    for index in range(4):
        entity = entity_base + 6_000 + index
        cases.append(QueryCase("unknown", entity, 1, 2, 75, 75, 4, QueryStatus.UNKNOWN, None))

    for index in range(4):
        entity = entity_base + 7_000 + index
        point = (float(index + 4), float(index + 5))
        add(
            subject=entity,
            relation="position",
            position=point,
            coordinate_frame=1,
            event_time=20,
            ingest_time=90,
            valid_from=0,
            valid_to=100,
        )
        cases.append(QueryCase("delayed_hidden", entity, 1, 2, 75, 75, 4, QueryStatus.UNKNOWN, None))
        cases.append(QueryCase("delayed_visible", entity, 1, 2, 75, 100, 4, QueryStatus.ANSWER, expected(point, 1, 75)))

    # Two differently framed facts that normalize to the same global claim.
    for index in range(4):
        entity = entity_base + 8_000 + index
        point_frame_one = (float(index + 2), float(index + 6))
        global_point = frames.to_global(1, point_frame_one, 75)
        assert global_point is not None
        point_frame_two = frames.from_global(2, global_point, 75)
        assert point_frame_two is not None
        add(subject=entity, relation="position", position=point_frame_one, coordinate_frame=1, valid_from=0, valid_to=100)
        add(subject=entity, relation="position", position=point_frame_two, coordinate_frame=2, valid_from=0, valid_to=100)
        cases.append(QueryCase("equivalent_evidence", entity, 1, 2, 75, 75, 4, QueryStatus.ANSWER, expected(point_frame_one, 1, 75)))

    # A low-confidence incompatible source must not create a conflict.
    for index in range(4):
        entity = entity_base + 9_000 + index
        trusted = (float(index + 3), float(index + 7))
        add(subject=entity, relation="position", position=trusted, coordinate_frame=1, confidence=1.0, valid_from=0, valid_to=100)
        add(subject=entity, relation="position", position=(99.0, 99.0), coordinate_frame=1, confidence=0.2, valid_from=0, valid_to=100)
        cases.append(QueryCase("low_confidence_noise", entity, 1, 2, 75, 75, 4, QueryStatus.ANSWER, expected(trusted, 1, 75)))

    if len(facts) > scale:
        raise ValueError("core V5.4 facts exceed requested scale")
    # Every filler record has a unique subject/object pair, so these are independent
    # facts rather than repeated stress copies.
    filler_count = scale - len(facts)
    for index in range(filler_count):
        add(
            subject=entity_base + 100_000 + index,
            relation="attribute",
            object_id=entity_base + 500_000 + index,
            event_time=index % 100,
            ingest_time=index % 100,
            valid_from=0,
            valid_to=100,
        )
    memory = VersionedMemory()
    memory.add_many(facts)
    assert memory.fact_count == scale
    return BuiltWorld(
        name=f"V54-S{scale}-W{world_index + 1}",
        memory=memory,
        frames=frames,
        cases=tuple(cases),
        independent_facts=scale,
    )
