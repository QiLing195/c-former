import torch

from cformer_v4.data import STATUS_ANSWER, STATUS_CONFLICT
from cformer_v5 import ConflictResolver, V5WorldSuite


def test_v5_generates_120_cases_per_world() -> None:
    suite = V5WorldSuite()
    for scale in (128, 512, 2048):
        cases = suite.cases(scale, 0)
        assert len(cases) == 120
        assert all(case.memory.shape == (scale, 4) for case in cases)


def test_conflict_resolver_separates_conflict_from_time_and_version() -> None:
    suite = V5WorldSuite()
    resolver = ConflictResolver()
    cases = suite.cases(128, 0)
    conflict = next(case for case in cases if case.category == "true_conflict")
    time_change = next(case for case in cases if case.category == "time_change")
    version = next(case for case in cases if case.category == "version_update")
    assert resolver.resolve(
        conflict.memory, conflict.metadata, conflict.question, conflict.query_time
    ).status == STATUS_CONFLICT
    assert resolver.resolve(
        time_change.memory, time_change.metadata, time_change.question, time_change.query_time
    ).status == STATUS_ANSWER
    assert resolver.resolve(version.memory, version.metadata, version.question, version.query_time).status == STATUS_ANSWER


def test_low_confidence_source_does_not_create_conflict() -> None:
    suite = V5WorldSuite()
    resolver = ConflictResolver()
    case = next(case for case in suite.cases(128, 0) if case.category == "low_confidence_source")
    resolution = resolver.resolve(case.memory, case.metadata, case.question, case.query_time)
    assert resolution.status == STATUS_ANSWER
    assert resolution.answer == case.expected_answer

