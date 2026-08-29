import json

from cformer_v63 import RecursiveResolver, RelationGraph

DATA = r"E:\deepseek\c-former\data\ai_models_dataset.json"


def _load():
    return json.load(open(DATA, encoding="utf-8"))["objects"]


def test_latest_of_gpt_series_walks_to_chain_end() -> None:
    objects = _load()
    graph = RelationGraph(objects)
    resolver = RecursiveResolver(graph, max_depth=4)
    result = resolver.latest_of_series("GPT")
    assert result.ok, result
    assert result.answer_id == "gpt-5-2"
    assert result.path[0] == "gpt-1"
    assert result.path[-1] == "gpt-5-2"


def test_predecessor_lookup() -> None:
    objects = _load()
    graph = RelationGraph(objects)
    resolver = RecursiveResolver(graph)
    result = resolver.predecessor_of("gpt-5-2")
    assert result.ok and result.answer_id == "gpt-5-1"


def test_cycle_detection() -> None:
    objects = [
        {"id": "a", "series": "S", "year": 1, "predecessor": None},
        {"id": "b", "series": "S", "year": 1, "predecessor": "a"},
        {"id": "c", "series": "S", "year": 1, "predecessor": "b"},
    ]
    graph = RelationGraph(objects)
    graph.successors["c"] = ["a"]  # 人为制造 a->b->c->a 环（年份相等，避免先触发时间检查）
    resolver = RecursiveResolver(graph)
    result = resolver.latest_of_series("S")
    assert not result.ok
    assert result.reason == "cycle"


def test_depth_limit() -> None:
    objects = [
        {"id": f"n{i}", "series": "S", "year": i, "predecessor": f"n{i-1}" if i > 0 else None}
        for i in range(16)
    ]
    graph = RelationGraph(objects)
    resolver = RecursiveResolver(graph, max_depth=4)
    result = resolver.chain("n0", hops=6)
    assert not result.ok
    assert result.reason == "depth_exceeded"


def test_time_violation_detected() -> None:
    objects = [
        {"id": "a", "series": "S", "year": 2020, "predecessor": None},
        {"id": "b", "series": "S", "year": 2019, "predecessor": "a"},  # 时间回退
    ]
    graph = RelationGraph(objects)
    resolver = RecursiveResolver(graph)
    result = resolver.latest_of_series("S")
    assert not result.ok
    assert result.reason == "time_violation"


def test_version_pinning_limits_walk() -> None:
    objects = _load()
    graph = RelationGraph(objects)
    resolver = RecursiveResolver(graph, max_depth=4)
    # 只允许 2024 及以前的版本：最新应为 2024 年的 GPT-4o mini
    result = resolver.latest_of_series("GPT", world_version=2024)
    assert result.ok, result
    assert graph.year_of(result.answer_id) <= 2024
    assert result.answer_id == "gpt-4o-mini"
