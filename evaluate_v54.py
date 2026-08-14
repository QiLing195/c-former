from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from cformer_v54 import QueryStatus, RecursiveQueryEngine, V54_SCALES, build_world


def evaluate_engine(engine: RecursiveQueryEngine, cases, mode: str = "full"):
    correct = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    depths = []
    results = []
    answer_expected = 0
    false_refusals = 0
    nonanswer_expected = 0
    unsafe_answers = 0
    for case in cases:
        observer_scope = 1 if mode == "no_observer_policy" else case.observer_scope
        observer_frame = 0 if mode == "no_spatial_alignment" else case.observer_frame
        query_time = 75 if mode == "no_time_alignment" else case.query_time
        ingest_cutoff = 100 if mode == "no_time_alignment" else case.ingest_cutoff
        max_depth = 0 if mode == "no_recursion" else case.max_depth
        result = engine.query_position(
            case.entity,
            observer_scope=observer_scope,
            observer_frame=observer_frame,
            query_time=query_time,
            ingest_cutoff=ingest_cutoff,
            max_depth=max_depth,
        )
        value_correct = case.expected_value is None or result.value == case.expected_value
        is_correct = result.status == case.expected_status and value_correct
        correct += int(is_correct)
        category_correct[case.category] += int(is_correct)
        category_total[case.category] += 1
        depths.append(result.depth)
        results.append(result)
        if case.expected_status == QueryStatus.ANSWER:
            answer_expected += 1
            false_refusals += int(result.status != QueryStatus.ANSWER)
        else:
            nonanswer_expected += 1
            unsafe_answers += int(result.status == QueryStatus.ANSWER)
    metrics = {
        "accuracy": correct / len(cases),
        "average_depth": sum(depths) / len(depths),
        "max_depth": max(depths),
        "false_refusal_rate": false_refusals / max(1, answer_expected),
        "unsafe_answer_rate": unsafe_answers / max(1, nonanswer_expected),
        "future_leakage_rate": sum(
            result.status == QueryStatus.ANSWER
            for case, result in zip(cases, results)
            if case.category == "delayed_hidden"
        )
        / max(1, sum(case.category == "delayed_hidden" for case in cases)),
    }
    for category in sorted(category_total):
        metrics[f"accuracy_{category}"] = category_correct[category] / category_total[category]
    return metrics, results


def main() -> None:
    all_results = {}
    for scale in V54_SCALES:
        aggregate = defaultdict(float)
        ablation_aggregate = defaultdict(float)
        indexed_times = []
        linear_times = []
        for world_index in range(5):
            world = build_world(scale, world_index)
            indexed = RecursiveQueryEngine(world.memory, world.frames, indexed=True)
            linear = RecursiveQueryEngine(world.memory, world.frames, indexed=False)
            indexed_metrics, indexed_results = evaluate_engine(indexed, world.cases)
            linear_metrics, linear_results = evaluate_engine(linear, world.cases)
            assert indexed_results == linear_results
            for mode in (
                "no_recursion",
                "no_time_alignment",
                "no_spatial_alignment",
                "no_observer_policy",
            ):
                metrics, _ = evaluate_engine(indexed, world.cases, mode)
                ablation_aggregate[mode] += metrics["accuracy"] / 5

            start = time.perf_counter()
            evaluate_engine(indexed, world.cases)
            indexed_times.append(1000.0 * (time.perf_counter() - start) / len(world.cases))
            start = time.perf_counter()
            evaluate_engine(linear, world.cases)
            linear_times.append(1000.0 * (time.perf_counter() - start) / len(world.cases))
            for key, value in indexed_metrics.items():
                aggregate[key] += value / 5
        all_results[scale] = {
            "metrics": dict(aggregate),
            "ablations": dict(ablation_aggregate),
            "indexed_query_ms": sum(indexed_times) / len(indexed_times),
            "linear_query_ms": sum(linear_times) / len(linear_times),
            "speedup_vs_linear": sum(linear_times) / sum(indexed_times),
            "independent_facts": scale,
            "estimated_compact_cache_mb": scale * 128 / 1024**2,
        }
        print(f"scale={scale} result={all_results[scale]}")

    output = Path("artifacts/v54_results.json")
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
