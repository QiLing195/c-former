from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from cformer_v55 import CognitiveEngine, CognitiveStatus, build_cognitive_worlds


def _contains(actual, expected) -> bool:
    values = dict(actual or ())
    return all(values.get(field) == value for field, value in expected)


def evaluate(**engine_options):
    total = 0
    correct = 0
    answer_expected = 0
    false_refusal = 0
    unsafe_simulation = 0
    future_leakage = 0
    identity_collapse = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    per_world = {}
    start = time.perf_counter()
    for world in build_cognitive_worlds():
        engine = CognitiveEngine(world.memory, world.frames, **engine_options)
        world_correct = 0
        for case in world.cases:
            if case.action == "observe":
                result = engine.observe(case.name, case.observer)
                status = result.status
                object_id = result.object_id
                properties = result.value
            elif case.action == "simulate":
                result = engine.simulate(
                    case.name, case.operator or "", case.observer, dict(case.context)
                )
                status = result.status
                object_id = result.target_object_id
                properties = result.predicted_properties
            else:
                raise ValueError(f"unknown action {case.action}")
            matched = (
                status == case.expected_status
                and (
                    case.expected_object_id is None
                    or object_id == case.expected_object_id
                )
                and _contains(properties, case.expected_properties)
            )
            total += 1
            correct += int(matched)
            world_correct += int(matched)
            category_total[case.category] += 1
            category_correct[case.category] += int(matched)
            if case.expected_status == CognitiveStatus.ANSWER:
                answer_expected += 1
                false_refusal += int(status != CognitiveStatus.ANSWER)
            if case.category in {"constraint_unsatisfied", "constraint_missing"}:
                unsafe_simulation += int(status == CognitiveStatus.ANSWER)
            if case.category == "delayed_ingest_hidden":
                future_leakage += int(status == CognitiveStatus.ANSWER)
            if case.category == "identity_split_conflict":
                identity_collapse += int(status == CognitiveStatus.ANSWER)
        per_world[world.name] = world_correct / len(world.cases)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "accuracy": correct / total,
        "cases": total,
        "false_refusal_rate": false_refusal / max(1, answer_expected),
        "unsafe_simulation_rate": unsafe_simulation
        / max(1, sum(category_total[key] for key in ("constraint_unsatisfied", "constraint_missing"))),
        "future_leakage_rate": future_leakage / max(1, category_total["delayed_ingest_hidden"]),
        "identity_collapse_rate": identity_collapse / max(1, category_total["identity_split_conflict"]),
        "elapsed_ms": elapsed_ms,
        "per_world": per_world,
        "per_category": {
            key: category_correct[key] / category_total[key]
            for key in sorted(category_total)
        },
    }


def main() -> None:
    results = {
        "full": evaluate(),
        "ablations": {
            "no_constraints": evaluate(enforce_constraints=False),
            "no_observer_projection": evaluate(project_observer=False),
            "no_ingest_cutoff": evaluate(enforce_ingest_cutoff=False),
            "no_identity_conflict_guard": evaluate(detect_identity_conflict=False),
        },
    }
    output = Path("artifacts/v55_cognitive_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
