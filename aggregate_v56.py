from __future__ import annotations

import json
import math
from pathlib import Path


SCALES = (2048, 8192, 32768)


def mean_ci(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance / len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> None:
    files = (Path("artifacts/v56_results.json"), Path("artifacts/v56_results_extra.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    seeds = [item for run in runs for item in run["seed_results"]]
    if len({item["seed"] for item in seeds}) != len(seeds):
        raise ValueError("duplicate seed in V5.6 aggregation")
    summary = {}
    for scale in SCALES:
        key = str(scale)
        cformer = [item["scales"][key]["cformer_accuracy"] for item in seeds]
        rag = [item["scales"][key]["rag_accuracy"] for item in seeds]
        no_observer = [item["scales"][key]["no_observer_accuracy"] for item in seeds]
        difference = [left - right for left, right in zip(cformer, rag)]
        summary[key] = {
            "cformer_accuracy": mean_ci(cformer),
            "rag_accuracy": mean_ci(rag),
            "paired_cformer_minus_rag": mean_ci(difference),
            "no_observer_accuracy": mean_ci(no_observer),
            "rough_recall_at_64": sum(
                item["scales"][key]["cformer_rough_recall_at_k"] for item in seeds
            )
            / len(seeds),
            "rerank_query_ms": sum(
                item["scales"][key]["cformer_rerank_query_ms"] for item in seeds
            )
            / len(seeds),
            "full_scan_query_ms": sum(
                item["scales"][key]["full_scan_query_ms"] for item in seeds
            )
            / len(seeds),
            "compact_cache_mb": seeds[0]["scales"][key]["compact_cache_mb"],
        }
    output = {
        "seeds": [item["seed"] for item in seeds],
        "worlds_per_scale_per_seed": runs[0]["configuration"]["worlds"],
        "queries_per_world": runs[0]["configuration"]["queries"],
        "parameters": runs[0]["parameters"],
        "summary": summary,
        "boundary_metrics": runs[0]["boundary_metrics_from_v55"],
    }
    path = Path("artifacts/v56_summary.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"saved={path}")


if __name__ == "__main__":
    main()
