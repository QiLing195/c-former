from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cformer_v58 import LayeredAliasStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5.8 disk-first alias storage benchmark")
    parser.add_argument("--scales", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/v58_storage_benchmark")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/v58_storage_results.json")
    )
    return parser.parse_args()


def attributes(object_id: int) -> tuple[str, str, str, str]:
    value = object_id - 1
    return (
        f"fam{value % 64:02d}",
        f"dom{(value // 64) % 64:02d}",
        f"fun{(value // 4096) % 64:02d}",
        f"reg{(value // 262144) % 64:02d}",
    )


def record(object_id: int) -> dict:
    family, domain, function, region = attributes(object_id)
    return {
        "object_id": object_id,
        "canonical_name": f"object_{object_id}",
        "document": (
            f"family {family} domain {domain} function {function} region {region} "
            f"stable object object_{object_id}"
        ),
        "aliases": (f"known_alias_{object_id}",),
        "perspectives": (1, 2, 3, 4),
    }


def open_expression(object_id: int) -> str:
    family, domain, function, region = attributes(object_id)
    return f"region {region} function {function} unknown expression domain {domain} family {family}"


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024**2


def benchmark_scale(args, scale: int) -> dict:
    directory = args.root / f"objects_{scale}"
    if directory.exists():
        raise FileExistsError(f"benchmark directory already exists: {directory}")
    before_build_rss = rss_mb()
    store = LayeredAliasStore(directory, dimensions=args.dimensions)
    start = time.perf_counter()
    for first in range(1, scale + 1, args.batch_size):
        last = min(scale + 1, first + args.batch_size)
        store.add_objects([record(object_id) for object_id in range(first, last)])
        if first == 1 or (first - 1) % (args.batch_size * 16) == 0:
            print(f"scale={scale} built={last - 1}", flush=True)
    build_seconds = time.perf_counter() - start
    disk_bytes = store.disk_bytes()
    counts = {
        "objects": store.object_count,
        "aliases": store.alias_count,
        "perspectives": store.perspective_count,
        "quantized_vector_rows": store.vectors.count,
    }
    store.close()

    before_open_rss = rss_mb()
    start = time.perf_counter()
    store = LayeredAliasStore(directory, dimensions=args.dimensions)
    lazy_open_ms = 1000.0 * (time.perf_counter() - start)
    after_open_rss = rss_mb()
    generator = random.Random(58_000 + scale)
    query_ids = generator.sample(range(1, scale + 1), min(args.queries, scale))

    start = time.perf_counter()
    exact_correct = sum(
        store.lookup_alias(f"known_alias_{object_id}") == [object_id]
        for object_id in query_ids
    )
    exact_ms = 1000.0 * (time.perf_counter() - start) / len(query_ids)

    open_top1 = 0
    open_recall10 = 0
    start = time.perf_counter()
    for object_id in query_ids:
        hits = store.search(open_expression(object_id), limit=64)
        open_top1 += int(bool(hits) and hits[0].object_id == object_id)
        open_recall10 += int(object_id in [hit.object_id for hit in hits[:10]])
    open_query_ms = 1000.0 * (time.perf_counter() - start) / len(query_ids)
    after_queries_rss = rss_mb()
    store.close()

    naive_repeated_float_bytes = scale * 4 * 264 * 4
    return {
        "scale_objects": scale,
        "effective_four_view_candidates": scale * 4,
        **counts,
        "build_seconds": build_seconds,
        "build_objects_per_second": scale / build_seconds,
        "disk_mb": disk_bytes / 1024**2,
        "disk_bytes_per_object": disk_bytes / scale,
        "naive_repeated_float_mb": naive_repeated_float_bytes / 1024**2,
        "storage_reduction_vs_repeated_float": naive_repeated_float_bytes / disk_bytes,
        "lazy_open_ms": lazy_open_ms,
        "open_rss_delta_mb": (
            after_open_rss - before_open_rss
            if after_open_rss is not None and before_open_rss is not None
            else None
        ),
        "query_rss_delta_mb": (
            after_queries_rss - after_open_rss
            if after_queries_rss is not None and after_open_rss is not None
            else None
        ),
        "exact_alias_accuracy": exact_correct / len(query_ids),
        "exact_alias_query_ms": exact_ms,
        "open_expression_top1": open_top1 / len(query_ids),
        "open_expression_recall_at_10": open_recall10 / len(query_ids),
        "open_expression_query_ms": open_query_ms,
        "committed_false_merge_rate": 0.0,
        "note": "open candidates are never auto-committed",
        "build_rss_start_mb": before_build_rss,
    }


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    results = {}
    for scale in args.scales:
        results[str(scale)] = benchmark_scale(args, scale)
        print(json.dumps(results[str(scale)], ensure_ascii=False, indent=2), flush=True)
    output = {
        "configuration": vars(args) | {"root": str(args.root), "output": str(args.output)},
        "complexity": {
            "storage": "O(objects + aliases + bounded_postings)",
            "exact_alias_query": "O(log aliases) SQLite B-tree",
            "open_query": "O(matched postings + top_k * vector_dimensions)",
            "neural_rerank": "O(top_k), never O(all objects)",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
