from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cformer_v58 import LayeredAliasStore
from evaluate_v58_storage import open_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebenchmark an existing V5.8 store")
    parser.add_argument("--input", type=Path, default=Path("artifacts/v58_storage_results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v58_storage_results_optimized.json"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/v58_storage_20260718"))
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--queries", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for scale_text, metrics in data["results"].items():
        scale = int(scale_text)
        store = LayeredAliasStore(args.root / f"objects_{scale}", dimensions=args.dimensions)
        generator = random.Random(58_000 + scale)
        ids = generator.sample(range(1, scale + 1), min(args.queries, scale))
        start = time.perf_counter()
        exact = sum(store.lookup_alias(f"known_alias_{object_id}") == [object_id] for object_id in ids)
        metrics["exact_alias_query_ms"] = 1000.0 * (time.perf_counter() - start) / len(ids)
        metrics["exact_alias_accuracy"] = exact / len(ids)
        top1 = recall10 = 0
        start = time.perf_counter()
        for object_id in ids:
            hits = store.search(open_expression(object_id), 64)
            top1 += int(bool(hits) and hits[0].object_id == object_id)
            recall10 += int(object_id in [hit.object_id for hit in hits[:10]])
        metrics["open_expression_query_ms"] = 1000.0 * (time.perf_counter() - start) / len(ids)
        metrics["open_expression_top1"] = top1 / len(ids)
        metrics["open_expression_recall_at_10"] = recall10 / len(ids)
        metrics["query_strategy"] = "high-information posting intersection then int8 Top-64"
        store.close()
        print(scale, metrics["open_expression_query_ms"], metrics["open_expression_top1"])
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
