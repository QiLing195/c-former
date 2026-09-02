# -*- coding: utf-8 -*-
"""链尾查询：打印指定系列的前代链（从链头到链尾），用于核对盲测期望值。"""
import json
import sys
from pathlib import Path

from cformer_v63 import RecursiveResolver, RelationGraph

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    series_filter = sys.argv[1] if len(sys.argv) > 1 else None
    all_objects = []
    for path in (ROOT / "data" / "ai_models_dataset.json",
                 ROOT / "data" / "movies_dataset.json",
                 ROOT / "data" / "countries_recursion.json"):
        all_objects.extend(json.loads(path.read_text(encoding="utf-8"))["objects"])
    graph = RelationGraph(all_objects)
    resolver = RecursiveResolver(graph, max_depth=4)
    by_series = {}
    for obj in all_objects:
        by_series.setdefault(obj["series"], []).append(obj)
    for series, members in by_series.items():
        if series_filter and series_filter not in series:
            continue
        if len(members) < 2:
            continue
        result = resolver.latest_of_series(series)
        tail = result.answer_id if result.ok else None
        print(f"{series}: n={len(members)} latest={tail}")
        for obj in sorted(members, key=lambda o: int(o.get('year', 0))):
            pred = obj.get("predecessor")
            print(f"    {obj['name']} ({obj.get('year')}) -> {pred}")
