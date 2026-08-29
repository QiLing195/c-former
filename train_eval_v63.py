# -*- coding: utf-8 -*-
"""V6.3 递归层评测：在真实 AI 模型数据集的关系图上验证 latest/predecessor/多跳。

确定性控制器（v1 无神经网络），秒级出结果。对比参考：身份解析层 predecessor/latest
仅 33%（V62 报告）——递归层应显著高于此。

用法：
    D:/conda/envs/cformer-gpu/python.exe train_eval_v63.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from cformer_v63 import RecursiveResolver, RelationGraph

ROOT = Path(__file__).resolve().parent
AI_DATA = ROOT / "data" / "ai_models_dataset.json"


def main() -> None:
    objects = json.loads(AI_DATA.read_text(encoding="utf-8"))["objects"]
    graph = RelationGraph(objects)
    resolver = RecursiveResolver(graph, max_depth=4)

    # 1) latest：每个有多成员的系列
    latest_ok = latest_total = 0
    series_chains = {}
    for obj in objects:
        series_chains.setdefault(obj["series"], []).append(obj["id"])
    for series, members in series_chains.items():
        if len(members) < 2:
            continue
        latest_total += 1
        result = resolver.latest_of_series(series)
        if result.ok and result.answer_id in members:
            latest_ok += 1

    # 2) predecessor：每个有前代的对象
    pred_ok = pred_total = 0
    for obj in objects:
        if not obj.get("predecessor"):
            continue
        pred_total += 1
        result = resolver.predecessor_of(obj["id"])
        if result.ok and result.answer_id == obj["predecessor"]:
            pred_ok += 1

    # 3) 多跳：从链头走 1/2/3 跳
    hop_ok = hop_total = 0
    for series, members in series_chains.items():
        head = resolver.graph.series_heads(series)
        if not head:
            continue
        start = head[0]
        for hops in (1, 2, 3):
            result = resolver.chain(start, hops)
            if result.ok and result.answer_id in members:
                hop_ok += 1
            hop_total += 1

    # 4) 控制探针：循环 / 深度 / 时间 / 版本
    cycle_objects = [
        {"id": "a", "series": "S", "year": 1, "predecessor": None},
        {"id": "b", "series": "S", "year": 1, "predecessor": "a"},
        {"id": "c", "series": "S", "year": 1, "predecessor": "b"},
    ]
    cycle_graph = RelationGraph(cycle_objects)
    cycle_graph.successors["c"] = ["a"]
    cycle_res = RecursiveResolver(cycle_graph).latest_of_series("S")
    depth_res = resolver.chain("gpt-1", hops=8)  # 超过 max_depth=4
    time_res = RecursiveResolver(RelationGraph([
        {"id": "x", "series": "T", "year": 2020, "predecessor": None},
        {"id": "y", "series": "T", "year": 2019, "predecessor": "x"},
    ])).latest_of_series("T")
    version_res = resolver.latest_of_series("GPT", world_version=2024)

    report = {
        "latest_accuracy": latest_ok / latest_total if latest_total else float("nan"),
        "predecessor_accuracy": pred_ok / pred_total if pred_total else float("nan"),
        "multi_hop_accuracy": hop_ok / hop_total if hop_total else float("nan"),
        "cycle_rejected": not cycle_res.ok and cycle_res.reason == "cycle",
        "depth_rejected": not depth_res.ok and depth_res.reason == "depth_exceeded",
        "time_violation_rejected": not time_res.ok and time_res.reason == "time_violation",
        "version_pinned_latest_within_2024": version_res.ok and graph.year_of(version_res.answer_id) <= 2024,
        "counts": {"latest": latest_total, "predecessor": pred_total, "multi_hop": hop_total},
        "identity_layer_baseline_predecessor_latest": 0.33,  # V62 报告参考值
    }
    output = ROOT / "artifacts" / "v63_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "report": report}, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
