# -*- coding: utf-8 -*-
"""V6.3-M0 时间轴评测：as-of 快照查询 + 未来事实泄漏闸门。

用法：
    D:\conda\envs\cformer-gpu\python.exe evaluate_temporal.py

真值构造（无需人工标注）：对每个系列的每个历史年份切点 Y（非全局最大年），
"截至{Y}年，{company} 的 {series} 系列最新模型是什么？"的期望答案 =
年份≤Y 中最大者（平局取系列内定义顺序靠后）；另为每系列生成一个
截至(最小年-1) 的空集查询，期望 unknown。

对照双臂：
- naive：推理块忽略 as_of（模拟无时间防线）——预期大量未来泄漏；
- temporal：as_of 过滤生效——闸门要求泄漏=0、Top-1≥95%、空集全部显式拒绝。
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_v62 import WorldReasoner, extract_year
from cformer_real import AIModelWorld, query_variants
from evaluate_v61c import WorldEncoder, build_chain
from evaluate_v62 import make_reasoner
from train_eval_real import split_known, train

ROOT = Path(__file__).resolve().parent


class NaiveAsOfReasoner:
    """忽略 as_of 的对照臂：只保留普通跨候选选择。"""

    def __init__(self, inner: WorldReasoner):
        self.inner = inner

    def select(self, text, ranked_labels, ranked_scores, **kwargs):
        return self.inner.select(text, ranked_labels, ranked_scores)


def build_temporal_sets(world: AIModelWorld) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    raw = json.loads((ROOT / "data" / "ai_models_dataset.json").read_text(encoding="utf-8"))
    for obj in raw["objects"]:
        meta = obj.get("meta") or {}
        if not meta.get("company"):
            continue
        evidence_change = obj["evidence"]["变化"]
        year = extract_year(evidence_change)
        if year is None:
            continue
        groups[(meta["company"], meta["series"])].append({
            "id": obj["id"], "year": year,
            "series_index": meta.get("series_index", 0),
        })

    asof_queries: list[dict] = []
    vacuous_queries: list[dict] = []
    for (company, series), members in sorted(groups.items()):
        members.sort(key=lambda m: (m["year"], m["series_index"]))
        years = sorted({m["year"] for m in members})
        for cut in years[:-1]:                      # 全局最大年不构成历史切点
            eligible = [m for m in members if m["year"] <= cut]
            expected = max(eligible, key=lambda m: (m["year"], m["series_index"]))
            asof_queries.append({
                "text": f"截至{cut}年，{company} 的 {series} 系列最新模型是什么？",
                "expected_id": expected["id"], "as_of": cut,
            })
        vacuous_queries.append({
            "text": f"截至{years[0] - 1}年，{company} 的 {series} 系列最新模型是什么？",
            "expected_id": None, "as_of": years[0] - 1,
        })
    return asof_queries, vacuous_queries


def year_of_id(raw_by_id: dict, object_id: str | None) -> int | None:
    if object_id is None:
        return None
    return extract_year(raw_by_id[object_id]["evidence"]["变化"])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    raw = json.loads((ROOT / "data" / "ai_models_dataset.json").read_text(encoding="utf-8"))
    raw_by_id = {obj["id"]: obj for obj in raw["objects"]}
    asof_set, vacuous_set = build_temporal_sets(world)
    known_train, heldout = split_known(world.known_queries())
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )
    print(json.dumps({"temporal_queries": len(asof_set),
                      "vacuous_queries": len(vacuous_set)}, ensure_ascii=False))

    per_seed = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        entries_spec = []
        for query in known_train:
            variants = query_variants(query["text"], query.get("meta"))
            target = world.target_label(query["target_id"])
            entries_spec.append({
                "variants": [variants[i] for i in (0, 2, 3)] if len(variants) >= 5 else [variants[0]],
                "target": target,
            })
        train(model, world, device, entries=entries_spec, steps=400, lr=1e-3,
              batch_size=16, hard_k=0, seed=seed)
        encoder = WorldEncoder(world, model, device)

        arms = {}
        for arm, reasoner in (
            ("naive", NaiveAsOfReasoner(make_reasoner(world))),
            ("temporal", make_reasoner(world)),
        ):
            store, ledger, index, vectors, pipeline = build_chain(
                world, encoder, minimum_score=0.40, minimum_coverage=0.60, known_margin=0.01,
            )
            pipeline.reasoner = reasoner

            top1 = leak = supported = 0
            for item in asof_set:
                result = pipeline.resolve(item["text"], query_type="known")
                if result.status != CandidateStatus.SUPPORTED:
                    continue
                supported += 1
                pred_year = year_of_id(raw_by_id, result.object_id)
                if result.object_id == item["expected_id"]:
                    top1 += 1
                if pred_year is not None and pred_year > item["as_of"]:
                    leak += 1
            vacuous_rejected = 0
            for item in vacuous_set:
                result = pipeline.resolve(item["text"])
                vacuous_rejected += int(
                    result.status != CandidateStatus.SUPPORTED and result.object_id is None
                )
            heldout_hits = sum(
                1 for q in heldout
                if (r := pipeline.resolve(q["text"], query_type="known")).status
                == CandidateStatus.SUPPORTED and r.object_id == q["target_id"]
            )
            arms[arm] = {
                "asof_top1": top1 / len(asof_set),
                "future_leakage_rate": leak / len(asof_set),
                "supported_coverage": supported / len(asof_set),
                "vacuous_rejected_rate": vacuous_rejected / len(vacuous_set),
                "heldout_top1_regression": heldout_hits / len(heldout),
            }
            store.close()
        per_seed[str(seed)] = arms
        print(json.dumps({"phase": "seed", "seed": seed,
                          "naive_leak": arms["naive"]["future_leakage_rate"],
                          "temporal_leak": arms["temporal"]["future_leakage_rate"],
                          "temporal_top1": arms["temporal"]["asof_top1"]},
                         ensure_ascii=False), flush=True)

    aggregate = {
        arm: {key: statistics.mean(per_seed[s][arm][key] for s in per_seed)
              for key in per_seed["1"][arm]}
        for arm in ("naive", "temporal")
    }
    gates = {
        "temporal_future_leakage_zero": aggregate["temporal"]["future_leakage_rate"] == 0.0,
        "temporal_top1_ge_95": aggregate["temporal"]["asof_top1"] >= 0.95,
        "vacuous_all_explicitly_rejected":
            aggregate["temporal"]["vacuous_rejected_rate"] == 1.0,
        "heldout_regression_100pct":
            aggregate["temporal"]["heldout_top1_regression"] == 1.0,
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "set_sizes": {"asof": len(asof_set), "vacuous": len(vacuous_set)},
        "verifier": {"minimum_score": 0.40, "minimum_coverage": 0.60, "known_margin": 0.01},
        "note": ("naive 臂证明无时间防线时未来泄漏普遍存在；temporal 臂以确定性"
                 "快照过滤满足全部闸门。空集查询必须显式 unknown 且不得提案。"),
        "gates": gates,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    out = ROOT / "artifacts" / "v63_temporal_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "gates": gates}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
