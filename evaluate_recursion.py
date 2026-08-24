# -*- coding: utf-8 -*-
"""V6.3-M0 Transformation 递归评测：多跳行走准确率 + 安全控制器 + 回归。

用法：
    D:\conda\envs\cformer-gpu\python.exe evaluate_recursion.py

真值构造（零标注）：每个系列按（年份, 定义顺序）成链，对成员位置 p：
- 后退 k∈1..min(p,8) 跳 → 期望 members[p-k]；
- 前进 k∈1..min(len-1-p,8) 跳 → 期望 members[p+k]。
按路线图 §9 要求，1—4 跳与 5—8 跳分开报告，不互相掩盖。

递归本身是确定性的（不依赖编码器），种子仅用于留出集回归检验。
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import re

import torch

from cformer_v59 import CandidateStatus
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_v63 import MultiHopResolver, TransformationGraph
from cformer_real import AIModelWorld
from evaluate_v61c import WorldEncoder, build_chain
from evaluate_v62 import make_reasoner
from train_eval_real import split_known, train

ROOT = Path(__file__).resolve().parent


def build_hop_sets(world: AIModelWorld):
    raw = json.loads((ROOT / "data" / "ai_models_dataset.json").read_text(encoding="utf-8"))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for obj in raw["objects"]:
        meta = obj.get("meta") or {}
        if not meta.get("company"):
            continue
        match = re.search(r"(?:19|20)\d{2}", obj["evidence"]["变化"])
        year = int(match.group()) if match else 0
        groups[(meta["company"], meta["series"])].append({
            "id": obj["id"], "name": obj["name"],
            "year": year, "series_index": meta.get("series_index", 0),
        })

    backward_14, backward_58, forward_14, forward_58 = [], [], [], []
    for (company, series), members in sorted(groups.items()):
        members.sort(key=lambda m: (m["year"], m["series_index"]))
        for position, member in enumerate(members):
            name = member["name"].replace(" ", "")
            max_back = min(position, 8)
            for hop in range(1, max_back + 1):
                expected = members[position - hop]["id"]
                item = {"text": f"{name} 往前数{hop}代是哪个模型？" if hop > 1
                        else f"{name} 的前一代是什么？",
                        "expected_id": expected, "hops": hop,
                        "direction": "backward"}
                if hop <= 4:
                    backward_14.append(item)
                else:
                    backward_58.append(item)
            max_fwd = min(len(members) - 1 - position, 8)
            for hop in range(1, max_fwd + 1):
                expected = members[position + hop]["id"]
                item = {"text": f"{name} 往后数{hop}代是哪个模型？",
                        "expected_id": expected, "hops": hop,
                        "direction": "forward"}
                if hop <= 4:
                    forward_14.append(item)
                else:
                    forward_58.append(item)
    return {"backward_1_4": backward_14, "backward_5_8": backward_58,
            "forward_1_4": forward_14, "forward_5_8": forward_58}


def score_set(pipeline, items: list[dict]) -> dict:
    hits = supported = 0
    for item in items:
        result = pipeline.resolve(item["text"])
        if result.status == CandidateStatus.SUPPORTED:
            supported += 1
            hits += int(result.object_id == item["expected_id"])
    return {
        "n": len(items),
        "supported_rate": supported / len(items) if items else 1.0,
        "top1": hits / len(items) if items else 1.0,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    hop_sets = build_hop_sets(world)
    print(json.dumps({name: len(items) for name, items in hop_sets.items()},
                     ensure_ascii=False))
    known_train, heldout = split_known(world.known_queries())
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )

    graph = TransformationGraph(world)
    multihop = MultiHopResolver(graph)

    # 确定性递归指标只算一次；种子循环只做留出集回归
    store, ledger, index, vectors, pipeline = build_chain(
        world, WorldEncoder(world, TokenCFormerResolver(config).eval(), device),
        minimum_score=0.40, minimum_coverage=0.60, known_margin=0.01,
    )
    pipeline.reasoner = make_reasoner(world)
    pipeline.multihop = multihop

    recursion_metrics = {
        name: score_set(pipeline, items) for name, items in hop_sets.items()
    }
    alias_smoke = pipeline.resolve("千问系列最新的模型是哪一个？")
    qwen_latest_expected = "qwen3-7-max"
    alias_ok = (
        alias_smoke.status == CandidateStatus.SUPPORTED
        and alias_smoke.object_id == qwen_latest_expected
    )
    store.close()

    per_seed = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        entries_spec = []
        from cformer_real import query_variants
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
        store, ledger, index, vectors, pipeline = build_chain(
            world, encoder, minimum_score=0.40, minimum_coverage=0.60, known_margin=0.01,
        )
        pipeline.reasoner = make_reasoner(world)
        pipeline.multihop = multihop
        held_hits = sum(
            1 for q in heldout
            if (r := pipeline.resolve(q["text"], query_type="known")).status
            == CandidateStatus.SUPPORTED and r.object_id == q["target_id"]
        )
        per_seed[str(seed)] = {"heldout_top1_regression": held_hits / len(heldout)}
        print(json.dumps({"phase": "seed", "seed": seed,
                          "heldout_regression": per_seed[str(seed)]["heldout_top1_regression"]},
                         ensure_ascii=False), flush=True)
        store.close()

    regression = statistics.mean(per_seed[s]["heldout_top1_regression"] for s in per_seed)
    gates = {
        "backward_1_4_ge_90pct": recursion_metrics["backward_1_4"]["top1"] >= 0.90,
        "forward_1_4_ge_90pct": recursion_metrics["forward_1_4"]["top1"] >= 0.90,
        "hop_5_8_reported_separately":
            "backward_5_8" in recursion_metrics and "forward_5_8" in recursion_metrics,
        "alias_coreference_resolves_qwen_latest": alias_ok,
        "heldout_regression_100pct": regression == 1.0,
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "verifier": {"minimum_score": 0.40, "minimum_coverage": 0.60, "known_margin": 0.01},
        "note": ("递归为确定性图行走（cycle/depth 控制器由单元测试覆盖）；"
                 "5—8 跳单独报告；别名指代消解以千问→Qwen 冒烟验证。"),
        "gates": gates,
        "recursion_metrics": recursion_metrics,
        "alias_smoke": {"ok": alias_ok, "predicted": alias_smoke.object_id},
        "per_seed_regression": per_seed,
    }
    out = ROOT / "artifacts" / "v63_recursion_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "gates": gates}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
