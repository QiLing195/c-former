# -*- coding: utf-8 -*-
"""V6.2 世界推理块 A/B 评测：同一链路、同一种子与阈值，仅开关 WorldReasoner。

用法：
    D:\conda\envs\cformer-gpu\python.exe evaluate_v62.py

对照假设（来自泛化消融负结果）：超级指代查询（"系列最新"）在 dual-encoder 上
留出集 ≈6.7%；接入跨候选年份比较后应大幅恢复，且不影响安全指标。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch

from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_v62 import WorldReasoner
from cformer_real import AIModelWorld
from evaluate_v61c import WorldEncoder, build_chain
from train_eval_real import split_known, train

ROOT = Path(__file__).resolve().parent
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def make_reasoner(world: AIModelWorld) -> WorldReasoner:
    def evidence_text_of(label: int) -> str:
        return "；".join(world.objects[label].evidence)

    lexicon = world.series_lexicon()

    def series_key_from_text(text: str):
        """词法锚定：系列名命中取最长者；否则唯一公司命中按公司级锚定。"""
        lowered = " ".join(text.split()).lower()
        series_hits = [
            (key, company, series) for key, company, series in lexicon if series in lowered
        ]
        if series_hits:
            key, _, _ = max(series_hits, key=lambda item: len(item[2]))
            return ("series", key)
        companies = {company for key, company, _ in lexicon if company in lowered}
        if len(companies) == 1:
            return ("company", next(iter(companies)))
        return None

    return WorldReasoner(
        series_key_of=world.series_key_of,
        evidence_text_of=evidence_text_of,
        series_index_of=world.series_index_of,
        series_key_from_text=series_key_from_text,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-score", type=float, default=0.40)
    parser.add_argument("--minimum-coverage", type=float, default=0.60)
    parser.add_argument("--known-margin", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "v62_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    blind = json.loads((ROOT / "data" / "ai_models_blindset.json").read_text(encoding="utf-8"))
    known_train, known_heldout = split_known(world.known_queries())
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )

    def run(pipeline, encoder, tag: str) -> dict:
        known_rows = []
        for query in world.known_queries():
            result = pipeline.resolve(query["text"], query_type="known")
            known_rows.append({
                "hit": result.status.value == "supported"
                       and result.object_id == query["target_id"],
                "path": result.path,
            })
        heldout_hits = []
        for query in known_heldout:
            result = pipeline.resolve(query["text"], query_type="known")
            heldout_hits.append(
                result.status.value == "supported"
                and result.object_id == query["target_id"]
            )
        blind_summary = {"known": [], "ambiguous": [], "unknown": []}
        for query in blind["queries"]:
            result = pipeline.resolve(query["text"], query.get("kind"))
            kind = query["kind"]
            if kind == "known":
                blind_summary[kind].append(
                    result.status.value == "supported" and result.object_id == query["target_id"]
                )
            elif kind == "ambiguous":
                blind_summary[kind].append(result.status.value == "ambiguous")
            else:
                blind_summary[kind].append(result.status.value != "supported")
        reasoned_share = statistics.mean(r["path"] == "reasoned" for r in known_rows)
        return {
            "tag": tag,
            "main_known_top1": statistics.mean(r["hit"] for r in known_rows),
            "heldout_top1": statistics.mean(heldout_hits),
            "blind_known_top1": statistics.mean(blind_summary["known"]),
            "blind_ambiguous_detected": statistics.mean(blind_summary["ambiguous"]),
            "blind_unknown_blocked": statistics.mean(blind_summary["unknown"]),
            "reasoned_path_share": reasoned_share,
        }

    ablation = {}
    per_seed = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        # 训练条目：与 calibrate 一致，T0/T2/T3 进训练，T1/T4 留作它用
        from cformer_real import query_variants
        entries_spec = []
        for query in known_train:
            variants = query_variants(query["text"], query.get("meta"))
            target = world.target_label(query["target_id"])
            if len(variants) >= 5:
                entries_spec.append({
                    "variants": [variants[i] for i in (0, 2, 3)],
                    "target": target,
                })
            else:
                entries_spec.append({"variants": [variants[0]], "target": target})
        train(model, world, device, entries=entries_spec, steps=400, lr=1e-3,
              batch_size=16, hard_k=0, seed=seed)
        encoder = WorldEncoder(world, model, device)

        arms = {}
        for arm, with_reasoner in (("baseline", False), ("reasoner", True)):
            store, ledger, index, vectors, pipeline = build_chain(
                world, encoder,
                minimum_score=args.minimum_score,
                minimum_coverage=args.minimum_coverage,
                known_margin=args.known_margin,
            )
            if with_reasoner:
                pipeline.reasoner = make_reasoner(world)
            arms[arm] = run(pipeline, encoder, arm)
            store.close()
        ablation[f"seed{seed}"] = arms
        print(json.dumps({"phase": "seed", "seed": seed,
                          "baseline_heldout": arms["baseline"]["heldout_top1"],
                          "reasoner_heldout": arms["reasoner"]["heldout_top1"],
                          "reasoned_share": arms["reasoner"]["reasoned_path_share"]},
                         ensure_ascii=False), flush=True)
        per_seed[str(seed)] = arms

    aggregate = {}
    for arm in ("baseline", "reasoner"):
        aggregate[arm] = {
            key: statistics.mean(per_seed[s][arm][key] for s in per_seed)
            for key in ("main_known_top1", "heldout_top1", "blind_known_top1",
                        "blind_ambiguous_detected", "blind_unknown_blocked",
                        "reasoned_path_share")
        }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "verifier": {"minimum_score": args.minimum_score,
                     "minimum_coverage": args.minimum_coverage,
                     "known_margin": args.known_margin},
        "note": ("A/B 仅差 WorldReasoner 开关。推理块为确定性跨候选年份比较"
                 "（含 series_index 平局裁决），无训练参数。"),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
