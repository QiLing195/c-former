# -*- coding: utf-8 -*-
"""V6.1c 真实库阈值校准：恢复 supported 覆盖率，同时零误支持。

方法（沿用 V6.0b 校准方法论）：
1. 三分切割 known 查询：训练 / 校准 / 留出，按内容哈希互斥；
2. 歧义查询对半切；unknown 用固定表合成补量后对半切；
3. 每种子训练编码器后，在校准集上采集 (score, runner_up, margin, coverage) 分布；
4. 网格搜索 (min_score, min_coverage, known_margin)：硬约束歧义/unknown 零误支持，
   最大化 known 支持率，平手取更保守值；
5. 跨 3 种子取中位数阈值，输出推荐值与风险—覆盖曲线。

用法：
    D:\conda\envs\cformer-gpu\python.exe calibrate_v61c.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_real import AIModelWorld
from evaluate_v61c import WorldEncoder, build_chain
from train_eval_real import build_training_entries, train

ROOT = Path(__file__).resolve().parent

SYNTHETIC_UNKNOWN = [
    "GPT-7", "GPT-8", "Claude Opus 9", "Claude Sonnet 8", "Gemini 4",
    "Gemini 3.9", "Llama 5", "Grok 7", "Qwen4", "Qwen5-Max",
    "DeepSeek-V5", "DeepSeek-R2", "GLM-6", "Kimi K3", "Mistral Large 4",
    "Phi-5", "Nova Ultra", "Command X", "Falcon 4", "OLMo 3",
    "豆包 3.0", "混元 T2", "星火 X2", "盘古 6.0",
]

GRID_MIN_SCORE = [round(0.30 + 0.04 * i, 2) for i in range(8)]
GRID_MIN_COVERAGE = [0.40, 0.50, 0.60]
GRID_KNOWN_MARGIN = [0.005, 0.01, 0.02, 0.03, 0.05]
SAFETY_MARGIN = 0.08  # 歧义/unknown 永不放松（V6.0b 结论）


def _bucket(text: str, modulus: int, residue: int) -> bool:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return digest[0] % modulus == residue


def three_way_split(known: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train, calibration, heldout = [], [], []
    for query in known:
        if _bucket(query["text"], 5, 0):
            heldout.append(query)
        elif _bucket(query["text"], 7, 3):
            calibration.append(query)
        else:
            train.append(query)
    return train, calibration, heldout


def half_split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    first = [i for i in items if _bucket(i["text"], 2, 0)]
    return first, [i for i in items if not _bucket(i["text"], 2, 0)]


@torch.inference_mode()
def collect_scores(encoder: WorldEncoder, index, nprobe: int, items: list[tuple[str, str]]) -> list[dict]:
    records = []
    for text, kind in items:
        vector, coverage = encoder.encode_query(text)
        scores, _ = index.search(vector, nprobe, 2)
        top1 = float(scores[0]) if len(scores) else 0.0
        top2 = float(scores[1]) if len(scores) > 1 else 0.0
        records.append({
            "text": text, "kind": kind, "score": top1,
            "runner_up": top2, "margin": top1 - top2, "coverage": float(coverage),
        })
    return records


def passes_gates(record: dict, min_score: float, min_coverage: float, margin: float) -> bool:
    return (
        record["coverage"] >= min_coverage
        and record["score"] >= min_score
        and record["margin"] >= margin
    )


def fit_thresholds(records: list[dict]) -> dict:
    """Max known coverage under zero false support on ambiguous/unknown.

    If no grid point achieves zero false support, fall back to the knee
    (minimum false support, then maximum coverage) and flag it.
    """
    known = [r for r in records if r["kind"] == "known"]
    safety = [r for r in records if r["kind"] != "known"]
    feasible = None
    knee = None
    curve = []
    for min_score in GRID_MIN_SCORE:
        for min_coverage in GRID_MIN_COVERAGE:
            for known_margin in GRID_KNOWN_MARGIN:
                supported = sum(passes_gates(r, min_score, min_coverage, known_margin) for r in known)
                false_support = sum(
                    passes_gates(r, min_score, min_coverage, SAFETY_MARGIN) for r in safety
                )
                entry = {
                    "min_score": min_score, "min_coverage": min_coverage,
                    "known_margin": known_margin,
                    "known_coverage": supported / max(1, len(known)),
                    "false_support": false_support,
                }
                curve.append(entry)
                if knee is None or (entry["false_support"], -entry["known_coverage"]) < (
                    knee["false_support"], -knee["known_coverage"]
                ):
                    knee = entry
                if false_support == 0 and (
                    feasible is None
                    or entry["known_coverage"] > feasible["known_coverage"]
                    or (entry["known_coverage"] == feasible["known_coverage"]
                        and entry["min_score"] > feasible["min_score"])
                ):
                    feasible = entry
    chosen = feasible if feasible is not None else knee
    # 每个 known_margin 档位的零误支持最优解（选点时权衡覆盖率 vs 近失风险）
    per_margin = {}
    for margin_value in GRID_KNOWN_MARGIN:
        candidates_at_margin = [
            e for e in curve
            if e["known_margin"] == margin_value and e["false_support"] == 0
        ]
        if candidates_at_margin:
            top = max(candidates_at_margin,
                      key=lambda e: (e["known_coverage"], e["min_score"], e["min_coverage"]))
            per_margin[str(margin_value)] = top
    # 记录当前选点下误支持的安全项样本，供诊断
    offenders = [
        {k: r[k] for k in ("text", "kind", "score", "margin", "coverage")}
        for r in safety
        if passes_gates(r, chosen["min_score"], chosen["min_coverage"], SAFETY_MARGIN)
   ][:10]
    curve.sort(key=lambda e: (e["false_support"], -e["known_coverage"]))
    return {
        "chosen": chosen,
        "constraint_met": feasible is not None,
        "per_margin_best": per_margin,
        "offenders": offenders,
        "curve_head": curve[:12],
        "n_known": len(known), "n_safety": len(safety),
        "records": records,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    train_q, calibration_q, heldout_q = three_way_split(world.known_queries())
    amb_cal, amb_eval = half_split(world.ambiguous_queries())
    unk_main = world.unknown_queries()
    unk_all = unk_main + [{"text": t, "target_id": None, "kind": "unknown"}
                          for t in SYNTHETIC_UNKNOWN]
    unk_cal, unk_eval = half_split(unk_all)

    print(json.dumps({"split": {
        "train": len(train_q), "calibration_known": len(calibration_q),
        "heldout": len(heldout_q), "ambiguous_cal": len(amb_cal),
        "ambiguous_eval": len(amb_eval), "unknown_cal": len(unk_cal),
        "unknown_eval": len(unk_eval)}}, ensure_ascii=False))

    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )
    # 校准 known 用训练查询的 T1/T4 改写体：表面形式与训练（T0/T2/T3）不相交，
    # 但目标对象相同——测的是"阈值跨措辞迁移"，需在文档中声明该近似。
    # 无 meta 的手写查询没有改写体，原句进训练、不进校准。
    from cformer_real import query_variants
    from cformer_v61c import has_selection_phrase

    calibration_items = []
    entries_spec = []
    for query in train_q:
        variants = query_variants(query["text"], query.get("meta"))
        target = world.target_label(query["target_id"])
        if len(variants) >= 5:
            calibration_items.append((variants[1], "known"))
            calibration_items.append((variants[4], "known"))
            entries_spec.append({"variants": [variants[i] for i in (0, 2, 3)], "target": target})
        else:
            entries_spec.append({"variants": [variants[0]], "target": target})
    # 裸系列指代由结构性歧义规则确定性拦截（见 pipeline），不进入神经安全集
    structural_caught = [
        q for q in amb_cal if not has_selection_phrase(q["text"])
    ]
    neural_amb_cal = [q for q in amb_cal if has_selection_phrase(q["text"])]
    calibration_items += [(q["text"], "ambiguous") for q in neural_amb_cal]
    calibration_items += [(q["text"], "unknown") for q in unk_cal]

    per_seed = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        train(model, world, device, entries=entries_spec, steps=400, lr=1e-3,
              batch_size=16, hard_k=0, seed=seed)
        encoder = WorldEncoder(world, model, device)
        _, _, index, _, _ = build_chain(world, encoder)
        result = fit_thresholds(collect_scores(encoder, index, 16, calibration_items))
        per_seed[str(seed)] = result
        print(json.dumps({"phase": "seed", "seed": seed, "chosen": result["chosen"],
                          "n_known": result["n_known"]}, ensure_ascii=False))

    def median_of(key: str) -> float:
        return statistics.median(result_seed[key] for result_seed in
                                 (per_seed[s]["chosen"] for s in per_seed))

    recommendation = {
        "minimum_score": median_of("min_score"),
        "minimum_coverage": median_of("min_coverage"),
        "known_margin": median_of("known_margin"),
        "safety_margin": SAFETY_MARGIN,
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__},
        "split_sizes": {"train": len(train_q), "calibration_known": len(calibration_q),
                        "heldout": len(heldout_q),
                        "ambiguous_structural_caught": len(structural_caught),
                        "ambiguous_neural_cal": len(neural_amb_cal)},
        "grid": {"min_score": GRID_MIN_SCORE, "min_coverage": GRID_MIN_COVERAGE,
                 "known_margin": GRID_KNOWN_MARGIN},
        "note": ("三分切割互斥；歧义/unknown 的 margin 固定 0.08 不放松；裸系列指代由"
                 "结构性规则拦截后不计入神经安全集；校准 known 用训练查询的 T1/T4 改写体。"),
        "recommendation": recommendation,
        "per_seed": per_seed,
    }
    out = ROOT / "artifacts" / "v61c_calibration.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "recommendation": recommendation}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
