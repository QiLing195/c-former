# -*- coding: utf-8 -*-
"""V6.5-M3 公平基线矩阵：同候选库/同训练预算/同决策层，仅换架构与检索路径。

臂定义：
    b1_dual        冻结 v60 TokenCFormer 直接最近邻（主架构的神经底座）
    b2_meanpool    MeanPoolMLPResolver（同预算，无词序能力）
    b3_flat        FlatTransformer（平铺注意力，无四证据门控）
    b4_fts_rag     FTS 召回候选内神经打分（无 ANN/reasoner/gate——RAG 式）
    unified        完整栈（reasoner+as_of+multihop+verifier+gate）

公平性：四臂共用同一 tokenizer 覆盖率、同一 EvidenceVerifier 决策层、同一训练
条目与步数；差异只在编码器架构（b1-b3）或检索路径（b4 vs unified）。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

import torch

from cformer_v59 import CandidateStatus, EvidenceVerifier
from cformer_v60 import (
    ChineseTransformerConfig,
    FlatTransformerResolver,
    MeanPoolMLPResolver,
    TokenCFormerResolver,
)
from cformer_v62 import ObserverGate
from cformer_v63 import MultiHopResolver, TransformationGraph
from cformer_real import AIModelWorld, query_variants
from evaluate_temporal import build_temporal_sets
from evaluate_v61c import WorldEncoder, build_chain
from evaluate_v62 import make_reasoner
from train_eval_real import split_known, train

ROOT = Path(__file__).resolve().parent
VERIFIER = EvidenceVerifier(
    minimum_score=0.40, minimum_margin=0.08, minimum_coverage=0.60,
    margin_by_type={"known": 0.01, "hard": 0.03, "disambiguated": 0.03,
                    "ambiguous": 0.08, "unknown": 0.08, "conflict": 0.08},
)


def build_model(arm: str, tokenizer_size: int):
    config = ChineseTransformerConfig(
        tokenizer_size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )
    if arm == "b3_flat":
        # 平铺基线把四证据拼接为单序列（4×field_length），与 V6.0 报告口径一致
        config = ChineseTransformerConfig(
            tokenizer_size, layers=2, d_model=64, heads=4,
            ffn_dimensions=128, output_dimensions=32, max_length=192,
        )
    return {
        "b1_dual": TokenCFormerResolver,
        "b2_meanpool": MeanPoolMLPResolver,
        "b3_flat": FlatTransformerResolver,
        "b4_fts_rag": TokenCFormerResolver,   # RAG 式复用同一编码器，仅换检索路径
        "unified": TokenCFormerResolver,
    }[arm](config)


@torch.inference_mode()
def bank_scores(encoder_vectors: torch.Tensor, query_vector: torch.Tensor,
                restrict_labels: list[int] | None = None):
    """Return ranked (labels, scores) descending; optionally restricted."""
    if restrict_labels is not None:
        sub = encoder_vectors[restrict_labels]
        scores = sub @ query_vector
        order = torch.argsort(scores, descending=True)
        return [restrict_labels[i] for i in order.tolist()], scores[order].tolist()
    scores = encoder_vectors @ query_vector
    order = torch.argsort(scores, descending=True)
    return order.tolist(), scores[order].tolist()


def decide(verifier, labels: list[int], scores: list[float], coverage: float,
           query_type: str | None):
    if not labels:
        return CandidateStatus.UNKNOWN, None, 0.0, 0.0
    runner = scores[1] if len(scores) > 1 else 0.0
    decision = verifier.decide(float(scores[0]), float(runner),
                               float(coverage), query_type)
    object_id = None
    if decision.status == CandidateStatus.SUPPORTED:
        object_id = labels[0]
    return decision.status, object_id, decision.score, decision.margin


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    blind = json.loads((ROOT / "data" / "ai_models_blindset.json").read_text(encoding="utf-8"))
    asof_set, vacuous_set = build_temporal_sets(world)
    raw = json.loads((ROOT / "data" / "ai_models_dataset.json").read_text(encoding="utf-8"))
    year_by_id = {}
    for obj in raw["objects"]:
        match = __import__("re").search(r"(?:19|20)\d{2}", obj["evidence"]["变化"])
        year_by_id[obj["id"]] = int(match.group()) if match else None

    known_train, heldout = split_known(world.known_queries())
    entries_spec = []
    for query in known_train:
        variants = query_variants(query["text"], query.get("meta"))
        target = world.target_label(query["target_id"])
        entries_spec.append({
            "variants": [variants[i] for i in (0, 2, 3)] if len(variants) >= 5 else [variants[0]],
            "target": target,
        })

    arms = ["b1_dual", "b2_meanpool", "b3_flat", "b4_fts_rag", "unified"]
    per_seed: dict[str, dict] = {}

    for seed in (1, 2, 3):
        results = {}
        for arm in arms:
            print(json.dumps({"phase": "start", "seed": seed, "arm": arm,
                              "device": str(device)}, ensure_ascii=False), flush=True)
            torch.manual_seed(seed)
            model = build_model(arm, world.tokenizer.size)
            train(model, world, device, entries=entries_spec, steps=400, lr=1e-3,
                  batch_size=16, hard_k=0, seed=seed)
            encoder = WorldEncoder(world, model, device)

            store, ledger, index, vectors, pipeline = build_chain(
                world, encoder, minimum_score=0.40,
                minimum_coverage=0.60, known_margin=0.01,
            )
            id_to_label = {obj.object_id: obj.label for obj in world.objects}
            if arm == "unified":
                pipeline.reasoner = make_reasoner(world)
                pipeline.multihop = MultiHopResolver(TransformationGraph(world))
                raw_meta = {obj["id"]: obj.get("meta") or {} for obj in raw["objects"]}
                pipeline.access_gate = ObserverGate(
                    company_of=lambda oid: raw_meta.get(oid, {}).get("company"),
                    region_of=lambda oid: raw_meta.get(oid, {}).get("region"),
                )

            @torch.inference_mode()
            def resolve_arm(text: str, query_type: str | None = None):
                started = time.perf_counter()
                if arm == "unified":
                    result = pipeline.resolve(text, query_type=query_type)
                    ms = (time.perf_counter() - started) * 1000
                    return (result.status.value, result.object_id, ms)

                tokens, coverage = world.encode_query(text)
                query_vector = encoder.encode_query(text)[0]
                if arm == "b4_fts_rag":
                    from cformer_real import apply_aliases
                    lowered = apply_aliases(text, world.company_aliases,
                                            world.series_aliases)
                    keywords = re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]{3,}", lowered)
                    match_query = " OR ".join(f'"{token}"' for token in keywords) or None
                    fts_ids = store.fts_candidates(lowered, limit=64,
                                                   match_query=match_query)
                    labels = [id_to_label[i] for i in fts_ids if i in id_to_label]
                    ranked_labels, ranked_scores = bank_scores(vectors, query_vector, labels)
                else:
                    ranked_labels, ranked_scores = bank_scores(vectors, query_vector)
                status, object_label, score, _margin = decide(
                    VERIFIER, ranked_labels[:16], ranked_scores[:16],
                    float(coverage), query_type,
                )
                ms = (time.perf_counter() - started) * 1000
                object_id = (world.objects[object_label].object_id
                             if object_label is not None else None)
                return (status.value, object_id, ms)

            def hit(text: str, expected: str | None, query_type="known"):
                status, object_id, _ms = resolve_arm(text, query_type)
                return int(status == "supported" and object_id == expected)

            latencies: list[float] = []
            known_hits = []
            for query in world.known_queries():
                status, object_id, ms = resolve_arm(query["text"], "known")
                latencies.append(ms)
                known_hits.append(int(status == "supported"
                                      and object_id == query["target_id"]))
            heldout_hits = [
                hit(q["text"], q["target_id"]) for q in heldout
            ]
            blind_known, blind_amb, blind_unk = [], [], []
            for query in blind["queries"]:
                status, object_id, ms = resolve_arm(query["text"], query.get("kind"))
                latencies.append(ms)
                if query["kind"] == "known":
                    blind_known.append(int(status == "supported"
                                           and object_id == query["target_id"]))
                elif query["kind"] == "ambiguous":
                    blind_amb.append(int(status == "ambiguous"))
                else:
                    blind_unk.append(int(status != "supported"))

            leaks = 0
            vacuous_ok = 0
            for item in asof_set:
                status, object_id, _ms = resolve_arm(item["text"], "known")
                if status == "supported" and object_id is not None:
                    if year_by_id[object_id] is not None and year_by_id[object_id] > item["as_of"]:
                        leaks += 1
            for item in vacuous_set:
                status, object_id, _ms = resolve_arm(item["text"])
                vacuous_ok += int(status != "supported" and object_id is None)

            params = sum(p.numel() for p in model.parameters())
            results[arm] = {
                "main_known_top1": statistics.mean(known_hits),
                "heldout_top1": statistics.mean(heldout_hits) if heldout_hits else 0.0,
                "blind_known_top1": statistics.mean(blind_known) if blind_known else 0.0,
                "blind_ambiguous_detected": statistics.mean(blind_amb) if blind_amb else 0.0,
                "blind_unknown_blocked": statistics.mean(blind_unk) if blind_unk else 0.0,
                "temporal_future_leakage": leaks / len(asof_set),
                "vacuous_explicit_rejection": vacuous_ok / len(vacuous_set),
                "params": params,
                "p50_latency_ms": statistics.quantiles(latencies, n=10)[4],
                "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
            }
            store.close()
            print(json.dumps({"phase": "seed-arm", "seed": seed, "arm": arm,
                              "main_known_top1": results[arm]["main_known_top1"],
                              "leakage": results[arm]["temporal_future_leakage"]},
                             ensure_ascii=False), flush=True)
        per_seed[str(seed)] = results

    aggregate = {
        arm: {key: statistics.mean(per_seed[s][arm][key] for s in per_seed)
              for key in per_seed["1"][arm]}
        for arm in arms
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "protocol": ("同候选库/同训练条目(steps=400)/同 EvidenceVerifier 决策层；"
                     "差异仅在编码器架构(b1-b3)或检索路径(b4/unified)。"),
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    out = ROOT / "artifacts" / "v65_baseline_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done",
                      "arms": {arm: {"top1": aggregate[arm]["main_known_top1"],
                                     "leak": aggregate[arm]["temporal_future_leakage"]}
                               for arm in arms}}, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
