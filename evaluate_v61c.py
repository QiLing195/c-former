# -*- coding: utf-8 -*-
"""V6.1c 端到端链路评测：统一存储 + IVF 粗召回 + 重排 + verifier + ledger。

用法：
    D:\conda\envs\cformer-gpu\python.exe evaluate_v61c.py

在 212 对象真实 AI 模型库上验证检索链路的正确性（不是速度）：
精确命中路径、ANN Recall@Top256 与穷举一致性、分类型 margin 决策、
未知短文本进 ledger 且绝不被自动 verified、延迟分解。
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from cformer_v59 import RECOMMENDED_MARGINS_V60B, CandidateLedger, CandidateStatus, EvidenceVerifier
from cformer_v60 import ChineseTransformerConfig, TokenCFormerResolver
from cformer_v61 import IVFConfig, IVFIndex
from cformer_v61c import ObjectRecord, UnifiedObjectStore, UnifiedResolutionPipeline
from cformer_real import AIModelWorld
from train_eval_real import build_training_entries, split_known, train

ROOT = Path(__file__).resolve().parent


class WorldEncoder:
    """Adapts AIModelWorld tokenizer + frozen V6.0 encoder to the pipeline API."""

    def __init__(self, world: AIModelWorld, model: TokenCFormerResolver, device: torch.device):
        self.world = world
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode_query(self, text: str) -> tuple[torch.Tensor, float]:
        tokens, coverage = self.world.encode_query(text)
        vector = self.model.encode_query(tokens[None].to(self.device))[0]
        return vector.cpu(), coverage

    @torch.inference_mode()
    def encode_objects(self) -> torch.Tensor:
        candidates = self.world.encode_candidates(self.world.objects)
        return self.model.encode_candidate(candidates.to(self.device)).cpu()

    def object_id_of(self, label: int) -> str:
        return self.world.objects[label].object_id


def build_chain(world: AIModelWorld, encoder: WorldEncoder):
    store = UnifiedObjectStore()
    for obj in world.objects:
        evidence = dict(zip(("名称", "属性", "关系", "变化"), obj.evidence))
        record = ObjectRecord(
            object_id=obj.object_id,
            canonical_name=obj.name,
            document=evidence,
            meta={},
        )
        aliases = [obj.name]
        if " " in obj.name:
            aliases.append(obj.name.replace(" ", ""))
        store.upsert_object(record, aliases)

    vectors = encoder.encode_objects()
    dimension = vectors.shape[1]
    index = IVFIndex(dimension, IVFConfig(
        n_centroids=min(16, vectors.shape[0]), n_iter=8, seed=61,
    ))
    index.train(vectors)
    index.add(vectors, list(range(len(world.objects))))

    ledger = CandidateLedger()
    verifier = EvidenceVerifier(
        minimum_score=0.50, minimum_margin=0.08, minimum_coverage=0.60,
        margin_by_type=RECOMMENDED_MARGINS_V60B,
    )
    pipeline = UnifiedResolutionPipeline(store, ledger, index, encoder, verifier,
                                         nprobe=min(16, max(1, vectors.shape[0] // 8)))
    return store, ledger, index, vectors, pipeline


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    world = AIModelWorld(ROOT / "data" / "ai_models_dataset.json")
    blind = json.loads((ROOT / "data" / "ai_models_blindset.json").read_text(encoding="utf-8"))
    known_train, known_heldout = split_known(world.known_queries())
    config = ChineseTransformerConfig(
        world.tokenizer.size, layers=2, d_model=64, heads=4,
        ffn_dimensions=128, output_dimensions=32,
    )

    per_seed = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = TokenCFormerResolver(config)
        entries = build_training_entries(world, known_train)
        train(model, world, device, entries=entries, steps=400, lr=1e-3,
              batch_size=16, hard_k=0, seed=seed)
        encoder = WorldEncoder(world, model, device)
        store, ledger, index, vectors, pipeline = build_chain(world, encoder)

        def exhaustive_top1(text: str) -> int:
            query, _ = encoder.encode_query(text)
            return int((vectors @ query).argmax())

        # 主数据 known 查询走完整链路
        rows = []
        for query in world.known_queries():
            result = pipeline.resolve(query["text"], query_type="known")
            target_label = world.target_label(query["target_id"])
            rows.append({
                "hit": result.status == CandidateStatus.SUPPORTED
                       and result.object_id == query["target_id"],
                "supported": result.status == CandidateStatus.SUPPORTED,
                "path": result.path,
                "chain_agrees_exhaustive":
                    result.status == CandidateStatus.SUPPORTED
                    and result.object_id == world.objects[exhaustive_top1(query["text"])].object_id,
                "latency_ms": sum(result.stage_ms.values()),
            })
        # 盲测集走完整链路（oracle intent 类型）
        blind_rows = []
        for query in blind["queries"]:
            result = pipeline.resolve(query["text"], query.get("kind"))
            item = {"kind": query["kind"], "status": result.status.value,
                    "path": result.path, "proposed": result.proposed_alias}
            if query["kind"] == "known":
                item["hit"] = (result.status == CandidateStatus.SUPPORTED
                               and result.object_id == query["target_id"])
            blind_rows.append(item)

        per_seed[str(seed)] = {
            "main_known_top1": statistics.mean(r["hit"] for r in rows),
            "main_supported_coverage": statistics.mean(r["supported"] for r in rows),
            "exact_path_share": statistics.mean(r["path"] == "exact" for r in rows),
            "ann_recall_target_in_candidates": _ann_recall(pipeline, world),
            "chain_vs_exhaustive_agreement": (
                statistics.mean(r["chain_agrees_exhaustive"] for r in rows if r["supported"])
                if any(r["supported"] for r in rows) else 0.0
            ),
            "heldout_top1_oracle_intent": statistics.mean(
                (lambda r: r.status == CandidateStatus.SUPPORTED
                 and r.object_id == world.objects[world.target_label(q["target_id"])].object_id)(
                    pipeline.resolve(q["text"], "known"))
                for q in known_heldout
            ),
            "blind": {
                kind: {
                    "n": sum(1 for r in blind_rows if r["kind"] == kind),
                    **_blind_metrics(blind_rows, kind),
                }
                for kind in ("known", "ambiguous", "unknown")
            },
            "ledger_proposals": ledger.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE status='proposed'").fetchone()[0],
            "ledger_auto_verified": ledger.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE status='verified'").fetchone()[0],
            "p50_latency_ms": statistics.quantiles([r["latency_ms"] for r in rows], n=10)[4]
            if len(rows) > 1 else rows[0]["latency_ms"],
            "p95_latency_ms": sorted(r["latency_ms"] for r in rows)[int(0.95 * len(rows)) - 1],
        }
        print(json.dumps({"phase": "seed", "seed": seed,
                          "main_known_top1": per_seed[str(seed)]["main_known_top1"],
                          "exact_share": per_seed[str(seed)]["exact_path_share"],
                          "p95_ms": per_seed[str(seed)]["p95_latency_ms"]}, ensure_ascii=False))
        store.close()

    aggregate = {
        key: statistics.mean(per_seed[s][key] for s in per_seed)
        for key in ("main_known_top1", "main_supported_coverage", "exact_path_share",
                    "ann_recall_target_in_candidates", "chain_vs_exhaustive_agreement",
                    "heldout_top1_oracle_intent", "p50_latency_ms", "p95_latency_ms")
    }
    payload = {
        "environment": {"device": str(device), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "note": ("V6.1c 链路正确性验证：212 对象真实库，verifier 使用 V6.0b 分类型 margin；"
                 "intent 类型为 oracle（评测上下文），生产需意图分类器。规模太小，"
                 "ANN/FTS 的性能优势不在本测范围内。"),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out = ROOT / "artifacts" / "v61c_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "done", "aggregate": aggregate}, ensure_ascii=False))
    print(f"wrote {out}")


def _ann_recall(pipeline: UnifiedResolutionPipeline, world: AIModelWorld) -> float:
    hits = 0
    for query in world.known_queries():
        vector, _ = pipeline.encoder.encode_query(query["text"])
        _, ids = pipeline.index.search(vector, pipeline.nprobe, pipeline.top_ann)
        target = world.target_label(query["target_id"])
        hits += int(target in ids.tolist())
    return hits / max(1, len(world.known_queries()))


def _blind_metrics(rows: list[dict], kind: str) -> dict:
    subset = [r for r in rows if r["kind"] == kind]
    if not subset:
        return {}
    out: dict[str, float] = {}
    if kind == "known":
        out["top1"] = statistics.mean(bool(r.get("hit")) for r in subset)
    if kind == "ambiguous":
        out["detected"] = statistics.mean(r["status"] == "ambiguous" for r in subset)
    if kind == "unknown":
        out["not_supported"] = statistics.mean(r["status"] != "supported" for r in subset)
        out["proposed_share"] = statistics.mean(bool(r["proposed"]) for r in subset)
    return out


if __name__ == "__main__":
    main()
